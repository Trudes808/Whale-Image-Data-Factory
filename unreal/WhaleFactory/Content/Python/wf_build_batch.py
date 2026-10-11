"""Turn a render batch (factory/shotlist.py) into one level sequence, LS_Batch, so Movie
Render Queue renders all its shots in a single game session.

    WF_BATCH_JSON=data/renders/run01/batches/batch_000.json tools/run_editor_script.sh build_batch

The batch's shared sea state and water look are applied to the level. Each shot gets
``frames_per_shot`` frames and its own camera cut, so MRQ warms up per shot (exposure,
Lumen, water) and nothing bleeds between shots. Each shot gets its own whale actor
(its deformed mesh from factory/deform.py, written by tools/render_batch.sh, and its skin
material instance), shown only in its frames. Keyed per shot, with constant
interpolation: camera transform, focal length, exposure bias, sun rotation and
intensity, fog density, and the visibility of that shot's whale and splash meshes. tools/render_batch.sh renders the result; the last
frame of each shot is the one to keep.
"""

import json
import os
import traceback

import unreal

import wf_shot

MAP_PATH = "/Game/WhaleFactory/Maps/OceanPOC"
SEQUENCE_PATH = "/Game/WhaleFactory/Sequences/LS_Batch"
CONSTANT = unreal.MovieSceneKeyInterpolation.CONSTANT


def log(message: str) -> None:
    unreal.log(f"[wf_build_batch] {message}")


def frame(n: int) -> unreal.FrameNumber:
    return unreal.FrameNumber(int(n))


def channels_by_name(section) -> dict:
    return {str(channel.get_editor_property("channel_name")): channel for channel in section.get_all_channels()}


def add_transform_track(sequence, binding, total: int):
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(0, total)
    return channels_by_name(section)


def add_float_track(binding, property_name: str, total: int):
    track = binding.add_track(unreal.MovieSceneFloatTrack)
    track.set_property_name_and_path(property_name, property_name)
    section = track.add_section()
    section.set_range(0, total)
    return section.get_all_channels()[0]


def add_visibility_track(binding, total: int):
    track = binding.add_track(unreal.MovieSceneVisibilityTrack)
    track.set_property_name_and_path("bHidden", "bHidden")
    section = track.add_section()
    section.set_range(0, total)
    return section.get_all_channels()[0]


def key_transform(channels: dict, at: int, location, rotation: dict) -> None:
    values = {
        "Location.X": location[0], "Location.Y": location[1], "Location.Z": location[2],
        "Rotation.X": rotation.get("roll", 0.0), "Rotation.Y": rotation.get("pitch", 0.0), "Rotation.Z": rotation.get("yaw", 0.0),
    }
    for name, value in values.items():
        channels[name].add_key(frame(at), float(value), interpolation=CONSTANT)


def main() -> None:
    path = os.environ.get("WF_BATCH_JSON")
    if not path:
        raise RuntimeError("WF_BATCH_JSON is not set")
    with open(path) as handle:
        batch = json.load(handle)
    shots = batch["shots"]
    per_shot = int(batch.get("frames_per_shot", 3))
    total = per_shot * len(shots)
    log(f"batch {batch['index']}: {len(shots)} shots, sea {batch['sea_state']}, look {batch['look']}, {total} frames")

    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not levels.load_level(MAP_PATH):
        raise RuntimeError(f"could not load {MAP_PATH}")
    wf_shot.apply_ocean_look(batch["ocean"])
    wf_shot.apply_waves(batch["waves"])
    wf_shot.clear_splash()
    wf_shot.clear_whale_variants()

    whale = wf_shot.find_actor(wf_shot.WHALE_LABEL)
    camera = wf_shot.find_actor(wf_shot.CAMERA_LABEL)
    sun = wf_shot.find_actor(wf_shot.SUN_LABEL)
    fog = wf_shot.find_actor(wf_shot.FOG_LABEL)
    cine = camera.get_cine_camera_component()
    post = cine.get_editor_property("post_process_settings")
    post.set_editor_property("override_auto_exposure_bias", True)
    cine.set_editor_property("post_process_settings", post)

    if unreal.EditorAssetLibrary.does_asset_exist(SEQUENCE_PATH):
        unreal.EditorAssetLibrary.delete_asset(SEQUENCE_PATH)
    folder, name = SEQUENCE_PATH.rsplit("/", 1)
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    sequence.set_display_rate(unreal.FrameRate(30, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(total)

    # Every shot gets its own whale actor (deformed mesh and skin), shown only in its frames;
    # the base whale stays hidden for the whole batch.
    whale.set_actor_hidden_in_game(True)
    camera_binding = sequence.add_possessable(camera)
    cine_binding = sequence.add_possessable(cine)
    sun_binding = sequence.add_possessable(sun)
    sun_light_binding = sequence.add_possessable(sun.get_component_by_class(unreal.DirectionalLightComponent))
    fog_binding = sequence.add_possessable(fog.get_component_by_class(unreal.ExponentialHeightFogComponent))
    camera_channels = add_transform_track(sequence, camera_binding, total)
    sun_channels = add_transform_track(sequence, sun_binding, total)
    focal = add_float_track(cine_binding, "CurrentFocalLength", total)
    exposure = add_float_track(cine_binding, "PostProcessSettings.AutoExposureBias", total)
    intensity = add_float_track(sun_light_binding, "Intensity", total)
    fog_density = add_float_track(fog_binding, "FogDensity", total)
    cuts = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    camera_id = sequence.get_binding_id(camera_binding)

    sprite_total = 0
    for index, shot in enumerate(shots):
        start = index * per_shot
        variant = wf_shot.spawn_whale_variant(shot, shot["id"], hidden=True)
        whale_visibility = add_visibility_track(sequence.add_possessable(variant), total)
        whale_visibility.add_key(frame(0), False)
        whale_visibility.add_key(frame(start), True)
        whale_visibility.add_key(frame(start + per_shot), False)
        key_transform(camera_channels, start, shot["camera"]["location"], shot["camera"]["rotation"])
        key_transform(sun_channels, start, [0.0, 0.0, 1000.0], {"roll": 0.0, "pitch": shot["sun"]["pitch"], "yaw": shot["sun"]["yaw"]})
        focal.add_key(frame(start), float(shot["camera"]["focal_mm"]), interpolation=CONSTANT)
        exposure.add_key(frame(start), float(shot.get("exposure_ev", 0.0)), interpolation=CONSTANT)
        intensity.add_key(frame(start), float(shot["sun"].get("intensity", 10.0)), interpolation=CONSTANT)
        fog_density.add_key(frame(start), float(shot["fog"]["density"]), interpolation=CONSTANT)
        cut = cuts.add_section()
        cut.set_range(start, start + per_shot)
        cut.set_camera_binding_id(camera_id)
        splash = shot.get("splash")
        if splash and splash.get("mesh_obj"):
            for actor in wf_shot.apply_splash(splash, label=shot["id"], hidden=True):
                visibility = add_visibility_track(sequence.add_possessable(actor), total)
                # Visibility track value: True = visible. Hidden everywhere except this shot.
                visibility.add_key(frame(0), False)
                visibility.add_key(frame(start), True)
                visibility.add_key(frame(start + per_shot), False)
                sprite_total += 1
    if not levels.save_current_level():
        raise RuntimeError("saving the level failed")
    unreal.EditorAssetLibrary.save_asset(SEQUENCE_PATH, only_if_is_dirty=False)
    log(f"saved {SEQUENCE_PATH}: {len(shots)} shots x {per_shot} frames, {sprite_total} splash actors")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_build_batch] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
