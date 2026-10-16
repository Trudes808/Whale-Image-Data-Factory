"""Turn a render batch (factory/shotlist.py) into one level sequence, LS_Batch, so Movie
Render Queue renders all its shots in a single game session.

    WF_BATCH_JSON=data/renders/run01/batches/batch_000.json tools/run_editor_script.sh build_batch

The batch's shared sea state and water look are applied to the level. Each shot gets
``frames_per_shot`` frames and its own camera cut, so MRQ warms up per shot (exposure,
Lumen, water) and nothing bleeds between shots. Each shot gets its own whale actor
(its deformed mesh from factory/deform.py, written by tools/render_batch.sh, and its skin
material instance), shown only in its frames. Keyed per shot, with constant
interpolation: camera transform, focal length, exposure bias, sun rotation and
intensity, fog density, focus distance and aperture (depth of field), the water's wave time
(a console-variable track setting r.Water.OverrideWavesTime, so each shot renders a known,
frozen wave phase) and the visibility of that shot's whale and splash meshes. Knowing the
phase, the whale, camera and splash are lifted onto the actual wave surface (ride_waves) and
the batch as rendered is written to WF_BATCH_ADJUSTED for the label pass. tools/render_batch.sh
renders the result; the last frame of each shot is the one to keep.
"""

import json
import math
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


def key_transform(channels: dict, at: int, location, rotation: dict, scale: float | None = None) -> None:
    values = {
        "Location.X": location[0], "Location.Y": location[1], "Location.Z": location[2],
        "Rotation.X": rotation.get("roll", 0.0), "Rotation.Y": rotation.get("pitch", 0.0), "Rotation.Z": rotation.get("yaw", 0.0),
    }
    if scale is not None:                      # a transform section without scale keys resets the actor's scale (the sky sphere needs its 20 km)
        values.update({"Scale.X": scale, "Scale.Y": scale, "Scale.Z": scale})
    for name, value in values.items():
        channels[name].add_key(frame(at), float(value), interpolation=CONSTANT)


GRAVITY_CM_S2 = 980.0


def read_gerstner_waves() -> list:
    """The waves the ocean renders (generated from the batch spectrum by wf_shot.apply_waves)."""
    asset = unreal.EditorAssetLibrary.load_asset(wf_shot.OCEAN_WAVES_ASSET)
    inner = wf_shot.asset_waves(asset)
    waves = []
    for wave in inner.get_editor_property("gerstner_waves"):
        direction = wave.get_editor_property("direction")
        waves.append({"wavelength": float(wave.get_editor_property("wave_length")),
                      "amplitude": float(wave.get_editor_property("amplitude")),
                      "steepness": float(wave.get_editor_property("steepness")),
                      "direction": [float(direction.x), float(direction.y)]})
    return waves


def gerstner_displacement(waves: list, x: float, y: float, t: float) -> tuple:
    """(dx, dy, dz) of the surface point whose rest position is (x, y) at time t, as
    GerstnerWaveFunctions.ush computes it (world cm, seconds)."""
    dx = dy = dz = 0.0
    for wave in waves:
        k = 2.0 * math.pi / wave["wavelength"]
        ux, uy = wave["direction"]
        phase = (x * ux + y * uy) * k - math.sqrt(k * GRAVITY_CM_S2) * t
        s, c = math.sin(phase), math.cos(phase)
        qa = wave["steepness"] / k           # q * amplitude, q = steepness / (amplitude * k)
        dx -= qa * s * ux
        dy -= qa * s * uy
        dz += wave["amplitude"] * c
    return dx, dy, dz


def surface_height(waves: list, x: float, y: float, t: float, iterations: int = 8) -> float:
    """Wave height (cm) above the mean plane at world (x, y): the horizontal Gerstner displacement
    is inverted by fixed-point iteration, as the label shader does (summed steepness < 1)."""
    qx, qy = x, y
    for _ in range(iterations):
        dx, dy, _ = gerstner_displacement(waves, qx, qy, t)
        qx, qy = x - dx, y - dy
    return gerstner_displacement(waves, qx, qy, t)[2]


def ride_waves(shot: dict, waves: list, t: float) -> dict:
    """factory/shotlist.py places the whale and its splash against the mean water plane. Lift the
    whale (and the camera with it, so the framing holds) by the wave height under the whale's
    above-water centroid, and the splash by the height at its centre, at the shot's wave time."""
    if shot.get("water_surface_z_cm"):
        return shot["water_surface_z_cm"]                  # factory.shotlist already placed it on the surface (factory.gerstner)
    aim = shot.get("stats", {}).get("aim_point") or shot["whale"]["location"]
    dz = surface_height(waves, float(aim[0]), float(aim[1]), t)
    shot["whale"]["location"][2] = round(shot["whale"]["location"][2] + dz, 2)
    shot["camera"]["location"][2] = round(shot["camera"]["location"][2] + dz, 2)
    surface = {"whale": round(dz, 2)}
    splash = shot.get("splash")
    if splash and splash.get("centre_cm"):
        ds = surface_height(waves, float(splash["centre_cm"][0]), float(splash["centre_cm"][1]), t)
        splash["centre_cm"][2] = round(splash["centre_cm"][2] + ds, 2)
        surface["splash"] = round(ds, 2)
    surface["editor_lifted"] = True                   # the splash actor still needs moving (its mesh was built on the mean plane)
    shot["water_surface_z_cm"] = surface
    return surface


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
    wf_shot.hide_all_skies()
    wf_shot.apply_islands(batch.get("islands"))
    wf_shot.clear_splash()
    wf_shot.clear_whale_variants()

    whale = wf_shot.find_actor(wf_shot.WHALE_LABEL)
    camera = wf_shot.find_actor(wf_shot.CAMERA_LABEL)
    sun = wf_shot.find_actor(wf_shot.SUN_LABEL)
    fog = wf_shot.find_actor(wf_shot.FOG_LABEL)
    cine = camera.get_cine_camera_component()
    wf_shot.unclamp_lens(cine)        # the focal-length track below must not be clamped at the lens's 1000 mm
    post = cine.get_editor_property("post_process_settings")
    post.set_editor_property("override_auto_exposure_bias", True)
    cine.set_editor_property("post_process_settings", post)
    # Depth of field like a field photo: manual focus, distance and f-number keyed per shot below.
    focus_settings = cine.get_editor_property("focus_settings")
    focus_settings.set_editor_property("focus_method", unreal.CameraFocusMethod.MANUAL)
    cine.set_editor_property("focus_settings", focus_settings)
    waves = read_gerstner_waves()
    log(f"{len(waves)} Gerstner waves read back, summed amplitude {sum(w['amplitude'] for w in waves):.1f} cm")

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
    # real skies: one sphere per HDRI, shown only in the shots that use it (and turned per shot); the
    # atmosphere and clouds shown only in the shots without one
    sky_specs = {s["sky"]["name"]: s["sky"] for s in shots if s.get("sky")}
    sky_tracks = {}
    for name, spec in sky_specs.items():
        sphere = wf_shot.ensure_sky_sphere(spec)
        sb = sequence.add_possessable(sphere)
        vis = add_visibility_track(sb, total); vis.add_key(frame(0), False)
        sky_tracks[name] = (vis, add_transform_track(sequence, sb, total))
    atmo_tracks = []
    if sky_specs:
        for a in wf_shot.atmosphere_actors():
            vis = add_visibility_track(sequence.add_possessable(a), total); vis.add_key(frame(0), True); atmo_tracks.append(vis)
    focal = add_float_track(cine_binding, "CurrentFocalLength", total)
    exposure = add_float_track(cine_binding, "PostProcessSettings.AutoExposureBias", total)
    intensity = add_float_track(sun_light_binding, "Intensity", total)
    fog_density = add_float_track(fog_binding, "FogDensity", total)
    focus_distance = add_float_track(cine_binding, "FocusSettings.ManualFocusDistance", total)
    aperture = add_float_track(cine_binding, "CurrentAperture", total)
    # The water's wave time, frozen per shot (r.Water.OverrideWavesTime drives the Time the water
    # material and the label pass evaluate the Gerstner waves at), so ride_waves knows the surface.
    cvars = sequence.add_track(unreal.MovieSceneCVarTrack)
    cuts = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    camera_id = sequence.get_binding_id(camera_binding)

    sprite_total = 0
    for index, shot in enumerate(shots):
        start = index * per_shot
        wave_time = float(shot.get("wave_time_s", 0.0))
        cvar = cvars.add_section()
        cvar.set_range(start, start + per_shot)
        cvar.set_from_string(f"r.Water.OverrideWavesTime={wave_time:.3f}")   # the same frozen time as DefaultEngine.ini (the track alone never reached the water)
        surface = ride_waves(shot, waves, wave_time)
        variant = wf_shot.spawn_whale_variant(shot, shot["id"], hidden=True)
        whale_visibility = add_visibility_track(sequence.add_possessable(variant), total)
        companion = shot.get("companion")
        if companion:
            comp_actor = wf_shot.spawn_whale_variant(companion, f"{shot['id']}c", hidden=True, stencil=wf_shot.COMPANION_STENCIL)
            comp_visibility = add_visibility_track(sequence.add_possessable(comp_actor), total)
            comp_visibility.add_key(frame(0), False)
            comp_visibility.add_key(frame(start), True)
            comp_visibility.add_key(frame(start + per_shot), False)
        if shot.get("skin_roughness_scale") is not None:
            try:
                comp_binding = sequence.add_possessable(variant.static_mesh_component)
                mat_track = comp_binding.add_track(unreal.MovieSceneComponentMaterialTrack)
                info = unreal.ComponentMaterialInfo()
                info.set_editor_property("material_slot_index", 0)
                info.set_editor_property("material_type", unreal.ComponentMaterialType.INDEXED_MATERIAL)
                mat_track.set_material_info(info)
                mat_section = mat_track.add_section(); mat_section.set_range(0, total)
                mat_section.add_scalar_parameter_key(unreal.MaterialParameterInfo(name="RoughnessScale"), frame(0), float(shot["skin_roughness_scale"]), "", "")
            except Exception as exc:
                log(f"skin wetness track failed for {shot['id']}: {exc}")
        whale_visibility.add_key(frame(0), False)
        whale_visibility.add_key(frame(start), True)
        whale_visibility.add_key(frame(start + per_shot), False)
        key_transform(camera_channels, start, shot["camera"]["location"], shot["camera"]["rotation"])
        key_transform(sun_channels, start, [0.0, 0.0, 1000.0], {"roll": 0.0, "pitch": shot["sun"]["pitch"], "yaw": shot["sun"]["yaw"]})
        sky = shot.get("sky")
        if sky and sky["name"] in sky_tracks:
            vis, channels = sky_tracks[sky["name"]]
            vis.add_key(frame(start), True); vis.add_key(frame(start + per_shot), False)
            key_transform(channels, start, [0.0, 0.0, 0.0], {"roll": 0.0, "pitch": 0.0, "yaw": float(sky.get("rotation_deg", 0.0))}, scale=wf_shot.SKY_SPHERE_SCALE)
        for vis in atmo_tracks:
            vis.add_key(frame(start), not bool(sky)); vis.add_key(frame(start + per_shot), True)
        focal.add_key(frame(start), float(shot["camera"]["focal_mm"]), interpolation=CONSTANT)
        exposure.add_key(frame(start), float(shot.get("exposure_ev", 0.0)), interpolation=CONSTANT)
        intensity.add_key(frame(start), float(shot["sun"].get("intensity", 10.0)), interpolation=CONSTANT)
        fog_density.add_key(frame(start), float(shot["fog"]["density"]), interpolation=CONSTANT)
        focus_distance.add_key(frame(start), 100.0 * float(shot["camera"].get("focus_distance_m", shot["camera"].get("distance_m", 100.0))),
                               interpolation=CONSTANT)
        aperture.add_key(frame(start), float(shot["camera"].get("aperture_f", 8.0)), interpolation=CONSTANT)
        cut = cuts.add_section()
        cut.set_range(start, start + per_shot)
        cut.set_camera_binding_id(camera_id)
        splash = shot.get("splash")
        ww_actors = wf_shot.apply_whitewater(shot.get("whitewater"), label=shot["id"], hidden=True)
        for actor in ww_actors:
            visibility = add_visibility_track(sequence.add_possessable(actor), total)
            visibility.add_key(frame(0), False)
            visibility.add_key(frame(start), True)
            visibility.add_key(frame(start + per_shot), False)
            sprite_total += 1
        if splash and splash.get("mesh_obj"):
            for actor in wf_shot.apply_splash(splash, label=shot["id"], hidden=True):
                if surface.get("editor_lifted") and "splash" in surface:
                    actor.set_actor_location(unreal.Vector(0.0, 0.0, float(surface["splash"])), False, False)
                visibility = add_visibility_track(sequence.add_possessable(actor), total)
                # Visibility track value: True = visible. Hidden everywhere except this shot.
                visibility.add_key(frame(0), False)
                visibility.add_key(frame(start), True)
                visibility.add_key(frame(start + per_shot), False)
                sprite_total += 1
    if not levels.save_current_level():
        raise RuntimeError("saving the level failed")
    unreal.EditorAssetLibrary.save_asset(SEQUENCE_PATH, only_if_is_dirty=False)
    adjusted = os.environ.get("WF_BATCH_ADJUSTED")
    if adjusted:
        # The batch as rendered (whale, camera and splash lifted onto the wave surface), for the label pass.
        with open(os.environ.get("WF_BATCH_SOURCE") or path) as handle:
            output = json.load(handle)
        by_id = {shot["id"]: shot for shot in shots}
        for shot in output["shots"]:
            source = by_id[shot["id"]]
            shot["whale"]["location"] = source["whale"]["location"]
            shot["camera"]["location"] = source["camera"]["location"]
            if shot.get("splash") and source.get("splash"):
                shot["splash"]["centre_cm"] = source["splash"]["centre_cm"]
            shot["water_surface_z_cm"] = source["water_surface_z_cm"]
            shot["waves_rendered"] = waves
        with open(adjusted, "w") as handle:
            json.dump(output, handle, indent=1)
        log(f"wrote the adjusted batch to {adjusted}")
    log(f"saved {SEQUENCE_PATH}: {len(shots)} shots x {per_shot} frames, {sprite_total} splash actors")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_build_batch] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
