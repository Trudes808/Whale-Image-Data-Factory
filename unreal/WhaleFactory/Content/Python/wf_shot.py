"""Apply a "shot" (whale pose, camera, ocean look, sun) to the loaded OceanPOC level.

Shared by wf_build_scene.py and wf_apply_shot.py. A shot is a JSON dict, normally written
by ``python3 -m factory.poses``; every key is optional:

    {
      "whale":  {"location": [x, y, z], "rotation": {"roll": 0, "pitch": 0, "yaw": 20}},
      "camera": {"location": [...], "rotation": {...}, "focal_mm": 85},
      "ocean":  {"scalars": {"Water Roughness": 0.02}, "vectors": {"Absorption": [r, g, b, a]},
                 "switches": {"Enable Ocean Foam": true}},
      "sun":    {"pitch": -35, "yaw": 40, "intensity": 10},
      "fog":    {"density": 0.002},
      "splash": {"kind": "crown", "mesh_obj": "/abs/path/to/000123.obj", ...},
      "waves":  {"num_waves": 32, "min_wavelength": 120, ...}   # GerstnerWaterWaveGeneratorSimple
    }

A splash is a mesh (factory/splash.py) rendered with M_Splash (what the camera sees) plus
a hidden twin with M_SplashCoverage that only writes custom depth with stencil
SPRAY_STENCIL, so dense white water in front of the whale labels as background, like any
other occluder.
"""

import json
import os

import unreal

WHALE_LABEL = "Whale"
WHALE_VARIANT_PREFIX = "WhaleVar_"
WHALE_MESH_DIR = "/Game/ThirdParty/Humpback/Gen"   # per-shot deformed whale meshes (gitignored)
HUMPBACK_DIR = "/Game/ThirdParty/Humpback"
WHALE_STENCIL = 1
CAMERA_LABEL = "BoatCamera"
SUN_LABEL = "Sun"
FOG_LABEL = "HeightFog"
OCEAN_LABEL = "Ocean"
OCEAN_MATERIAL_INSTANCE = "/Game/WhaleFactory/Water/MI_OceanDeep"
OCEAN_WAVES_ASSET = "/Game/WhaleFactory/Water/WW_Ocean"
SPRAY_DIR = "/Game/WhaleFactory/Splash"
SPLASH_MESH_DIR = f"{SPRAY_DIR}/Gen"  # imported per-shot splash meshes (gitignored)
SPRAY_MATERIAL = f"{SPRAY_DIR}/M_Splash"
SPRAY_COVERAGE_MATERIAL = f"{SPRAY_DIR}/M_SplashCoverage"
SPRAY_LABEL_PREFIX = "Spray_"
SPRAY_STENCIL = 2
SPRAY_COVERAGE_ALPHA = 0.35  # splash alpha above which white water counts as hiding the whale
MESH_IMPORT_PIPELINE = "/Game/WhaleFactory/Import/IP_StaticMeshNoNanite"
ENGINE_OBJ_PIPELINE = "/Interchange/Pipelines/DefaultFBXOBJAssetsPipeline"


def mesh_import_options() -> unreal.InterchangePipelineStackOverride:
    """Interchange pipeline for static meshes with Nanite off.

    Interchange builds Nanite by default, but we render with Vulkan SM5 (no Nanite), where a
    Nanite mesh draws as its coarse fallback. The mesh's Nanite settings are not scriptable
    after import, so disable it at import time with a copy of the engine's FBX/OBJ pipeline.
    """
    if not unreal.EditorAssetLibrary.does_asset_exist(MESH_IMPORT_PIPELINE):
        if not unreal.EditorAssetLibrary.duplicate_asset(ENGINE_OBJ_PIPELINE, MESH_IMPORT_PIPELINE):
            raise RuntimeError(f"could not duplicate {ENGINE_OBJ_PIPELINE}")
        pipeline = unreal.EditorAssetLibrary.load_asset(MESH_IMPORT_PIPELINE)
        pipeline.get_editor_property("mesh_pipeline").set_editor_property("build_nanite", False)
        pipeline.get_editor_property("mesh_pipeline").set_editor_property("import_static_meshes", True)
        unreal.EditorAssetLibrary.save_asset(MESH_IMPORT_PIPELINE, only_if_is_dirty=False)
    override = unreal.InterchangePipelineStackOverride()
    override.add_pipeline(unreal.EditorAssetLibrary.load_asset(MESH_IMPORT_PIPELINE))
    return override


def import_asset(filename, destination_dir: str, name: str, options=None) -> unreal.Object:
    task = unreal.AssetImportTask()
    if options is not None:
        task.set_editor_property("options", options)
    task.set_editor_property("filename", str(filename))
    task.set_editor_property("destination_path", destination_dir)
    task.set_editor_property("destination_name", name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported = [str(p) for p in task.get_editor_property("imported_object_paths")]
    asset = unreal.EditorAssetLibrary.load_asset(f"{destination_dir}/{name}")
    if asset is None:
        # Interchange names meshes after the OBJ's objects, not the destination name, and
        # also creates placeholder materials. Take the one static mesh and rename it.
        meshes = [p for p in imported if isinstance(unreal.EditorAssetLibrary.load_asset(p), unreal.StaticMesh)]
        if len(meshes) != 1:
            raise RuntimeError(f"import of {filename} failed; expected one static mesh, got {imported}")
        if not unreal.EditorAssetLibrary.rename_asset(meshes[0], f"{destination_dir}/{name}"):
            raise RuntimeError(f"could not rename {meshes[0]} to {name}")
        asset = unreal.EditorAssetLibrary.load_asset(f"{destination_dir}/{name}")
        unreal.EditorAssetLibrary.save_asset(f"{destination_dir}/{name}", only_if_is_dirty=False)
    return asset


def log(message: str) -> None:
    unreal.log(f"[wf_shot] {message}")


def load_shot_from_env() -> dict | None:
    path = os.environ.get("WF_SHOT_JSON")
    if not path:
        return None
    with open(path) as handle:
        shot = json.load(handle)
    log(f"loaded shot {shot.get('name', '?')} from {path}")
    return shot


def find_actor(label: str) -> unreal.Actor:
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    for actor in actors:
        if actor.get_actor_label() == label:
            return actor
    raise RuntimeError(f"no actor labelled {label!r} in the level")


def _vector(values) -> unreal.Vector:
    return unreal.Vector(float(values[0]), float(values[1]), float(values[2]))


def _rotator(values: dict) -> unreal.Rotator:
    return unreal.Rotator(roll=float(values.get("roll", 0.0)), pitch=float(values.get("pitch", 0.0)),
                          yaw=float(values.get("yaw", 0.0)))


def apply_transform(actor: unreal.Actor, spec: dict) -> None:
    if "location" in spec:
        actor.set_actor_location(_vector(spec["location"]), False, False)
    if "rotation" in spec:
        actor.set_actor_rotation(_rotator(spec["rotation"]), False)


def apply_camera(camera: unreal.Actor, spec: dict) -> None:
    apply_transform(camera, spec)
    if "focal_mm" in spec:
        camera.get_cine_camera_component().set_editor_property("current_focal_length", float(spec["focal_mm"]))


def apply_ocean_look(spec: dict, instance_path: str = OCEAN_MATERIAL_INSTANCE) -> None:
    """Set parameters on the ocean's material instance (made by wf_build_scene)."""
    instance = unreal.EditorAssetLibrary.load_asset(instance_path)
    if instance is None:
        raise RuntimeError(f"{instance_path} missing; run tools/run_editor_script.sh build_scene first")
    mel = unreal.MaterialEditingLibrary
    # The MaterialEditingLibrary setters always return False (UE 5.8), so check names
    # against the base material instead.
    base = instance.get_base_material()
    known = {
        "scalars": set(map(str, mel.get_scalar_parameter_names(base))),
        "vectors": set(map(str, mel.get_vector_parameter_names(base))),
        "switches": set(map(str, mel.get_static_switch_parameter_names(base))),
    }
    for kind in known:
        unknown = sorted(set(spec.get(kind, {})) - known[kind])
        if unknown:
            log(f"warning: {kind} not on the ocean material, ignored: {unknown}")
    for name, value in spec.get("scalars", {}).items():
        mel.set_material_instance_scalar_parameter_value(instance, name, float(value))
    for name, value in spec.get("vectors", {}).items():
        mel.set_material_instance_vector_parameter_value(instance, name, unreal.LinearColor(*[float(v) for v in value]))
    for name, value in spec.get("switches", {}).items():
        mel.set_material_instance_static_switch_parameter_value(instance, name, bool(value))
    mel.update_material_instance(instance)
    log(f"ocean look applied: {sorted(spec.get('scalars', {}))} {sorted(spec.get('vectors', {}))} {sorted(spec.get('switches', {}))}")
    unreal.EditorAssetLibrary.save_asset(instance_path, only_if_is_dirty=False)


def asset_waves(asset: unreal.Object) -> unreal.Object:
    """The GerstnerWaterWaves a WaterWavesAsset actually uses. Not find_object(asset,
    "WaterWaves"): a duplicated asset also carries the constructor's default subobject of
    that name, which the asset's property does not point to. The property is not
    Blueprint-visible, but set/get_editor_property accept its raw C++ name."""
    return asset.get_editor_property("WaterWaves")


def apply_waves(params: dict) -> None:
    """Set the ocean's Gerstner spectrum (GerstnerWaterWaveGeneratorSimple parameters).

    The ocean references WW_Ocean through the plugin's own WaterWavesAssetReference (set up
    by wf_build_scene with the SET console command), so editing the asset's inner
    GerstnerWaterWaves and saving it is all that is needed. Never hand the inner object to
    the ocean directly: that moves it out of the asset.
    """
    asset = unreal.EditorAssetLibrary.load_asset(OCEAN_WAVES_ASSET)
    inner = asset_waves(asset) if asset else None
    if not isinstance(inner, unreal.GerstnerWaterWaves):
        raise RuntimeError(f"no GerstnerWaterWaves inside {OCEAN_WAVES_ASSET}: {inner!r}; rebuild the scene")
    generator = unreal.new_object(unreal.GerstnerWaterWaveGeneratorSimple, outer=inner)
    for name, value in params.items():
        generator.set_editor_property(name, value)
    inner.set_editor_property("gerstner_wave_generator", generator)  # recomputes the waves
    if not unreal.EditorAssetLibrary.save_asset(OCEAN_WAVES_ASSET, only_if_is_dirty=False):
        raise RuntimeError(f"could not save {OCEAN_WAVES_ASSET}")
    log(f"waves: {params}; max wave height {inner.get_editor_property('max_wave_height'):.0f} cm")


def apply_sun(spec: dict) -> None:
    sun = find_actor(SUN_LABEL)
    rotation = sun.get_actor_rotation()
    sun.set_actor_rotation(unreal.Rotator(roll=0.0, pitch=float(spec.get("pitch", rotation.pitch)),
                                          yaw=float(spec.get("yaw", rotation.yaw))), False)
    if "intensity" in spec:
        sun.get_component_by_class(unreal.DirectionalLightComponent).set_editor_property("intensity", float(spec["intensity"]))


def apply_fog(spec: dict) -> None:
    fog = find_actor(FOG_LABEL).get_component_by_class(unreal.ExponentialHeightFogComponent)
    if "density" in spec:
        fog.set_editor_property("fog_density", float(spec["density"]))


def clear_splash() -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    old = [a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith(SPRAY_LABEL_PREFIX)]
    for actor in old:
        subsystem.destroy_actor(actor)
    if unreal.EditorAssetLibrary.does_directory_exist(SPLASH_MESH_DIR):
        unreal.EditorAssetLibrary.delete_directory(SPLASH_MESH_DIR)
    if old:
        log(f"removed {len(old)} splash actors")


def apply_splash(spec: dict | None, label: str = "shot", hidden: bool = False) -> list:
    """Import the shot's splash mesh and spawn it twice: visible (M_Splash) and the hidden
    coverage twin (M_SplashCoverage, custom depth only). Returns the two actors."""
    if not spec or not spec.get("mesh_obj"):
        return []
    visible = unreal.EditorAssetLibrary.load_asset(SPRAY_MATERIAL)
    coverage = unreal.EditorAssetLibrary.load_asset(SPRAY_COVERAGE_MATERIAL)
    if visible is None or coverage is None:
        raise RuntimeError("splash materials missing; run tools/run_editor_script.sh build_scene first")
    mesh = import_asset(spec["mesh_obj"], SPLASH_MESH_DIR, f"SM_Splash_{label}", options=mesh_import_options())
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"splash mesh import failed for {spec['mesh_obj']}: {mesh!r}")
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    for suffix, material, is_coverage in (("v", visible, False), ("c", coverage, True)):
        actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator())
        actor.set_actor_label(f"{SPRAY_LABEL_PREFIX}{label}{suffix}")
        if hidden:
            actor.set_actor_hidden_in_game(True)
        smc = actor.static_mesh_component
        smc.set_mobility(unreal.ComponentMobility.MOVABLE)
        smc.set_static_mesh(mesh)
        smc.set_material(0, material)
        smc.set_cast_shadow(False)
        if is_coverage:
            smc.set_render_in_main_pass(False)
            smc.set_render_in_depth_pass(False)
            smc.set_render_custom_depth(True)
            smc.set_custom_depth_stencil_value(SPRAY_STENCIL)
        else:
            smc.set_render_custom_depth(False)
        actors.append(actor)
    log(f"splash {label}: {spec.get('kind')} mesh with {mesh.get_num_triangles(0)} triangles")
    return actors


def skin_instance_path(skin: str) -> str:
    return f"{HUMPBACK_DIR}/MI_HumpbackSkin_{skin}"


def clear_whale_variants() -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    old = [a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith(WHALE_VARIANT_PREFIX)]
    for actor in old:
        subsystem.destroy_actor(actor)
    if unreal.EditorAssetLibrary.does_directory_exist(WHALE_MESH_DIR):
        unreal.EditorAssetLibrary.delete_directory(WHALE_MESH_DIR)
    if old:
        log(f"removed {len(old)} whale variants")


def spawn_whale_variant(shot: dict, label: str, hidden: bool = False) -> unreal.Actor:
    """A whale actor for one shot: its deformed mesh (whale.mesh_obj) or the base mesh, with
    the shot's skin material instance, custom stencil 1, at the shot's transform."""
    base = find_actor(WHALE_LABEL)
    mesh_obj = shot["whale"].get("mesh_obj")
    if mesh_obj:
        mesh = import_asset(mesh_obj, WHALE_MESH_DIR, f"SM_Whale_{label}", options=mesh_import_options())
        if not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"whale mesh import failed for {mesh_obj}: {mesh!r}")
    else:
        mesh = base.static_mesh_component.static_mesh
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, _vector(shot["whale"]["location"]),
                                             _rotator(shot["whale"]["rotation"]))
    actor.set_actor_label(f"{WHALE_VARIANT_PREFIX}{label}")
    if hidden:
        actor.set_actor_hidden_in_game(True)
    smc = actor.static_mesh_component
    smc.set_mobility(unreal.ComponentMobility.MOVABLE)
    smc.set_static_mesh(mesh)
    skin = str(shot.get("skin", "00"))
    skin_material = unreal.EditorAssetLibrary.load_asset(skin_instance_path(skin)) if skin != "00" else None
    base_smc = base.static_mesh_component
    for slot in range(base_smc.get_num_materials()):
        material = base_smc.get_material(slot)
        # Slots: body, barnacle, eye (see wf_build_scene.import_humpback); the skin covers the first two.
        if skin_material is not None and slot < 2:
            material = skin_material
        smc.set_material(slot, material)
    smc.set_render_custom_depth(True)
    smc.set_custom_depth_stencil_value(WHALE_STENCIL)
    return actor


def apply_shot(shot: dict) -> None:
    clear_whale_variants()
    base = find_actor(WHALE_LABEL)
    if "whale" in shot:
        apply_transform(base, shot["whale"])
    uses_variant = bool(shot.get("whale", {}).get("mesh_obj")) or str(shot.get("skin", "00")) != "00"
    base.set_actor_hidden_in_game(uses_variant)
    if uses_variant:
        spawn_whale_variant(shot, str(shot.get("id", shot.get("name", "shot"))))
    if "camera" in shot:
        apply_camera(find_actor(CAMERA_LABEL), shot["camera"])
    if "ocean" in shot:
        apply_ocean_look(shot["ocean"])
    if "waves" in shot:
        apply_waves(shot["waves"])
    if "sun" in shot:
        apply_sun(shot["sun"])
    if "fog" in shot:
        apply_fog(shot["fog"])
    clear_splash()
    if shot.get("splash"):
        apply_splash(shot["splash"], label=str(shot.get("id", shot.get("name", "shot"))))
    log(f"applied shot {shot.get('name', '?')}: {sorted(k for k in shot if k in ('whale', 'camera', 'ocean', 'waves', 'sun', 'fog', 'splash') and shot[k])}")
