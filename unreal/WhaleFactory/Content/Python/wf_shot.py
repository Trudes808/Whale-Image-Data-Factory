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
from pathlib import Path
import os

import unreal

PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()))
REPO_ROOT = PROJECT_DIR.parents[1]

WHALE_LABEL = "Whale"
WHALE_VARIANT_PREFIX = "WhaleVar_"
WHALE_MESH_DIR = "/Game/ThirdParty/Humpback/Gen"   # per-shot deformed whale meshes (gitignored)
HUMPBACK_DIR = "/Game/ThirdParty/Humpback"
WHALE_STENCIL = 1
COMPANION_STENCIL = 3        # a second whale in the shot (mom + calf): 0.6 in the label pass
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


LENS_FOCAL_RANGE_MM = (4.0, 4000.0)


def unclamp_lens(cine) -> None:
    """Widen the cine camera's lens range. The component clamps current_focal_length to its
    LensSettings (the default "Universal Zoom" lens stops at 1000 mm), so without this a
    1500 mm shot renders at 1000 mm while the labels assume 1500 (run03 smoke, 2026-10-12)."""
    lens = cine.get_editor_property("lens_settings")
    lens.set_editor_property("min_focal_length", LENS_FOCAL_RANGE_MM[0])
    lens.set_editor_property("max_focal_length", LENS_FOCAL_RANGE_MM[1])
    cine.set_editor_property("lens_settings", lens)


def apply_camera(camera: unreal.Actor, spec: dict) -> None:
    apply_transform(camera, spec)
    if "focal_mm" in spec:
        cine = camera.get_cine_camera_component()
        unclamp_lens(cine)
        cine.set_editor_property("current_focal_length", float(spec["focal_mm"]))


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


SKY_DIR = "/Game/WhaleFactory/Sky"
SKY_MATERIAL = f"{SKY_DIR}/M_SkyHDRI"
SKY_INSTANCE = f"{SKY_DIR}/MI_SkyHDRI"
SKY_SPHERE_LABEL = "SkySphere"
SKY_SPHERE_MESH = "/Engine/BasicShapes/Sphere.Sphere"      # 50 cm radius; scaled up to a sky dome
SKY_SPHERE_SCALE = float(os.environ.get("WF_SKY_SPHERE_SCALE", "40000"))   # x 50 cm: 40000 = 20 km radius, beyond the islands
ATMOSPHERE_LABELS = ("SkyAtmosphere", "Clouds")
SKY_UV_HLSL = r"""
// lat-long lookup by view direction: u = azimuth from +X (0.5 straight ahead), v = 0 at the zenith
float3 d = normalize(P - Cam);
float u = atan2(d.y, d.x) / 6.28318530718 + 0.5;
float v = 0.5 - asin(clamp(d.z, -1.0, 1.0)) / 3.14159265359;
return float2(u, v);
"""


def ensure_sky_material(default_texture: unreal.Texture2D) -> None:
    """An unlit sky material sampling a lat-long HDRI by view direction (texture parameter "Sky",
    scalar "Intensity"), flagged as sky so fog and aerial perspective leave it alone, and an instance of it.
    default_texture: an HDR (linear) sky texture, so the linear-colour sampler compiles."""
    if unreal.EditorAssetLibrary.does_asset_exist(SKY_INSTANCE):
        return
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    if unreal.EditorAssetLibrary.does_asset_exist(SKY_MATERIAL):
        unreal.EditorAssetLibrary.delete_asset(SKY_MATERIAL)
    material = tools.create_asset("M_SkyHDRI", SKY_DIR, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("two_sided", True)
    if os.environ.get("WF_SKY_IS_SKY", "1") == "1":
        try:
            material.set_editor_property("is_sky", True)
        except Exception as exc:                                      # older property name
            log(f"is_sky not settable ({exc}); the sky sphere may receive fog")
    else:
        log("sky material without the is_sky flag (WF_SKY_IS_SKY=0)")
    world_pos = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -1300, 0)
    cam = mel.create_material_expression(material, unreal.MaterialExpressionCameraPositionWS, -1300, 150)
    uv = mel.create_material_expression(material, unreal.MaterialExpressionCustom, -1000, 50)
    uv.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT2)
    uv.set_editor_property("description", "LatLongUV"); uv.set_editor_property("code", SKY_UV_HLSL)
    items = []
    for item_name in ("P", "Cam"):
        item = unreal.CustomInput(); item.set_editor_property("input_name", item_name); items.append(item)
    uv.set_editor_property("inputs", items)
    mel.connect_material_expressions(world_pos, "", uv, "P"); mel.connect_material_expressions(cam, "", uv, "Cam")
    tex = mel.create_material_expression(material, unreal.MaterialExpressionTextureSampleParameter2D, -700, 0)
    tex.set_editor_property("parameter_name", "Sky")
    tex.set_editor_property("texture", default_texture)
    tex.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    mel.connect_material_expressions(uv, "", tex, "UVs")
    intensity = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -700, 250)
    intensity.set_editor_property("parameter_name", "Intensity"); intensity.set_editor_property("default_value", 1.0)
    mult = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -400, 100)
    mel.connect_material_expressions(tex, "RGB", mult, "A"); mel.connect_material_expressions(intensity, "", mult, "B")
    # the HDRI's sun disc is clamped: the directional light is the sun (otherwise glitter is counted twice)
    clamp = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -400, 250)
    clamp.set_editor_property("parameter_name", "Clamp"); clamp.set_editor_property("default_value", 8.0)
    mn = mel.create_material_expression(material, unreal.MaterialExpressionMin, -200, 100)
    mel.connect_material_expressions(mult, "", mn, "A"); mel.connect_material_expressions(clamp, "", mn, "B")
    mel.connect_material_property(mn, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"M_SkyHDRI failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(SKY_MATERIAL, only_if_is_dirty=False)
    instance = tools.create_asset("MI_SkyHDRI", SKY_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(instance, material)
    unreal.EditorAssetLibrary.save_asset(SKY_INSTANCE, only_if_is_dirty=False)
    log("made M_SkyHDRI / MI_SkyHDRI")


def ensure_sky_texture(name: str, path: str) -> unreal.Texture2D:
    asset_path = f"{SKY_DIR}/T_Sky_{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return unreal.EditorAssetLibrary.load_asset(asset_path)
    texture = import_asset(path, SKY_DIR, f"T_Sky_{name}")          # a 16-bit linear PNG (factory.skies.write_png16): .hdr would import as a cubemap
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"sky import failed for {path}: {texture!r}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_HDR)
    texture.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_CLAMP)
    unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False)
    log(f"imported sky {name}")
    return texture


def atmosphere_actors() -> list:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return [a for a in subsystem.get_all_level_actors() if a.get_actor_label() in ATMOSPHERE_LABELS]


def sky_spheres() -> dict:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return {a.get_actor_label()[len(SKY_SPHERE_LABEL) + 1:]: a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith(SKY_SPHERE_LABEL + "_")}


def ensure_sky_sphere(spec: dict) -> unreal.Actor:
    """The sky sphere for one HDRI (made on first use): a huge inside-out sphere with its own instance
    of M_SkyHDRI, labelled SkySphere_<name>. Hidden by default; batches key its visibility per shot."""
    name = spec["name"]
    texture = ensure_sky_texture(name, spec["file"])
    ensure_sky_material(texture)
    inst_path = f"{SKY_DIR}/MI_Sky_{name}"
    mel = unreal.MaterialEditingLibrary
    if not unreal.EditorAssetLibrary.does_asset_exist(inst_path):
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        instance = tools.create_asset(f"MI_Sky_{name}", SKY_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(instance, unreal.EditorAssetLibrary.load_asset(SKY_MATERIAL))
    else:
        instance = unreal.EditorAssetLibrary.load_asset(inst_path)
    mel.set_material_instance_texture_parameter_value(instance, "Sky", texture)
    mel.set_material_instance_scalar_parameter_value(instance, "Intensity", float(spec.get("intensity", 1.0)))
    mel.update_material_instance(instance)
    unreal.EditorAssetLibrary.save_asset(inst_path, only_if_is_dirty=False)
    existing = sky_spheres().get(name)
    if existing is not None:                                      # refresh: material and scale may have changed since it was made
        if existing.static_mesh_component.static_mesh is None:
            log(f"sky sphere {name} had no mesh: respawning")
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(existing)
        else:
            existing.static_mesh_component.set_material(0, instance)
            existing.set_actor_scale3d(unreal.Vector(SKY_SPHERE_SCALE, SKY_SPHERE_SCALE, SKY_SPHERE_SCALE))
            log(f"sky sphere {name}: reused, mesh {existing.static_mesh_component.static_mesh.get_name()}, hidden {existing.is_hidden_ed() if hasattr(existing, 'is_hidden_ed') else '?'}, material {instance.get_name()}")
            return existing
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator())
    actor.set_actor_label(f"{SKY_SPHERE_LABEL}_{name}")
    smc = actor.static_mesh_component
    smc.set_mobility(unreal.ComponentMobility.MOVABLE)
    mesh = unreal.EditorAssetLibrary.load_asset(SKY_SPHERE_MESH) or unreal.load_object(None, SKY_SPHERE_MESH)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"sky sphere mesh {SKY_SPHERE_MESH} not found: {mesh!r}")
    smc.set_static_mesh(mesh)
    smc.set_material(0, instance)
    smc.set_cast_shadow(False)
    smc.set_render_custom_depth(False)
    actor.set_actor_scale3d(unreal.Vector(SKY_SPHERE_SCALE, SKY_SPHERE_SCALE, SKY_SPHERE_SCALE))
    actor.set_actor_hidden_in_game(True)
    log(f"sky sphere {name}: mesh {mesh.get_name()} intensity {spec.get('intensity', 1.0)}")
    return actor


def hide_all_skies() -> None:
    """Every sky sphere hidden, the atmosphere and clouds shown: the state outside a shot."""
    for a in sky_spheres().values():
        a.set_actor_hidden_in_game(True)
    for a in atmosphere_actors():
        a.set_actor_hidden_in_game(False)


def apply_sky(spec: dict | None) -> None:
    """A real sky for a single shot (tools/render_shot.sh): its sphere shown and turned, atmosphere hidden."""
    hide_all_skies()
    if not spec or not spec.get("file"):
        return
    sphere = ensure_sky_sphere(spec)
    sphere.set_actor_hidden_in_game(False)
    sphere.set_actor_rotation(unreal.Rotator(roll=0.0, pitch=0.0, yaw=float(spec.get("rotation_deg", 0.0))), False)
    for a in atmosphere_actors():
        a.set_actor_hidden_in_game(True)


ISLAND_DIR = "/Game/WhaleFactory/Land"
ISLAND_MESH_DIR = f"{ISLAND_DIR}/Gen"
ISLAND_MATERIAL = f"{ISLAND_DIR}/M_Island"
ISLAND_LABEL_PREFIX = "Island_"
ISLAND_COLOUR_HLSL = r"""
// UV.x = height fraction, UV.y = slope. Vegetation on gentle mid slopes, rock on steep ground, dark
// wet rock and sand at the shore; a little noise so it is not flat.
float h = UV.x; float sl = UV.y;
float n = saturate(0.5 * N1 + 0.5);
float3 veg = float3(0.10, 0.16, 0.07) * (0.7 + 0.6 * n);
float3 rock = float3(0.22, 0.19, 0.16) * (0.7 + 0.6 * n);
float3 shore = float3(0.12, 0.11, 0.10);
float3 c = lerp(veg, rock, smoothstep(0.25, 0.7, sl + 0.4 * (1 - n)));
c = lerp(shore, c, smoothstep(0.0, 0.08, h));
return c;
"""


def ensure_island_material() -> unreal.Material:
    if unreal.EditorAssetLibrary.does_asset_exist(ISLAND_MATERIAL):
        return unreal.EditorAssetLibrary.load_asset(ISLAND_MATERIAL)
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    material = tools.create_asset("M_Island", ISLAND_DIR, unreal.Material, unreal.MaterialFactoryNew())
    uv = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -900, 0)
    world_pos = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -1200, 150)
    noise = mel.create_material_expression(material, unreal.MaterialExpressionNoise, -900, 150)
    noise.set_editor_property("scale", 0.0002); noise.set_editor_property("levels", 5); noise.set_editor_property("quality", 2)
    noise.set_editor_property("noise_function", unreal.NoiseFunction.NOISEFUNCTION_VALUE_ALU)
    mel.connect_material_expressions(world_pos, "", noise, "Position")
    col = _custom(material, ISLAND_COLOUR_HLSL, ["UV", "N1"], unreal.CustomMaterialOutputType.CMOT_FLOAT3, -500, 0, "LandColour")
    mel.connect_material_expressions(uv, "", col, "UV"); mel.connect_material_expressions(noise, "", col, "N1")
    mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -500, 200); rough.set_editor_property("r", 0.92)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"M_Island failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(ISLAND_MATERIAL, only_if_is_dirty=False)
    log("made M_Island")
    return material


def clear_islands() -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in subsystem.get_all_level_actors():
        if a.get_actor_label().startswith(ISLAND_LABEL_PREFIX):
            subsystem.destroy_actor(a)


def apply_islands(specs: list | None) -> list:
    """Distant land for the batch (factory.islands): import each island mesh once, place the actors."""
    clear_islands()
    if not specs:
        return []
    material = ensure_island_material()
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    for k, spec in enumerate(specs):
        asset_path = f"{ISLAND_MESH_DIR}/SM_{spec['name']}"
        mesh = unreal.EditorAssetLibrary.load_asset(asset_path) if unreal.EditorAssetLibrary.does_asset_exist(asset_path) else None
        if mesh is None:
            mesh = import_asset(spec["mesh_obj"], ISLAND_MESH_DIR, f"SM_{spec['name']}", options=mesh_import_options())
            if not isinstance(mesh, unreal.StaticMesh):
                raise RuntimeError(f"island import failed for {spec['mesh_obj']}: {mesh!r}")
        actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, _vector(spec["location_cm"]), unreal.Rotator(roll=0.0, pitch=0.0, yaw=float(spec.get("yaw_deg", 0.0))))
        actor.set_actor_label(f"{ISLAND_LABEL_PREFIX}{k}_{spec['name']}")
        smc = actor.static_mesh_component
        smc.set_mobility(unreal.ComponentMobility.MOVABLE)
        smc.set_static_mesh(mesh); smc.set_material(0, material)
        smc.set_render_custom_depth(False)
        sc = float(spec.get("scale", 1.0)); actor.set_actor_scale3d(unreal.Vector(sc, sc, sc))
        actors.append(actor)
    log(f"islands: {[(s['name'], s['distance_m'], s['bearing_deg']) for s in specs]}")
    return actors


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
    old = [a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith((SPRAY_LABEL_PREFIX, FOAM_LABEL_PREFIX))]
    for actor in old:
        subsystem.destroy_actor(actor)
    if unreal.EditorAssetLibrary.does_directory_exist(SPLASH_MESH_DIR):
        unreal.EditorAssetLibrary.delete_directory(SPLASH_MESH_DIR)
    if old:
        log(f"removed {len(old)} splash actors")


FOAM_LABEL_PREFIX = "Foam_"
FOAM_MATERIAL = f"{SPRAY_DIR}/M_Foam"
FOAM_COVERAGE_MATERIAL = f"{SPRAY_DIR}/M_FoamCoverage"
CHURN_MATERIAL = f"{SPRAY_DIR}/M_Churn"
FOAM_TEXTURE = f"{SPRAY_DIR}/T_Foam"
FOAM_TEXTURE_PNG = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).parents[1] / "third_party_assets" / "generated" / "T_Foam.png"
FOAM_COVERAGE_ALPHA = 0.6    # foam denser than this hides the whale in the label (it is white water, not skin)
FOAM_TILE_CM = 160.0         # the foam texture repeats every 1.6 m on the water

FOAM_UV_HLSL = r"""
return float2(P.x, P.y) / Scale;
"""
FOAM_ALPHA_HLSL = r"""
// Foam carpet: the per-vertex foam intensity (UV.x, from factory.whitewater) shaped by the foam texture
// (R density, A large-scale coverage) and a world-space noise: dense cores, lacy edges, holes.
float f = UV.x;
float cover = lerp(0.35, 1.0, Tex.a);
float dens = Tex.r * cover * (0.7 + 0.5 * saturate(0.5 * N1 + 0.5));
// never a solid sheet: even dense foam stays a little translucent, thin foam is lace
// the texture's cell density must carve real holes (its R runs 0.25..0.7), else the carpet is a sheet
float holes = smoothstep(0.30, 0.62, dens * (0.55 + 0.65 * f));
float a = holes * lerp(0.55, 0.9, f);
return a * Opacity;
"""
FOAM_COLOUR_HLSL = r"""
// bubble rims a little brighter, interiors a little darker and bluer
float l = 0.84 + 0.16 * Tex.g - 0.14 * Tex.b;
return float3(l * 0.96, l * 0.985, l);
"""
CHURN_ALPHA_HLSL = r"""
// Aerated water under the surface: a soft milky cloud (UV.x = churn intensity), no sharp structure.
float n = saturate(0.5 * N1 + 0.5);
return saturate(UV.x * (0.35 + 0.5 * Tex.r) * (0.6 + 0.6 * n)) * Opacity;
"""


def ensure_foam_texture() -> unreal.Texture2D:
    if unreal.EditorAssetLibrary.does_asset_exist(FOAM_TEXTURE):
        return unreal.EditorAssetLibrary.load_asset(FOAM_TEXTURE)
    if not FOAM_TEXTURE_PNG.exists():
        raise RuntimeError(f"{FOAM_TEXTURE_PNG} missing: python3 -m factory.foamtex third_party_assets/generated/T_Foam.png")
    texture = import_asset(str(FOAM_TEXTURE_PNG), SPRAY_DIR, "T_Foam")
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"foam texture import failed: {texture!r}")
    texture.set_editor_property("srgb", False)
    unreal.EditorAssetLibrary.save_asset(FOAM_TEXTURE, only_if_is_dirty=False)
    log("imported T_Foam")
    return texture


def _custom(material, code: str, inputs: list[str], out_type, x: int, y: int, description: str = "Custom"):
    mel = unreal.MaterialEditingLibrary
    node = mel.create_material_expression(material, unreal.MaterialExpressionCustom, x, y)
    node.set_editor_property("output_type", out_type)
    node.set_editor_property("description", description)
    node.set_editor_property("code", code)
    items = []
    for item_name in inputs:
        item = unreal.CustomInput(); item.set_editor_property("input_name", item_name); items.append(item)
    node.set_editor_property("inputs", items)
    return node


def _foam_material(path: str, alpha_hlsl: str, masked: bool, colour_hlsl: str | None, emissive, roughness: float, opacity_default: float,
                   tile_cm: float, clip: float | None = None):
    """Translucent (or masked twin) material: opacity = alpha_hlsl(UV, Tex, N1, Opacity) with the foam
    texture sampled in world space (tile_cm per repeat) and a world-space value noise (80 cm)."""
    name = path.rsplit("/", 1)[1]
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    material = tools.create_asset(name, SPRAY_DIR, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED if masked else unreal.BlendMode.BLEND_TRANSLUCENT)
    material.set_editor_property("two_sided", True)
    if masked:
        material.set_editor_property("opacity_mask_clip_value", clip if clip is not None else FOAM_COVERAGE_ALPHA)
    mel = unreal.MaterialEditingLibrary
    uv = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -1100, 0)
    world_pos = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -1400, 150)
    scale = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -1400, 300)
    scale.set_editor_property("r", tile_cm)
    wuv = _custom(material, FOAM_UV_HLSL, ["P", "Scale"], unreal.CustomMaterialOutputType.CMOT_FLOAT2, -1100, 200, "WorldUV")
    mel.connect_material_expressions(world_pos, "", wuv, "P")
    mel.connect_material_expressions(scale, "", wuv, "Scale")
    tex = mel.create_material_expression(material, unreal.MaterialExpressionTextureSample, -850, 200)
    tex.set_editor_property("texture", ensure_foam_texture())
    tex.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    mel.connect_material_expressions(wuv, "", tex, "UVs")
    noise = mel.create_material_expression(material, unreal.MaterialExpressionNoise, -850, 450)
    noise.set_editor_property("scale", 0.0125)
    noise.set_editor_property("levels", 4)
    noise.set_editor_property("quality", 2)
    noise.set_editor_property("noise_function", unreal.NoiseFunction.NOISEFUNCTION_VALUE_ALU)
    mel.connect_material_expressions(world_pos, "", noise, "Position")
    opacity = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -850, 600)
    opacity.set_editor_property("parameter_name", "Opacity")
    opacity.set_editor_property("default_value", opacity_default)
    alpha = _custom(material, alpha_hlsl, ["UV", "Tex", "N1", "Opacity"], unreal.CustomMaterialOutputType.CMOT_FLOAT1, -500, 200, "Alpha")
    mel.connect_material_expressions(uv, "", alpha, "UV")
    mel.connect_material_expressions(tex, "RGBA", alpha, "Tex")    # float4: R density, G rim, B dark, A coverage
    mel.connect_material_expressions(noise, "", alpha, "N1")
    mel.connect_material_expressions(opacity, "", alpha, "Opacity")
    mel.connect_material_property(alpha, "", unreal.MaterialProperty.MP_OPACITY_MASK if masked else unreal.MaterialProperty.MP_OPACITY)
    if not masked:
        if colour_hlsl:
            col = _custom(material, colour_hlsl, ["Tex"], unreal.CustomMaterialOutputType.CMOT_FLOAT3, -500, -150, "Colour")
            mel.connect_material_expressions(tex, "RGBA", col, "Tex")
            mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
        rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -500, 0)
        rough.set_editor_property("r", roughness)
        mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
        glow = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -500, 80)
        glow.set_editor_property("constant", unreal.LinearColor(*emissive, 1.0))
        mel.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"{name} failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return material


def ensure_whitewater_materials(rebuild: bool = False) -> None:
    """Foam carpet (lit white, lacy, textured), its coverage twin for the label pass, and the aerated water cloud.
    WF_REBUILD_WW_MATERIALS=1 rebuilds them (after editing the HLSL above or the foam texture)."""
    rebuild = rebuild or os.environ.get("WF_REBUILD_WW_MATERIALS") == "1"
    if rebuild and unreal.EditorAssetLibrary.does_asset_exist(FOAM_TEXTURE):
        unreal.EditorAssetLibrary.delete_asset(FOAM_TEXTURE)
    if rebuild or not unreal.EditorAssetLibrary.does_asset_exist(FOAM_MATERIAL):
        _foam_material(FOAM_MATERIAL, FOAM_ALPHA_HLSL, False, FOAM_COLOUR_HLSL, (0.18, 0.185, 0.2), 0.8, 0.95, FOAM_TILE_CM)
        log("made M_Foam")
    if rebuild or not unreal.EditorAssetLibrary.does_asset_exist(FOAM_COVERAGE_MATERIAL):
        _foam_material(FOAM_COVERAGE_MATERIAL, FOAM_ALPHA_HLSL, True, None, None, 0.8, 0.95, FOAM_TILE_CM, clip=FOAM_COVERAGE_ALPHA)
        log("made M_FoamCoverage")
    if rebuild or not unreal.EditorAssetLibrary.does_asset_exist(CHURN_MATERIAL):
        _foam_material(CHURN_MATERIAL, CHURN_ALPHA_HLSL, False, None, (0.05, 0.065, 0.075), 0.7, 0.7, 4.0 * FOAM_TILE_CM)
        log("made M_Churn")
    os.environ.pop("WF_REBUILD_WW_MATERIALS", None)


def apply_whitewater(spec: dict | None, label: str = "shot", hidden: bool = False) -> list:
    """Import a shot's white-water meshes: the foam carpet (visible + coverage twin, like a splash)
    and the aerated-water cloud (visible only; it is under the surface). Returns the actors."""
    if not spec or not spec.get("mesh_obj"):
        return []
    ensure_whitewater_materials()
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    parts = [("foam", spec["mesh_obj"], FOAM_MATERIAL, False), ("foamc", spec["mesh_obj"], FOAM_COVERAGE_MATERIAL, True)]
    if spec.get("churn_obj"):
        parts.append(("churn", spec["churn_obj"], CHURN_MATERIAL, False))
    meshes = {}
    for suffix, obj, material_path, is_coverage in parts:
        if obj not in meshes:
            mesh = import_asset(obj, SPLASH_MESH_DIR, f"SM_WW_{label}_{suffix if suffix != 'foamc' else 'foam'}", options=mesh_import_options())
            if not isinstance(mesh, unreal.StaticMesh):
                raise RuntimeError(f"white-water mesh import failed for {obj}: {mesh!r}")
            meshes[obj] = mesh
        material = unreal.EditorAssetLibrary.load_asset(material_path)
        actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator())
        actor.set_actor_label(f"{FOAM_LABEL_PREFIX}{label}{suffix}")
        if hidden:
            actor.set_actor_hidden_in_game(True)
        smc = actor.static_mesh_component
        smc.set_mobility(unreal.ComponentMobility.MOVABLE)
        smc.set_static_mesh(meshes[obj])
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
    log(f"white water {label}: {spec.get('category')} foam {spec.get('mesh')} churn {spec.get('churn_mesh')}")
    return actors


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


def skin_instance_path(skin: str, kind: str = "k9239") -> str:
    if kind == "humpback4k":
        return f"{HUMPBACK4K_DIR}/MI_Humpback4kSkin_{skin}"
    return f"{HUMPBACK_DIR}/MI_HumpbackSkin_{skin}"


# ---- the purchased rigged humpback (third_party_assets/humpback_4k, factory/humpback4k.py) -------------
# Its assets are made lazily on the first shot that uses it, so the editor scene build stays as it is.
HUMPBACK4K_DIR = "/Game/ThirdParty/Humpback4k"
HUMPBACK4K_SRC = REPO_ROOT / "third_party_assets" / "humpback_4k" / "converted"
HUMPBACK4K_SKINS = REPO_ROOT / "third_party_assets" / "humpback_4k" / "skins"
_humpback4k_materials: dict = {}


def import_texture_2d(path: Path, directory: str, name: str, srgb: bool, compression=None) -> unreal.Texture2D:
    asset_path = f"{directory}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return unreal.EditorAssetLibrary.load_asset(asset_path)
    texture = import_asset(str(path), directory, name)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"texture import failed for {path}: {texture!r}")
    texture.set_editor_property("srgb", srgb)
    if compression is not None:
        texture.set_editor_property("compression_settings", compression)
    unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False)
    return texture


def ensure_humpback4k_materials() -> dict:
    """{"body": skin material, "eye": eye material}: the same graph as the K9239 skin (BaseColor texture
    parameter, RoughnessScale scalar parameter for the wetness track, normal map) on the 4k textures."""
    if _humpback4k_materials:
        return _humpback4k_materials
    mel = unreal.MaterialEditingLibrary
    skin_path = f"{HUMPBACK4K_DIR}/M_Humpback4kSkin"
    eye_path = f"{HUMPBACK4K_DIR}/M_Humpback4kEye"
    # import stamp: the source textures' mtimes; a change (new skins, new individuals) rebuilds everything
    sources = sorted(list(HUMPBACK4K_SRC.glob("T_Humpback4k_*.png")) + list(HUMPBACK4K_SKINS.glob("T_Humpback_BaseColor_*.png")))
    stamp = {str(f): f.stat().st_mtime for f in sources}
    stamp_file = PROJECT_DIR / "Saved" / "wf_humpback4k_import.json"
    previous = json.loads(stamp_file.read_text()) if stamp_file.exists() else None
    if previous == stamp and unreal.EditorAssetLibrary.does_asset_exist(skin_path) and unreal.EditorAssetLibrary.does_asset_exist(eye_path):
        _humpback4k_materials.update(body=unreal.EditorAssetLibrary.load_asset(skin_path), eye=unreal.EditorAssetLibrary.load_asset(eye_path))
        return _humpback4k_materials
    if unreal.EditorAssetLibrary.does_directory_exist(HUMPBACK4K_DIR):
        unreal.EditorAssetLibrary.delete_directory(HUMPBACK4K_DIR)
        log("humpback4k assets: sources changed, rebuilding")
    if not (HUMPBACK4K_SRC / "SM_Humpback4k.obj").exists():
        raise RuntimeError(f"{HUMPBACK4K_SRC}/SM_Humpback4k.obj missing: run factory/humpback4k.py")
    base_tex = import_texture_2d(HUMPBACK4K_SRC / "T_Humpback4k_BaseColor.png", HUMPBACK4K_DIR, "T_Humpback4k_BaseColor", True)
    rough_tex = import_texture_2d(HUMPBACK4K_SRC / "T_Humpback4k_Roughness.png", HUMPBACK4K_DIR, "T_Humpback4k_Roughness", False,
                                  unreal.TextureCompressionSettings.TC_GRAYSCALE)
    normal_tex = import_texture_2d(HUMPBACK4K_SRC / "T_Humpback4k_Normal.png", HUMPBACK4K_DIR, "T_Humpback4k_Normal", False,
                                   unreal.TextureCompressionSettings.TC_NORMALMAP)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    if unreal.EditorAssetLibrary.does_asset_exist(skin_path):
        unreal.EditorAssetLibrary.delete_asset(skin_path)
    skin = tools.create_asset("M_Humpback4kSkin", HUMPBACK4K_DIR, unreal.Material, unreal.MaterialFactoryNew())
    base = mel.create_material_expression(skin, unreal.MaterialExpressionTextureSampleParameter2D, -500, -200)
    base.set_editor_property("parameter_name", "BaseColor")
    base.set_editor_property("texture", base_tex)
    base.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    mel.connect_material_property(base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = mel.create_material_expression(skin, unreal.MaterialExpressionTextureSample, -700, 100)
    rough.set_editor_property("texture", rough_tex)
    rough.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
    scale = mel.create_material_expression(skin, unreal.MaterialExpressionScalarParameter, -700, 300)
    scale.set_editor_property("parameter_name", "RoughnessScale")
    scale.set_editor_property("default_value", 0.85)
    mult = mel.create_material_expression(skin, unreal.MaterialExpressionMultiply, -400, 150)
    mel.connect_material_expressions(rough, "R", mult, "A")
    mel.connect_material_expressions(scale, "", mult, "B")
    mel.connect_material_property(mult, "", unreal.MaterialProperty.MP_ROUGHNESS)
    normal = mel.create_material_expression(skin, unreal.MaterialExpressionTextureSample, -500, 450)
    normal.set_editor_property("texture", normal_tex)
    normal.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    mel.connect_material_property(normal, "RGB", unreal.MaterialProperty.MP_NORMAL)
    errors = mel.recompile_material(skin)
    if errors:
        raise RuntimeError(f"humpback4k skin material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(skin_path, only_if_is_dirty=False)
    if unreal.EditorAssetLibrary.does_asset_exist(eye_path):
        unreal.EditorAssetLibrary.delete_asset(eye_path)
    eye = tools.create_asset("M_Humpback4kEye", HUMPBACK4K_DIR, unreal.Material, unreal.MaterialFactoryNew())
    color = mel.create_material_expression(eye, unreal.MaterialExpressionConstant3Vector, -400, 0)
    color.set_editor_property("constant", unreal.LinearColor(0.01, 0.009, 0.008, 1.0))
    mel.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    gloss = mel.create_material_expression(eye, unreal.MaterialExpressionConstant, -400, 200)
    gloss.set_editor_property("r", 0.08)
    mel.connect_material_property(gloss, "", unreal.MaterialProperty.MP_ROUGHNESS)
    errors = mel.recompile_material(eye)
    if errors:
        raise RuntimeError(f"humpback4k eye material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(eye_path, only_if_is_dirty=False)
    # skin variants (factory/skin.py on the 4k base colour): MI_Humpback4kSkin_<k>
    for png in sorted(HUMPBACK4K_SKINS.glob("T_Humpback_BaseColor_*.png")):
        key = png.stem.rsplit("_", 1)[1]
        if key == "00":
            continue
        texture = import_texture_2d(png, HUMPBACK4K_DIR, f"T_Humpback4k_BaseColor_{key}", True)
        path = skin_instance_path(key, "humpback4k")
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            unreal.EditorAssetLibrary.delete_asset(path)
        instance = tools.create_asset(path.rsplit("/", 1)[1], HUMPBACK4K_DIR, unreal.MaterialInstanceConstant,
                                      unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(instance, skin)
        mel.set_material_instance_texture_parameter_value(instance, "BaseColor", texture)
        mel.update_material_instance(instance)
        unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text(json.dumps(stamp))
    log(f"humpback4k materials made ({len(sources)} textures)")
    _humpback4k_materials.update(body=skin, eye=eye)
    return _humpback4k_materials


def clear_whale_variants() -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    old = [a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith(WHALE_VARIANT_PREFIX)]
    for actor in old:
        subsystem.destroy_actor(actor)
    if unreal.EditorAssetLibrary.does_directory_exist(WHALE_MESH_DIR):
        unreal.EditorAssetLibrary.delete_directory(WHALE_MESH_DIR)
    if old:
        log(f"removed {len(old)} whale variants")


def spawn_whale_variant(shot: dict, label: str, hidden: bool = False, stencil: int = WHALE_STENCIL) -> unreal.Actor:
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
    kind = shot.get("whale_kind", "k9239")
    if kind == "humpback4k":
        materials = ensure_humpback4k_materials()
        skin_material = unreal.EditorAssetLibrary.load_asset(skin_instance_path(skin, kind)) if skin != "00" else None
        slots = [str(m.get_editor_property("material_slot_name")).lower() for m in mesh.get_editor_property("static_materials")]
        for slot, name in enumerate(slots):
            # OBJ groups body / eye (factory.humpback4k.write_obj); anything unnamed is body
            material = materials["eye"] if "eye" in name else (skin_material or materials["body"])
            smc.set_material(slot, material)
    else:
        skin_material = unreal.EditorAssetLibrary.load_asset(skin_instance_path(skin)) if skin != "00" else None
        base_smc = base.static_mesh_component
        for slot in range(base_smc.get_num_materials()):
            material = base_smc.get_material(slot)
            # Slots: body, barnacle, eye (see wf_build_scene.import_humpback); the skin covers the first two.
            if skin_material is not None and slot < 2:
                material = skin_material
            smc.set_material(slot, material)
    smc.set_render_custom_depth(True)
    smc.set_custom_depth_stencil_value(stencil)
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
