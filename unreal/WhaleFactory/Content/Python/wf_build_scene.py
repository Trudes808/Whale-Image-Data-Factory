"""Build the POC scene: an open ocean, sky and sun, the humpback whale, and a boat camera.

Run headless with ``tools/run_editor_script.sh build_scene``. The script is idempotent:
it re-imports the whale and rebuilds ``/Game/WhaleFactory/Maps/OceanPOC`` from scratch.

Whale source (``WF_WHALE``): ``humpback`` (default when the converted CC-BY asset exists,
see THIRD_PARTY.md) or ``procedural`` (the stand-in mesh from factory/procedural_whale.py).
``WF_SHOT_JSON`` (optional, from ``python3 -m factory.poses``) sets the whale pose, camera
and ocean look; otherwise the defaults below are used.
"""

import math
import os
from pathlib import Path

import unreal

import wf_shot

PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()))
REPO_ROOT = PROJECT_DIR.parents[1]

MAP_PATH = "/Game/WhaleFactory/Maps/OceanPOC"
WHALE_DIR = "/Game/WhaleFactory/Whale"
WHALE_MESH = f"{WHALE_DIR}/procedural_whale"
WHALE_MATERIAL = f"{WHALE_DIR}/M_WhaleSkin"
WHALE_OBJ = PROJECT_DIR / "SourceAssets" / "procedural_whale.obj"
# The CC-BY humpback (THIRD_PARTY.md), converted by factory/humpback_asset.py. Its Unreal
# assets live under Content/ThirdParty, which is gitignored.
HUMPBACK_SRC = REPO_ROOT / "third_party_assets" / "humpback_k9239" / "converted"
HUMPBACK_DIR = "/Game/ThirdParty/Humpback"
HUMPBACK_MESH = f"{HUMPBACK_DIR}/SM_Humpback"
WHALE_SOURCE = os.environ.get("WF_WHALE") or ("humpback" if (HUMPBACK_SRC / "SM_Humpback.obj").exists() else "procedural")
OCEAN_WAVES = "/Water/Waves/GerstnerWaves_Ocean"
OCEAN_DIR = "/Game/WhaleFactory/Water"
# The plugin's ocean material instance (parent of ours) and our tunable child instance.
OCEAN_PARENT_MATERIAL = "/Water/Materials/WaterSurface/Water_Material_Ocean"
OCEAN_MATERIAL_INSTANCE = wf_shot.OCEAN_MATERIAL_INSTANCE
# Default open-ocean look. The plugin default is clear turquoise shallow-tropical water:
# Absorption (10, 150, 350, 8) are per-channel absorption distances (times alpha) in cm, so
# red dies in 0.8 m but blue carries 28 m, and Scattering/Albedo are high. Open ocean off a
# boat is dark blue-grey: shorter green/blue absorption, much less in-scattering.
# Override per shot via the "ocean" key of WF_SHOT_JSON (see wf_shot.py).
OCEAN_LOOK = {
    "vectors": {
        "Absorption": [5.0, 25.0, 90.0, 8.0],
        "Scattering": [0.5, 0.7, 1.0, 0.06],
        "Water Albedo": [0.25, 0.45, 0.8, 0.5],
    },
    "scalars": {
        "Water Roughness": 0.04,
        "Default Near Normal Strength": 0.7,
        "Default Distant Normal Strength": 0.5,
    },
}
# Gerstner wave spectrum for the ocean (GerstnerWaterWaveGeneratorSimple; cm and degrees).
# A light breeze: swell up to 50 m long, up to 20 cm amplitude per wave, wind chop down to 1.2 m.
OCEAN_WAVES_PARAMS = {
    "num_waves": 16,
    "seed": 7,
    "min_wavelength": 120.0,
    "max_wavelength": 5000.0,
    "wavelength_falloff": 1.5,
    "min_amplitude": 1.0,
    "max_amplitude": 20.0,
    "amplitude_falloff": 2.0,
    "wind_angle_deg": 35.0,
    "direction_angular_spread_deg": 140.0,  # keep >= 120, see factory/seastate.py
    "small_wave_steepness": 0.4,
    "large_wave_steepness": 0.2,
    "steepness_falloff": 2.0,
}
SEQUENCE_DIR = "/Game/WhaleFactory/Sequences"
PREVIEW_SEQUENCE = f"{SEQUENCE_DIR}/LS_BoatPreview"
PREVIEW_FRAMES = 3
TOP_DEBUG_SEQUENCE = f"{SEQUENCE_DIR}/LS_TopDebug"
LABEL_DIR = "/Game/WhaleFactory/Labels"
SPRAY_DIR = wf_shot.SPRAY_DIR
LABEL_MATERIAL = f"{LABEL_DIR}/M_WhaleVisibleMask"
# Depth-only variant used with the opaque ocean (see make_ocean_opaque_instance).
LABEL_DEPTH_MATERIAL = f"{LABEL_DIR}/M_WhaleVisibleMaskDepth"

# Custom stencil value that marks whale pixels in the label pass (PLAN.md, "Core Technique").
WHALE_STENCIL = 1
# Ocean surface: base height (the ocean actor's Z) and the plugin's water body index, used
# by the label shader to evaluate the same Gerstner waves the water surface renders with.
WATER_SURFACE_Z = 0.0
WATER_BODY_INDEX = 0
WATER_MPC = "/Water/Materials/MPC/MPC_Water"
# Where the ocean actor (and the island its spline encloses) sits; see build_ocean.
ISLAND_CENTER = unreal.Vector(150000.0, 150000.0, 0.0)

# Scene layout (cm). The whale floats with its back about 0.8 m above calm water.
WHALE_LOCATION = unreal.Vector(0.0, 0.0, -60.0)
WHALE_ROTATION = unreal.Rotator(roll=0.0, pitch=0.0, yaw=20.0)
# About 40 m away, roughly broadside (bearing 110 deg from the whale's heading), 3 m up.
BOAT_CAMERA_LOCATION = unreal.Vector(-2570.0, 3065.0, 300.0)
BOAT_CAMERA_FOCAL_MM = 70.0
# Open ocean has nothing under the water; the Water material then refracts the sky and
# looks transparent. A dark floor 60 m down gives the water depth to absorb light.
SEABED_DEPTH_CM = 6000.0
SEABED_SIZE_CM = 800000.0

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def log(message: str) -> None:
    unreal.log(f"[wf_build_scene] {message}")


import_asset = wf_shot.import_asset
mesh_import_options = wf_shot.mesh_import_options


def import_whale_mesh() -> unreal.StaticMesh:
    mesh = import_asset(WHALE_OBJ, WHALE_DIR, "procedural_whale")
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"whale import failed; got {mesh!r}")
    box = mesh.get_bounding_box()
    log(f"whale mesh bounds min={box.min} max={box.max}")
    return mesh


def import_texture(filename: Path, name: str, *, srgb: bool, compression=None) -> unreal.Texture2D:
    texture = import_asset(filename, HUMPBACK_DIR, name)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"texture import failed for {filename}: {texture!r}")
    texture.set_editor_property("srgb", srgb)
    if compression is not None:
        texture.set_editor_property("compression_settings", compression)
    # factory/humpback_asset.py already flipped the normal map's green channel.
    texture.set_editor_property("flip_green_channel", False)
    unreal.EditorAssetLibrary.save_asset(f"{HUMPBACK_DIR}/{name}", only_if_is_dirty=False)
    return texture


def texture_sample(material, texture, sampler_type, x, y):
    node = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionTextureSample, x, y)
    node.set_editor_property("texture", texture)
    node.set_editor_property("sampler_type", sampler_type)
    return node


def make_humpback_materials(textures: dict) -> dict:
    """Textured skin (also used by the barnacles) and a dark glossy eye, by OBJ group name."""
    mel = unreal.MaterialEditingLibrary
    skin_path = f"{HUMPBACK_DIR}/M_HumpbackSkin"
    if unreal.EditorAssetLibrary.does_asset_exist(skin_path):
        unreal.EditorAssetLibrary.delete_asset(skin_path)
    skin = asset_tools.create_asset("M_HumpbackSkin", HUMPBACK_DIR, unreal.Material, unreal.MaterialFactoryNew())
    base = texture_sample(skin, textures["base_color"], unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -500, -200)
    mel.connect_material_property(base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = texture_sample(skin, textures["roughness"], unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE, -700, 100)
    # Wet skin at the surface: scale the authored roughness down. A parameter so shots can
    # randomise sheen later (dry/drying skin after a long surfacing is rougher).
    scale = mel.create_material_expression(skin, unreal.MaterialExpressionScalarParameter, -700, 300)
    scale.set_editor_property("parameter_name", "RoughnessScale")
    scale.set_editor_property("default_value", 0.85)
    mult = mel.create_material_expression(skin, unreal.MaterialExpressionMultiply, -400, 150)
    mel.connect_material_expressions(rough, "R", mult, "A")
    mel.connect_material_expressions(scale, "", mult, "B")
    mel.connect_material_property(mult, "", unreal.MaterialProperty.MP_ROUGHNESS)
    normal = texture_sample(skin, textures["normal"], unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, -500, 450)
    mel.connect_material_property(normal, "RGB", unreal.MaterialProperty.MP_NORMAL)
    errors = mel.recompile_material(skin)
    if errors:
        raise RuntimeError(f"humpback skin material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(skin_path, only_if_is_dirty=False)

    eye_path = f"{HUMPBACK_DIR}/M_HumpbackEye"
    if unreal.EditorAssetLibrary.does_asset_exist(eye_path):
        unreal.EditorAssetLibrary.delete_asset(eye_path)
    eye = asset_tools.create_asset("M_HumpbackEye", HUMPBACK_DIR, unreal.Material, unreal.MaterialFactoryNew())
    color = mel.create_material_expression(eye, unreal.MaterialExpressionConstant3Vector, -400, 0)
    color.set_editor_property("constant", unreal.LinearColor(0.01, 0.009, 0.008, 1.0))
    mel.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    gloss = mel.create_material_expression(eye, unreal.MaterialExpressionConstant, -400, 200)
    gloss.set_editor_property("r", 0.08)
    mel.connect_material_property(gloss, "", unreal.MaterialProperty.MP_ROUGHNESS)
    errors = mel.recompile_material(eye)
    if errors:
        raise RuntimeError(f"humpback eye material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(eye_path, only_if_is_dirty=False)
    return {"body": skin, "barnacle": skin, "eye": eye}


def import_humpback() -> tuple:
    """Import the converted humpback OBJ and textures; returns (mesh, {slot name: material})."""
    if not (HUMPBACK_SRC / "SM_Humpback.obj").exists():
        raise RuntimeError(f"{HUMPBACK_SRC}/SM_Humpback.obj missing: see THIRD_PARTY.md")
    # Start clean: the importer also drops placeholder materials next to the mesh.
    if unreal.EditorAssetLibrary.does_directory_exist(HUMPBACK_DIR):
        unreal.EditorAssetLibrary.delete_directory(HUMPBACK_DIR)
    textures = {
        "base_color": import_texture(HUMPBACK_SRC / "T_Humpback_BaseColor.png", "T_Humpback_BaseColor", srgb=True),
        "roughness": import_texture(HUMPBACK_SRC / "T_Humpback_Roughness.png", "T_Humpback_Roughness", srgb=False,
                                    compression=unreal.TextureCompressionSettings.TC_GRAYSCALE),
        "normal": import_texture(HUMPBACK_SRC / "T_Humpback_Normal.png", "T_Humpback_Normal", srgb=False,
                                 compression=unreal.TextureCompressionSettings.TC_NORMALMAP),
    }
    materials = make_humpback_materials(textures)
    mesh = import_asset(HUMPBACK_SRC / "SM_Humpback.obj", HUMPBACK_DIR, "SM_Humpback", options=mesh_import_options())
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"humpback import failed; got {mesh!r}")
    box = mesh.get_bounding_box()
    slots = [str(m.get_editor_property("material_slot_name")) for m in mesh.get_editor_property("static_materials")]
    log(f"humpback mesh bounds min={box.min} max={box.max}; triangles={mesh.get_num_triangles(0)}; slots={slots}")
    by_slot = {}
    for index, slot in enumerate(slots):
        key = next((g for g in materials if g in slot.lower()), None)
        if key is None:
            key = list(materials)[min(index, len(materials) - 1)]
            log(f"warning: slot {slot!r} matched no group name; using {key!r} by position")
        by_slot[index] = materials[key]
    return mesh, by_slot


def make_whale_material() -> unreal.Material:
    if unreal.EditorAssetLibrary.does_asset_exist(WHALE_MATERIAL):
        unreal.EditorAssetLibrary.delete_asset(WHALE_MATERIAL)
    material = asset_tools.create_asset("M_WhaleSkin", WHALE_DIR, unreal.Material, unreal.MaterialFactoryNew())
    mel = unreal.MaterialEditingLibrary
    color = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -400, 0)
    color.set_editor_property("constant", unreal.LinearColor(0.022, 0.025, 0.03, 1.0))
    mel.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    # Wet skin: fairly glossy.
    rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -400, 200)
    rough.set_editor_property("r", 0.28)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"whale material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(WHALE_MATERIAL, only_if_is_dirty=False)
    return material


LABEL_SHADER = r"""
// Whale pixel?  (custom stencil written only by the whale)
if (abs(Stencil.r - WHALE_STENCIL) > 0.5) return 0;
// Something opaque (boat, island, ...) in front of the whale?
if (CDepth.r > SDepth.r * 1.0005 + 1.0) return 0;

#if !ANALYTIC_WAVES
return 1;  // the ocean is opaque in this pass, so the SceneDepth test above covered it
#endif
// The water surface is not in SceneDepth here, so test it analytically: rebuild the whale
// point from custom depth and march the camera ray, comparing against the exact Gerstner
// surface the water renders (same waves, same time). Any sample under water -> hidden.
float3 Dir = normalize(-CamVec);
float RayLength = CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4);
float3 P = CamPos + Dir * RayLength;

FWaterBodyData Body = GetWaterBodyData(BODY_INDEX);
float MaxAmplitude = 0;
for (int w = 0; w < Body.NumWaves; w++) MaxAmplitude += GetWaveDataNew(w, Body).Amplitude;
float Ceiling = BASE_Z + MaxAmplitude + 1.0;

const int Steps = 128;
for (int i = Steps; i >= 1; i--)  // whale point first, then back toward the camera
{
    float3 S = CamPos + Dir * (RayLength * i / Steps);
    if (S.z > Ceiling) break;  // the rest of the ray toward the camera is above every crest
    // Height at S.xy: invert the horizontal Gerstner displacement by fixed-point iteration.
    float2 Q = S.xy;
    WaveOutput Wave = (WaveOutput)0;
    for (int k = 0; k < 4; k++)
    {
        Wave = GetAllGerstnerWavesNew(BODY_INDEX, WaterTime, DFPromote(Q));
        Q = S.xy - Wave.WPO.xy;
    }
    Wave = GetAllGerstnerWavesNew(BODY_INDEX, WaterTime, DFPromote(Q));
    if (S.z < BASE_Z + Wave.WPO.z - 0.5) return 0;
}
return 1;
"""


def make_label_material(path: str = LABEL_MATERIAL, analytic_waves: bool = True) -> unreal.Material:
    """Post-process material: 1 where the whale is directly visible, else 0.

    whale_visible = stencil is whale, nothing opaque in front (CustomDepth <= SceneDepth),
    and (analytic_waves) the camera ray to the whale point never goes below the wave
    surface. With analytic_waves=False the water is expected to be in SceneDepth, which is
    the case when the ocean renders with MI_OceanOpaque (the label render pass).
    """
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    folder, name = path.rsplit("/", 1)
    material = asset_tools.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    # After the tonemapper (not replacing it): Movie Render Queue's extra post-process passes
    # never complete with BL_REPLACING_TONEMAPPER. Emissive 0/1 still passes through as-is.
    material.set_editor_property("blendable_location", unreal.BlendableLocation.BL_SCENE_COLOR_AFTER_TONEMAPPING)
    mel = unreal.MaterialEditingLibrary

    def scene_texture(texture_id, y):
        node = mel.create_material_expression(material, unreal.MaterialExpressionSceneTexture, -700, y)
        node.set_editor_property("scene_texture_id", texture_id)
        return node, "Color"

    water_time = mel.create_material_expression(material, unreal.MaterialExpressionCollectionParameter, -700, 600)
    water_time.set_editor_property("collection", unreal.EditorAssetLibrary.load_asset(WATER_MPC))
    water_time.set_editor_property("parameter_name", "Time")
    sources = {
        "Stencil": scene_texture(unreal.SceneTextureId.PPI_CUSTOM_STENCIL, 0),
        "CDepth": scene_texture(unreal.SceneTextureId.PPI_CUSTOM_DEPTH, 150),
        "SDepth": scene_texture(unreal.SceneTextureId.PPI_SCENE_DEPTH, 300),
        "CamPos": (mel.create_material_expression(material, unreal.MaterialExpressionCameraPositionWS, -700, 450), ""),
        "CamVec": (mel.create_material_expression(material, unreal.MaterialExpressionCameraVectorWS, -700, 525), ""),
        "WaterTime": (water_time, ""),
    }

    custom = mel.create_material_expression(material, unreal.MaterialExpressionCustom, -300, 200)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1)
    custom.set_editor_property("description", "WhaleVisible")
    custom.set_editor_property("include_file_paths", ["/Plugin/Water/Private/GerstnerWaveFunctions.ush"])
    defines = []
    for name, value in (("WHALE_STENCIL", f"{WHALE_STENCIL}.0"), ("BODY_INDEX", str(WATER_BODY_INDEX)),
                        ("BASE_Z", f"{WATER_SURFACE_Z:.3f}"), ("ANALYTIC_WAVES", "1" if analytic_waves else "0")):
        define = unreal.CustomDefine()
        define.set_editor_property("define_name", name)
        define.set_editor_property("define_value", value)
        defines.append(define)
    custom.set_editor_property("additional_defines", defines)
    custom.set_editor_property("code", LABEL_SHADER)
    inputs = []
    for name in sources:
        item = unreal.CustomInput()
        item.set_editor_property("input_name", name)
        inputs.append(item)
    custom.set_editor_property("inputs", inputs)
    for name, (node, pin) in sources.items():
        if not mel.connect_material_expressions(node, pin, custom, name):
            raise RuntimeError(f"could not connect {name}")
    mel.connect_material_property(custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"label material failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return material


SPLASH_ALPHA_HLSL = r"""
// White water: soft noise, torn more and more toward the top (UV.y = height fraction) and
// toward the sides of a sheet (UV.x runs 0..1 across a plume), so no straight edges show.
float n = smoothstep(0.15, 0.85, Noise * 0.5 + 0.5);
float fade = saturate(1.0 - 0.9 * UV.y);
float side = 1.0 - pow(abs(frac(UV.x) * 2.0 - 1.0), 3.0);
float tear = smoothstep(0.0, 0.6, n + fade * side - 0.55);   // gaps open where both are low
return tear * (0.35 + 0.65 * n) * fade * side * Opacity;
"""


def make_spray_material(path: str, masked: bool) -> unreal.Material:
    """Splash mesh material. The translucent one is what the camera sees; the masked one
    is its hidden twin that writes custom depth/stencil for the label pass."""
    name = path.rsplit("/", 1)[1]
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    material = asset_tools.create_asset(name, SPRAY_DIR, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED if masked else unreal.BlendMode.BLEND_TRANSLUCENT)
    material.set_editor_property("two_sided", True)
    if masked:
        # Coverage threshold: spray with alpha above this hides the whale in the label.
        material.set_editor_property("opacity_mask_clip_value", wf_shot.SPRAY_COVERAGE_ALPHA)
    mel = unreal.MaterialEditingLibrary
    uv = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -900, 0)
    noise = mel.create_material_expression(material, unreal.MaterialExpressionNoise, -900, 150)
    noise.set_editor_property("scale", 0.035)
    noise.set_editor_property("levels", 5)
    noise.set_editor_property("quality", 2)
    noise.set_editor_property("noise_function", unreal.NoiseFunction.NOISEFUNCTION_VALUE_ALU)
    world_pos = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -1200, 150)
    mel.connect_material_expressions(world_pos, "", noise, "Position")
    opacity = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -900, 320)
    opacity.set_editor_property("parameter_name", "Opacity")
    opacity.set_editor_property("default_value", 0.9)
    alpha = mel.create_material_expression(material, unreal.MaterialExpressionCustom, -500, 150)
    alpha.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1)
    alpha.set_editor_property("description", "SprayAlpha")
    alpha.set_editor_property("code", SPLASH_ALPHA_HLSL)
    alpha_inputs = []
    for item_name in ("UV", "Noise", "Opacity"):
        item = unreal.CustomInput()
        item.set_editor_property("input_name", item_name)
        alpha_inputs.append(item)
    alpha.set_editor_property("inputs", alpha_inputs)
    mel.connect_material_expressions(uv, "", alpha, "UV")
    mel.connect_material_expressions(noise, "", alpha, "Noise")
    mel.connect_material_expressions(opacity, "", alpha, "Opacity")
    mel.connect_material_property(alpha, "", unreal.MaterialProperty.MP_OPACITY_MASK if masked else unreal.MaterialProperty.MP_OPACITY)
    if not masked:
        white = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -500, -150)
        white.set_editor_property("constant", unreal.LinearColor(0.88, 0.93, 0.97, 1.0))
        mel.connect_material_property(white, "", unreal.MaterialProperty.MP_BASE_COLOR)
        rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -500, 0)
        rough.set_editor_property("r", 0.45)
        mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
        # Lit translucency comes out grey under cloud; a little emissive keeps spray white.
        glow = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -500, 80)
        glow.set_editor_property("constant", unreal.LinearColor(0.35, 0.36, 0.38, 1.0))
        mel.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"{name} failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return material


def spawn(actor_class, location=unreal.Vector(), rotation=unreal.Rotator(), label=None):
    actor = actors.spawn_actor_from_class(actor_class, location, rotation)
    if label:
        actor.set_actor_label(label)
    return actor


def build_sky() -> None:
    sun = spawn(unreal.DirectionalLight, unreal.Vector(0, 0, 1000),
                unreal.Rotator(roll=0.0, pitch=-35.0, yaw=40.0), "Sun")
    sun_light = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_light.set_mobility(unreal.ComponentMobility.MOVABLE)
    sun_light.set_editor_property("atmosphere_sun_light", True)
    sun_light.set_editor_property("intensity", 10.0)

    spawn(unreal.SkyAtmosphere, label="SkyAtmosphere")
    sky = spawn(unreal.SkyLight, unreal.Vector(0, 0, 500), label="SkyLight")
    sky_light = sky.get_component_by_class(unreal.SkyLightComponent)
    sky_light.set_mobility(unreal.ComponentMobility.MOVABLE)
    sky_light.set_editor_property("real_time_capture", True)
    spawn(unreal.VolumetricCloud, label="Clouds")
    fog = spawn(unreal.ExponentialHeightFog, label="HeightFog")
    fog.get_component_by_class(unreal.ExponentialHeightFogComponent).set_editor_property("fog_density", 0.002)


def make_ocean_material_instance() -> unreal.MaterialInstanceConstant:
    """Our tunable child of the plugin's ocean material, with the OCEAN_LOOK defaults."""
    if unreal.EditorAssetLibrary.does_asset_exist(OCEAN_MATERIAL_INSTANCE):
        unreal.EditorAssetLibrary.delete_asset(OCEAN_MATERIAL_INSTANCE)
    folder, name = OCEAN_MATERIAL_INSTANCE.rsplit("/", 1)
    instance = asset_tools.create_asset(name, folder, unreal.MaterialInstanceConstant,
                                        unreal.MaterialInstanceConstantFactoryNew())
    parent = unreal.EditorAssetLibrary.load_asset(OCEAN_PARENT_MATERIAL)
    unreal.MaterialEditingLibrary.set_material_instance_parent(instance, parent)
    unreal.EditorAssetLibrary.save_asset(OCEAN_MATERIAL_INSTANCE, only_if_is_dirty=False)
    wf_shot.apply_ocean_look(OCEAN_LOOK, OCEAN_MATERIAL_INSTANCE)
    return instance


ENGINE_WAVES_ASSET = "/Water/Waves/GerstnerWaves_Ocean"
OCEAN_WAVES_ASSET = f"{OCEAN_DIR}/WW_Ocean"


def make_ocean_waves(ocean: unreal.Actor) -> None:
    """Point the ocean at our own wave asset and set its spectrum (OCEAN_WAVES_PARAMS, or
    WF_OCEAN_WAVES_JSON overrides).

    The ocean spawns with the plugin's WaterWavesAssetReference -> /Water/Waves/
    GerstnerWaves_Ocean. We replace it with our own reference to WW_Ocean, a fresh copy of
    the plugin asset made each build; wf_shot.apply_waves edits its inner GerstnerWaterWaves
    per shot. Other routes tried on 2026-10-10 failed: a GerstnerWaterWaves instanced on the
    actor rendered a surface that disagreed with the label pass; handing the asset's inner
    object to the ocean moved it out of the asset and resolved to nothing on reload; the SET
    console command did nothing.
    """
    params = dict(OCEAN_WAVES_PARAMS)
    if os.environ.get("WF_OCEAN_WAVES_JSON"):  # diagnostics / per-run override
        import json
        with open(os.environ["WF_OCEAN_WAVES_JSON"]) as handle:
            params.update(json.load(handle))
    if unreal.EditorAssetLibrary.does_asset_exist(OCEAN_WAVES_ASSET):
        unreal.EditorAssetLibrary.delete_asset(OCEAN_WAVES_ASSET)
    if not unreal.EditorAssetLibrary.duplicate_asset(ENGINE_WAVES_ASSET, OCEAN_WAVES_ASSET):
        raise RuntimeError(f"could not duplicate {ENGINE_WAVES_ASSET} to {OCEAN_WAVES_ASSET}")
    wf_shot.apply_waves(params)
    # Python cannot see WaterWavesAssetReference.WaterWavesAsset under its snake-case name
    # (it is not Blueprint-visible), but set_editor_property accepts the raw property name.
    # So make our own reference object, point it at WW_Ocean and give it to the ocean: the
    # same mechanism the plugin uses for its default waves.
    asset = unreal.EditorAssetLibrary.load_asset(OCEAN_WAVES_ASSET)
    reference_class = unreal.load_class(None, "/Script/Water.WaterWavesAssetReference")
    reference = unreal.new_object(reference_class, outer=ocean, name="WhaleFactoryWavesRef")
    reference.set_editor_property("WaterWavesAsset", asset)
    ocean.set_water_waves(reference)
    component = ocean.get_editor_property("water_body_component")
    wanted = wf_shot.asset_waves(asset).get_editor_property("max_wave_height")
    got = component.get_max_wave_height()
    if abs(got - wanted) > 0.5:
        raise RuntimeError(f"ocean max wave height {got} cm, expected {wanted} cm from {OCEAN_WAVES_ASSET}")
    log(f"ocean waves object: {reference} -> {OCEAN_WAVES_ASSET}; max wave height {got:.1f} cm")


def build_ocean() -> unreal.Actor:
    zone = spawn(unreal.WaterZone, label="WaterZone")
    zone.set_editor_property("zone_extent", unreal.Vector2D(400000.0, 400000.0))
    # The ocean's spline is a coastline: everything inside it is an island with no water. The
    # default spline is a 200 m square around the actor, so put the actor (and that island)
    # far from the whale; the ocean still fills the whole water zone at height WATER_SURFACE_Z.
    ocean = spawn(unreal.WaterBodyOcean, unreal.Vector(ISLAND_CENTER.x, ISLAND_CENTER.y, WATER_SURFACE_Z), label="Ocean")
    component = ocean.get_editor_property("water_body_component")
    waves = unreal.EditorAssetLibrary.load_asset(OCEAN_WAVES)
    log(f"ocean default waves={ocean.get_editor_property('water_waves')}; preset={waves}; "
        f"water_body_index={component.get_editor_property('water_body_index')} (label shader assumes {WATER_BODY_INDEX})")
    if os.environ.get("WF_OCEAN_WAVES", "custom") == "custom":
        make_ocean_waves(ocean)
    else:
        log("keeping the plugin's wave preset (WF_OCEAN_WAVES != custom)")
    component.set_editor_property("water_material", unreal.EditorAssetLibrary.load_asset(OCEAN_MATERIAL_INSTANCE))
    # Set extents via set_editor_property so the editor change event regenerates the ocean.
    component.set_editor_property("ocean_extents", unreal.Vector2D(400000.0, 400000.0))
    return ocean


def build_seabed() -> unreal.Actor:
    plane = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane")
    seabed = spawn(unreal.StaticMeshActor, unreal.Vector(0, 0, -SEABED_DEPTH_CM), label="Seabed")
    seabed.static_mesh_component.set_static_mesh(plane)
    scale = SEABED_SIZE_CM / 100.0  # the engine plane is 100 cm across
    seabed.set_actor_scale3d(unreal.Vector(scale, scale, 1.0))
    return seabed


def build_preview_sequence(camera: unreal.Actor, path: str = PREVIEW_SEQUENCE) -> None:
    """Short level sequence that cuts to one camera, for Movie Render Queue."""
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    folder, name = path.rsplit("/", 1)
    sequence = asset_tools.create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    sequence.set_display_rate(unreal.FrameRate(30, 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(PREVIEW_FRAMES)
    binding = sequence.add_possessable(camera)
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, PREVIEW_FRAMES)
    cut.set_camera_binding_id(sequence.get_binding_id(binding))
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)


def build_whale(mesh: unreal.StaticMesh, materials: dict, location=WHALE_LOCATION, rotation=WHALE_ROTATION) -> unreal.Actor:
    whale = spawn(unreal.StaticMeshActor, location, rotation, "Whale")
    smc = whale.static_mesh_component
    smc.set_mobility(unreal.ComponentMobility.MOVABLE)
    smc.set_static_mesh(mesh)
    for slot, material in materials.items():
        smc.set_material(slot, material)
    smc.set_render_custom_depth(True)
    smc.set_custom_depth_stencil_value(WHALE_STENCIL)
    return whale


def build_camera(target: unreal.Vector) -> unreal.Actor:
    rotation = unreal.MathLibrary.find_look_at_rotation(BOAT_CAMERA_LOCATION, target)
    camera = spawn(unreal.CineCameraActor, BOAT_CAMERA_LOCATION, rotation, "BoatCamera")
    cine = camera.get_cine_camera_component()
    cine.set_editor_property("current_focal_length", BOAT_CAMERA_FOCAL_MM)
    focus = cine.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    cine.set_editor_property("focus_settings", focus)
    return camera


def main() -> None:
    log(f"whale source: {WHALE_SOURCE}")
    if WHALE_SOURCE == "humpback":
        mesh, materials = import_humpback()
    elif WHALE_SOURCE == "procedural":
        mesh, materials = import_whale_mesh(), {0: make_whale_material()}
    else:
        raise RuntimeError(f"WF_WHALE must be 'humpback' or 'procedural', not {WHALE_SOURCE!r}")
    make_label_material()
    make_spray_material(wf_shot.SPRAY_MATERIAL, masked=False)
    make_spray_material(wf_shot.SPRAY_COVERAGE_MATERIAL, masked=True)
    if unreal.EditorAssetLibrary.does_directory_exist("/Game/WhaleFactory/Spray"):  # old sprite materials
        unreal.EditorAssetLibrary.delete_directory("/Game/WhaleFactory/Spray")
    make_ocean_material_instance()
    # Leftovers of a tried-and-dropped opaque-ocean label pass (the ocean's displacement
    # lives in the water mesh, not the material, so an opaque instance renders flat).
    for stale in ("/Game/WhaleFactory/Water/MI_OceanOpaque", LABEL_DEPTH_MATERIAL, "/Game/WhaleFactory/Render/MRQ_LabelDepth"):
        if unreal.EditorAssetLibrary.does_asset_exist(stale):
            unreal.EditorAssetLibrary.delete_asset(stale)
    shot = wf_shot.load_shot_from_env() or {}
    # new_level refuses to overwrite, so rebuild from scratch by deleting the old map first.
    if unreal.EditorAssetLibrary.does_asset_exist(MAP_PATH):
        unreal.EditorAssetLibrary.delete_asset(MAP_PATH)
    if not levels.new_level(MAP_PATH, False):
        raise RuntimeError(f"could not create {MAP_PATH}")
    build_sky()
    build_ocean()
    build_seabed()
    whale = build_whale(mesh, materials)
    camera = build_camera(whale.get_actor_location() + unreal.Vector(0, 0, 50))
    if shot:
        wf_shot.apply_shot(shot)
    # Debug view straight down from 200 m, for checking water coverage and whale placement.
    top = spawn(unreal.CineCameraActor, unreal.Vector(0, 0, 20000), unreal.Rotator(roll=0.0, pitch=-90.0, yaw=0.0), "TopCamera")
    top.get_cine_camera_component().set_editor_property("current_focal_length", 12.0)
    if not levels.save_current_level():
        raise RuntimeError("saving the level failed")
    build_preview_sequence(camera)
    build_preview_sequence(top, TOP_DEBUG_SEQUENCE)
    log(f"saved {MAP_PATH}")
    log("OK")


try:
    main()
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
