"""Round 20f (run06): the research boat's bow (rail, tape, deck, bitt, pontoons) and land for one shot (factory.boat_scene).

wf_build_batch calls clear() once per batch (actors with the Boat20_ label prefix; the per-shot meshes too when the batch
has boat20 content) and spawn(shot) for each shot with shot["boat20"] content: the shot's OBJ meshes imported to
/Game/WhaleFactory/Boat20/Gen, placed at the shot's transform, hidden outside the shot by the sequence's visibility keys.
Everything here is a NEW asset under /Game/WhaleFactory/Boat20 (materials M_BF20_*, instances MI_BF20_*); no existing
material, the ocean, the sky, post-process or ini setting is touched. The boat writes no custom depth / stencil: it is an
opaque occluder for the whale label pass, and its own labels come from factory.boat_scene's projection.
"""
import os

import unreal

import wf_shot

BOAT_DIR = "/Game/WhaleFactory/Boat20"
GEN_DIR = f"{BOAT_DIR}/Gen"
LABEL_PREFIX = "Boat20_"
M_PAINT = f"{BOAT_DIR}/M_BF20_Paint_v1"
M_TAPE = f"{BOAT_DIR}/M_BF20_Tape_v1"
M_LAND = f"{BOAT_DIR}/M_BF20_Land_v1"

# linear base colours (coarse appearance only): slightly glossy white paint, matte grey Hypalon tubes, vinyl tape
PAINT_PRESETS = {
    "bfpaint": ((0.80, 0.80, 0.78), 0.30, 0.10),        # colour, roughness, grime amount
    "bfdeck": ((0.76, 0.76, 0.74), 0.48, 0.16),
    "bfdark": ((0.045, 0.045, 0.045), 0.6, 0.05),
    "bfpontoon": ((0.11, 0.12, 0.13), 0.55, 0.12),
}
TAPE_SHADES = {
    "blue": [(0.010, 0.060, 0.45), (0.015, 0.09, 0.55), (0.02, 0.12, 0.62)],
    "red": [(0.50, 0.06, 0.07), (0.60, 0.13, 0.15), (0.65, 0.22, 0.25)],
    "green": [(0.005, 0.06, 0.03), (0.010, 0.09, 0.05), (0.02, 0.12, 0.07)],
    "yellow": [(0.55, 0.32, 0.02), (0.62, 0.40, 0.04), (0.50, 0.30, 0.05)],
}

PAINT_HLSL = r"""
// white paint over metal / gelcoat: fine mottling and grime from two world-space noises
float n = saturate(0.5 * N1 + 0.5); float g = saturate(0.5 * N2 + 0.5);
float3 c = Colour * (1.0 - Grime * (0.6 * g * g + 0.4 * n));
return c;
"""
PAINT_ROUGH_HLSL = r"""
float n = saturate(0.5 * N1 + 0.5);
return saturate(Rough + 0.12 * (n - 0.5));
"""
TAPE_HLSL = r"""
// vinyl tape band: UV.y runs across the band; worn, lifted edges show lighter and dirtier, the face fades in patches
float v = UV.y; float e = min(v, 1.0 - v);
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float edge = 1.0 - smoothstep(0.0, 0.05 + 0.08 * n, e);
float3 faded = lerp(Colour, Colour * 0.6 + 0.25, 0.35 * m);
float3 c = lerp(faded, float3(0.62, 0.60, 0.55), edge * 0.7);
return c;
"""
LAND_HLSL = r"""
// UV.x = gully greenness, UV.y = slope (factory.boat_scene.land_mesh). Dry grass and scrub on the spurs, green in the
// gullies, bare red-brown rock on steep ground; world noise breaks it up.
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float3 dry = float3(0.15, 0.12, 0.065) * (0.8 + 0.4 * m);
float3 green = float3(0.050, 0.085, 0.030) * (0.8 + 0.4 * n);
float3 rock = float3(0.14, 0.095, 0.075) * (0.75 + 0.5 * n);
float3 c = lerp(dry, green, saturate(UV.x * 1.3 + 0.35 * (n - 0.5)));
c = lerp(c, rock, smoothstep(0.55, 0.95, UV.y + 0.25 * (m - 0.5)));
return c;
"""


# ---- round 20h (look round 2): new assets *_v2 / Cloud / sea floor; the v1 assets above stay as they were (old runs) ----
M_PAINT2 = f"{BOAT_DIR}/M_BF20_Paint_v2"
M_TAPE2 = f"{BOAT_DIR}/M_BF20_Tape_v2"
M_LAND2 = f"{BOAT_DIR}/M_BF20_Land_v2"
M_CLOUD = f"{BOAT_DIR}/M_BF20_Cloud_v1"
M_TAPE3 = f"{BOAT_DIR}/M_BF20_Tape_v3"
M_LAND3 = f"{BOAT_DIR}/M_BF20_Land_v3"
M_CLOUD2 = f"{BOAT_DIR}/M_BF20_Cloud_v2"
# colour, roughness, grime, chip threshold (higher = fewer chips)
PAINT2_PRESETS = {
    "bfpaint": ((0.80, 0.80, 0.78), 0.32, 0.12, 0.975),        # tubes, bitt: slight grime, rare small chips / specks
    "bfdeck": ((0.76, 0.76, 0.74), 0.55, 0.16, 0.95),          # non-slip deck paint: rougher, scuffed, a few more specks
    "bfdark": ((0.040, 0.040, 0.040), 0.6, 0.05, 1.10),
    "bfpontoon": ((0.060, 0.065, 0.070), 0.80, 0.10, 1.10),    # Hypalon: dark grey, matte (v1 0.11 / 0.55 read silver)
    "bfseafloor": ((0.010, 0.028, 0.085), 0.95, 0.0, 1.10),    # under-water stand-in for the sea's back-scatter (markers)
}
TAPE2_SHADES = {     # luminance 0.06-0.19 like real vinyl against white paint (the 20h test's lighter tapes read L* 57-84)
    "blue": [(0.010, 0.070, 0.45), (0.012, 0.085, 0.52), (0.016, 0.10, 0.55)],
    "red": [(0.30, 0.062, 0.062), (0.34, 0.075, 0.078), (0.27, 0.055, 0.058)],  # washed pink / coral (v1 magenta, 20h-b1 orange)
    "green": [(0.012, 0.075, 0.030), (0.016, 0.090, 0.036), (0.008, 0.055, 0.022)],
    "yellow": [(0.24, 0.115, 0.008), (0.28, 0.14, 0.011), (0.22, 0.10, 0.012)],     # mustard (20h-b1 read bright yellow)
}
PAINT2_HLSL = r"""
// white paint (round 20h): mottled grime, yellowed scuffs, small dark chips / specks, a rare rust bloom
float n = saturate(0.5 * N1 + 0.5); float g = saturate(0.5 * N2 + 0.5);
float s = saturate(0.5 * N3 + 0.5); float k = saturate(0.5 * N4 + 0.5);
float3 c = Colour * (1.0 - Grime * (0.55 * g * g + 0.35 * n)) * (0.975 + 0.05 * s);
c = lerp(c, c * float3(0.94, 0.91, 0.83), saturate(Grime * 4.0) * smoothstep(0.55, 0.85, k));
float chip = smoothstep(Chip, Chip + 0.03, s) * smoothstep(0.35, 0.6, k);
c = lerp(c, float3(0.16, 0.14, 0.12), 0.8 * chip);
float rust = smoothstep(Chip + 0.05, Chip + 0.08, s) * smoothstep(0.7, 0.9, g);
c = lerp(c, float3(0.30, 0.13, 0.05), 0.7 * rust * saturate(Grime * 5.0));
return c;
"""
PAINT2_ROUGH_HLSL = r"""
float n = saturate(0.5 * N1 + 0.5); float s = saturate(0.5 * N3 + 0.5); float k = saturate(0.5 * N4 + 0.5);
float chip = smoothstep(Chip, Chip + 0.03, s) * smoothstep(0.35, 0.6, k);
return saturate(Rough + 0.14 * (n - 0.5) + 0.08 * (s - 0.5) + 0.8 * Grime * smoothstep(0.55, 0.9, k) + 0.3 * chip);
"""
TAPE2_HLSL = r"""
// vinyl tape (round 20h): worn lighter edges, faded patches, light creases along the wrinkles (stripes across the band)
float v = UV.y; float e = min(v, 1.0 - v);
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float x = UV.x * 41.0 + 13.0 * n; float i0 = floor(x); float fx = frac(x); fx = fx * fx * (3.0 - 2.0 * fx);
float h0 = frac(sin(i0 * 127.1) * 43758.5453); float h1 = frac(sin((i0 + 1.0) * 127.1) * 43758.5453);
float streak = lerp(h0, h1, fx);
float edge = 1.0 - smoothstep(0.0, 0.04 + 0.08 * n, e);
float3 faded = lerp(Colour, Colour * 0.65 + 0.22, 0.3 * m);
faded = lerp(faded, faded * 0.75 + 0.15, 0.4 * smoothstep(0.62, 0.92, streak));
return lerp(faded, float3(0.60, 0.58, 0.52), edge * 0.6);
"""
TAPE3_HLSL = r"""
// round 20h b: vinyl tape with less washing (v2 faded toward white and read L* 57-84): faded patches, faint creases,
// worn edges toward a dirty grey, all mild
float v = UV.y; float e = min(v, 1.0 - v);
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float x = UV.x * 41.0 + 13.0 * n; float i0 = floor(x); float fx = frac(x); fx = fx * fx * (3.0 - 2.0 * fx);
float h0 = frac(sin(i0 * 127.1) * 43758.5453); float h1 = frac(sin((i0 + 1.0) * 127.1) * 43758.5453);
float streak = lerp(h0, h1, fx);
float edge = 1.0 - smoothstep(0.0, 0.03 + 0.06 * n, e);
float3 faded = lerp(Colour, Colour * 0.85 + 0.05, 0.5 * m);
faded = lerp(faded, faded * 0.85 + 0.06, 0.5 * smoothstep(0.65, 0.92, streak));
return lerp(faded, float3(0.40, 0.38, 0.34), edge * 0.45);
"""
TAPE3_ROUGH_HLSL = r"""
float m = saturate(0.5 * N2 + 0.5);
return saturate(0.58 + 0.2 * m);
"""
TAPE2_ROUGH_HLSL = r"""
float m = saturate(0.5 * N2 + 0.5);
return saturate(0.40 + 0.25 * m);
"""
LAND2_HLSL = r"""
// round 20h: UV.x = gully greenness, UV.y = height / crest (factory.boat_scene.land_mesh); slope from the normal.
// Green upper slopes and gullies, dark forest in the deep gullies, dry tan grass low on the spurs, bare rock only on
// the steepest faces (v1's red-brown rock and dry grass read pink / lilac under the haze).
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float slope = saturate(1.0 - NW.z);
float3 lush = float3(0.032, 0.062, 0.018) * (0.8 + 0.4 * n);
float3 forest = float3(0.016, 0.032, 0.012) * (0.85 + 0.3 * m);
float3 dry = float3(0.12, 0.10, 0.050) * (0.8 + 0.4 * m);
float3 rock = float3(0.085, 0.075, 0.060) * (0.8 + 0.4 * n);
float wet = saturate(UV.x * 1.2 + 0.9 * UV.y - 0.35 + 0.35 * (n - 0.5));
float3 c = lerp(dry, lush, smoothstep(0.25, 0.65, wet));
c = lerp(c, forest, smoothstep(0.55, 0.9, UV.x + 0.2 * (m - 0.5)));
c = lerp(c, rock, smoothstep(0.45, 0.7, slope + 0.2 * (m - 0.5)));
return c;
"""
LAND3_HLSL = r"""
// round 20h b: as v2 with the dry grass olive (tan read lilac under the haze) and more of the slope green
float n = saturate(0.5 * N1 + 0.5); float m = saturate(0.5 * N2 + 0.5);
float slope = saturate(1.0 - NW.z);
float3 lush = float3(0.030, 0.060, 0.016) * (0.8 + 0.4 * n);
float3 forest = float3(0.014, 0.030, 0.010) * (0.85 + 0.3 * m);
float3 dry = float3(0.085, 0.085, 0.035) * (0.8 + 0.4 * m);
float3 rock = float3(0.070, 0.062, 0.050) * (0.8 + 0.4 * n);
float wet = saturate(UV.x * 1.2 + 1.1 * UV.y - 0.25 + 0.35 * (n - 0.5));
float3 c = lerp(dry, lush, smoothstep(0.2, 0.6, wet));
c = lerp(c, forest, smoothstep(0.55, 0.9, UV.x + 0.2 * (m - 0.5)));
c = lerp(c, rock, smoothstep(0.5, 0.75, slope + 0.2 * (m - 0.5)));
return c;
"""
CLOUD2_ALPHA_HLSL = r"""
// round 20h b: as v1 with a sin-free hash (v1's sin hash lost precision at large uv: square patches)
float2 p = float2(UV.x * 1.6, UV.y * 2.2);
float a = 0.0; float amp = 0.5; float norm = 0.0;
for (int o = 0; o < 6; o++) {
  float2 i = floor(p); float2 f = frac(p); f = f * f * (3.0 - 2.0 * f);
  float3 q;
  q = frac(float3(i.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); float h00 = frac((q.x + q.y) * q.z);
  q = frac(float3((i + float2(1.0, 0.0)).xyx) * 0.1031); q += dot(q, q.yzx + 33.33); float h10 = frac((q.x + q.y) * q.z);
  q = frac(float3((i + float2(0.0, 1.0)).xyx) * 0.1031); q += dot(q, q.yzx + 33.33); float h01 = frac((q.x + q.y) * q.z);
  q = frac(float3((i + float2(1.0, 1.0)).xyx) * 0.1031); q += dot(q, q.yzx + 33.33); float h11 = frac((q.x + q.y) * q.z);
  a += amp * lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
  norm += amp; amp *= 0.5; p = p * 2.07 + float2(1.73, 0.91);
}
a /= norm;
float v = UV.y;
float thr = 0.44 + 0.30 * pow(saturate(v), 1.3);
float base = smoothstep(0.0, 0.04, v);
return saturate((a - thr) * 7.0) * base * Opacity;
"""
CLOUD_ALPHA_HLSL = r"""
// trade-wind cumulus on a card (factory.boat_scene.cloud_card): UV.y = 0 at the flat base, 1 at the tops; UV.x in the
// same units along the card. Value-noise fbm, a threshold rising with height (towers narrow upward), a sharp flat base.
float2 p = float2(UV.x * 1.6, UV.y * 2.2);
float a = 0.0; float amp = 0.5; float norm = 0.0;
for (int o = 0; o < 6; o++) {
  float2 i = floor(p); float2 f = frac(p); f = f * f * (3.0 - 2.0 * f);
  float h00 = frac(sin(dot(i, float2(127.1, 311.7))) * 43758.5453);
  float h10 = frac(sin(dot(i + float2(1.0, 0.0), float2(127.1, 311.7))) * 43758.5453);
  float h01 = frac(sin(dot(i + float2(0.0, 1.0), float2(127.1, 311.7))) * 43758.5453);
  float h11 = frac(sin(dot(i + float2(1.0, 1.0), float2(127.1, 311.7))) * 43758.5453);
  a += amp * lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
  norm += amp; amp *= 0.5; p = p * 2.07 + float2(17.3, 9.1);
}
a /= norm;
float v = UV.y;
float thr = 0.44 + 0.30 * pow(saturate(v), 1.3);
float base = smoothstep(0.0, 0.04, v);
return saturate((a - thr) * 7.0) * base * Opacity;
"""
CLOUD_COLOUR_HLSL = r"""
float v = saturate(UV.y);
return lerp(float3(0.52, 0.55, 0.60), float3(0.95, 0.95, 0.96), smoothstep(0.0, 0.45, v));
"""


def log(message: str) -> None:
    unreal.log(f"[wf_boat20] {message}")


def _noise(material, x, y, scale):
    mel = unreal.MaterialEditingLibrary
    wp = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, x - 300, y)
    nz = mel.create_material_expression(material, unreal.MaterialExpressionNoise, x, y)
    nz.set_editor_property("scale", scale); nz.set_editor_property("levels", 4); nz.set_editor_property("quality", 2)
    nz.set_editor_property("noise_function", unreal.NoiseFunction.NOISEFUNCTION_VALUE_ALU)
    mel.connect_material_expressions(wp, "", nz, "Position")
    return nz


def _build(path: str, colour_hlsl: str, rough_hlsl: str | None, scalars: dict, vectors: dict, noise_scales, rough_const=None):
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    folder, name = path.rsplit("/", 1)
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    material = tools.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("two_sided", True)
    uv = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -900, -200)
    n1 = _noise(material, -900, 0, noise_scales[0]); n2 = _noise(material, -900, 200, noise_scales[1])
    nodes = {"UV": uv, "N1": n1, "N2": n2}
    y = 400
    for pname, default in scalars.items():
        p = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -900, y); y += 120
        p.set_editor_property("parameter_name", pname); p.set_editor_property("default_value", float(default)); nodes[pname] = p
    for pname, default in vectors.items():
        p = mel.create_material_expression(material, unreal.MaterialExpressionVectorParameter, -900, y); y += 120
        p.set_editor_property("parameter_name", pname)
        p.set_editor_property("default_value", unreal.LinearColor(*[float(c) for c in default], 1.0)); nodes[pname] = p
    col = wf_shot._custom(material, colour_hlsl, list(nodes), unreal.CustomMaterialOutputType.CMOT_FLOAT3, -450, 0, "Colour")
    for k, node in nodes.items():
        mel.connect_material_expressions(node, "", col, k)
    mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
    if rough_hlsl:
        rough = wf_shot._custom(material, rough_hlsl, list(nodes), unreal.CustomMaterialOutputType.CMOT_FLOAT1, -450, 300, "Rough")
        for k, node in nodes.items():
            mel.connect_material_expressions(node, "", rough, k)
    else:
        rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -450, 300); rough.set_editor_property("r", rough_const)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"{name} failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    log(f"made {path}")
    return material


def _instance(name: str, parent, scalars: dict, vectors: dict):
    path = f"{BOAT_DIR}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    mel = unreal.MaterialEditingLibrary
    inst = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, BOAT_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, parent)
    for k, v in scalars.items():
        mel.set_material_instance_scalar_parameter_value(inst, k, float(v))
    for k, v in vectors.items():
        mel.set_material_instance_vector_parameter_value(inst, k, unreal.LinearColor(*[float(c) for c in v], 1.0))
    mel.update_material_instance(inst)
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return inst


def _build2(path: str, colour_hlsl: str, rough_hlsl: str | None, scalars: dict, vectors: dict, noise_scales, rough_const=None,
            normal_ws: bool = False, translucent: dict | None = None):
    """Round 20h material builder: like _build with any number of world-space noises (N1..Nk), the vertex normal (NW,
    world space) on request and an optional translucent mode {"alpha_hlsl", "lighting_mode", "normal_up"}."""
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    folder, name = path.rsplit("/", 1)
    tools = unreal.AssetToolsHelpers.get_asset_tools(); mel = unreal.MaterialEditingLibrary
    material = tools.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("two_sided", True)
    if translucent:
        material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        material.set_editor_property("translucency_lighting_mode", getattr(unreal.TranslucencyLightingMode, translucent.get("lighting_mode", "TLM_SURFACE_PER_PIXEL_LIGHTING")))
        for prop, value in (("compute_fog_per_pixel", True), ("tangent_space_normal", not translucent.get("normal_up", True))):
            try:
                material.set_editor_property(prop, value)
            except Exception as exc:
                log(f"{name}: {prop} not set ({exc})")
    uv = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -900, -200)
    nodes = {"UV": uv}
    for k, sc in enumerate(noise_scales):
        nodes[f"N{k + 1}"] = _noise(material, -900, 200 * k, sc)
    if normal_ws:
        nodes["NW"] = mel.create_material_expression(material, unreal.MaterialExpressionVertexNormalWS, -900, -400)
    y = 200 * len(noise_scales) + 200
    for pname, default in scalars.items():
        p = mel.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -900, y); y += 120
        p.set_editor_property("parameter_name", pname); p.set_editor_property("default_value", float(default)); nodes[pname] = p
    for pname, default in vectors.items():
        p = mel.create_material_expression(material, unreal.MaterialExpressionVectorParameter, -900, y); y += 120
        p.set_editor_property("parameter_name", pname)
        p.set_editor_property("default_value", unreal.LinearColor(*[float(c) for c in default], 1.0)); nodes[pname] = p

    def custom(code, out_type, yy, desc):
        node = wf_shot._custom(material, code, list(nodes), out_type, -450, yy, desc)
        for k, nd in nodes.items():
            mel.connect_material_expressions(nd, "", node, k)
        return node
    col = custom(colour_hlsl, unreal.CustomMaterialOutputType.CMOT_FLOAT3, 0, "Colour")
    mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
    if rough_hlsl:
        rough = custom(rough_hlsl, unreal.CustomMaterialOutputType.CMOT_FLOAT1, 300, "Rough")
    else:
        rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -450, 300); rough.set_editor_property("r", rough_const)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    if translucent:
        alpha = custom(translucent["alpha_hlsl"], unreal.CustomMaterialOutputType.CMOT_FLOAT1, 600, "Alpha")
        mel.connect_material_property(alpha, "", unreal.MaterialProperty.MP_OPACITY)
        if translucent.get("normal_up", True):
            up = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -450, 800)
            up.set_editor_property("constant", unreal.LinearColor(0.0, 0.0, 1.0, 1.0))
            mel.connect_material_property(up, "", unreal.MaterialProperty.MP_NORMAL)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(f"{name} failed to compile: {errors}")
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    log(f"made {path}")
    return material


def _instance2(name: str, parent, scalars: dict, vectors: dict):
    """Round 20h: like _instance, but an existing instance (made by this module) gets its parameters set again, so the
    presets above are what renders (tuning without new asset names)."""
    path = f"{BOAT_DIR}/{name}"
    mel = unreal.MaterialEditingLibrary
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        inst = unreal.EditorAssetLibrary.load_asset(path)
    else:
        inst = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, BOAT_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    if inst.get_editor_property("parent") != parent:
        mel.set_material_instance_parent(inst, parent)
    for k, v in scalars.items():
        mel.set_material_instance_scalar_parameter_value(inst, k, float(v))
    for k, v in vectors.items():
        mel.set_material_instance_vector_parameter_value(inst, k, unreal.LinearColor(*[float(c) for c in v], 1.0))
    mel.update_material_instance(inst)
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return inst


_MATERIALS = {}


def ensure_materials() -> dict:
    """slot name (factory.boat_scene material) -> material instance; made once, reused afterwards. Round 20h: the v2
    materials (worn paint, wrinkled tape, greener land, cumulus card, sea-floor stand-in); v1 assets are left alone."""
    if _MATERIALS:
        return _MATERIALS
    paint = _build2(M_PAINT2, PAINT2_HLSL, PAINT2_ROUGH_HLSL, {"Rough": 0.3, "Grime": 0.1, "Chip": 0.9}, {"Colour": (0.8, 0.8, 0.78)},
                    (0.08, 0.02, 1.6, 0.35))
    # round 20h b: the tape, land and cloud graphs revised after the first test render (v2 / Cloud_v1 assets kept, unused)
    tape = _build2(M_TAPE3, TAPE3_HLSL, TAPE3_ROUGH_HLSL, {}, {"Colour": (0.5, 0.1, 0.1)}, (0.6, 0.15))
    land = _build2(M_LAND3, LAND3_HLSL, None, {}, {}, (0.0004, 0.0025), rough_const=0.95, normal_ws=True)
    cloud = _build2(M_CLOUD2, CLOUD_COLOUR_HLSL, None, {"Opacity": 0.95}, {}, (), rough_const=1.0,
                    translucent={"alpha_hlsl": CLOUD2_ALPHA_HLSL, "lighting_mode": "TLM_SURFACE_PER_PIXEL_LIGHTING", "normal_up": True})
    for slot, (colour, rough, grime, chip) in PAINT2_PRESETS.items():
        _MATERIALS[slot] = _instance2(f"MI_BF20v2_{slot}", paint, {"Rough": rough, "Grime": grime, "Chip": chip}, {"Colour": colour})
    for colour, shades in TAPE2_SHADES.items():
        for k, c in enumerate(shades):
            _MATERIALS[f"bftape_{colour}_{k}"] = _instance2(f"MI_BF20v2_tape_{colour}_{k}", tape, {}, {"Colour": c})
    # horizon ridges keep round 20f's land (mesh uv = greenness, slope) and material; marker land uses v2 (slot bfland2)
    _MATERIALS["bfland"] = _build(M_LAND, LAND_HLSL, None, {}, {}, (0.0004, 0.0025), rough_const=0.95)
    _MATERIALS["bfland2"] = land
    _MATERIALS["bfcloud"] = cloud
    return _MATERIALS


def clear(delete_meshes: bool = False) -> None:
    """Remove boat20 actors (and the per-shot meshes); never raises (it runs in every batch of every run)."""
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        old = [a for a in subsystem.get_all_level_actors() if a.get_actor_label().startswith(LABEL_PREFIX)]
        for actor in old:
            subsystem.destroy_actor(actor)
        if delete_meshes and unreal.EditorAssetLibrary.does_directory_exist(GEN_DIR):
            unreal.EditorAssetLibrary.delete_directory(GEN_DIR)
        if old:
            log(f"removed {len(old)} boat20 actors")
    except Exception as exc:
        unreal.log_warning(f"[wf_boat20] clear failed: {exc}")


def _material_for(slot_name: str, materials: dict):
    s = slot_name.lower()
    best = None
    for key in materials:
        if key in s and (best is None or len(key) > len(best)):
            best = key
    return materials.get(best) if best else None


def spawn(shot: dict) -> list:
    """The shot's boat and / or land actors (hidden; the caller keys their visibility)."""
    b = shot.get("boat20") or {}
    specs = [(k, b[k]) for k in ("boat", "land", "floor", "clouds") if b.get(k)]     # floor / clouds: round 20h
    if not specs:
        return []
    materials = ensure_materials()
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    for kind, spec in specs:
        name = f"SM_BF20_{shot['id']}_{kind}"
        mesh = wf_shot.import_asset(spec["obj"], GEN_DIR, name, options=wf_shot.mesh_import_options())
        if not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"boat20 import failed for {spec['obj']}: {mesh!r}")
        loc = spec["location_cm"]
        actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(float(loc[0]), float(loc[1]), float(loc[2])),
                                                 unreal.Rotator(roll=0.0, pitch=0.0, yaw=float(spec["yaw_deg"])))
        actor.set_actor_label(f"{LABEL_PREFIX}{shot['id']}_{kind}")
        actor.set_actor_hidden_in_game(True)
        smc = actor.static_mesh_component
        smc.set_mobility(unreal.ComponentMobility.MOVABLE)
        smc.set_static_mesh(mesh)
        smc.set_render_custom_depth(False)
        if kind in ("floor", "clouds"):
            smc.set_cast_shadow(False)
        slots = [str(m.get_editor_property("material_slot_name")) for m in mesh.get_editor_property("static_materials")]
        for k, slot in enumerate(slots):
            m = _material_for(slot, materials)
            if m is None:
                log(f"no material for slot {slot} of {name}")
                continue
            smc.set_material(k, m)
        actors.append(actor)
        log(f"{shot['id']} {kind}: {spec.get('variant', '')} faces {spec.get('faces')} slots {slots}")
    return actors


def key_marker_grade(cine_binding, shots: list, per_shot: int, total: int, batch_grade: dict | None, frame, constant) -> None:
    """Round 20h: marker shots' shadow saturation (boat20.grade.saturation_shadows; the run06 grade's 0.4 is for whale skin
    and turns the dark sea behind a rail grey) keyed per shot on the cine camera's post-process; every other shot of the
    batch is keyed the batch grade's value, so it renders as before. Called only for batches with such a marker. Never
    raises: on any failure the batch renders with the batch grade (logged)."""
    try:
        base = (batch_grade or {}).get("saturation_shadows")
        if base is None:
            log("no batch saturation_shadows: marker grade not keyed")
            return
        base = float(base if not isinstance(base, (list, tuple)) else base[0])
        track = cine_binding.add_track(unreal.MovieSceneDoubleVectorTrack)
        track.set_property_name_and_path("PostProcessSettings.ColorSaturationShadows", "PostProcessSettings.ColorSaturationShadows")
        track.set_num_channels_used(4)
        if int(track.get_num_channels_used()) != 4:
            raise RuntimeError("vector track channels not set")
        section = track.add_section()
        section.set_range(0, total)
        channels = section.get_all_channels()
        if len(channels) != 4:
            raise RuntimeError(f"{len(channels)} channels")
        for index, shot in enumerate(shots):
            v = (((shot.get("boat20") or {}).get("grade") or {}).get("saturation_shadows"))
            v = base if v is None else float(v)
            for k, ch in enumerate(channels):
                ch.add_key(frame(index * per_shot), v if k < 3 else 1.0, interpolation=constant)
        log(f"marker grade keyed: {[((s.get('boat20') or {}).get('grade') or {}).get('saturation_shadows', base) for s in shots]}")
    except Exception as exc:
        unreal.log_warning(f"[wf_boat20] marker grade not keyed: {exc}")
