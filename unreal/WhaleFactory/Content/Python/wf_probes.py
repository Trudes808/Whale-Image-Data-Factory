"""Diagnostic: spawn striped poles (1 m bands, red/white, unlit) around the whale so the
rendered water level can be read off the RGB image. WF_PROBES=clear removes them.

    tools/run_editor_script.sh probes && WF_RENDER_CONFIG=MRQ_RgbOnly tools/render_sequence.sh
"""
import os
import traceback

import unreal

import wf_shot

MAP_PATH = "/Game/WhaleFactory/Maps/OceanPOC"
PROBE_DIR = "/Game/WhaleFactory/Debug"
PROBE_MATERIAL = f"{PROBE_DIR}/M_Probe"
STRIPES_HLSL = r"""
float band = floor(WorldZ / 100.0);
float odd = fmod(abs(band), 2.0);
float3 red = float3(1, 0.05, 0.05); float3 white = float3(1, 1, 1);
float3 c = odd > 0.5 ? red : white;
if (abs(WorldZ) < 12.0) c = float3(0, 1, 0);  // green band at z = 0 (calm water level)
return c;
"""


def log(m):
    unreal.log(f"[wf_probes] {m}")


def make_material():
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    if unreal.EditorAssetLibrary.does_asset_exist(PROBE_MATERIAL):
        unreal.EditorAssetLibrary.delete_asset(PROBE_MATERIAL)
    material = tools.create_asset("M_Probe", PROBE_DIR, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    mel = unreal.MaterialEditingLibrary
    pos = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -700, 0)
    custom = mel.create_material_expression(material, unreal.MaterialExpressionCustom, -300, 0)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("code", STRIPES_HLSL)
    item = unreal.CustomInput()
    item.set_editor_property("input_name", "WorldZ")
    custom.set_editor_property("inputs", [item])
    mel.connect_material_expressions(pos, "Z", custom, "WorldZ")
    mel.connect_material_property(custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(material)
    if errors:
        raise RuntimeError(errors)
    unreal.EditorAssetLibrary.save_asset(PROBE_MATERIAL, only_if_is_dirty=False)
    return material


def main():
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    assert levels.load_level(MAP_PATH)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in actors.get_all_level_actors():
        if a.get_actor_label().startswith("Probe_"):
            actors.destroy_actor(a)
    if os.environ.get("WF_PROBES", "") != "clear":
        material = make_material()
        cylinder = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder")  # 100 cm tall, 100 wide
        whale = wf_shot.find_actor("Whale").get_actor_location()
        offsets = [(0, 0), (1000, 0), (-1000, 0), (0, 1000), (0, -1000), (2000, 2000), (-2000, -2000), (0, 3000)]
        for i, (dx, dy) in enumerate(offsets):
            actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(whale.x + dx, whale.y + dy, 0.0))
            actor.set_actor_label(f"Probe_{i}")
            actor.set_actor_scale3d(unreal.Vector(0.25, 0.25, 16.0))  # 25 cm wide, 16 m tall, centred on z = 0
            smc = actor.static_mesh_component
            smc.set_mobility(unreal.ComponentMobility.MOVABLE)
            smc.set_static_mesh(cylinder)
            smc.set_material(0, material)
            smc.set_cast_shadow(False)
        log(f"spawned {len(offsets)} probes around whale at {whale}")
    assert levels.save_current_level()
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_probes] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
