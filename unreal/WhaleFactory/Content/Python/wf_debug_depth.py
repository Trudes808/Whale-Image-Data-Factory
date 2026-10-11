"""Debug: make depth-visualisation post-process materials and an MRQ config that renders them.

Writes M_DebugSceneDepth (SceneDepth / 100 m) and M_DebugCustomDepth, plus config
MRQ_DebugDepth. Render with: WF_RENDER_CONFIG=MRQ_DebugDepth tools/render_sequence.sh
"""
import os, traceback
import unreal

DIR = "/Game/WhaleFactory/Debug"
tools = unreal.AssetToolsHelpers.get_asset_tools()
mel = unreal.MaterialEditingLibrary


def log(m): unreal.log(f"[wf_debug_depth] {m}")


def depth_material(name, texture_id, scale_cm):
    path = f"{DIR}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    m = tools.create_asset(name, DIR, unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    m.set_editor_property("blendable_location", unreal.BlendableLocation.BL_SCENE_COLOR_AFTER_TONEMAPPING)
    tex = mel.create_material_expression(m, unreal.MaterialExpressionSceneTexture, -600, 0)
    tex.set_editor_property("scene_texture_id", texture_id)
    custom = mel.create_material_expression(m, unreal.MaterialExpressionCustom, -300, 0)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("code", f"return saturate(D.r / {scale_cm:.1f}).xxx;")
    i = unreal.CustomInput(); i.set_editor_property("input_name", "D")
    custom.set_editor_property("inputs", [i])
    mel.connect_material_expressions(tex, "Color", custom, "D")
    mel.connect_material_property(custom, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    errors = mel.recompile_material(m)
    if errors:
        raise RuntimeError(errors)
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    return m


def main():
    passes = []
    for name, tid in (("M_DebugSceneDepth", unreal.SceneTextureId.PPI_SCENE_DEPTH),
                      ("M_DebugCustomDepth", unreal.SceneTextureId.PPI_CUSTOM_DEPTH)):
        p = unreal.MoviePipelinePostProcessPass()
        p.set_editor_property("enabled", True)
        p.set_editor_property("material", depth_material(name, tid, 10000.0))
        passes.append(p)
    path = "/Game/WhaleFactory/Render/MRQ_DebugDepth"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    base = unreal.EditorAssetLibrary.load_asset("/Game/WhaleFactory/Render/MRQ_RgbAndLabel")
    config = unreal.EditorAssetLibrary.duplicate_asset("/Game/WhaleFactory/Render/MRQ_RgbAndLabel", path)
    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    deferred.set_editor_property("additional_post_process_materials",
                                 list(deferred.get_editor_property("additional_post_process_materials")) + passes)
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_debug_depth] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
