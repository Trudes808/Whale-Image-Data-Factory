"""Diagnostic: apply a shot with a whale variant and log the imported mesh bounds and actor transform."""
import os, traceback
import unreal
import wf_shot
try:
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    assert levels.load_level("/Game/WhaleFactory/Maps/OceanPOC")
    shot = wf_shot.load_shot_from_env()
    wf_shot.apply_shot(shot)
    for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        if a.get_actor_label().startswith(wf_shot.WHALE_VARIANT_PREFIX) or a.get_actor_label() == "Whale":
            smc = a.static_mesh_component
            box = smc.static_mesh.get_bounding_box()
            unreal.log(f"[wf_variant_check] {a.get_actor_label()} hidden={a.is_hidden_ed() or a.get_editor_property('hidden')} loc={a.get_actor_location()} rot={a.get_actor_rotation()} scale={a.get_actor_scale3d()} mesh={smc.static_mesh.get_name()} bounds min={box.min} max={box.max} rel={smc.get_relative_transform()}")
    assert levels.save_current_level()
    unreal.log("[wf_variant_check] OK")
except Exception:
    unreal.log_error("[wf_variant_check] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
