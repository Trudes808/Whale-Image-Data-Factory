"""Apply the shot in $WF_SHOT_JSON to the saved OceanPOC level (whale pose, camera, ocean
look, sun), then save the level. Headless:

    WF_SHOT_JSON=data/shots/poses/fluke_up.json tools/run_editor_script.sh apply_shot

tools/render_shot.sh does this and then renders.
"""

import os
import traceback

import unreal

import wf_shot

MAP_PATH = "/Game/WhaleFactory/Maps/OceanPOC"


def main() -> None:
    shot = wf_shot.load_shot_from_env()
    if shot is None:
        raise RuntimeError("WF_SHOT_JSON is not set")
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not levels.load_level(MAP_PATH):
        raise RuntimeError(f"could not load {MAP_PATH}")
    wf_shot.apply_shot(shot)
    if not levels.save_current_level():
        raise RuntimeError("saving the level failed")
    unreal.log("[wf_apply_shot] OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_apply_shot] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
