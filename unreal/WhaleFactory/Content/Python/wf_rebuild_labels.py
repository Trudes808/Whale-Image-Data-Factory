"""Rebuild the label-pass material (M_WhaleVisibleMask) in place after wf_build_scene.LABEL_SHADER
changes, without touching the rest of the scene.

    tools/run_editor_script.sh rebuild_labels
"""
import os

import unreal

os.environ["WF_IMPORT_ONLY"] = "1"
import wf_build_scene  # noqa: E402


def log(message: str) -> None:
    unreal.log(f"[wf_rebuild_labels] {message}")


def main() -> None:
    material = wf_build_scene.make_label_material()
    log(f"rebuilt {material.get_path_name()}")
    log("OK")


try:
    main()
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
