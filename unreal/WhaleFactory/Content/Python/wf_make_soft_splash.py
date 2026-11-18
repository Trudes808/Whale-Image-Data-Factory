"""run06 realism: the "soft" white-water materials M_SplashSoft (translucent, shaded grey-blue, two-scale noise erosion) and
M_SplashSoftCoverage (its masked label twin), as NEW assets; splash specs with "style": "soft" use them (wf_shot.apply_splash).

    tools/run_editor_script.sh make_soft_splash
"""
import os
import traceback

import unreal

os.environ["WF_IMPORT_ONLY"] = "1"


def main() -> None:
    import wf_build_scene
    import wf_shot
    wf_build_scene.make_spray_material(wf_shot.SPRAY_SOFT_MATERIAL, masked=False, soft=True)
    wf_build_scene.make_spray_material(wf_shot.SPRAY_SOFT_COVERAGE_MATERIAL, masked=True, soft=True)
    unreal.log("[wf_make_soft_splash] OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_make_soft_splash] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
