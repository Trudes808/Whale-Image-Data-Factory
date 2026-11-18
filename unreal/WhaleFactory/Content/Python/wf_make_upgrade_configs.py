"""run06 engine upgrade test only: path-traced MRQ configs as NEW assets (nothing existing is rebuilt).

    WF_UPGRADE_CONFIGS="32:1280x853,32:2560x1707,128:1280x853" WF_UPGRADE_WARMUP=8 tools/run_editor_script.sh make_upgrade_configs

makes /Game/WhaleFactory/Render/MRQ_BatchInst_PT<spp>w<warmup>_<w>x<h>: the instance label pass on a path-traced
beauty pass (denoised), <spp> samples per pixel. Render with SM6 + ray tracing switched on from the command line
(WF_RENDER_EXTRA_ARGS, tools/render_sequence.sh); the default renders never use these assets.
"""
import os
import traceback

import unreal

os.environ["WF_IMPORT_ONLY"] = "1"


def log(message: str) -> None:
    unreal.log(f"[wf_make_upgrade_configs] {message}")


def main() -> None:
    import wf_build_scene  # noqa: E402
    import wf_make_render_config  # noqa: E402
    warmup = int(os.environ.get("WF_UPGRADE_WARMUP", "8"))
    for item in filter(None, os.environ.get("WF_UPGRADE_CONFIGS", "").split(",")):
        spp, res = item.split(":")
        w, h = (int(v) for v in res.lower().split("x"))
        name = f"MRQ_BatchInst_PT{int(spp)}w{warmup}_{w}x{h}"
        wf_make_render_config.make_config(name, with_label=True, label_material=wf_build_scene.INSTANCE_LABEL_MATERIAL,
                                          warmup=warmup, spatial_samples=int(spp), resolution=(w, h), path_tracer=True)
        log(f"made {name}")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_make_upgrade_configs] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
