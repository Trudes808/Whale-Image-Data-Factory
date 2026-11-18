"""run06: the instance label pass, as NEW assets next to the existing ones (nothing existing is replaced):

    tools/run_editor_script.sh make_instance_labels

* /Game/WhaleFactory/Labels/M_WhaleInstanceMask: wf_build_scene.INSTANCE_LABEL_SHADER, an RGB post-process material that
  writes each visible whale instance as an exact colour code (factory.postprocess.ID_CODES: 1 white, 2 red, 3 green,
  4 blue, 5 yellow, 6 magenta, 7 cyan; stencil 1 = instance 1, stencil 3 = instance 2 (run04/05 calf), stencil 16 + k =
  instance k of a run06 scene), with the same visibility tests (opaque occluders, analytic Gerstner water, spray / foam
  coverage stencil 2) as M_WhaleVisibleMask;
* /Game/WhaleFactory/Render/MRQ_BatchInst and MRQ_BatchInst_AA2: MRQ_Batch / MRQ_Batch_AA2 with that material as the
  label pass. Its frames are named FinalImageM_WhaleInstanceMask.<frame>.png, which tells factory.postprocess to decode
  them as instance codes (label encoding rgb_id_v1).

Render run06 shot lists with WF_RENDER_CONFIG=MRQ_BatchInst_AA2 (the legacy material does not see stencils >= 16).
"""
import os
import traceback

import unreal

os.environ["WF_IMPORT_ONLY"] = "1"


def log(message: str) -> None:
    unreal.log(f"[wf_make_instance_labels] {message}")


def main() -> None:
    import wf_build_scene  # noqa: E402  (inside the try below: an import error must still quit the editor)
    import wf_make_render_config  # noqa: E402
    material = wf_build_scene.make_label_material(wf_build_scene.INSTANCE_LABEL_MATERIAL, shader=wf_build_scene.INSTANCE_LABEL_SHADER, instance_codes=True)
    log(f"built {material.get_path_name()}")
    for name, samples in (("MRQ_BatchInst", 1), ("MRQ_BatchInst_AA2", 2)):
        wf_make_render_config.make_config(name, with_label=True, label_material=wf_build_scene.INSTANCE_LABEL_MATERIAL,
                                          warmup=wf_make_render_config.BATCH_WARMUP_FRAMES, spatial_samples=samples)
    # run06 realism: 3:2 frames (WF_INSTANCE_RESOLUTIONS="2560x1707,1280x853" -> MRQ_BatchInst_AA2_2560x1707 ...)
    for res in filter(None, os.environ.get("WF_INSTANCE_RESOLUTIONS", "").split(",")):
        w, h = (int(v) for v in res.lower().split("x"))
        wf_make_render_config.make_config(f"MRQ_BatchInst_AA2_{w}x{h}", with_label=True, label_material=wf_build_scene.INSTANCE_LABEL_MATERIAL,
                                          warmup=wf_make_render_config.BATCH_WARMUP_FRAMES, spatial_samples=2, resolution=(w, h))
        log(f"made MRQ_BatchInst_AA2_{w}x{h}")
        if os.environ.get("WF_PATHTRACER_SPP"):          # engine upgrade test: a path-traced variant (not a default)
            spp = int(os.environ["WF_PATHTRACER_SPP"])
            wf_make_render_config.make_config(f"MRQ_BatchInst_PT{spp}_{w}x{h}", with_label=True, label_material=wf_build_scene.INSTANCE_LABEL_MATERIAL,
                                              warmup=wf_make_render_config.BATCH_WARMUP_FRAMES, spatial_samples=spp, resolution=(w, h), path_tracer=True)
            log(f"made MRQ_BatchInst_PT{spp}_{w}x{h}")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_make_instance_labels] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
