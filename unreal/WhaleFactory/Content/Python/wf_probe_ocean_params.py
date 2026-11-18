"""run06 realism round 2, read-only probe: the ocean material's scalar / vector / texture parameter names and the values
MI_OceanDeep holds (for the small-scale roughness try). Writes Saved/wf_probe_ocean_params.json; changes nothing.

    tools/run_editor_script.sh probe_ocean_params
"""
import json
import os
import traceback

import unreal


def log(message: str) -> None:
    unreal.log(f"[wf_probe_ocean_params] {message}")


def main() -> None:
    mel = unreal.MaterialEditingLibrary
    mi = unreal.EditorAssetLibrary.load_asset("/Game/WhaleFactory/Water/MI_OceanDeep")
    base = mi.get_base_material()
    out = {"base": base.get_path_name(), "scalars": {}, "vectors": {}, "textures": {}, "switches": []}
    for n in mel.get_scalar_parameter_names(base):
        out["scalars"][str(n)] = mel.get_material_instance_scalar_parameter_value(mi, n)
    for n in mel.get_vector_parameter_names(base):
        v = mel.get_material_instance_vector_parameter_value(mi, n)
        out["vectors"][str(n)] = [v.r, v.g, v.b, v.a]
    for n in mel.get_texture_parameter_names(base):
        t = mel.get_material_instance_texture_parameter_value(mi, n)
        out["textures"][str(n)] = t.get_path_name() if t else None
    out["switches"] = [str(n) for n in mel.get_static_switch_parameter_names(base)]
    path = os.path.join(unreal.Paths.project_saved_dir(), "wf_probe_ocean_params.json")
    with open(path, "w") as h:
        json.dump(out, h, indent=1)
    log(f"wrote {path}: {len(out['scalars'])} scalars, {len(out['textures'])} textures")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_probe_ocean_params] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
