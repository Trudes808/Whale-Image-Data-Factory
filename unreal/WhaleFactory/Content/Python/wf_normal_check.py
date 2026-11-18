"""run06 round 9 diagnosis: import one posed whale OBJ (env WF_NC_OBJ) the way batches import whales and write what the
imported static mesh carries (build settings; per-vertex positions, normals, tangents of LOD 0 section 0) to WF_NC_OUT
(JSON), to compare with the normals written into the OBJ.  tools/run_editor_script.sh normal_check"""
import json
import os
import traceback

import unreal

import wf_shot


def main() -> None:
    obj, out = os.environ["WF_NC_OBJ"], os.environ["WF_NC_OUT"]
    mesh = wf_shot.import_asset(obj, "/Game/WhaleFactory/Debug", "SM_NormCheck", options=wf_shot.mesh_import_options())
    info = {"nsections": unreal.EditorStaticMeshLibrary.get_number_materials(mesh) if hasattr(unreal.EditorStaticMeshLibrary, "get_number_materials") else None}
    try:
        bs = unreal.EditorStaticMeshLibrary.get_lod_build_settings(mesh, 0)
        for k in ("recompute_normals", "recompute_tangents", "use_mikk_t_space", "compute_weighted_normals", "remove_degenerates",
                  "use_high_precision_tangent_basis", "use_full_precision_u_vs"):
            try:
                info[k] = bool(bs.get_editor_property(k))
            except Exception as exc:
                info[k] = f"n/a {exc}"
    except Exception as exc:
        info["build_settings_error"] = str(exc)
    secs = []
    for sec in range(4):
        try:
            v, tri, n, uv, tan = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, 0, sec)
        except Exception as exc:
            secs.append({"error": str(exc)}); break
        if not v:
            break
        secs.append({"v": [[p.x, p.y, p.z] for p in v], "n": [[q.x, q.y, q.z] for q in n],
                     "t": [[q.tangent_x.x, q.tangent_x.y, q.tangent_x.z] for q in tan] if tan else [], "ntri": len(tri) // 3})
    info["sections"] = secs
    with open(out, "w") as fh:
        json.dump(info, fh)
    unreal.log("[wf_normal_check] OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_normal_check] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
