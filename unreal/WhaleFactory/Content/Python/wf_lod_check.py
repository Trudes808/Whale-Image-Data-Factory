"""run06 round 9 morning check: import one posed whale OBJ (env WF_NC_OBJ) the way batches do and write its LODs (count,
vertices and triangles per LOD, screen sizes, Nanite) to WF_NC_OUT (JSON).  Copy to Content/Python, then
tools/run_editor_script.sh lod_check"""
import json
import os
import traceback

import unreal

import wf_shot


def main() -> None:
    obj, out = os.environ["WF_NC_OBJ"], os.environ["WF_NC_OUT"]
    mesh = wf_shot.import_asset(obj, "/Game/WhaleFactory/Debug", "SM_LodCheck", options=wf_shot.mesh_import_options())
    sml = unreal.EditorStaticMeshLibrary
    info = {"lods": sml.get_lod_count(mesh)}
    info["per_lod"] = []
    for lod in range(info["lods"]):
        e = {"lod": lod}
        try:
            e["verts"] = sml.get_number_verts(mesh, lod)
        except Exception as exc:
            e["verts"] = str(exc)
        try:
            v, tri, *_ = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, lod, 0)
            e["tris_section0"] = len(tri) // 3
        except Exception as exc:
            e["tris_section0"] = str(exc)
        info["per_lod"].append(e)
    for name in ("get_lod_screen_sizes",):
        try:
            info["screen_sizes"] = [float(x) for x in getattr(sml, name)(mesh)]
        except Exception as exc:
            info["screen_sizes"] = str(exc)
    try:
        info["nanite"] = bool(mesh.get_editor_property("nanite_settings").get_editor_property("enabled"))
    except Exception as exc:
        info["nanite"] = str(exc)
    try:
        info["lod_group"] = str(mesh.get_editor_property("lod_group"))
    except Exception as exc:
        info["lod_group"] = str(exc)
    with open(out, "w") as fh:
        json.dump(info, fh, indent=1)
    unreal.log("[wf_lod_check] OK " + json.dumps(info))


try:
    main()
except Exception:
    unreal.log_error("[wf_lod_check] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
