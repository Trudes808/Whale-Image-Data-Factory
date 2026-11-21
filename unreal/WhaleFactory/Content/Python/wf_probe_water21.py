"""run06 round 21b, read-only probe (water helper): the ocean material graph (Water_Material and every material function
it calls, each expression with its class, parameter name, key properties and input links), the values MI_OceanDeep holds
now, and the water zone / ocean / sky light settings that decide the far water and its reflection. Writes the JSON file
named by WF_PROBE_OUT (default Saved/wf_probe_water21.json); changes and saves nothing.

    WF_PROBE_OUT=/path/out.json tools/run_editor_script.sh probe_water21
"""
import json
import os
import traceback

import unreal

MI = "/Game/WhaleFactory/Water/MI_OceanDeep"
LEVEL = os.environ.get("WF_PROBE_LEVEL", "")
PROPS = ("parameter_name", "default_value", "r", "g", "b", "a", "constant", "texture", "material_function", "code", "description",
         "const_a", "const_b", "exponent", "sampler_type", "coordinate_index", "u_tiling", "v_tiling", "input_name",
         "output_name", "mask_r", "mask_g", "mask_b", "mask_a", "slider_min", "slider_max", "group", "speed_x", "speed_y",
         "base_reflect_fraction", "default_value_x", "default_value_y", "world_position_shader_offset", "period",
         "const_coordinate", "mip_value_mode", "texture_object_override", "scalar_value_source", "input_type", "preview_value",
         "use_preview_value_as_default", "sort_priority", "time", "ignore_pause", "fraction", "radius", "is_default", "parameter")


def log(message: str) -> None:
    unreal.log(f"[wf_probe_water21] {message}")


def jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, unreal.LinearColor):
        return [v.r, v.g, v.b, v.a]
    if isinstance(v, (unreal.Vector, unreal.Vector4)):
        return [getattr(v, k) for k in ("x", "y", "z", "w") if hasattr(v, k)]
    if isinstance(v, unreal.Object):
        return v.get_path_name()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return str(v)


def describe(expr, owner, mel, is_function):
    d = {"name": expr.get_name(), "class": expr.get_class().get_name()}
    for p in PROPS:
        try:
            d[p] = jsonable(expr.get_editor_property(p))
        except Exception:
            pass
    try:
        names = list(mel.get_material_expression_input_names(expr))
        ins = list(mel.get_inputs_for_material_function_expression(owner, expr) if is_function else mel.get_inputs_for_material_expression(owner, expr))
        links = []
        for k, n in enumerate(names):
            src = ins[k] if k < len(ins) else None
            out_name = ""
            if src is not None:
                try:
                    ok, out_name = mel.get_input_node_output_name_for_material_expression(expr, src)
                except Exception:
                    out_name = "?"
            links.append([str(n), src.get_name() if src is not None else None, out_name])
        d["inputs"] = links
    except Exception as exc:
        d["inputs_error"] = str(exc)
    return d


def main() -> None:
    mel = unreal.MaterialEditingLibrary
    out = {"instance": {}, "graphs": {}, "material_outputs": {}}
    mi = unreal.EditorAssetLibrary.load_asset(MI)
    base = mi.get_base_material()
    inst = {"parent": mi.get_editor_property("parent").get_path_name(), "scalars": {}, "vectors": {}, "switches": {}}
    for n in mel.get_scalar_parameter_names(base):
        inst["scalars"][str(n)] = mel.get_material_instance_scalar_parameter_value(mi, n)
    for n in mel.get_vector_parameter_names(base):
        v = mel.get_material_instance_vector_parameter_value(mi, n)
        inst["vectors"][str(n)] = [v.r, v.g, v.b, v.a]
    for n in mel.get_static_switch_parameter_names(base):
        inst["switches"][str(n)] = bool(mel.get_material_instance_static_switch_parameter_value(mi, n))
    out["instance"] = inst
    todo = [("material", base)]
    seen = set()
    while todo:
        kind, owner = todo.pop()
        key = owner.get_path_name()
        if key in seen:
            continue
        seen.add(key)
        exprs = list(mel.get_material_function_expressions(owner) if kind == "function" else mel.get_material_expressions(owner))
        g = []
        for e in exprs:
            d = describe(e, owner, mel, kind == "function")
            g.append(d)
            if e.get_class().get_name() == "MaterialExpressionMaterialFunctionCall":
                f = e.get_editor_property("material_function")
                if f is not None and isinstance(f, unreal.MaterialFunction):
                    todo.append(("function", f))
        out["graphs"][key] = g
        log(f"{kind} {key}: {len(g)} expressions")
    for prop in ("MP_BASE_COLOR", "MP_NORMAL", "MP_ROUGHNESS", "MP_SPECULAR", "MP_METALLIC", "MP_EMISSIVE_COLOR", "MP_OPACITY",
                 "MP_WORLD_POSITION_OFFSET", "MP_REFRACTION", "MP_AMBIENT_OCCLUSION", "MP_PIXEL_DEPTH_OFFSET"):
        try:
            node = mel.get_material_property_input_node(base, getattr(unreal.MaterialProperty, prop))
            out["material_outputs"][prop] = [node.get_name() if node else None, mel.get_material_property_input_node_output_name(base, getattr(unreal.MaterialProperty, prop)) if node else None]
        except Exception as exc:
            out["material_outputs"][prop] = str(exc)
    # level-side settings (read only; the level is not saved)
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
        out["world"] = world.get_path_name()
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        lv = {}
        for a in subsystem.get_all_level_actors():
            label = a.get_actor_label()
            if label in ("WaterZone", "Ocean", "SkyLight", "SkyAtmosphere") or label.startswith("Island"):
                d = {"class": a.get_class().get_name(), "location": jsonable(a.get_actor_location())}
                for comp in a.get_components_by_class(unreal.ActorComponent):
                    cd = {}
                    for p in ("water_material", "far_distance_material", "far_distance_mesh_extent", "zone_extent", "lower_hemisphere_is_black",
                              "lower_hemisphere_color", "intensity", "real_time_capture", "source_type", "sky_distance_threshold",
                              "capture_emissive_only", "TessellationFactor", "LODScale", "ocean_extents", "far_distance_mesh_extent"):
                        try:
                            cd[p] = jsonable(comp.get_editor_property(p))
                        except Exception:
                            pass
                    if cd:
                        d[comp.get_class().get_name() + ":" + comp.get_name()] = cd
                for p in ("far_distance_material", "far_distance_mesh_extent", "zone_extent", "enable_far_mesh"):
                    try:
                        d[p] = jsonable(a.get_editor_property(p))
                    except Exception:
                        pass
                lv[label] = d
        out["level"] = lv
    except Exception as exc:
        out["level_error"] = traceback.format_exc()
    path = os.environ.get("WF_PROBE_OUT") or os.path.join(unreal.Paths.project_saved_dir(), "wf_probe_water21.json")
    with open(path, "w") as h:
        json.dump(out, h, indent=1)
    log(f"wrote {path}")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_probe_water21] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
