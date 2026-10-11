"""Diagnostic: dump the ocean's Gerstner wave list and water body settings to JSON."""
import json, os, traceback
import unreal
import wf_shot

def main():
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    assert levels.load_level("/Game/WhaleFactory/Maps/OceanPOC")
    ocean = wf_shot.find_actor("Ocean")
    comp = ocean.get_editor_property("water_body_component")
    waves = comp.get_water_waves() if hasattr(comp, "get_water_waves") else ocean.get_editor_property("water_waves")
    inner = waves.get_water_waves() if hasattr(waves, "get_water_waves") else waves
    out = {"max_wave_height": comp.get_max_wave_height(), "waves": []}
    for p in ("target_wave_mask_depth", "fixed_water_depth", "water_body_index"):
        try:
            out[p] = comp.get_editor_property(p)
        except Exception as e:  # noqa: BLE001
            out[p] = f"n/a: {e}"
    try:
        out["constant_depth"] = comp.get_constant_depth()
    except Exception as e:  # noqa: BLE001
        out["constant_depth"] = f"n/a: {e}"
    try:
        for w in inner.get_editor_property("gerstner_waves"):
            out["waves"].append({k: (list(w.get_editor_property(k)) if k == "direction" else w.get_editor_property(k)) for k in ("wave_length", "amplitude", "steepness", "direction")})
    except Exception as e:  # noqa: BLE001
        out["waves_error"] = str(e)
    path = os.environ.get("WF_WAVES_OUT", "/tmp/waves.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=1)
    unreal.log(f"[wf_dump_waves] wrote {path}: {len(out['waves'])} waves, max height {out['max_wave_height']}")
    unreal.log("[wf_dump_waves] OK")

try:
    main()
except Exception:
    unreal.log_error("[wf_dump_waves] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
