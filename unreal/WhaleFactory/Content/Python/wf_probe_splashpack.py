"""Probe the purchased Niagara splash pack (README "run05 prep") headless, without changing anything else:

    (copy Niagra_splash_pack/data/Content/SplashPack to unreal/WhaleFactory/Content/SplashPack first; gitignored)
    tools/run_editor_script.sh probe_splashpack

Lists the pack's Niagara systems, their emitters' mesh renderers (meshes, materials, blend modes) and the materials'
parameters, so the integration (a deterministic frame per shot, a label coverage twin) can be planned on facts.
Writes Saved/wf_splashpack_probe.json.
"""
import json
import os
import traceback

import unreal

ROOT = "/Game/SplashPack"


def log(message: str) -> None:
    unreal.log(f"[wf_probe_splashpack] {message}")


def describe_material(path: str) -> dict:
    mat = unreal.EditorAssetLibrary.load_asset(path)
    if mat is None:
        return {"path": path, "loaded": False}
    base = mat.get_base_material() if hasattr(mat, "get_base_material") else mat
    mel = unreal.MaterialEditingLibrary
    out = {"path": path, "loaded": True, "class": mat.get_class().get_name()}
    try:
        out["blend_mode"] = str(base.get_editor_property("blend_mode"))
        out["shading_model"] = str(base.get_editor_property("shading_model"))
        out["scalars"] = [str(n) for n in mel.get_scalar_parameter_names(base)]
        out["textures"] = [str(n) for n in mel.get_texture_parameter_names(base)]
        out["used_with_niagara_meshes"] = bool(base.get_editor_property("used_with_niagara_mesh_particles"))
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> None:
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    assets = registry.get_assets_by_path(ROOT, recursive=True)
    by_class = {}
    for a in assets:
        by_class.setdefault(str(a.asset_class_path.asset_name), []).append(str(a.package_name))
    log(f"{len(assets)} assets: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_class.items())))
    report = {"counts": {k: len(v) for k, v in by_class.items()}, "systems": {}, "materials": {}}
    for path in sorted(by_class.get("NiagaraSystem", [])):
        system = unreal.EditorAssetLibrary.load_asset(path)
        entry = {"loaded": system is not None}
        if system is not None:
            try:
                entry["emitters"] = [str(h.get_editor_property("name")) for h in system.get_editor_property("emitter_handles")]
            except Exception as exc:
                entry["emitters_error"] = str(exc)
        report["systems"][path] = entry
        log(f"system {path}: {entry}")
    for path in sorted(p for p in by_class.get("Material", []) + by_class.get("MaterialInstanceConstant", []) if "/FX/" in p or "/DEBUG/" in p):
        report["materials"][path] = describe_material(path)
        try:
            base = unreal.EditorAssetLibrary.load_asset(path)
            base = base.get_base_material() if hasattr(base, "get_base_material") else base
            report["materials"][path]["allow_custom_depth_writes"] = bool(base.get_editor_property("allow_translucent_custom_depth_writes"))
        except Exception as exc:
            report["materials"][path]["custom_depth_prop"] = str(exc)[:120]
    report["static_meshes"] = sorted(by_class.get("StaticMesh", []))
    # the scene's colour controls (read only): ocean material parameters, sky atmosphere, fog, clouds
    mel = unreal.MaterialEditingLibrary
    ocean = unreal.EditorAssetLibrary.load_asset("/Game/WhaleFactory/Water/MI_OceanDeep")
    if ocean is not None:
        base = ocean.get_base_material()
        report["ocean"] = {"scalars": {str(n): mel.get_material_instance_scalar_parameter_value(ocean, n) for n in mel.get_scalar_parameter_names(base)},
                           "vectors": {str(n): str(mel.get_material_instance_vector_parameter_value(ocean, n)) for n in mel.get_vector_parameter_names(base)}}
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    levels.load_level("/Game/WhaleFactory/Maps/OceanPOC")
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    report["actors"] = sorted({a.get_actor_label() + ":" + a.get_class().get_name() for a in actors if not a.get_actor_label().startswith(("WhaleVar_", "Foam_", "Spray_", "Island_"))})
    for a in actors:
        for comp_cls, key in ((unreal.SkyAtmosphereComponent, "sky_atmosphere"), (unreal.ExponentialHeightFogComponent, "fog"),
                              (unreal.VolumetricCloudComponent, "clouds"), (unreal.SkyLightComponent, "sky_light")):
            c = a.get_component_by_class(comp_cls)
            if c is not None and key not in report:
                props = {}
                for name in ("rayleigh_scattering", "rayleigh_scattering_scale", "rayleigh_exponential_distribution", "mie_scattering_scale",
                             "mie_absorption_scale", "mie_anisotropy", "mie_exponential_distribution", "multi_scattering_factor", "sky_luminance_factor",
                             "fog_density", "fog_height_falloff", "fog_inscattering_luminance", "fog_inscattering_color", "directional_inscattering_luminance",
                             "start_distance", "fog_max_opacity", "intensity", "source_type", "real_time_capture", "layer_bottom_altitude", "layer_height"):
                    try:
                        props[name] = str(c.get_editor_property(name))
                    except Exception:
                        pass
                report[key] = {"actor": a.get_actor_label(), **props}
    out = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()), "wf_splashpack_probe.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    log(f"wrote {out}")
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_probe_splashpack] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
