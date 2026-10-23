"""Seeded shot sampler: config + mesh -> shotlist.jsonl and render batches.

    python3 -m factory.shotlist factory/config/poc_boat.json MESH.obj -o data/renders/run01 --count 2000 --seed 1

Every shot is a dict that ``wf_shot.apply_shot`` understands (whale, camera, splash, sun,
fog, exposure, waves, ocean) plus bookkeeping (id, pose category, sea state, look). Shots
sharing a sea state and water look are grouped into batches (``batches/batch_NNN.json``),
each rendered in one editor session by ``wf_build_batch.py``.

Variance covered: pose category and continuous pitch/roll/yaw/height jitter (so the same
category appears at many submersion depths), camera bearing all around the whale,
distance, boat and elevated heights, focal length chosen from how much of the frame the
whale should fill, aim offset, roll, sea state, water colour, sun elevation and azimuth,
overcast or clear, haze, exposure, spray, and negatives (submerged, or whale out of frame).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from factory import deform, gerstner, islands, poses, seastate, skies, splash, whitewater
from factory.whalemodel import WhaleModel, sample_clip_time

REPO_ROOT = Path(__file__).resolve().parents[1]
_SKY_CATALOGUE: dict | None = None
from factory.project import SENSOR_WIDTH_MM, Camera, transform_mesh

WHALE_APPARENT_WIDTH_M = 13.0  # typical on-screen extent of a 16 m whale seen obliquely


def weighted_choice(rng: np.random.Generator, weights: dict) -> str:
    names = list(weights)
    probabilities = np.array([weights[n] for n in names], float)
    return names[rng.choice(len(names), p=probabilities / probabilities.sum())]


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_pose(rng: np.random.Generator, config: dict, name: str | None = None) -> tuple[str, poses.Pose]:
    name = name or weighted_choice(rng, config["pose_weights"])
    base = poses.POSES[name]
    jitter = config["pose_jitter"]
    roll_jitter = jitter["breach_roll_deg"] if name == "breach" else jitter.get("roll_deg_by_category", {}).get(name, jitter["roll_deg"])
    pitch_jitter = jitter.get("pitch_deg_by_category", {}).get(name, jitter["pitch_deg"])     # a level surfacing stays level
    lo, hi = jitter["clearance_scale"]
    clearance = base.clearance_m * rng.uniform(lo, hi) if base.clearance_m > 0 else base.clearance_m * rng.uniform(0.5, 3.0)
    cap = jitter.get("clearance_max_m", {}).get(name)          # resting / breathing whales never show more than the back
    if cap is not None:
        clearance = min(clearance, float(cap))
    pose = replace(base,
                   pitch_deg=base.pitch_deg + rng.uniform(-pitch_jitter, pitch_jitter),
                   roll_deg=base.roll_deg + rng.uniform(-roll_jitter, roll_jitter),
                   clearance_m=clearance,
                   yaw_deg=float(rng.uniform(0.0, 360.0)))
    return name, pose


def sample_camera(rng: np.random.Generator, config: dict, pose: poses.Pose, placed: dict, water_z_cm: float,
                  focal_scale: float = 1.0) -> dict:
    """One camera draw. focal_scale multiplies the focal length the fill fraction asks for
    (sample_camera_for_size zooms in with it when the above-water whale is too small)."""
    cam = config["camera"]
    distance_m = log_uniform(rng, *cam["distance_m"])
    elevated = rng.uniform() < cam["elevated_fraction"]
    height_m = rng.uniform(*(cam["elevated_height_m"] if elevated else cam["boat_height_m"]))
    # Camera bearing about the whale: 0 = ahead of the nose, 90 / 270 = alongside, 180 = behind. Field photos are
    # mostly taken following the whale or running alongside it, so camera.bearing_sectors weights the sectors
    # ([start, end, weight] in degrees, uniform within a sector); without it the bearing is uniform all round.
    sectors = cam.get("bearing_sectors")
    if sectors:
        weights = np.array([float(sec[2]) for sec in sectors]); sec = sectors[int(rng.choice(len(sectors), p=weights / weights.sum()))]
        bearing = float(rng.uniform(sec[0], sec[1])) % 360.0
    else:
        bearing = rng.uniform(0.0, 360.0)
    aim = np.array(placed["stats"]["aim_point"], float)
    angle = math.radians(pose.yaw_deg + bearing)
    location = aim + np.array([math.cos(angle) * distance_m * 100.0, math.sin(angle) * distance_m * 100.0, 0.0])
    location[2] = water_z_cm + height_m * 100.0
    fill = rng.uniform(*cam["fill_fraction"])
    focal = fill * SENSOR_WIDTH_MM * distance_m / WHALE_APPARENT_WIDTH_M * focal_scale
    focal = float(np.clip(focal, *cam["focal_mm"]))
    rotation = poses.look_at_rotation(location, aim)
    # Aim jitter: shift the view by a fraction of the frame, in angle.
    hfov = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / 2 / focal))
    rotation["yaw"] = round(rotation["yaw"] + rng.uniform(-1, 1) * cam["aim_jitter_frame_fraction"] * hfov / 2, 4)
    rotation["pitch"] = round(rotation["pitch"] + rng.uniform(-1, 1) * cam["aim_jitter_frame_fraction"] * hfov * 9 / 16 / 2, 4)
    rotation["roll"] = round(float(rng.uniform(-cam["roll_deg"], cam["roll_deg"])), 3)
    return {"location": location.round(2).tolist(), "rotation": rotation, "focal_mm": round(focal, 2), "bearing_deg": round(bearing, 2),
            "bearing_deg": round(bearing, 2), "distance_m": round(distance_m, 2), "height_m": round(height_m, 2),
            "elevated": bool(elevated), "fill_fraction": round(fill, 3)}


def visible_extent(world_cm: np.ndarray, camera: dict, resolution, water_z_cm: float = 0.0) -> dict:
    """How much of the frame the above-water part of the whale spans, from projecting its
    vertices: the larger side of their bounding box as a fraction of the frame (width for
    horizontal, height for vertical), and the fraction of those vertices inside the frame."""
    width, height = resolution
    above = world_cm[world_cm[:, 2] > water_z_cm]
    if len(above) == 0:
        return {"extent_frame_fraction": 0.0, "in_frame_fraction": 0.0}
    cam = Camera(camera["location"], camera["rotation"], camera["focal_mm"], width, height)
    uv, depth = cam.project(above)
    in_front = depth > 1.0
    inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        return {"extent_frame_fraction": 0.0, "in_frame_fraction": 0.0}
    pts = uv[inside]
    extent = max((pts[:, 0].max() - pts[:, 0].min()) / width, (pts[:, 1].max() - pts[:, 1].min()) / height)
    return {"extent_frame_fraction": round(float(extent), 4), "in_frame_fraction": round(float(inside.mean()), 4)}


def sample_camera_for_size(rng: np.random.Generator, config: dict, pose: poses.Pose, placed: dict, world_cm: np.ndarray,
                           water_z_cm: float = 0.0) -> dict:
    """sample_camera, re-drawn until the above-water whale spans at least
    camera.min_extent_frame_fraction of the frame with camera.min_in_frame_fraction of it in
    view (when the config sets them). The fill fraction assumes a whole whale, so when only a
    back or a fluke is out of the water the lens is zoomed in by the shortfall (a photographer
    frames what is visible), within camera.focal_mm. After max_camera_tries the best draw is
    kept."""
    cam = config["camera"]
    min_extent = cam.get("min_extent_frame_fraction", 0.0)
    min_inside = cam.get("min_in_frame_fraction", 0.0)
    # run03: a per-shot target for how much of the frame the *visible* whale spans (log-uniform in
    # camera.extent_target, e.g. 0.3 .. 1.6 of the frame, i.e. from a small whale to one cut by the
    # frame edges as in long-lens field photos); the lens zooms in or out towards it.
    target = log_uniform(rng, *cam["extent_target"]) if cam.get("extent_target") else None
    tries = int(cam.get("max_camera_tries", 1)) if (min_extent > 0 or min_inside > 0 or target) else 1
    best, best_score = None, -20.0
    focal_scale = 1.0
    for _ in range(max(tries, 1)):
        camera = sample_camera(rng, config, pose, placed, water_z_cm, focal_scale)
        extent = visible_extent(world_cm, camera, config["resolution"], water_z_cm)
        camera.update(extent)
        if target:
            camera["extent_target"] = round(target, 4)
        ext, inside = extent["extent_frame_fraction"], extent["in_frame_fraction"]
        if target:
            score = -abs(math.log(max(ext, 1e-3) / target)) if inside >= min_inside else -10.0 + inside
        else:
            score = ext if inside >= min_inside else -1.0 + inside
        if score > best_score:
            best, best_score = camera, score
        if target:
            if 0.8 * target <= ext <= 1.25 * target and inside >= min_inside:
                break
            if ext > 0:
                focal_scale *= float(np.clip(target / ext, 0.3, 3.0))
            continue
        if ext >= min_extent and inside >= min_inside:
            break
        if 0 < ext < min_extent:
            focal_scale *= min_extent / ext * 1.1
        elif ext >= min_extent:
            focal_scale = 1.0          # big enough but cut off by the frame edge: re-aim at normal zoom
    return best


def sample_lighting(rng: np.random.Generator, config: dict, camera_yaw: float) -> tuple[dict, dict, float, bool]:
    light = config["lighting"]
    overcast = rng.uniform() < light["overcast_fraction"]
    elevation = rng.uniform(*light["sun_elevation_deg"])
    # Azimuth relative to the camera's view direction: 0 = sun behind the camera, 180 = backlit.
    relative = rng.uniform(0.0, 360.0)
    sun_yaw = (camera_yaw + 180.0 + relative) % 360.0  # the light's forward vector points *from* the sun
    intensity = rng.uniform(*(light["overcast_intensity"] if overcast else light["clear_intensity"]))
    sun = {"pitch": round(-elevation, 2), "yaw": round(sun_yaw, 2), "intensity": round(intensity, 2),
           "elevation_deg": round(elevation, 2), "relative_azimuth_deg": round(relative, 2)}
    fog_lo, fog_hi = light["fog_density"]
    fog = {"density": round(log_uniform(rng, fog_lo * (4 if overcast else 1), fog_hi), 5)}
    exposure = round(float(rng.uniform(*light["exposure_ev"])), 3)
    return sun, fog, exposure, overcast


def sample_sky(rng: np.random.Generator, config: dict, light: dict) -> tuple[dict | None, dict | None, bool | None]:
    """A real sky HDRI (config "skies": directory, fraction, intensity jitter) with the sun light that
    matches it, or None to keep the atmosphere model. Returns (sky spec, sun spec, overcast)."""
    global _SKY_CATALOGUE
    cfg = config.get("skies")
    if not cfg:
        return None, None, None
    if _SKY_CATALOGUE is None:
        d = Path(cfg["directory"]); d = d if d.is_absolute() else REPO_ROOT / d
        _SKY_CATALOGUE = {k: dict(v, path=str((d / v.get("png", v["file"])).resolve())) for k, v in skies.load_catalogue(d).items()}
    if not _SKY_CATALOGUE or rng.uniform() > cfg.get("fraction", 1.0):
        return None, None, None
    overcast = rng.uniform() < light["overcast_fraction"]
    pool = [k for k, v in _SKY_CATALOGUE.items() if v["overcast"] == overcast] or list(_SKY_CATALOGUE)
    name = pool[int(rng.integers(len(pool)))]; sky = _SKY_CATALOGUE[name]
    rotation = float(rng.uniform(0.0, 360.0))
    # the sky's own sun: the light's forward vector points from the sun, so yaw = azimuth + 180
    elevation = max(float(sky["sun_elevation_deg"]), 2.0)
    yaw = (float(sky["sun_azimuth_deg"]) + rotation + 180.0) % 360.0
    intensity = rng.uniform(*(light["overcast_intensity"] if sky["overcast"] else light["clear_intensity"]))
    sun = {"pitch": round(-elevation, 2), "yaw": round(yaw, 2), "intensity": round(float(intensity), 2),
           "elevation_deg": round(elevation, 2), "from_sky": name}
    lo, hi = cfg.get("intensity_jitter", [0.8, 1.25])
    emissive = float(np.clip(cfg.get("sky_median_target", 1.5) / max(float(sky["sky_median_luminance"]), 1e-3), 0.3, 12.0) * rng.uniform(lo, hi))
    emissive *= float(sky.get("png_scale", 1.0))                           # the PNG holds radiance / png_scale
    spec = {"name": name, "file": sky["path"], "rotation_deg": round(rotation, 2), "intensity": round(emissive, 3), "overcast": bool(sky["overcast"])}
    return spec, sun, bool(sky["overcast"])


# pectoral fin droop (deg below the clip's straight-out fins) per category: surface whales carry them down; the
# rolled categories keep them out so the roll raises one
PEC_DROOP_DEG = {"pec_up": (-35.0, 5.0), "fin_tip": (-25.0, 10.0), "roll_belly": (-20.0, 10.0), "breach": (-20.0, 20.0),
                 "spyhop": (0.0, 20.0), "spyhop_low": (0.0, 20.0)}
WAVE_TIME_S = 0.0                 # the frozen wave time of every render (DefaultEngine.ini r.Water.OverrideWavesTime)
SCENE_OFFSET_CM = 15000.0         # the scene is translated by up to this (each axis) to vary the wave phase under it
HEAD_UNDER_CATEGORIES = {"fluke_down", "fluke_up", "fluke_edge"}       # fluking dives: the head is never out of the water
COMPANION_CATEGORIES = {"logging", "blow", "blow_sliver", "awash", "arch", "back_arch", "surface", "peduncle", "fluke_up", "fluke_down", "pec_up", "fin_tip"}
CALF_CATEGORIES = {"logging": 0.3, "blow": 0.3, "awash": 0.15, "back_arch": 0.25, "surface": 0.25}
# Procedural surface poses for the rig: a straight keyframe plus a hump along the spine (deform.DeformParams; on this
# model arch < 0 = back up, head_pitch < 0 = snout down, tail_curl < 0 = flukes down). Ranges per category; a config
# "procedural_poses" section adds or overrides categories.
PROCEDURAL_POSES = {
    # the breath-then-arch series: from a mild roll-through (-30) to the full arch of the middle of the motion (-80:
    # head and tail stock under, only back and dorsal fin out); apex at the dorsal fin (body fraction ~0.39)
    "back_arch": dict(arch_deg=(-80.0, -30.0), tail_curl_deg=(-30.0, -5.0), head_pitch_deg=(-20.0, -5.0), lateral_deg=(-6.0, 6.0),
                      arch_centre=(0.36, 0.48), arch_span=(0.45, 0.6)),
    # level surfacing (run05): a mild hump so the back from the peduncle to the blowhole is out while the head stays low,
    # as in real surfacing photos (the rostrum is 1 % of the whale pixels photographed in the field, 39 % of run04's)
    # (the straight keyframes carry the head high and the dorsal fin low, so the apex must sit at the dorsal fin, body
    # fraction ~0.39, and the head must pitch well down: a milder back_arch, not a level body)
    "surface": dict(arch_deg=(-55.0, -30.0), tail_curl_deg=(-25.0, -8.0), head_pitch_deg=(-30.0, -12.0), lateral_deg=(-6.0, 6.0),
                    arch_centre=(0.36, 0.42), arch_span=(0.45, 0.6)),
    # the breathing / resting categories (run05): the same back-apex recipe, the clearance cap in the config sets how
    # much back shows (blow: apex a little forward so the blowhole breaks the surface; sliver / awash: the top of the back)
    "logging": dict(arch_deg=(-45.0, -25.0), tail_curl_deg=(-20.0, -5.0), head_pitch_deg=(-28.0, -12.0), lateral_deg=(-5.0, 5.0),
                    arch_centre=(0.36, 0.42), arch_span=(0.5, 0.65)),
    "blow": dict(arch_deg=(-55.0, -30.0), tail_curl_deg=(-25.0, -8.0), head_pitch_deg=(-30.0, -14.0), lateral_deg=(-6.0, 6.0),
                 arch_centre=(0.38, 0.44), arch_span=(0.45, 0.6)),
    "blow_sliver": dict(arch_deg=(-55.0, -30.0), tail_curl_deg=(-25.0, -8.0), head_pitch_deg=(-30.0, -12.0), lateral_deg=(-6.0, 6.0),
                        arch_centre=(0.36, 0.44), arch_span=(0.45, 0.6)),
    "awash": dict(arch_deg=(-55.0, -30.0), tail_curl_deg=(-25.0, -8.0), head_pitch_deg=(-30.0, -12.0), lateral_deg=(-6.0, 6.0),
                  arch_centre=(0.36, 0.44), arch_span=(0.45, 0.6)),
}


def sample_companion(rng: np.random.Generator, config: dict, model, category: str, pose: poses.Pose, placed: dict,
                     geometry: np.ndarray) -> dict | None:
    """A calf beside the whale (config "companion": {"fraction", "scale", "lateral_cm", "along_cm"}): its own
    individual (c..), body scale, surface pose category and clip time, yawed roughly with its mother and
    placed off her flank on the mean water plane. Returns a shot-like dict (whale, pose, body_scale,
    individual, skin, whale_kind) plus "_world" (its vertices, for the camera framing) or None."""
    cfg = config.get("companion") or {}
    if model is None or model.kind != "humpback4k" or category not in COMPANION_CATEGORIES:
        return None
    if rng.uniform() >= float(cfg.get("fraction", 0.0)):
        return None
    calves = sorted(k for k in model.individuals() if k.startswith("c"))
    if not calves:
        return None
    calf = calves[int(rng.integers(len(calves)))]
    scale = float(rng.uniform(*cfg.get("scale", [0.35, 0.5])))
    ccat = weighted_choice(rng, CALF_CATEGORIES)
    base = poses.POSES[ccat]; jitter = config["pose_jitter"]
    cclear = (base.clearance_m * rng.uniform(*jitter["clearance_scale"]) if base.clearance_m > 0 else base.clearance_m * rng.uniform(0.5, 2.0))
    ccap = jitter.get("clearance_max_m", {}).get(ccat)
    cpose = replace(base, pitch_deg=base.pitch_deg + rng.uniform(-jitter["pitch_deg"], jitter["pitch_deg"]),
                    roll_deg=base.roll_deg + rng.uniform(-jitter["roll_deg"], jitter["roll_deg"]),
                    clearance_m=(min(cclear, float(ccap)) if ccap is not None else cclear) * scale,
                    yaw_deg=float((pose.yaw_deg + rng.normal(0.0, 20.0)) % 360.0), splash=None)
    clip_t = sample_clip_time(rng, ccat)
    cshot = {"body_scale": round(scale, 4), "deform": None, "pose": {"clip_t": clip_t} if clip_t is not None else {}, "individual": calf,
             "pec_droop_deg": round(float(rng.uniform(10.0, 40.0)), 2)}
    cgeom = model.posed(cshot)
    cplaced = poses.place(cgeom, cpose)
    # beside the mother: lateral offset past both half-widths, anywhere along her flank, in her heading frame
    side = float(rng.choice([-1.0, 1.0]))
    half_mom = float(np.percentile(np.abs(geometry[:, 1]), 97))          # body frame: half-width, whatever her heading
    half_calf = float(np.percentile(np.abs(cgeom[:, 1]), 97))
    lat_lo, lat_hi = cfg.get("lateral_cm", [40.0, 300.0])
    lateral = side * (half_mom + half_calf + float(rng.uniform(lat_lo, lat_hi)))
    along = float(rng.uniform(*cfg.get("along_cm", [-500.0, 500.0])))
    yaw = math.radians(pose.yaw_deg)
    dx = along * math.cos(yaw) - lateral * math.sin(yaw); dy = along * math.sin(yaw) + lateral * math.cos(yaw)
    mom = placed["whale"]["location"]
    location = [round(mom[0] + dx, 2), round(mom[1] + dy, 2), cplaced["whale"]["location"][2]]
    cshot.update({"whale": {"location": location, "rotation": cplaced["whale"]["rotation"]}, "pose": cshot["pose"] | {k: v for k, v in poses.asdict(cpose).items() if k != "camera"},
                  "category": ccat, "skin": calf, "whale_kind": model.kind, "role": "calf"})
    cshot["_world"] = transform_mesh(cgeom, cshot["whale"])
    return cshot


def make_shot(index: int, rng: np.random.Generator, config: dict, vertices_cm: np.ndarray,
              fin_regions: dict | None = None, model: WhaleModel | None = None) -> dict:
    cycle = config.get("pose_cycle")                 # review renders: after the individual cycle, categories in turn
    n_cycle = max(1, len([k for k in model.individuals() if k.startswith("i")])) if (cycle and model is not None and config.get("individual_cycle")) else 1
    category, pose = sample_pose(rng, config, name=cycle[(index // n_cycle) % len(cycle)] if cycle else None)
    clip_t = sample_clip_time(rng, category) if (model is not None and model.kind == "humpback4k") else None
    if clip_t is not None and pose.pitch_deg < 0 and category not in poses.BACK_HEIGHT_CATEGORIES and category not in HEAD_UNDER_CATEGORIES:
        # tail-up categories: the keyframe already lifts the fluke; take most of that out of the nose-down pitch
        lift = model.fluke_lift_deg(clip_t)
        pose = replace(pose, pitch_deg=min(0.0, pose.pitch_deg + 0.5 * max(lift, 0.0)))
    if category in HEAD_UNDER_CATEGORIES:
        pose = replace(pose, pitch_deg=min(pose.pitch_deg, -15.0))
    # The sea is chosen first so that a pose meant to show above the water clears the waves: the
    # clearance is lifted by a fraction of the crest height (config body.wave_clearance_fraction).
    sea = weighted_choice(rng, config["sea_state_weights"])
    lift = float(config.get("body", {}).get("wave_clearance_fraction", 0.0))
    if lift > 0 and pose.clearance_m > 0:
        pose = replace(pose, clearance_m=pose.clearance_m + lift * seastate.crest_height_m(sea))
    # Body deformation (the "animation") and skin variant, both optional in the config.
    body = config.get("body", {})
    params = deform.DeformParams()
    rigged = model is not None and model.kind == "humpback4k"
    if not rigged and rng.uniform() < body.get("deform_fraction", 0.0):
        params = deform.sample_params(rng, category, strength=float(rng.uniform(*body.get("strength", [0.6, 1.2]))))
    # Body size: humpback length varies from calf to large adult; the canonical labels stay on the unit mesh.
    scale = float(rng.uniform(*body["length_scale"])) if body.get("length_scale") else 1.0
    procedural_bend = False
    pec_droop = None; fluke_curl = None
    if model is not None and model.kind == "humpback4k":
        pec_droop = round(float(rng.uniform(*PEC_DROOP_DEG.get(category, (10.0, 40.0)))), 2)
        # fluke blade angle through the dive: early in the dive (clip_t <= 1.2) the tips hang down as the whale
        # doubles over, later they point up; fluke_edge always shows the tips pointing up
        if category == "fluke_up":
            lo, hi = (-70.0, -10.0) if (clip_t is not None and clip_t <= 1.25) else (-15.0, 20.0)
        elif category == "fluke_edge":
            lo, hi = (0.0, 25.0)
        elif category == "fluke_down":
            lo, hi = (-95.0, -60.0)           # the blade hangs from the raised tail stock
        else:
            lo, hi = (-6.0, 6.0)
        fluke_curl = round(float(rng.uniform(lo, hi)), 2)
    procedural = PROCEDURAL_POSES | {k: {kk: tuple(vv) for kk, vv in v.items()} for k, v in config.get("procedural_poses", {}).items()}
    if model is not None and model.kind == "humpback4k" and category in procedural:
        # straight keyframe + hump: how far the back humps, how the tail stock curls down behind it and the head pitches
        # down; the clearance jitter (measured on the back only, poses.BACK_HEIGHT_CATEGORIES) sets how much back shows
        spec = procedural[category]
        params = deform.DeformParams(**{key: float(rng.uniform(*spec[key])) for key in spec})
        procedural_bend = True
    if model is not None:
        geometry = None   # computed below, once the individual is known
    else:
        base = vertices_cm * scale if abs(scale - 1.0) > 1e-6 else vertices_cm
        regions = fin_regions if abs(scale - 1.0) < 1e-6 else deform.fin_regions(base)
        geometry = base if params.is_identity() else deform.deform(base, params, regions)
    skin_weights = body.get("skin_weights", {"00": 1})
    skin = weighted_choice(rng, skin_weights)
    individual = None
    if model is not None and model.kind == "humpback4k":
        adults = sorted(k for k in model.individuals() if k.startswith("i"))
        if adults:
            # individual_cycle (review renders): shot i gets adult i mod N, so every individual appears in turn
            individual = adults[index % len(adults)] if config.get("individual_cycle") else adults[int(rng.integers(len(adults)))]
            skin = individual                      # the individual's texture is its skin instance
    # Wet skin: a whale that just surfaced is glossy (water sheeting); it dulls as it dries. RoughnessScale
    # multiplies the authored roughness (0.85 was the fixed value through run03).
    wet_cfg = body.get("wetness", {})
    wetness = float(rng.uniform(*wet_cfg.get("drying", [0.15, 0.5]))) if rng.uniform() < wet_cfg.get("drying_fraction", 0.15) else float(rng.uniform(*wet_cfg.get("wet", [0.55, 1.0])))
    roughness_scale = round(0.85 - 0.6 * wetness, 3)
    if model is not None:
        geometry = model.posed({"body_scale": scale, "deform": deform.params_to_dict(params), "pec_droop_deg": pec_droop, "fluke_curl_deg": fluke_curl,
                                "pose": {"clip_t": clip_t} | ({"bend": "procedural"} if procedural_bend else {}), "individual": individual})
    spray_cfg = config["splash"]
    if pose.splash and rng.uniform() > spray_cfg["probability"].get(category, 1.0):
        pose = replace(pose, splash=None)                      # this time, no white water
    elif pose.splash:
        lo, hi = spray_cfg["scale_jitter"]
        pose = replace(pose, splash_scale=pose.splash_scale * float(rng.uniform(lo, hi)),
                       splash_amount=pose.splash_amount * float(rng.uniform(*spray_cfg["amount_jitter"])))
    elif pose.clearance_m > 0 and rng.uniform() < spray_cfg["extra_wash_fraction"]:
        pose = replace(pose, splash="wash", splash_scale=float(rng.uniform(0.3, 0.8)), splash_amount=float(rng.uniform(0.3, 0.8)))
    top_mask = ((geometry[:, 0] > poses.BACK_X_MIN_CM) & (geometry[:, 0] < poses.BACK_X_MAX_CM)) if category in poses.BACK_HEIGHT_CATEGORIES else None
    placed = poses.place(geometry, pose, top_mask=top_mask)
    if category in HEAD_UNDER_CATEGORIES:
        # the head and front body stay under water in a fluking dive: pitch the nose down until they are
        head = geometry[:, 0] > 150.0
        for _ in range(8):
            world_h = transform_mesh(geometry, placed["whale"])[head]
            if (world_h[:, 2] > 0.0).mean() < 0.01:
                break
            pose = replace(pose, pitch_deg=pose.pitch_deg - 6.0)
            placed = poses.place(geometry, pose, top_mask=top_mask)
    world = transform_mesh(geometry, placed["whale"])
    companion = sample_companion(rng, config, model, category, pose, placed, geometry)
    frame_world = np.concatenate([world, companion.pop("_world")]) if companion else world
    camera = sample_camera_for_size(rng, config, pose, placed, frame_world, 0.0)
    no_whale = rng.uniform() < config["no_whale_in_frame_fraction"]
    if no_whale:
        camera["rotation"]["yaw"] = round(camera["rotation"]["yaw"] + rng.choice([-1, 1]) * rng.uniform(30.0, 70.0), 4)
    sun, fog, exposure, overcast = sample_lighting(rng, config, camera["rotation"]["yaw"])
    sky, sky_sun, sky_overcast = sample_sky(rng, config, config["lighting"])
    if sky:
        sun, overcast = sky_sun, sky_overcast                                   # the sun follows the real sky
    # Depth of field like a long-lens field photo: focus on the whale's above-water centroid, the
    # f-number log-uniform in camera.aperture_f. wf_build_batch keys both on the cine camera.
    aim = np.asarray(placed["stats"]["aim_point"], float)
    camera["focus_distance_m"] = round(float(np.linalg.norm(aim - np.asarray(camera["location"], float))) / 100.0, 2)
    f_lo, f_hi = config["camera"].get("aperture_f", [4.0, 11.0])
    camera["aperture_f"] = round(float(log_uniform(rng, f_lo, f_hi)), 2)
    look = weighted_choice(rng, {name: spec["weight"] for name, spec in config["looks"].items()})
    splash_spec = placed["splash"]
    shot = {
        "id": f"{index:06d}",
        "name": f"{index:06d}_{category}_{sea}",
        "category": category,
        "negative": bool(no_whale or placed["stats"]["vertex_fraction_above_water"] == 0.0),
        "no_whale_in_frame": bool(no_whale),
        "overcast": bool(overcast),
        "look": look,
        "pose": {k: v for k, v in poses.asdict(pose).items() if k != "camera"} | ({"clip_t": clip_t} if clip_t is not None else {})
                | ({"bend": "procedural"} if procedural_bend else {}),
        "whale_kind": model.kind if model is not None else "k9239",
        "individual": individual,
        "pec_droop_deg": pec_droop,
        "fluke_curl_deg": fluke_curl,
        "companion": companion,
        "deform": deform.params_to_dict(params),
        "body_scale": round(scale, 4),
        "skin": skin,
        "skin_wetness": round(wetness, 3),
        "skin_roughness_scale": roughness_scale,
        "whale": placed["whale"],
        "camera": camera,
        "splash": splash_spec,
        "sun": sun,
        "sky": sky,
        "fog": fog,
        "exposure_ev": exposure,
        # Wave phase. The render freezes the water at this time (r.Water.OverrideWavesTime, keyed per
        # shot by wf_build_batch), so the editor can put the whale and the splash on the actual surface.
        "wave_time_s": None,                      # chosen below, with the waves in hand
        "stats": placed["stats"],
        "ocean": {"vectors": dict(config["looks"][look]["vectors"])},
    }
    # The ocean renders ONE wave set per batch (wf_build_batch applies batch["waves"]), and batches group shots
    # by (sea state, look): the wave seed must therefore be a function of those, never of the shot, or the lift
    # onto the surface, the white water and the through-water labels are computed on waves the render never
    # shows (that was the case up to run03 for every shot but the first of each batch). Wave phase still
    # varies per shot through wave_time_s.
    seastate.apply_to_shot(shot, sea, seed=sum(ord(ch) for ch in look) % 1000)
    # The water surface at the shot's frozen wave time (factory.gerstner = the engine's own waves): lift the
    # whale (with its camera, so the framing holds) by the height under its above-water centroid and the splash
    # by the height at its centre; then the white water along the contact line on that surface.
    waves = gerstner.generate_waves(shot["waves"])
    # Wave phase. The render freezes the ocean at WAVE_TIME_S for every shot (r.Water.OverrideWavesTime in
    # DefaultEngine.ini: a per-shot cvar track never reached the water, it ran on the sequence clock), so the
    # phase varies by translating the whole scene (whale, camera, calf, splash, white water) across the
    # frozen wave field instead. A 16 m body placed rigidly on 10-30 m waves: of several candidate offsets
    # keep the one where the surface along the body is most even (a real whale rides the swell; a crest
    # cutting through a surfacing back is what a random phase produced), preferring the whale high on it.
    t = WAVE_TIME_S
    aim0 = np.asarray(placed["stats"]["aim_point"], float)
    n_candidates = {"rough": 6, "moderate": 4, "light": 2}.get(sea, 1)
    yaw_r = math.radians(pose.yaw_deg); fwd = np.array([math.cos(yaw_r), math.sin(yaw_r)])
    along = np.linspace(-0.4, 0.4, 9) * float(geometry[:, 0].max() - geometry[:, 0].min())
    best, best_score = None, None
    for _ in range(n_candidates):
        off = rng.uniform(-SCENE_OFFSET_CM, SCENE_OFFSET_CM, 2)
        probe = aim0[None, :2] + off[None, :] + along[:, None] * fwd[None, :]
        hs = gerstner.surface_height(waves, probe[:, 0], probe[:, 1], t)
        score = float(hs.max() - hs.min()) - 0.5 * float(gerstner.surface_height(waves, aim0[0] + off[0], aim0[1] + off[1], t))
        if best_score is None or score < best_score:
            best, best_score = off, score
    ox, oy = (round(float(best[0]), 2), round(float(best[1]), 2))
    for loc in (shot["whale"]["location"], camera["location"]) + ((companion["whale"]["location"],) if companion else ()) + ((splash_spec["centre_cm"],) if splash_spec else ()):
        loc[0] = round(loc[0] + ox, 2); loc[1] = round(loc[1] + oy, 2)
    placed["stats"]["aim_point"] = [round(aim0[0] + ox, 2), round(aim0[1] + oy, 2), round(float(aim0[2]), 2)]
    aim = placed["stats"]["aim_point"]
    world = transform_mesh(geometry, shot["whale"])
    shot["scene_offset_cm"] = [ox, oy]
    shot["wave_time_s"] = t
    dz = float(gerstner.surface_height(waves, aim[0], aim[1], t))
    # low cameras in rough water see little but the nearest crest: keep the boat camera up
    floor_cm = 100.0 * {"rough": 2.0, "moderate": 1.5}.get(sea, 0.0)
    if camera["location"][2] < floor_cm:
        camera["location"][2] = round(floor_cm, 2)
        if not no_whale:                           # the no-whale shots keep their deliberately turned-away yaw
            camera["rotation"] = poses.look_at_rotation(np.asarray(camera["location"], float), np.asarray(aim, float))
    shot["whale"]["location"][2] = round(shot["whale"]["location"][2] + dz, 2)
    camera["location"][2] = round(camera["location"][2] + dz, 2)
    shot["water_surface_z_cm"] = {"whale": round(dz, 2)}
    if companion:
        cx, cy = companion["whale"]["location"][:2]
        dzc = float(gerstner.surface_height(waves, cx, cy, t))
        companion["whale"]["location"][2] = round(companion["whale"]["location"][2] + dzc, 2)
        shot["water_surface_z_cm"]["companion"] = round(dzc, 2)
    if splash_spec:
        ds = float(gerstner.surface_height(waves, splash_spec["centre_cm"][0], splash_spec["centre_cm"][1], t))
        splash_spec["centre_cm"][2] = round(splash_spec["centre_cm"][2] + ds, 2)
        shot["water_surface_z_cm"]["splash"] = round(ds, 2)
    shot["whitewater"] = whitewater.spec(shot, world + np.array([0.0, 0.0, dz]), waves, t, rng, config)
    # Under a long lens the fine chop is sub-pixel: scale the water's normal-map strength down with the
    # focal length (camera.normal_strength_focal_mm), otherwise it renders as aliased streaks.
    ref = config["camera"].get("normal_strength_focal_mm")
    if ref:
        k = float(np.clip(math.sqrt(ref / max(camera["focal_mm"], 1e-6)), 0.35, 1.0))
        scalars = shot["ocean"].setdefault("scalars", {})
        for key in ("Default Near Normal Strength", "Default Distant Normal Strength"):
            if key in scalars:
                scalars[key] = round(scalars[key] * k, 4)
        shot["ocean"]["normal_strength_scale"] = round(k, 4)
    return shot


def posed_world(shot: dict, vertices_cm: np.ndarray, fin_regions: dict | None, model: WhaleModel | None = None) -> np.ndarray:
    """The shot's whale in world cm (scaled, bent, posed and lifted onto the wave surface), rebuilt
    from the shot dict exactly as make_shot built it: for mesh writers after sampling."""
    if model is not None:
        return transform_mesh(model.posed(shot), shot["whale"])
    scale = float(shot.get("body_scale", 1.0))
    base = vertices_cm * scale if abs(scale - 1.0) > 1e-6 else vertices_cm
    params = deform.DeformParams(**shot["deform"]) if shot.get("deform") else deform.DeformParams()
    if params.is_identity():
        geometry = base
    else:
        regions = fin_regions if (abs(scale - 1.0) < 1e-6 and fin_regions is not None) else deform.fin_regions(base)
        geometry = deform.deform(base, params, regions)
    return transform_mesh(geometry, shot["whale"])


def make_batches(shots: list[dict], batch_size: int) -> list[dict]:
    # Waves and the ocean material are applied once per batch (wf_build_batch), from the first
    # shot, so everything that varies them must be part of the grouping key: sea state, look and
    # the focal-length normal-strength scale (binned to 0.1).
    groups: dict[tuple, list] = {}
    for shot in shots:
        scale_bin = round(float(shot.get("ocean", {}).get("normal_strength_scale", 1.0)), 1)
        groups.setdefault((shot["sea_state"], shot["look"], scale_bin), []).append(shot)
    batches = []
    for (sea, look, scale_bin), members in sorted(groups.items()):
        for start in range(0, len(members), batch_size):
            chunk = members[start:start + batch_size]
            batches.append({
                "index": len(batches),
                "sea_state": sea,
                "look": look,
                "normal_strength_scale": scale_bin,
                "waves": chunk[0]["waves"],
                "ocean": chunk[0]["ocean"],
                "shots": chunk,
            })
    return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    model = WhaleModel.from_config(config)
    vertices = model.canonical if model.kind != "k9239" else poses.load_obj_vertices(args.mesh)
    regions = deform.fin_regions(vertices) if config.get("body", {}).get("deform_fraction", 0.0) > 0 else None
    rng = np.random.default_rng(args.seed)
    shots = [make_shot(i, rng, config, vertices, regions, model if model.kind != "k9239" else None) for i in range(args.count)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        splash.ensure_obj(shot.get("splash"), args.out_dir / "splash", shot["id"])
        if shot.get("whitewater"):
            whitewater.ensure_obj(shot["whitewater"], args.out_dir / "whitewater", shot["id"], posed_world(shot, vertices, regions, model if model.kind != "k9239" else None),
                                  gerstner.generate_waves(shot["waves"]), shot.get("splash"))
    (args.out_dir / "run_config.json").write_text(json.dumps({"config": config, "seed": args.seed, "count": args.count,
                                                              "mesh": str(args.mesh)}, indent=2))
    with open(args.out_dir / "shotlist.jsonl", "w") as handle:
        for shot in shots:
            handle.write(json.dumps(shot) + "\n")
    batch_dir = args.out_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    batches = make_batches(shots, config["batch_size"])
    library = islands.make_library(args.out_dir / "islands", np.random.default_rng(args.seed + 7919), config["islands"].get("library", 6)) if config.get("islands") else []
    for batch in batches:
        batch["resolution"] = config["resolution"]
        batch["whale_kind"] = model.kind
        batch["islands"] = islands.place(np.random.default_rng([args.seed, batch["index"]]), library, config) if library else []
        batch["frames_per_shot"] = config["frames_per_shot"]
        (batch_dir / f"batch_{batch['index']:03d}.json").write_text(json.dumps(batch))
    categories = {}
    for shot in shots:
        categories[shot["category"]] = categories.get(shot["category"], 0) + 1
    print(f"{len(shots)} shots in {len(batches)} batches -> {args.out_dir}")
    print("categories:", dict(sorted(categories.items())))
    print("negatives:", sum(s["negative"] for s in shots), "no whale in frame:", sum(s["no_whale_in_frame"] for s in shots))
    extents = np.array([s["camera"].get("extent_frame_fraction", 0.0) for s in shots if not s["negative"]])
    if len(extents):
        print(f"above-water extent (fraction of frame, whale shots): min {extents.min():.2f} p10 {np.percentile(extents, 10):.2f} "
              f"median {np.median(extents):.2f} max {extents.max():.2f}; "
              f"below {config['camera'].get('min_extent_frame_fraction', 0.0):.2f}: {(extents < config['camera'].get('min_extent_frame_fraction', 0.0)).sum()}")
    print("deformed:", sum(not deform.params_from_dict(s["deform"]).is_identity() for s in shots),
          "skins:", dict(sorted(Counter(s["skin"] for s in shots).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
