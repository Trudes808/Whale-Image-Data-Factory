"""Shots with several whales (run06): scene types, per-animal bodies, placement next to and overlapping each other,
a capsule collision test, and the camera framing of the group.

A config with a "scenes" section switches factory.shotlist.make_shot to this path; configs without it (run03-run05)
keep the single-companion sampler (shotlist.sample_companion) and produce the same shot lists as before.

    "scenes": {
      "weights": {"single": 30, "mother_calf": 15, "mother_calf_escort": 15, "pair": 20, "competitive": 20},
      "competitive_size": [3, 6],                 # adults in a competitive group, the primary included
      "adult_pose_weights": {...}, "calf_pose_weights": {...},   # categories of the further animals
      "calf_scale": [0.35, 0.5],
      "gap_m": [1.0, 6.0],                        # body-to-body gap of neighbours (side by side, in line, staggered)
      "arrangement_weights": {"side_by_side": 40, "in_line": 25, "staggered": 35},
      "calf_touch_fraction": 0.5,                 # the calf touching the mother's flank (else 0.4-3 m off it)
      "escort_gap_m": [2.0, 10.0],
      "heading_spread_deg": 20, "crossing_fraction": 0.12,
      "group_framing_fraction": 0.55,             # camera frames the whole group, else the primary (others partly in frame)
      "max_tries": 25
    }

The primary (instance 1) is always an adult: the mother in the mother-calf scenes. Every further animal is a dict
like the primary shot (whale transform, pose, clip_t, body_scale, individual, skin, wetness, deform, splash,
whitewater, role, instance_id) in shot["companions"]; shot["scene"] records the type, arrangement and framing.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from factory import deform, poses, seastate
from factory.project import transform_mesh

SCENE_TYPES = ("single", "mother_calf", "mother_calf_escort", "pair", "competitive")
DEFAULTS = {
    "weights": {"single": 30, "mother_calf": 15, "mother_calf_escort": 15, "pair": 20, "competitive": 20},
    "competitive_size": [3, 6],
    "calf_scale": [0.35, 0.5],
    "gap_m": [1.0, 6.0],
    "arrangement_weights": {"side_by_side": 40, "in_line": 25, "staggered": 35},
    "calf_touch_fraction": 0.5,
    "escort_gap_m": [2.0, 10.0],
    "heading_spread_deg": 20.0,
    "crossing_fraction": 0.12,
    "group_framing_fraction": 0.55,
    "max_tries": 25,
}
NEGATIVE_CATEGORIES = {"submerged"}
CAPSULE_MARGIN_CM = 10.0
N_SPINE = 10


def cfg_of(config: dict) -> dict:
    return DEFAULTS | (config.get("scenes") or {})


# ---- collision: capsules along the spine, the pectoral fins and the fluke blades --------------------------------

def capsules(model, geometry: np.ndarray, whale: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A (K, 3), B (K, 3), radius (K,)) world cm: segments between spine slice centroids of the body (pectorals and
    flukes excluded), one per pectoral fin (base -> tip) and two for the fluke blades (spine end -> tips)."""
    groups = model.part_groups()
    world = transform_mesh(geometry, whale)
    x = model.canonical[:, 0]
    fin = np.zeros(len(x), bool)
    for name in ("pectoral", "fluke"):
        fin[groups.get(name, [])] = True
    body = ~fin
    edges = np.linspace(x[body].min(), x[body].max(), N_SPINE + 1)
    cents, radii = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = body & (x >= a) & (x <= b)
        if not sel.any():
            continue
        c = world[sel].mean(0)
        cents.append(c); radii.append(float(np.percentile(np.linalg.norm(world[sel] - c, axis=1), 85)))
    A = [cents[i] for i in range(len(cents) - 1)]; B = [cents[i + 1] for i in range(len(cents) - 1)]
    R = [max(radii[i], radii[i + 1]) for i in range(len(cents) - 1)]
    idx = model.landmark_indices()

    def pt(name):
        v = idx.get(name)
        if v is None:
            return None
        return world[v].mean(0) if isinstance(v, list) else world[v]
    scale = float(np.ptp(geometry[:, 0]) / max(np.ptp(model.rest[:, 0]), 1e-6))
    for a, b, r in (("pectoral_base_left", "pectoral_tip_left", 30.0), ("pectoral_base_right", "pectoral_tip_right", 30.0),
                    ("spine_end", "fluke_tip_left", 20.0), ("spine_end", "fluke_tip_right", 20.0)):
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            A.append(pa); B.append(pb); R.append(r * scale)
    return np.array(A), np.array(B), np.array(R)


def segment_distances(A1, B1, A2, B2) -> np.ndarray:
    """(K1, K2) closest distances between segments A1B1[i] and A2B2[j] (clamped closest-point of two segments)."""
    p1 = A1[:, None, :]; d1 = (B1 - A1)[:, None, :]
    p2 = A2[None, :, :]; d2 = (B2 - A2)[None, :, :]
    r = p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = np.where(denom > 1e-9, np.clip((b * f - c * e) / np.where(denom > 1e-9, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / np.where(e > 1e-9, e, 1.0)
    t_c = np.clip(t, 0.0, 1.0)
    s = np.where((t < 0.0) | (t > 1.0), np.clip((b * t_c - c) / np.where(a > 1e-9, a, 1.0), 0.0, 1.0), s)
    q1 = p1 + d1 * s[..., None]; q2 = p2 + d2 * t_c[..., None]
    return np.linalg.norm(q1 - q2, axis=-1)


def collides(caps_a, caps_b, margin_cm: float = CAPSULE_MARGIN_CM) -> bool:
    A1, B1, R1 = caps_a; A2, B2, R2 = caps_b
    d = segment_distances(A1, B1, A2, B2)
    return bool((d < R1[:, None] + R2[None, :] + margin_cm).any())


# ---- per-animal body sampling (mirrors shotlist.make_shot for the primary) ------------------------------------------

def sample_body(rng: np.random.Generator, config: dict, model, role: str, individual: str, sea: str, category: str) -> dict:
    """One further animal's body: pose category, clip keyframe, pitch rules, clearance (wave lift, calves scaled),
    procedural hump, pectoral droop, fluke curl, length scale, wetness, splash. Returns the shot-like animal dict plus
    the private keys _pose (poses.Pose), _geometry (whale frame) and _top_mask."""
    from factory import shotlist as sl
    sc = cfg_of(config)
    category, pose = sl.sample_pose(rng, config, name=category)
    clip_t = sl.sample_clip_time(rng, category)
    if clip_t is not None and pose.pitch_deg < 0 and category not in poses.BACK_HEIGHT_CATEGORIES and category not in sl.HEAD_UNDER_CATEGORIES:
        lift = model.fluke_lift_deg(clip_t)
        pose = replace(pose, pitch_deg=min(0.0, pose.pitch_deg + 0.5 * max(lift, 0.0)))
    if category in sl.HEAD_UNDER_CATEGORIES:
        pose = replace(pose, pitch_deg=min(pose.pitch_deg, -15.0))
    body = config.get("body", {})
    if role == "calf":
        scale = float(rng.uniform(*sc["calf_scale"]))
    else:
        scale = float(rng.uniform(*body["length_scale"])) if body.get("length_scale") else 1.0
    lift = float(body.get("wave_clearance_fraction", 0.0))
    if role == "calf":
        pose = replace(pose, clearance_m=pose.clearance_m * scale)          # a calf shows as much of itself, at its size
    if lift > 0 and pose.clearance_m > 0:
        pose = replace(pose, clearance_m=pose.clearance_m + lift * seastate.crest_height_m(sea) * (scale if role == "calf" else 1.0))
    pec_droop = round(float(rng.uniform(*sl.PEC_DROOP_DEG.get(category, (10.0, 40.0)))), 2)
    if category == "fluke_up":
        lo, hi = (-70.0, -10.0) if (clip_t is not None and clip_t <= 1.25) else (-15.0, 20.0)
    elif category == "fluke_edge":
        lo, hi = (0.0, 25.0)
    elif category == "fluke_down":
        lo, hi = (-95.0, -60.0)
    else:
        lo, hi = (-6.0, 6.0)
    fluke_curl = round(float(rng.uniform(lo, hi)), 2)
    procedural = sl.PROCEDURAL_POSES | {k: {kk: tuple(vv) for kk, vv in v.items()} for k, v in config.get("procedural_poses", {}).items()}
    params = deform.DeformParams(); bend = False
    if category in procedural:
        spec = procedural[category]
        params = deform.DeformParams(**{key: float(rng.uniform(*spec[key])) for key in spec})
        bend = True
    wet_cfg = body.get("wetness", {})
    wetness = float(rng.uniform(*wet_cfg.get("drying", [0.15, 0.5]))) if rng.uniform() < wet_cfg.get("drying_fraction", 0.15) else float(rng.uniform(*wet_cfg.get("wet", [0.55, 1.0])))
    spray_cfg = config["splash"]
    if pose.splash and rng.uniform() > spray_cfg["probability"].get(category, 1.0):
        pose = replace(pose, splash=None)
    elif pose.splash:
        pose = replace(pose, splash_scale=pose.splash_scale * float(rng.uniform(*spray_cfg["scale_jitter"])) * (scale if role == "calf" else 1.0),
                       splash_amount=pose.splash_amount * float(rng.uniform(*spray_cfg["amount_jitter"])))
    animal = {"role": role, "category": category, "individual": individual, "skin": individual, "whale_kind": model.kind,
              "body_scale": round(scale, 4), "deform": deform.params_to_dict(params), "pec_droop_deg": pec_droop, "fluke_curl_deg": fluke_curl,
              "pose": {"clip_t": clip_t} | ({"bend": "procedural"} if bend else {}),
              "skin_wetness": round(wetness, 3), "skin_roughness_scale": round(0.85 - 0.6 * wetness, 3)}
    geometry = model.posed(animal)
    top_mask = ((geometry[:, 0] > poses.BACK_X_MIN_CM * scale) & (geometry[:, 0] < poses.BACK_X_MAX_CM * scale)) if category in poses.BACK_HEIGHT_CATEGORIES else None
    animal.update(_pose=pose, _geometry=geometry, _top_mask=top_mask)
    return animal


def finish_placement(model, animal: dict, yaw_deg: float, xy) -> None:
    """Rigid placement at heading yaw_deg with its body origin at xy (on the mean water plane), the head-under rule of
    the fluking dives, and the splash moved with the body. Fills whale, pose, splash, stats, _world."""
    from factory import shotlist as sl
    pose = replace(animal["_pose"], yaw_deg=float(yaw_deg % 360.0))
    geometry, top_mask = animal["_geometry"], animal["_top_mask"]
    placed = poses.place(geometry, pose, top_mask=top_mask)
    if animal["category"] in sl.HEAD_UNDER_CATEGORIES:
        head = geometry[:, 0] > 150.0 * float(animal["body_scale"])
        for _ in range(8):
            if (transform_mesh(geometry, placed["whale"])[head][:, 2] > 0.0).mean() < 0.01:
                break
            pose = replace(pose, pitch_deg=pose.pitch_deg - 6.0)
            placed = poses.place(geometry, pose, top_mask=top_mask)
    dx, dy = float(xy[0]), float(xy[1])
    loc = placed["whale"]["location"]
    placed["whale"]["location"] = [round(loc[0] + dx, 2), round(loc[1] + dy, 2), loc[2]]
    splash = placed["splash"]
    if splash:
        splash["centre_cm"][0] = round(splash["centre_cm"][0] + dx, 2); splash["centre_cm"][1] = round(splash["centre_cm"][1] + dy, 2)
    aim = placed["stats"]["aim_point"]
    placed["stats"]["aim_point"] = [round(aim[0] + dx, 2), round(aim[1] + dy, 2), aim[2]]
    animal["whale"] = placed["whale"]
    animal["pose"] = animal["pose"] | {k: v for k, v in poses.asdict(pose).items() if k != "camera"}
    animal["splash"] = splash
    animal["stats"] = placed["stats"]
    animal["_pose"] = pose
    animal["_world"] = transform_mesh(geometry, placed["whale"])


def body_extent(geometry: np.ndarray, model) -> tuple[float, float]:
    """(half length, half width) of the body in its own frame, pectorals excluded (cm)."""
    groups = model.part_groups()
    keep = np.ones(len(geometry), bool); keep[groups.get("pectoral", [])] = False
    g = geometry[keep]
    return float(np.ptp(g[:, 0]) / 2.0), float(np.percentile(np.abs(g[:, 1]), 97))


def offset_xy(anchor_xy, anchor_yaw_deg: float, along_cm: float, lateral_cm: float) -> np.ndarray:
    yaw = math.radians(anchor_yaw_deg)
    return np.array([anchor_xy[0] + along_cm * math.cos(yaw) - lateral_cm * math.sin(yaw),
                     anchor_xy[1] + along_cm * math.sin(yaw) + lateral_cm * math.cos(yaw)])


def relative_offset(rng, arrangement: str, a_ext, b_ext, gap_cm: float) -> tuple[float, float]:
    """(along, lateral) cm of a neighbour's origin from an anchor's, in the anchor's heading frame."""
    (la, wa), (lb, wb) = a_ext, b_ext
    side = float(rng.choice([-1.0, 1.0]))
    if arrangement == "side_by_side":
        return float(rng.uniform(-0.25, 0.25)) * 2 * la, side * (wa + wb + gap_cm)
    if arrangement == "in_line":                                         # one behind the other along the heading
        sign = -1.0 if rng.uniform() < 0.6 else 1.0
        return sign * (la + lb + gap_cm), float(rng.uniform(-0.6, 0.6)) * (wa + wb)
    # staggered: overlapping in length, off to the side
    sign = -1.0 if rng.uniform() < 0.6 else 1.0
    return sign * float(rng.uniform(0.3, 0.75)) * (la + lb), side * (wa + wb + gap_cm) * float(rng.uniform(0.6, 1.0))


def sample_scene(rng: np.random.Generator, config: dict, model, category: str, pose, placed: dict, geometry: np.ndarray,
                 sea: str, primary: dict) -> dict:
    """The further animals of a shot. primary: {"individual", "body_scale"} of the shot's own whale (already placed at
    the origin by make_shot). Returns {"scene": {...}, "companions": [animal dicts], "frame_world": (M, 3) world cm,
    "aim_point": [x, y, z]} (frame_world / aim_point: what the camera frames)."""
    sc = cfg_of(config)
    stype = _weighted(rng, sc["weights"])
    adults = sorted(k for k in model.individuals() if k.startswith("i"))
    calves = sorted(k for k in model.individuals() if k.startswith("c"))
    used = {primary.get("individual")}
    roles = []
    if stype == "mother_calf":
        roles = ["calf"]
    elif stype == "mother_calf_escort":
        roles = ["calf", "escort"]
    elif stype == "pair":
        roles = ["adult"]
    elif stype == "competitive":
        lo, hi = sc["competitive_size"]
        roles = ["adult"] * (int(rng.integers(int(lo), int(hi) + 1)) - 1)
    adult_weights = sc.get("adult_pose_weights") or {k: v for k, v in config["pose_weights"].items() if k not in NEGATIVE_CATEGORIES}
    from factory import shotlist as sl
    calf_weights = sc.get("calf_pose_weights") or sl.CALF_CATEGORIES
    spread = float(sc["heading_spread_deg"])
    group_yaw = float(pose.yaw_deg)
    prim_caps = capsules(model, geometry, placed["whale"])
    anchors = [{"xy": np.array(placed["whale"]["location"][:2], float), "yaw": group_yaw, "ext": body_extent(geometry, model), "caps": prim_caps, "role": "primary"}]
    companions, arrangements, dropped = [], [], 0
    for role in roles:
        pool = calves if role == "calf" else [a for a in adults if a not in used]
        if not pool:
            dropped += 1
            continue
        individual = pool[int(rng.integers(len(pool)))]
        used.add(individual)
        cat = _weighted(rng, calf_weights if role == "calf" else adult_weights)
        animal = sample_body(rng, config, model, role, individual, sea, cat)
        ext = body_extent(animal["_geometry"], model)
        ok = False
        for _ in range(int(sc["max_tries"])):
            crossing = role != "calf" and rng.uniform() < float(sc["crossing_fraction"])
            yaw = group_yaw + (float(rng.choice([-1.0, 1.0])) * float(rng.uniform(45.0, 135.0)) if crossing else float(rng.normal(0.0, spread * (0.5 if role == "calf" else 1.0))))
            if role == "calf":
                a_idx = 0; anchor = anchors[0]                               # the mother
                if rng.uniform() < float(sc["calf_touch_fraction"]):
                    arrangement, gap = "calf_flank", float(rng.uniform(0.0, 40.0))
                    along = float(rng.uniform(-0.15, 0.35)) * 2 * anchor["ext"][0]
                    lateral = float(rng.choice([-1.0, 1.0])) * (anchor["ext"][1] + ext[1] + gap)
                else:
                    arrangement, gap = "calf_near", float(rng.uniform(40.0, 300.0))
                    along = float(rng.uniform(-0.4, 0.5)) * 2 * anchor["ext"][0]
                    lateral = float(rng.choice([-1.0, 1.0])) * (anchor["ext"][1] + ext[1] + gap)
            elif role == "escort":
                a_idx = 0; anchor = anchors[0]
                arrangement = _weighted(rng, {"in_line": 40, "staggered": 40, "side_by_side": 20})
                gap = 100.0 * float(rng.uniform(*sc["escort_gap_m"]))
                along, lateral = relative_offset(rng, arrangement, anchor["ext"], ext, gap)
            else:
                a_idx = int(rng.integers(len(anchors))); anchor = anchors[a_idx]   # groups grow from any member
                arrangement = _weighted(rng, sc["arrangement_weights"])
                gap = 100.0 * float(rng.uniform(*sc["gap_m"]))
                along, lateral = relative_offset(rng, arrangement, anchor["ext"], ext, gap)
            xy = offset_xy(anchor["xy"], anchor["yaw"], along, lateral)
            finish_placement(model, animal, yaw, xy)
            caps = capsules(model, animal["_geometry"], animal["whale"])
            if not any(collides(caps, other["caps"]) for other in anchors):
                ok = True
                break
        if not ok:
            dropped += 1
            continue
        anchors.append({"xy": np.array(animal["whale"]["location"][:2], float), "yaw": float(animal["_pose"].yaw_deg), "ext": ext, "caps": caps, "role": role})
        arrangements.append({"role": role, "arrangement": arrangement, "gap_cm": round(gap, 1), "relative_to": a_idx + 1})
        companions.append(animal)
    for k, animal in enumerate(companions):
        animal["instance_id"] = k + 2
    group = bool(companions) and rng.uniform() < float(sc["group_framing_fraction"])
    worlds = [transform_mesh(geometry, placed["whale"])] + [a["_world"] for a in companions]
    if group:
        frame_world = np.concatenate(worlds)
        above = frame_world[frame_world[:, 2] > 0.0]
        aim = above.mean(0) if len(above) else frame_world.mean(0)
        aim_point = [round(float(c), 2) for c in aim]
    else:
        frame_world = worlds[0]
        aim_point = list(placed["stats"]["aim_point"])
    scene = {"type": stype, "animals": 1 + len(companions), "requested": 1 + len(roles), "dropped": dropped,
             "framing": "group" if group else "primary", "arrangements": arrangements}
    return {"scene": scene, "companions": companions, "frame_world": frame_world, "aim_point": aim_point}


def public(animal: dict) -> dict:
    """The JSON part of an animal dict (private _ keys dropped)."""
    return {k: v for k, v in animal.items() if not k.startswith("_")}


def _weighted(rng, weights: dict) -> str:
    names = list(weights)
    p = np.array([float(weights[n]) for n in names]); p = p / p.sum()
    return names[int(rng.choice(len(names), p=p))]
