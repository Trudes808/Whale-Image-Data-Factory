"""run06 realism: the shot sampler's settings matched to the field photos, and the hard real situations as categories.
Active only for a config with a "realism" section (factory/config/run06_realism.json); every other config takes the
old code paths unchanged.

    "realism": {
      "sequences": true,                 # dive and surfacing categories from one continuous phase each (below)
      "sun_elevation_quantiles": [[0, 18], [0.05, 25], [0.5, 46], [0.95, 67], [1, 75]],
      "exposure": {"ev100": 1.0, "jitter": 0.25, "sun_ref_elevation_deg": 46, "sun_weight": 0.5},
      "extent_quantiles": [[0, 0.04], [0.08, 0.08], ...],     # target share of the frame the visible whale spans
      "bearing_by_category": {"fluke_up": [[130, 230, 90], ...]},   # camera sectors about the whale per category
      "end_on": {"fraction": 0.15, "categories": [...], "sectors": [[155, 205, 70], [-25, 25, 30]]},
      "splash_seed": true                # every generated splash its own seed (was 0 for all)
    }

Sequences: the rig's clip (after levelling the front half) is a tail stroke, so the dive and the surfacing roll are made
procedurally from a phase in [0, 1] that drives the clip time, the body pitch, the fluke angle and how high the body
stands out of the water:
  dive:      0 fluke rolling out flat (tail stock arched, blade level, just clear)  ->  0.5 the raised fluke  ->
             1 the fluke sinking (vertical, only the tips / outer blades above the water)
             categories fluke_rollout [0, 0.3], fluke_up [0.3, 0.78], fluke_sinking [0.78, 1]
  surfacing: 0 blow (blowhole and head just out)  ->  the back  ->  the fin  ->  1 the peduncle arched high
             categories blow, surface, back_arch, peduncle (overlapping windows)
New posed categories (rig-level, procedural): pec_slap (body on its side, one flipper up or slapping), head_rise
(rostrum / head out), tail_slap (belly up, tail flexed out of the water), peduncle_throw (on its side, tail stock
thrown sideways out of the water).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from factory import deform

DIVE_WINDOWS = {"fluke_rollout": (0.0, 0.3), "fluke_up": (0.3, 0.78), "fluke_sinking": (0.78, 1.0)}
# phase knots and the values they take (linear in between); jitter added per shot
DIVE_KNOTS = np.array([0.0, 0.15, 0.3, 0.5, 0.65, 0.78, 0.9, 1.0])
DIVE_PITCH = np.array([-14.0, -20.0, -30.0, -50.0, -62.0, -70.0, -74.0, -78.0])
DIVE_BLADE_ELEV = np.array([-18.0, -12.0, 10.0, 55.0, 75.0, 85.0, 88.0, 90.0])   # roll-out: blades trail down from the arched stock     # blade direction (stock -> notch) above horizontal
DIVE_CLEARANCE_M = np.array([0.30, 0.45, 0.9, 2.2, 2.4, 1.6, 0.55, 0.15])
DIVE_CLIP_T = np.array([0.65, 0.75, 0.9, 1.15, 1.3, 1.4, 1.45, 1.5])

SURF_WINDOWS = {"blow": (0.0, 0.3), "surface": (0.2, 0.55), "back_arch": (0.45, 0.8), "logging": (0.25, 0.5), "peduncle": (0.75, 1.0),
                "tail_stock": (0.82, 1.0)}        # round 8: the tail stock arching out with the fin already under water
SURF_KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SURF = {"arch_centre": [0.56, 0.47, 0.40, 0.31, 0.24], "arch_deg": [-22.0, -32.0, -48.0, -65.0, -85.0],
        "head_pitch_deg": [-8.0, -18.0, -26.0, -32.0, -38.0], "tail_curl_deg": [-4.0, -10.0, -18.0, -26.0, -30.0],
        "clearance_m": [0.55, 0.65, 0.85, 1.05, 1.3], "arch_span": [0.5, 0.5, 0.52, 0.55, 0.58]}
# where the clearance is measured (body-frame x, cm at unit scale): the front of the back early, the tail stock late
SURF_TOP_X = {"blow": (-350.0, 520.0), "early": (-350.0, 380.0), "late": (-650.0, 150.0), "tail_stock": (-480.0, -230.0)}   # the head (x > 443) sets the clearance only at the breath

NEW_CATEGORIES = ("fluke_rollout", "fluke_sinking", "pec_slap", "head_rise", "tail_slap", "peduncle_throw")
HARD_TAG = {"fluke_rollout": "A_fluke_rollout", "fluke_sinking": "B_fluke_sinking", "pec_slap": "C_pec", "head_rise": "C_head",
            "tail_slap": "C_tail_slap", "peduncle_throw": "C_peduncle_throw", "breach": "C_breach", "pec_up": "C_pec",
            "spyhop": "C_head", "spyhop_low": "C_head", "roll_belly": "C_pec"}


def cfg(config: dict) -> dict:
    return config.get("realism") or {}


def qsample(rng: np.random.Generator, table, log: bool = False) -> float:
    """A draw from the distribution given by quantile points [[q, value], ...] (piecewise linear in value, or in
    log value with log=True)."""
    t = np.asarray(table, float)
    u = float(rng.uniform())
    v = np.log(t[:, 1]) if log else t[:, 1]
    x = float(np.interp(u, t[:, 0], v))
    return float(math.exp(x)) if log else x


def _interp(knots, values, phase):
    return float(np.interp(phase, knots, values))


def blade_elevation(model, geometry: np.ndarray, pitch_deg: float) -> float:
    """Angle (deg) of the fluke blade's direction (spine end -> notch) in the whale's vertical plane after it is pitched: 0
    = level pointing backward, 90 = straight up, above 90 = folded over toward the head (no ambiguity past vertical)."""
    from factory.poses import rotation_matrix
    idx = model.landmark_indices()

    def at(name):
        v = idx[name]
        return geometry[v].mean(0) if isinstance(v, list) else geometry[v]
    d = (at("fluke_notch") - at("spine_end")) @ rotation_matrix(0.0, pitch_deg, 0.0)
    return float(math.degrees(math.atan2(d[2], -d[0])))


def dive(rng, model, body: dict, category: str, pose, scale: float, crest_lift_m: float, rollout_clearance=(0.3, 0.9),
         opts: dict | None = None) -> tuple:
    """(pose, clip_t, fluke_curl_deg, phase, top, params) of a dive-sequence shot (top "fluke": place by the fluke's highest
    point; params: the procedural bend, identity unless opts asks for one); body = the posed() keys of the shot
    (individual, body_scale, pec_droop_deg). opts = the realism section (round 18, all optional; absent = as before):
      dive_blade_elev  8 values replacing DIVE_BLADE_ELEV (the blade angle along the dive phase)
      rollout_bend     {"arch_deg": [lo, hi], "arch_centre": [lo, hi], "arch_span": [lo, hi], "tail_curl_deg": [lo, hi],
                        "head_pitch_deg": [lo, hi]}: the roll-out (phase < 0.3) gets a procedural arch of the tail stock
                        (drawn after every other draw of this function, so the old draws are unchanged)"""
    opts = opts or {}
    lo, hi = DIVE_WINDOWS[category]
    phase = float(rng.uniform(lo, hi))
    pitch = _interp(DIVE_KNOTS, DIVE_PITCH, phase) + float(rng.uniform(-4.0, 4.0))
    elev = np.asarray(opts["dive_blade_elev"], float) if opts.get("dive_blade_elev") else DIVE_BLADE_ELEV
    target = _interp(DIVE_KNOTS, elev, phase) + float(rng.uniform(-8.0, 8.0))
    clear = _interp(DIVE_KNOTS, DIVE_CLEARANCE_M, phase) * float(rng.uniform(0.7, 1.3))
    t = round(_interp(DIVE_KNOTS, DIVE_CLIP_T, phase) + float(rng.uniform(-0.05, 0.05)), 3)
    # roll-out: the blades just clear of the water (the arched stock stands higher); sinking: only the tips out. Both are
    # placed by the fluke's own highest point (top "fluke"), with little of the wave lift (they ride the crest)
    top = None
    if phase < 0.3:
        clear, top = float(rng.uniform(*rollout_clearance)), None   # the arched stock is the top; the blades trail to the water
    elif phase > 0.78:
        clear, top = float(rng.uniform(0.05, 0.5)), "fluke"
    params = deform.DeformParams()
    bend = opts.get("rollout_bend")
    if bend and phase < 0.3:                             # round 18: the tail stock arched as the flukes come up
        params = deform.DeformParams(**{k: float(rng.uniform(*bend[k])) for k in sorted(bend)})
    curl = 0.0
    shot = dict(body, pose={"clip_t": t} | ({"bend": "procedural"} if not params.is_identity() else {}), fluke_curl_deg=curl,
                deform=deform.params_to_dict(params) if not params.is_identity() else None)
    for _ in range(3):                                   # the fluke curl that puts the blade at the target angle
        g = model.posed(shot)
        err = target - blade_elevation(model, g, pitch)
        if abs(err) < 2.0:
            break
        curl = float(np.clip(curl + err, -120.0, 90.0))
        shot["fluke_curl_deg"] = curl
    lift = crest_lift_m if (top is None and phase >= 0.3) else 0.25 * crest_lift_m
    pose = replace(pose, pitch_deg=pitch, clearance_m=clear * scale + lift)
    return pose, t, round(curl, 2), round(phase, 4), top, params


def surfacing(rng, category: str, pose, scale: float, crest_lift_m: float, windows: dict | None = None) -> tuple:
    """(pose, DeformParams, phase, top_x (lo, hi) cm scaled) of a surfacing-roll shot. windows (round 18, config
    realism.surf_windows): {category: [lo, hi]} replacing SURF_WINDOWS entries (e.g. the peduncle without the last,
    most arched part of the roll, whose end-on dome reads as a head)."""
    lo, hi = (windows or {}).get(category) or SURF_WINDOWS[category]
    phase = float(rng.uniform(lo, hi))
    vals = {k: _interp(SURF_KNOTS, v, phase) for k, v in SURF.items()}
    params = deform.DeformParams(arch_deg=vals["arch_deg"] * float(rng.uniform(0.85, 1.15)),
                                 arch_centre=vals["arch_centre"] + float(rng.uniform(-0.03, 0.03)),
                                 arch_span=vals["arch_span"] + float(rng.uniform(-0.04, 0.04)),
                                 head_pitch_deg=vals["head_pitch_deg"] + float(rng.uniform(-5.0, 5.0)),
                                 tail_curl_deg=vals["tail_curl_deg"] + float(rng.uniform(-5.0, 5.0)),
                                 lateral_deg=float(rng.uniform(-6.0, 6.0)))
    clear = vals["clearance_m"] * float(rng.uniform(0.6, 1.25))
    top = SURF_TOP_X["late" if phase > 0.7 else "early" if phase > 0.2 else "blow"]
    if category == "tail_stock":
        # round 8 (NOT WORKING YET: in r9 the fin stayed visible in all 6 tail_stock shots; weight 0 in run06_realism):
        # meant to let only the tail stock behind the fin set the clearance so the fin is under water (a finless back)
        clear = float(rng.uniform(0.1, 0.45)); top = SURF_TOP_X["tail_stock"]
    pose = replace(pose, clearance_m=clear * scale + crest_lift_m, pitch_deg=float(rng.uniform(-4.0, 4.0)))
    return pose, params, round(phase, 4), (top[0] * scale, top[1] * scale)


def posed_category(rng, category: str, pose, scale: float, crest_lift_m: float, opts: dict | None = None) -> dict:
    """Pose, clip time, bend and fin settings of the new rig-level categories (pec_slap, head_rise, tail_slap,
    peduncle_throw). Returns {"pose", "clip_t", "params", "pec_droop", "pec_side", "top": "pec" | None}.
    opts (round 18, config realism.pec_slap): {"pitch_deg": [lo, hi]} replaces the pec slap's body pitch range (-8, 8)
    (one draw either way); "fluke_under_cm" is applied by shotlist.make_shot after the placement."""
    opts = opts or {}
    side = float(rng.choice([-1.0, 1.0]))
    out = {"params": deform.DeformParams(), "pec_side": None, "top": None, "pec_droop": None, "clip_t": None}
    if category == "pec_slap":
        roll = side * float(rng.uniform(70.0, 110.0))
        droop = float(rng.uniform(15.0, 40.0))
        up = float(rng.uniform(-35.0, 55.0)) - droop                 # the upper flipper: up (0) to slapping down / back (+)
        out.update(pec_droop=droop, pec_side=[up, 0.0] if roll > 0 else [0.0, up], top="pec",
                   clip_t=float(rng.choice([0.0, 0.3, 12.0])) + float(rng.uniform(-0.05, 0.05)))
        out["pose"] = replace(pose, roll_deg=roll, pitch_deg=float(rng.uniform(*(opts.get("pitch_deg") or (-8.0, 8.0)))),
                              clearance_m=float(rng.uniform(1.0, 2.8)) * scale + 0.5 * crest_lift_m)
    elif category == "head_rise":
        out.update(clip_t=float(rng.choice([0.0, 0.3, 12.0])) + float(rng.uniform(-0.05, 0.05)), pec_droop=float(rng.uniform(-10.0, 30.0)))
        out["pose"] = replace(pose, pitch_deg=float(rng.uniform(15.0, 45.0)), roll_deg=float(rng.uniform(-15.0, 15.0)),
                              clearance_m=float(rng.uniform(0.8, 2.6)) * scale + 0.5 * crest_lift_m)
    elif category == "tail_slap":
        # belly up, the tail flexed toward the belly (clip 3.5-5.5 s: ventral flexion), which is up when inverted
        out.update(clip_t=round(float(rng.uniform(3.5, 5.5)), 3), pec_droop=float(rng.uniform(-10.0, 20.0)))
        out["pose"] = replace(pose, roll_deg=180.0 + float(rng.uniform(-25.0, 25.0)), pitch_deg=float(rng.uniform(-5.0, 12.0)),
                              clearance_m=float(rng.uniform(1.0, 2.8)) * scale + 0.5 * crest_lift_m)
    elif category == "peduncle_throw":
        # on its side; the lateral bend of the tail stock is then vertical: the stock and flukes thrown out sideways
        roll = side * float(rng.uniform(75.0, 100.0))
        out.update(clip_t=float(rng.choice([0.0, 0.3, 12.0])) + float(rng.uniform(-0.05, 0.05)), pec_droop=float(rng.uniform(10.0, 40.0)))
        out["params"] = deform.DeformParams(lateral_deg=float(rng.uniform(35.0, 70.0)), tail_curl_deg=float(rng.uniform(-10.0, 10.0)))
        out["lateral_sign_from_roll"] = True
        out["pose"] = replace(pose, roll_deg=roll, pitch_deg=float(rng.uniform(-10.0, 5.0)),
                              clearance_m=float(rng.uniform(1.0, 3.0)) * scale + 0.5 * crest_lift_m)
    else:
        raise ValueError(category)
    return out


def choose_lateral_sign(model, body: dict, params: deform.DeformParams, pose) -> deform.DeformParams:
    """The lateral bend direction that lifts the tail for this roll (tested on the posed mesh)."""
    from factory.poses import rotation_matrix
    best, best_z = params, -1e9
    tail = model.canonical[:, 0] < -450.0
    for sgn in (1.0, -1.0):
        p = replace(params, lateral_deg=sgn * abs(params.lateral_deg))
        g = model.posed(dict(body, deform=deform.params_to_dict(p), pose=dict(body.get("pose") or {}, bend="procedural")))
        z = float((g[tail] @ rotation_matrix(pose.roll_deg, pose.pitch_deg, 0.0))[:, 2].mean())
        if z > best_z:
            best, best_z = p, z
    return best


def bearing_sectors(config: dict, category: str, rng) -> tuple[list | None, bool]:
    """Camera bearing sectors for this shot (None: the config's default) and whether the shot is an end-on back view."""
    r = cfg(config)
    endon = r.get("end_on") or {}
    if category in endon.get("categories", []) and rng.uniform() < float(endon.get("fraction", 0.0)):
        return endon["sectors"], True
    return (r.get("bearing_by_category") or {}).get(category), False


def exposure_ev100(rng, config: dict, sun: dict, overcast: bool) -> float:
    """Fixed per-shot exposure (EV100), following the light like a photographer's meter would, narrowly jittered."""
    e = cfg(config).get("exposure") or {}
    ev = float(e.get("ev100", 0.0))
    ref = math.radians(float(e.get("sun_ref_elevation_deg", 46.0)))
    el = math.radians(max(float(sun.get("elevation_deg", 46.0)), 5.0))
    ev += float(e.get("sun_weight", 0.5)) * math.log2(math.sin(el) / math.sin(ref))
    ev += math.log2(max(float(sun.get("intensity", 10.0)), 0.1) / float(e.get("intensity_ref", 10.0)))
    if overcast:
        ev += float(e.get("overcast_offset", 0.0))
    ev += float(rng.uniform(-1.0, 1.0)) * float(e.get("jitter", 0.25))
    return round(ev, 3)


def subject_metering(config: dict, category: str, extent: float) -> float:
    """EV offset that meters for the subject like a photographer: white undersides (raised flukes, flippers, slaps) get
    less exposure so they keep their shading; a dark back that fills the frame gets more so its skin shows."""
    e = cfg(config).get("exposure") or {}
    off = float((e.get("subject_offsets") or {}).get(category, 0.0))
    if extent > 0.6:
        off += float(e.get("fill_frame_offset", 0.0)) * (1.0 if category in ("surface", "back_arch", "blow", "logging", "peduncle", "tail_stock", "awash", "blow_sliver") else 0.0)
    return off


def hard_tags(shot: dict, endon: bool) -> list[str]:
    tags = []
    if shot["category"] in HARD_TAG:
        tags.append(HARD_TAG[shot["category"]])
    if endon:
        tags.append("D_end_on")
    ext = (shot.get("camera") or {}).get("extent_frame_fraction", 0.0)
    if ext > 0.8:
        tags.append("E_fills_frame")
    elif 0 < ext < 0.08:
        tags.append("F_small_far")
    if shot.get("companions"):
        tags.append("several_whales")
    return tags


# The splash a body part makes, sized by it (radius, height in cm at body scale 1): a pectoral slap about one flipper
# length, a tail slap one to two fluke widths, a breach re-entry the largest; the wash where a back or fluke breaks
# the surface stays low.
SPLASH_SIZE = {"pec_slap": ((60.0, 150.0), (60.0, 220.0)), "pec_up": ((60.0, 150.0), (60.0, 200.0)),
               "tail_slap": ((180.0, 380.0), (150.0, 400.0)), "peduncle_throw": ((180.0, 380.0), (150.0, 400.0)),
               "breach": ((300.0, 650.0), (300.0, 800.0)), "head_rise": ((120.0, 260.0), (60.0, 180.0)),
               "fluke_rollout": ((120.0, 280.0), (30.0, 90.0)), "fluke_up": ((100.0, 220.0), (30.0, 100.0)),
               "fluke_sinking": ((80.0, 180.0), (20.0, 70.0)), "fluke_down": ((100.0, 220.0), (30.0, 90.0))}
DEFAULT_WASH = ((80.0, 200.0), (20.0, 60.0))


def water_detail(shot: dict, water: dict) -> None:
    """Round 2: the sea's centimetre roughness that the field photos always show (no whale reflected as in a pond, no
    venetian-blind bands under long lenses). water = {"keep_normals": true (undo the long-lens normal-strength scaling),
    "all": {scalar: value}, "<sea state>": {scalar: value}} merged into the shot's ocean scalars."""
    sc = shot["ocean"].setdefault("scalars", {})
    if water.get("keep_normals") and shot["ocean"].get("normal_strength_scale"):
        k = float(shot["ocean"]["normal_strength_scale"])
        for key in ("Default Near Normal Strength", "Default Distant Normal Strength"):
            if key in sc:
                sc[key] = round(sc[key] / k, 4)
    sc.update(water.get("all") or {})
    sc.update(water.get(shot.get("sea_state")) or {})


def resize_splash(rng, category: str, spec: dict, scale: float) -> None:
    """Size the shot's main splash by the body part that makes it (SPLASH_SIZE) and give it the soft material."""
    (r0, r1), (h0, h1) = SPLASH_SIZE.get(category, DEFAULT_WASH)
    spec["radius_cm"] = round(float(rng.uniform(r0, r1)) * scale, 2)
    spec["height_cm"] = round(float(rng.uniform(h0, h1)) * scale, 2)
    spec["amount"] = round(min(float(spec.get("amount", 1.0)), float(rng.uniform(0.45, 1.0))), 3)
    spec["style"] = "soft"


BLOW_CATEGORIES = {"blow": 0.7, "surface": 0.35, "logging": 0.3, "blow_sliver": 0.5, "head_rise": 0.2}
STREAM_CATEGORIES = {"fluke_rollout", "fluke_up", "fluke_sinking", "fluke_down", "tail_slap", "peduncle_throw", "fluke_edge"}
SPRAY_CATEGORIES = {"fluke_rollout", "fluke_sinking", "fluke_up", "pec_slap", "tail_slap", "peduncle_throw", "breach", "head_rise",
                    "back_arch", "surface", "peduncle", "spyhop", "tail_stock"}


def extra_splashes(rng, config: dict, model, category: str, geometry: np.ndarray, whale: dict, camera_location, scale: float,
                   camera=None) -> list[dict]:
    """White water beyond the shot's main splash (run06 realism): the blow (a mist column at the blowhole of a breathing
    whale) and a spray curtain thrown up on the camera side of the body-water contact, which stands in front of the whale
    and hides part of it (its coverage twin makes it background in the labels where it is dense). Mean-water-plane
    coordinates (shotlist lifts them onto the waves and moves them with the scene)."""
    from factory import splash as splash_mod
    from factory.project import transform_mesh, landmark_positions
    r = cfg(config)
    out = []
    world = transform_mesh(geometry, whale)
    cam = np.asarray(camera_location, float)
    p_blow = float(r.get("blow_probability_scale", 1.0)) * BLOW_CATEGORIES.get(category, 0.0)
    if p_blow > 0 and rng.uniform() < p_blow:
        bh = transform_mesh(landmark_positions({"b": model.landmark_indices()["blowhole"]}, geometry)["b"][None, :], whale)[0]
        if bh[2] > -30.0:
            wind = float(rng.uniform(0, 2 * math.pi)); h = float(rng.uniform(250.0, 550.0)) * scale
            out.append(splash_mod.splash_spec("blow", [bh[0], bh[1], max(bh[2], 0.0)], float(rng.uniform(25.0, 60.0)) * scale, h,
                                              seed=int(rng.integers(1 << 30)), amount=float(rng.uniform(0.6, 1.2)),
                                              drift_cm=[round(math.cos(wind) * h * 0.4, 1), round(math.sin(wind) * h * 0.4, 1)], role="blow", style="soft"))
    if category in STREAM_CATEGORIES and rng.uniform() < float(r.get("fluke_stream_probability", 0.0)):
        idx = model.landmark_indices()
        names = sorted(k for k in idx if k.startswith("fluke_te_"))
        pts = transform_mesh(np.array([landmark_positions({n: idx[n]}, geometry)[n] for n in names]), whale)
        pts = pts[pts[:, 2] > 25.0]                                  # trailing-edge points clear of the water
        if camera is not None and len(pts):
            uv, depth = camera.project(pts)                           # only an edge the camera sees: no orphan streams
            inside = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
            pts = pts[inside]
        if len(pts) >= 4:
            c = pts.mean(0); c[2] = 0.0
            strands = r.get("stream_strands")       # round 2: many thin strands breaking into drops (splash._strands)
            out.append(splash_mod.splash_spec("streams", [c[0], c[1], 0.0], 50.0, float(pts[:, 2].max()), seed=int(rng.integers(1 << 30)),
                                              points_rel_cm=[[round(float(v), 1) for v in (p - c)] for p in pts], role="fluke_streams",
                                              style="strands" if strands else "soft", width_cm=[1.0, 3.5], keep_fraction=0.45,
                                              fall_fraction=[0.25, 0.8], **({"strands": strands} if strands else {}),
                                              **({"opacity": strands["opacity"]} if strands and "opacity" in strands else {})))
    if category in SPRAY_CATEGORIES and rng.uniform() < float(r.get("front_spray_probability", 0.0)):
        band = world[np.abs(world[:, 2]) < 40.0]
        if len(band) > 20:
            d = np.linalg.norm(band[:, :2] - cam[None, :2], axis=1)
            near = band[np.argsort(d)[: max(5, len(band) // 20)]].mean(0)       # the contact on the camera side
            toward = math.degrees(math.atan2(cam[1] - near[1], cam[0] - near[0]))
            out.append(splash_mod.splash_spec("spray", [near[0], near[1], 0.0], float(rng.uniform(40.0, 150.0)) * scale,
                                              float(rng.uniform(30.0, 150.0)) * scale, seed=int(rng.integers(1 << 30)),
                                              amount=float(rng.uniform(0.35, 0.9)), azimuth_deg=round(toward, 1), role="front_spray", style="soft"))
    return out


PACK = "/Game/SplashPack/FX/FX_System/"
PACK_SYSTEMS = {"breach": ["NS_Breach_Mid", "NS_Breach_Small", "NS_Breach_Mid_Angle"],
                "slap": ["NS_Splash_Small", "NS_Splash_Mid", "NS_Splash_Mid_Angle", "NS_Splash_XL"]}
PACK_CATEGORY = {"breach": "breach", "pec_slap": "slap", "tail_slap": "slap", "peduncle_throw": "slap", "fluke_rollout": "slap",
                 "head_rise": "slap", "pec_up": "slap"}


def niagara_specs(rng, config: dict, category: str, shot: dict) -> list[dict]:
    """The purchased splash pack (Niagara, wf_build_batch.add_niagara): system, place, heading, scale and age per shot,
    for breaches and slaps (config realism.splash_pack.probability)."""
    sp = cfg(config).get("splash_pack") or {}
    group = PACK_CATEGORY.get(category)
    if not group or rng.uniform() >= float(sp.get("probability", 0.0)):
        return []
    centre = (shot.get("splash") or {}).get("centre_cm") or shot["stats"]["aim_point"]
    name = PACK_SYSTEMS[group][int(rng.integers(len(PACK_SYSTEMS[group])))]
    scale = float(rng.uniform(*sp.get("scale", [0.5, 1.3]))) * float(shot.get("body_scale", 1.0))
    water = float((shot.get("water_surface_z_cm") or {}).get("splash", (shot.get("water_surface_z_cm") or {}).get("whale", 0.0)))
    return [{"system": PACK + name, "location": [round(float(centre[0]), 1), round(float(centre[1]), 1), round(water, 1)],
             "yaw_deg": round(float(rng.uniform(0, 360)), 1), "scale": round(scale, 3),
             "age_s": round(float(rng.uniform(*sp.get("age_s", [0.25, 1.6]))), 3)}]
