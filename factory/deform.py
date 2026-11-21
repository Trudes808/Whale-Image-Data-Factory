"""Procedural body deformation of the whale mesh: the "animation" a rig would give.

The humpback mesh is rigid, so this module bends it. Along the body axis (x, tail at
-L/2 to snout at +L/2) a smooth bending profile turns the spine: an arch over the back, a
curl of the tail stock and flukes (up or down), a head pitch, a lateral flex (swim stroke)
and a fluke twist. The pectoral fins rotate about their roots (raise, sweep). Vertices
keep their indices, so canonical body coordinates, part ids and landmarks carry over.

    params = DeformParams(arch_deg=12, tail_curl_deg=-35, head_pitch_deg=8, lateral_deg=10, ...)
    deformed = deform(vertices_cm, params)     # same shape as vertices_cm

Angles are degrees; positive arch bends the back upward (nose and tail down), positive
tail curl lifts the flukes, positive head pitch lifts the snout, positive lateral bends
the tail toward +y (the whale's right), positive fin raise lifts the fins toward the back.
Each shot stores its parameters; the post-processing step regenerates the deformed mesh
from them (deterministic), so no per-shot mesh is kept in the dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

# Body axis milestones on the K9239 humpback mesh, as fractions of length from the tail.
TAIL_START, TAIL_END = 0.0, 0.32      # flukes and peduncle
BODY_START, BODY_END = 0.2, 0.85      # arch region
HEAD_START = 0.75                     # head pitch region (hinge at HEAD_START, moves the snout)


@dataclass
class DeformParams:
    arch_deg: float = 0.0        # back arched up (+) or sagging (-), total bend across the body
    tail_curl_deg: float = 0.0   # flukes curled up (+) or down (-)
    head_pitch_deg: float = 0.0  # snout up (+) or down (-)
    lateral_deg: float = 0.0     # tail swept toward +y (+): the swim stroke
    fluke_twist_deg: float = 0.0 # roll of the flukes about the body axis
    fin_raise_deg: float = 0.0   # pectoral fins rotated up toward the back (+) or down (-)
    fin_sweep_deg: float = 0.0   # pectoral fins swept back (+) or forward (-)
    arch_centre: float = 0.0     # body fraction (0 tail .. 1 snout) at the arch's apex; 0 = the default BODY_START..BODY_END region
    arch_span: float = 0.0       # length of the arch region (body fraction) when arch_centre is set

    def is_identity(self) -> bool:
        return all(abs(getattr(self, f.name)) < 1e-9 for f in fields(self))


def sample_params(rng: np.random.Generator, category: str, strength: float = 1.0) -> DeformParams:
    """Plausible deformation for a pose category (degrees), scaled by strength."""
    u = lambda lo, hi: float(rng.uniform(lo, hi)) * strength  # noqa: E731
    category = {"fin_tip": "pec_up", "fluke_edge": "fluke_up", "fluke_down": "fluke_up", "peduncle": "arch", "tail_stock": "arch", "back_arch": "arch", "surface": "arch", "spyhop_low": "spyhop", "roll_belly": "breach", "blow_sliver": "blow"}.get(category, category)
    if category == "fluke_up":
        p = DeformParams(arch_deg=u(0, 10), tail_curl_deg=u(-25, 25), lateral_deg=u(-15, 15), fluke_twist_deg=u(-15, 15))
    elif category == "arch":
        p = DeformParams(arch_deg=u(10, 30), tail_curl_deg=u(-20, 10), head_pitch_deg=u(-10, 5), lateral_deg=u(-8, 8))
    elif category == "breach":
        p = DeformParams(arch_deg=u(-15, 15), tail_curl_deg=u(-30, 30), lateral_deg=u(-25, 25), fluke_twist_deg=u(-25, 25),
                         fin_raise_deg=u(-20, 60), fin_sweep_deg=u(-20, 20))
    elif category == "pec_up":
        p = DeformParams(arch_deg=u(-5, 10), lateral_deg=u(-15, 15), fin_raise_deg=u(20, 70), fin_sweep_deg=u(-15, 15))
    elif category == "spyhop":
        p = DeformParams(arch_deg=u(-10, 10), head_pitch_deg=u(-5, 10), fin_raise_deg=u(-10, 40))
    else:  # logging, blow, awash, submerged: gentle
        p = DeformParams(arch_deg=u(-6, 8), tail_curl_deg=u(-12, 12), head_pitch_deg=u(-5, 5), lateral_deg=u(-10, 10),
                         fin_raise_deg=u(-10, 25), fin_sweep_deg=u(-10, 10))
    return p


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bend_profile(frac: np.ndarray, params: DeformParams) -> tuple[np.ndarray, np.ndarray]:
    """Pitch (about y) and yaw (about z) of the spine, in radians, per body fraction."""
    pitch = np.zeros_like(frac)
    # Arch: a bump of curvature over the body, so the angle goes from -A/2 at the tail end of
    # the region to +A/2 at the head end (back curves upward in between).
    if params.arch_deg:
        lo, hi = BODY_START, BODY_END
        if params.arch_centre:
            span = params.arch_span or (BODY_END - BODY_START)
            lo, hi = params.arch_centre - span / 2, params.arch_centre + span / 2
        t = _smooth((frac - lo) / (hi - lo))
        pitch += np.radians(params.arch_deg) * (t - 0.5)
    # Tail curl: angle grows toward the fluke tips (frac -> 0).
    if params.tail_curl_deg:
        t = _smooth((TAIL_END - frac) / (TAIL_END - TAIL_START))
        pitch -= np.radians(params.tail_curl_deg) * t      # minus: lifting the tail means the spine points up toward -x
    # Head pitch: angle grows toward the snout.
    if params.head_pitch_deg:
        t = _smooth((frac - HEAD_START) / (1.0 - HEAD_START))
        pitch += np.radians(params.head_pitch_deg) * t
    yaw = np.zeros_like(frac)
    if params.lateral_deg:
        t = _smooth((TAIL_END + 0.15 - frac) / (TAIL_END + 0.15))
        yaw -= np.radians(params.lateral_deg) * t
    return pitch, yaw


def deform(vertices_cm: np.ndarray, params: DeformParams, fin_regions: dict | None = None) -> np.ndarray:
    """Bend the mesh. Returns a new (N, 3) array."""
    v = np.asarray(vertices_cm, float)
    if params.is_identity():
        return v.copy()
    x = v[:, 0]
    x_min, x_max = x.min(), x.max()
    length = x_max - x_min
    frac = (x - x_min) / length
    # Integrate the spine from the body's mid-length outward on a fine grid, then map vertices.
    grid = np.linspace(0.0, 1.0, 513)
    pitch_g, yaw_g = _bend_profile(grid, params)
    mid = 0.5
    mid_i = int(np.argmin(np.abs(grid - mid)))
    pos = np.zeros((len(grid), 3))
    dx = length / (len(grid) - 1)
    for i in range(mid_i + 1, len(grid)):
        pitch_m, yaw_m = 0.5 * (pitch_g[i] + pitch_g[i - 1]), 0.5 * (yaw_g[i] + yaw_g[i - 1])
        pos[i] = pos[i - 1] + dx * np.array([np.cos(pitch_m) * np.cos(yaw_m), np.cos(pitch_m) * np.sin(yaw_m), np.sin(pitch_m)])
    for i in range(mid_i - 1, -1, -1):
        pitch_m, yaw_m = 0.5 * (pitch_g[i] + pitch_g[i + 1]), 0.5 * (yaw_g[i] + yaw_g[i + 1])
        pos[i] = pos[i + 1] - dx * np.array([np.cos(pitch_m) * np.cos(yaw_m), np.cos(pitch_m) * np.sin(yaw_m), np.sin(pitch_m)])
    # Per vertex: spine point and local frame at its body fraction.
    spine = np.stack([np.interp(frac, grid, pos[:, k]) for k in range(3)], axis=1)
    pitch = np.interp(frac, grid, pitch_g)
    yaw = np.interp(frac, grid, yaw_g)
    roll = np.zeros_like(frac)
    if params.fluke_twist_deg:
        roll = np.radians(params.fluke_twist_deg) * _smooth((TAIL_END - frac) / (TAIL_END - TAIL_START))
    # Local offsets from the straight spine (y, z), rotated by the local frame.
    offset = np.stack([np.zeros_like(x), v[:, 1], v[:, 2]], axis=1)
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    # roll about x, then pitch about y, then yaw about z (applied to the offset vector)
    oy = offset[:, 1] * cr - offset[:, 2] * sr
    oz = offset[:, 1] * sr + offset[:, 2] * cr
    ox = np.zeros_like(oy)
    # pitch about y: x' = x cos + z sin ; z' = -x sin + z cos
    px = ox * cp + oz * sp
    pz = -ox * sp + oz * cp
    py = oy
    # yaw about z
    qx = px * cy - py * sy
    qy = px * sy + py * cy
    qz = pz
    out = spine + (x_min + mid * length) * np.array([1.0, 0.0, 0.0]) + np.stack([qx, qy, qz], axis=1)
    # Re-anchor so the mid-body point stays where it was (spine integration started at 0).
    out[:, 0] += 0.0
    if fin_regions and (params.fin_raise_deg or params.fin_sweep_deg):
        out = rotate_fins(out, fin_regions, params)
    return out


def _frame(theta: np.ndarray, psi: np.ndarray, roll: np.ndarray | None = None):
    """Unit tangent, lateral (binormal) and up (normal) vectors of a line with pitch theta and yaw psi (radians); the
    tangent is (cos th cos ps, cos th sin ps, sin th) as in deform()'s spine integration. roll turns (b, n) about t."""
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    t = np.stack([ct * cp, ct * sp, st], -1)
    b = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    n = np.stack([-st * cp, -st * sp, ct], -1)
    if roll is not None:
        cr, sr = np.cos(roll)[..., None], np.sin(roll)[..., None]
        b, n = b * cr + n * sr, -b * sr + n * cr
    return t, b, n


def centre_line(v: np.ndarray, exclude: np.ndarray | None = None, bins: int = 96, neutral: float = 0.0) -> dict:
    """The body's own centre line in whatever pose the mesh is in (run06 round 20c): per slice along x the mid point of
    the cross-section's extent (y and z), fins (exclude mask: pectorals, old dorsal fin) left out, smoothed along the
    body. neutral (0..1) lifts the line toward the top of the section (0 = centre, 1 = dorsal outline): the bend's
    neutral line, the one that keeps its length. Returns {"s", "c", "theta", "psi", "S"} (arc length cm, points, pitch,
    yaw)."""
    x = v[:, 0]
    x_min, x_max = float(x.min()), float(x.max())
    keep = np.ones(len(v), bool) if exclude is None else ~np.asarray(exclude, bool)
    edges = np.linspace(x_min, x_max, bins + 1); xc = 0.5 * (edges[1:] + edges[:-1])
    k = np.clip(np.searchsorted(edges, x[keep], side="right") - 1, 0, bins - 1)
    vy, vz = v[keep, 1], v[keep, 2]
    ymin = np.full(bins, np.inf); ymax = np.full(bins, -np.inf); zmin = np.full(bins, np.inf); zmax = np.full(bins, -np.inf)
    np.minimum.at(ymin, k, vy); np.maximum.at(ymax, k, vy); np.minimum.at(zmin, k, vz); np.maximum.at(zmax, k, vz)
    ok = np.isfinite(zmin)
    ym, zm = 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    zm = zm + neutral * (zmax - zm)
    # the line stays in the vertical plane y = 0 (no yaw): the keyframe's own sideways flex is kept in the section
    # offsets as in deform(), and the 5.8 m fluke span never projects onto a laterally curved line
    ym = np.zeros(bins); zm = np.interp(xc, xc[ok], zm[ok])
    # smooth (Gaussian, 3 % of the length), edges padded with the end values
    sig = max(1.0, 0.03 * bins); r = int(3 * sig); w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); w /= w.sum()
    ys = np.convolve(np.pad(ym, r, mode="edge"), w, "valid"); zs = np.convolve(np.pad(zm, r, mode="edge"), w, "valid")
    c = np.stack([xc, ys, zs], 1)
    # extend to the mesh ends along the end tangents so every vertex projects inside
    d = np.diff(c, axis=0)
    t0 = d[0] / np.linalg.norm(d[0]); t1 = d[-1] / np.linalg.norm(d[-1])
    c = np.vstack([c[0] - t0 * (xc[0] - x_min + 0.02 * (x_max - x_min)) / max(t0[0], 0.2), c,
                   c[-1] + t1 * (x_max - xc[-1] + 0.02 * (x_max - x_min)) / max(t1[0], 0.2)])
    seg = np.diff(c, axis=0); sl = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(sl)])
    tan = np.gradient(c, s, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    theta = np.arcsin(np.clip(tan[:, 2], -1, 1)); psi = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    return {"s": s, "c": c, "theta": theta, "psi": psi, "S": float(s[-1])}


def _line_at(line: dict, sv: np.ndarray):
    """Point, pitch and yaw of the centre line at arc lengths sv (linear extrapolation past the ends)."""
    s, c = line["s"], line["c"]
    p = np.stack([np.interp(sv, s, c[:, k]) for k in range(3)], 1)
    th = np.interp(sv, s, line["theta"]); ps = np.interp(sv, s, line["psi"])
    lo, hi = sv < s[0], sv > s[-1]
    if lo.any() or hi.any():
        t, _, _ = _frame(th, ps)
        p[lo] = c[0] + (sv[lo] - s[0])[:, None] * t[lo]
        p[hi] = c[-1] + (sv[hi] - s[-1])[:, None] * t[hi]
    return p, th, ps


def _extra_pitch(line: dict, sv: np.ndarray, extra: dict) -> np.ndarray:
    """run06 round 20i (whale.fore_arch): added pitch (radians) along the centre line at arc lengths sv. extra =
    {"anchor_x": cm (the fin's front base, posed x), "terms": [[deg, s0_cm, s1_cm], ...]}: each term turns the line by
    deg (> 0 convex: the line ahead turns down) spread as a smoothstep between s0 and s1 cm of arc ahead of the anchor;
    0 behind s0, so the body behind the first term keeps its orientation."""
    sa = float(np.interp(float(extra["anchor_x"]), line["c"][:, 0], line["s"]))
    out = np.zeros_like(np.asarray(sv, float))
    for deg, s0, s1 in extra.get("terms") or []:
        out -= np.radians(float(deg)) * _smooth((np.asarray(sv, float) - sa - float(s0)) / max(float(s1) - float(s0), 1e-6))
    return out


def deform_centreline(vertices_cm: np.ndarray, params: DeformParams, exclude: np.ndarray | None = None,
                      neutral: float = 0.0, extra: dict | None = None) -> np.ndarray:
    """deform() done about the body's own centre line (run06 round 20c, config whale.arch_bend). deform() bends rigid
    sections about the mesh's z = 0 axis and turns them the wrong way (a section's 'up' tilts by +pitch while the spine
    tilts by +pitch, so sections sit 2 x pitch off perpendicular): over a hump the top converges (the head of a clip
    keyframe, 1-3 m above the axis, is crushed to ~0.4 of its area). Here every vertex is expressed in the frame of its
    own section of the centre line (perpendicular to the line, found by projection), the line is bent by the same pitch /
    yaw profile (_bend_profile, the same arch_deg / arch_centre / ... meaning) and the vertex is put back in the bent
    frame: sections stay perpendicular, the neutral line keeps its length. The pectoral rotation of deform() (K9239 fin
    regions) is not applied (the rig path never used it)."""
    v = np.asarray(vertices_cm, float)
    if params.is_identity() and not extra:
        return v.copy()
    line = centre_line(v, exclude, neutral=neutral)
    S = line["S"]
    # project every vertex onto the line (Newton steps on (v - c(s)) . t(s) = 0), starting from its x
    sv = np.interp(v[:, 0], line["c"][:, 0], line["s"])
    for _ in range(6):
        p, th, ps = _line_at(line, sv)
        t, _, _ = _frame(th, ps)
        step = np.einsum("ij,ij->i", v - p, t)
        sv = sv + step
        if np.abs(step).max() < 1e-3:
            break
    p, th0, ps0 = _line_at(line, sv)
    t0, b0, n0 = _frame(th0, ps0)
    d = v - p
    ut, ub, un = (np.einsum("ij,ij->i", d, e) for e in (t0, b0, n0))
    # the bent line: the original segments, each turned by the profile's added pitch / yaw, chained from mid-body
    grid = np.linspace(min(0.0, float(sv.min())), max(S, float(sv.max())), 1025)
    gp, gth, gps = _line_at(line, grid)
    dp, dy = _bend_profile(np.clip(grid / S, 0.0, 1.0), params)
    if extra:                                    # round 20i: whale.fore_arch (None = as before)
        dp = dp + _extra_pitch(line, grid, extra)
    seg = np.diff(gp, axis=0)
    mth, mps = 0.5 * (gth[1:] + gth[:-1]), 0.5 * (gps[1:] + gps[:-1])
    mdp, mdy = 0.5 * (dp[1:] + dp[:-1]), 0.5 * (dy[1:] + dy[:-1])
    a0, a1, a2 = _frame(mth, mps); e0, e1, e2 = _frame(mth + mdp, mps + mdy)
    seg_new = (np.einsum("ij,ij->i", seg, a0)[:, None] * e0 + np.einsum("ij,ij->i", seg, a1)[:, None] * e1
               + np.einsum("ij,ij->i", seg, a2)[:, None] * e2)
    mid = int(np.argmin(np.abs(grid - 0.5 * S)))
    newp = np.zeros_like(gp); newp[mid] = gp[mid]
    newp[mid + 1:] = gp[mid] + np.cumsum(seg_new[mid:], axis=0)
    newp[:mid] = gp[mid] - np.cumsum(seg_new[:mid][::-1], axis=0)[::-1]
    cnew = np.stack([np.interp(sv, grid, newp[:, k]) for k in range(3)], 1)
    fr = np.clip(sv / S, 0.0, 1.0)
    vdp, vdy = _bend_profile(fr, params)
    if extra:
        vdp = vdp + _extra_pitch(line, sv, extra)
    roll = np.radians(params.fluke_twist_deg) * _smooth((TAIL_END - fr) / (TAIL_END - TAIL_START)) if params.fluke_twist_deg else None
    t1, b1, n1 = _frame(th0 + vdp, ps0 + vdy, roll)
    return cnew + ut[:, None] * t1 + ub[:, None] * b1 + un[:, None] * n1


def fin_regions(vertices_cm: np.ndarray) -> dict:
    """Vertex indices of the left and right pectoral fins and their root hinge points.

    On this model the fins hang down from the body between 10 % and 45 % of the length from
    the snout; fin vertices are those well outside the body's half-width or well below the
    belly in that range."""
    v = np.asarray(vertices_cm, float)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    x_min, x_max = x.min(), x.max()
    frac = (x - x_min) / (x_max - x_min)
    in_range = (frac > 0.45) & (frac < 0.78)
    body_core = in_range & (np.abs(y) < 60)
    belly = np.percentile(z[body_core], 5) if body_core.any() else z.min()
    regions = {}
    for name, side in (("left", -1), ("right", 1)):
        sel = in_range & (side * y > 90) & (z < belly + 60)
        idx = np.where(sel)[0]
        if len(idx) < 20:
            continue
        root = np.array([x[idx].mean(), side * 100.0, belly + 40.0])
        regions[name] = {"indices": idx, "root": root, "side": side}
    return regions


def rotate_fins(vertices: np.ndarray, regions: dict, params: DeformParams) -> np.ndarray:
    out = vertices.copy()
    for name, region in regions.items():
        idx, root, side = region["indices"], region["root"], region["side"]
        rel = out[idx] - root
        # Raise: rotate about the body axis (x) so the fin tip moves up toward the back.
        a = np.radians(params.fin_raise_deg) * side
        y, z = rel[:, 1], rel[:, 2]
        rel[:, 1], rel[:, 2] = y * np.cos(a) - z * np.sin(a), y * np.sin(a) + z * np.cos(a)
        # Sweep: rotate about the vertical axis so the tip moves back toward the tail.
        b = np.radians(params.fin_sweep_deg) * side
        x, y = rel[:, 0], rel[:, 1]
        rel[:, 0], rel[:, 1] = x * np.cos(b) - y * np.sin(b), x * np.sin(b) + y * np.cos(b)
        # Blend near the root so the fin does not tear off the body.
        dist = np.linalg.norm(vertices[idx] - root, axis=1)
        w = _smooth(dist / 150.0)[:, None]
        out[idx] = root + rel * w + (vertices[idx] - root) * (1 - w)
    return out


def params_to_dict(params: DeformParams) -> dict:
    return {k: round(float(v), 3) for k, v in asdict(params).items()}


def params_from_dict(d: dict | None) -> DeformParams:
    return DeformParams(**{k: float(v) for k, v in (d or {}).items() if k in {f.name for f in fields(DeformParams)}})


def write_obj_with_vertices(source_obj, vertices_cm: np.ndarray, out_path) -> None:
    """Copy an OBJ, replacing its vertex positions (same order), keeping uv/normals/faces."""
    out = []
    index = 0
    with open(source_obj) as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = vertices_cm[index]
                out.append(f"v {x:.4f} {-y:.4f} {z:.4f}\n")   # Unreal's OBJ import mirrors Y
                index += 1
            elif line.startswith("vn "):
                continue  # normals no longer match; Unreal recomputes them
            else:
                out.append(line)
    if index != len(vertices_cm):
        raise ValueError(f"{source_obj} has {index} vertices, got {len(vertices_cm)}")
    # faces reference v/vt/vn; drop the vn index so the importer computes normals
    fixed = []
    for line in out:
        if line.startswith("f "):
            parts = [t.split("/") for t in line.split()[1:]]
            line = "f " + " ".join(f"{p[0]}/{p[1]}" if len(p) > 1 else p[0] for p in parts) + "\n"
        fixed.append(line)
    with open(out_path, "w") as handle:
        handle.writelines(fixed)


def write_batch_meshes(batch_path, base_obj, out_dir) -> dict:
    """Write the deformed whale OBJ of every shot in a batch that has a deformation, and
    return a copy of the batch with ``whale.mesh_obj`` filled in (saved as batch.json in
    out_dir). Used by tools/render_batch.sh right before building the batch."""
    import json
    from pathlib import Path

    from factory import project

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = json.loads(Path(batch_path).read_text())
    if batch.get("whale_kind", "k9239") != "k9239":
        from factory.whalemodel import WhaleModel

        model = WhaleModel(batch["whale_kind"])
        if batch.get("individuals"):                     # run06 realism: the run's own individual library
            from factory.whalemodel import REPO_ROOT
            model.library_path = Path(batch["individuals"]) if Path(batch["individuals"]).is_absolute() else REPO_ROOT / batch["individuals"]
        if batch.get("dorsal_front_ratio"):              # run06 round 5: the hump runs to this front base (as in shotlist / postprocess)
            model.dorsal_front_ratio = float(batch["dorsal_front_ratio"])
        if batch.get("ridge_soften"):                    # run06 round 8: ridge knobs toned down (as in shotlist / postprocess)
            model.ridge_soften = float(batch["ridge_soften"])
        if batch.get("blade_weld"):                      # run06 round 12: the blade welded into the back
            model.blade_weld = True
            from factory import weld as _wd              # round 15: blade_weld = 15 (true = round 13)
            if _wd.design_of(batch["blade_weld"]) == 15:
                model.weld_design = 15
        rc_path = Path(batch_path).resolve().parent.parent / "run_config.json"
        if rc_path.exists():                             # run06 round 18b: whale.winding_fix (faces wound for the engine; mesh files only)
            model.winding_fix = bool(((json.loads(rc_path.read_text()).get("config") or {}).get("whale") or {}).get("winding_fix", False))
            _ab = ((json.loads(rc_path.read_text()).get("config") or {}).get("whale") or {}).get("arch_bend")
            if _ab:                                      # run06 round 20c: procedural bends about the centre line
                model.arch_bend = dict(_ab)
            model.apply_r20d(((json.loads(rc_path.read_text()).get("config") or {}).get("whale") or {}))   # round 20d keys
            _wh = ((json.loads(rc_path.read_text()).get("config") or {}).get("whale") or {})
            if _wh.get("dorsal_hump"):                   # round 20g: whale.dorsal_hump (round 20e) reaches the batch meshes too
                model.dorsal_hump = dict(_wh["dorsal_hump"])
            if _wh.get("obj_shared_vertices"):           # round 20g: one OBJ vertex per position at UV seams (engine normals)
                model.obj_shared_vertices = dict(_wh["obj_shared_vertices"])
            if _wh.get("blade_uv_match"):                # round 20g: blade UV charts placed to match the patch's base colour
                model.blade_uv_match = dict(_wh["blade_uv_match"])
            if _wh.get("fore_arch"):                     # round 20i: convex bend ahead of the fin (surfacing poses)
                model.fore_arch = dict(_wh["fore_arch"])
        if batch.get("blade_templates"):                 # run06 round 7: the parametric dorsal blade (as in shotlist / postprocess)
            bt = Path(batch["blade_templates"])
            from factory.whalemodel import REPO_ROOT as _R
            model.enable_blade(bt if bt.is_absolute() else _R / bt)
        for shot in batch["shots"]:
            path = out_dir / f"{shot['id']}.obj"
            model.write_obj(model.posed(shot), path, f"{batch['whale_kind']} posed for shot {shot['id']} (clip t {(shot.get('pose') or {}).get('clip_t')})",
                            individual=shot.get("individual"))
            shot["whale"]["mesh_obj"] = str(path.resolve())
            if shot.get("companion"):
                cpath = out_dir / f"{shot['id']}c.obj"
                model.write_obj(model.posed(shot["companion"]), cpath, f"{batch['whale_kind']} companion of shot {shot['id']}",
                                individual=shot["companion"].get("individual"))
                shot["companion"]["whale"]["mesh_obj"] = str(cpath.resolve())
            for c in shot.get("companions") or []:          # run06 scenes: every further whale
                cpath = out_dir / f"{shot['id']}a{c['instance_id']}.obj"
                model.write_obj(model.posed(c), cpath, f"{batch['whale_kind']} {c.get('role')} {c['instance_id']} of shot {shot['id']}",
                                individual=c.get("individual"))
                c["whale"]["mesh_obj"] = str(cpath.resolve())
        (out_dir / "batch.json").write_text(json.dumps(batch))
        return {"shots": len(batch["shots"]), "meshes": len(batch["shots"]) + sum(len(s.get("companions") or []) + bool(s.get("companion")) for s in batch["shots"])}
    vertices, _ = project.load_obj(base_obj)
    regions = fin_regions(vertices)
    count = 0
    for shot in batch["shots"]:
        params = params_from_dict(shot.get("deform"))
        scale = float(shot.get("body_scale", 1.0))
        if params.is_identity() and abs(scale - 1.0) < 1e-6:
            continue
        path = out_dir / f"{shot['id']}.obj"
        base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
        reg = regions if abs(scale - 1.0) < 1e-6 else fin_regions(base)
        write_obj_with_vertices(base_obj, deform(base, params, reg) if not params.is_identity() else base, path)
        shot["whale"]["mesh_obj"] = str(path.resolve())
        count += 1
    (out_dir / "batch.json").write_text(json.dumps(batch))
    return {"shots": len(batch["shots"]), "meshes": count}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        sys.exit("usage: python3 -m factory.deform BATCH.json BASE_WHALE.obj OUT_DIR")
    print(write_batch_meshes(sys.argv[1], sys.argv[2], sys.argv[3]))
