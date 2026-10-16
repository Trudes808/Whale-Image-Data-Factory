"""White water where the whale meets the sea: a foam carpet on the actual wave surface along the
body-water contact line (bow wave ahead, wash along the flanks, a wake streak behind, the boil of
a dive, the impact patch of a breach or slap) and a cloud of aerated water just below it. Both
are meshes written per shot like the splashes, lit by their own materials (wf_shot) and, for the
label pass, a coverage twin of the foam hides the whale where the foam is dense.

Geometry is computed in the factory: the posed whale, the shot's Gerstner waves (factory.gerstner,
the engine's own waves) and its frozen wave time give the surface and the contact line exactly.
The foam intensity is stored per vertex in the first UV channel (u = foam, v = aerated water);
the materials multiply it with procedural noise so it breaks into patches and streaks.

    spec = whitewater.spec(shot, world_cm, waves, t, rng)   # in factory.shotlist.make_shot
    whitewater.ensure_obj(spec, run_dir / "whitewater", shot["id"], world_cm, waves)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from factory import gerstner

CONTACT_BAND_CM = 45.0          # body vertices this close to the local surface are "in contact"
GRID_CM = 30.0                  # carpet vertex spacing (lumps of froth need it finer than the old 40)
MAX_VERTICES = 60000

# Per pose category: strength of the contact wash, the wake behind the body (and its length in m),
# the boil of a diving tail stock, and the impact patch of a body hitting the water.
CATEGORY = {
    "logging":     dict(contact=0.55, wake=0.45, wake_len=12.0, boil=0.0, impact=0.0),
    "blow":        dict(contact=0.85, wake=0.80, wake_len=18.0, boil=0.0, impact=0.0),
    "blow_sliver": dict(contact=0.65, wake=0.50, wake_len=10.0, boil=0.0, impact=0.0),
    "arch":        dict(contact=0.70, wake=0.65, wake_len=15.0, boil=0.3, impact=0.0),
    "back_arch":   dict(contact=0.65, wake=0.6, wake_len=14.0, boil=0.25, impact=0.0),
    "surface":     dict(contact=0.55, wake=0.5, wake_len=12.0, boil=0.2, impact=0.0),
    "fluke_down":  dict(contact=0.8, wake=0.4, wake_len=8.0, boil=0.5, impact=0.0),     # water streaming off the stock and tips
    "peduncle":    dict(contact=0.70, wake=0.50, wake_len=10.0, boil=0.6, impact=0.0),
    "fluke_up":    dict(contact=0.60, wake=0.30, wake_len=8.0, boil=0.9, impact=0.0),
    "fluke_edge":  dict(contact=0.50, wake=0.25, wake_len=6.0, boil=0.9, impact=0.0),
    "breach":      dict(contact=0.30, wake=0.00, wake_len=0.0, boil=0.0, impact=1.0),
    "pec_up":      dict(contact=0.60, wake=0.20, wake_len=6.0, boil=0.0, impact=0.7),
    "fin_tip":     dict(contact=0.50, wake=0.20, wake_len=5.0, boil=0.0, impact=0.3),
    "roll_belly":  dict(contact=0.60, wake=0.10, wake_len=4.0, boil=0.0, impact=0.4),
    "spyhop":      dict(contact=0.60, wake=0.10, wake_len=4.0, boil=0.0, impact=0.3),
    "spyhop_low":  dict(contact=0.50, wake=0.10, wake_len=4.0, boil=0.0, impact=0.0),
    "awash":       dict(contact=0.45, wake=0.30, wake_len=8.0, boil=0.0, impact=0.0),
    "submerged":   dict(contact=0.20, wake=0.30, wake_len=8.0, boil=0.0, impact=0.0),
}


def value_noise(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, cell_cm: float, stretch=(1.0, 1.0)) -> np.ndarray:
    """Smooth value noise in [0, 1] over world cm, cell size cell_cm (anisotropic with stretch)."""
    u = x / (cell_cm * stretch[0]); v = y / (cell_cm * stretch[1])
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
    seed = int(rng.integers(1 << 30))

    def lattice(i, j):
        h = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ seed
        h = (h * 2654435761) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF).astype(np.float64)
    a, b, c, d = lattice(i0, j0), lattice(i0 + 1, j0), lattice(i0, j0 + 1), lattice(i0 + 1, j0 + 1)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def spec(shot: dict, world_cm: np.ndarray, waves: list[dict], t: float, rng: np.random.Generator, config: dict | None = None) -> dict | None:
    """Decide the white water of a shot (parameters only; geometry comes from build)."""
    cfg = (config or {}).get("whitewater", {})
    if rng.uniform() > cfg.get("probability", 1.0):
        return None
    cat = shot["category"]; base = CATEGORY.get(cat, CATEGORY["logging"])
    amount = float(rng.uniform(*cfg.get("amount_jitter", [0.4, 1.3])))
    if rng.uniform() < cfg.get("calm_fraction", 0.15):
        amount *= 0.3                                                     # the quiet moments between strokes
    surf = gerstner.surface_height(waves, world_cm[:, 0], world_cm[:, 1], t)
    contact = np.abs(world_cm[:, 2] - surf) < CONTACT_BAND_CM
    rot = shot["whale"]["rotation"]
    yaw = math.radians(float(rot.get("yaw", 0.0)))
    heading = [math.cos(yaw), math.sin(yaw)]                             # the snout's direction on the surface
    out = {"category": cat, "amount": round(amount, 3), "heading": [round(h, 4) for h in heading], "seed": int(rng.integers(1 << 31)),
           "contact_points": int(contact.sum()), "wave_time_s": float(t), **{k: round(v * amount, 3) if k != "wake_len" else v for k, v in base.items()}}
    if contact.sum() < 20:
        out["contact"] = 0.0; out["wake"] = 0.0; out["boil"] = 0.0           # body clear of the water (mid-breach) or well under it
        if shot.get("splash") and out["impact"] > 0:
            return out
        return None if out["impact"] <= 0 else out
    return out


def build(spec: dict, world_cm: np.ndarray, waves: list[dict], splash: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Foam carpet and aerated-water meshes: (vertices (N, 3) cm, faces (M, 3), uv (N, 2) = (foam, churn), keep (N,) bool)."""
    rng = np.random.default_rng(spec["seed"]); t = spec["wave_time_s"]
    hx, hy = spec["heading"]
    surf = gerstner.surface_height(waves, world_cm[:, 0], world_cm[:, 1], t)
    pts = world_cm[np.abs(world_cm[:, 2] - surf) < CONTACT_BAND_CM, :2]
    centre = pts.mean(0) if len(pts) else np.asarray(splash["centre_cm"][:2], float) if splash else world_cm[:, :2].mean(0)
    # the region: around the contact (or impact) with margin, extended behind for the wake
    lo = (pts.min(0) if len(pts) else centre) - 700.0; hi = (pts.max(0) if len(pts) else centre) + 700.0
    if splash:
        c = np.asarray(splash["centre_cm"][:2], float); r = 1.6 * float(splash["radius_cm"]) + 300.0
        lo = np.minimum(lo, c - r); hi = np.maximum(hi, c + r)
    wake_len = 100.0 * spec["wake_len"] * (1.0 if spec["wake"] > 0 else 0.0)
    if wake_len > 0:
        tail = centre - np.array([hx, hy]) * wake_len
        lo = np.minimum(lo, tail - 500.0); hi = np.maximum(hi, tail + 500.0)
    step = GRID_CM
    n = int(np.prod(np.ceil((hi - lo) / step) + 1))
    if n > MAX_VERTICES:
        step *= math.sqrt(n / MAX_VERTICES)
    xs = np.arange(lo[0], hi[0] + step, step); ys = np.arange(lo[1], hi[1] + step, step)
    gx, gy = np.meshgrid(xs, ys); gx = gx.ravel(); gy = gy.ravel()
    foam = np.zeros_like(gx); churn = np.zeros_like(gx)
    if len(pts):
        d = cKDTree(pts).query(np.stack([gx, gy], 1))[0]                  # distance to the contact line (cm)
        along = (gx - centre[0]) * hx + (gy - centre[1]) * hy               # position along the heading
        sigma = 60.0 + 70.0 * np.clip(along / 600.0, 0, 1)                  # wash along the flanks, wider at the bow
        foam += spec["contact"] * np.exp(-(d / sigma) ** 2)
        churn += spec["contact"] * np.exp(-(d / (3.0 * sigma)) ** 2)
        if spec["wake"] > 0 and wake_len > 0:
            rear = pts[np.argmin((pts - centre) @ np.array([hx, hy]))]      # the rearmost contact point
            s = -((gx - rear[0]) * hx + (gy - rear[1]) * hy)                 # metres behind, as cm
            w = (gx - rear[0]) * -hy + (gy - rear[1]) * hx
            width = 70.0 + 0.15 * np.clip(s, 0, None)
            wk = spec["wake"] * np.exp(-np.clip(s, 0, None) / wake_len) * np.exp(-(w / width) ** 2) * (s > -100.0)
            foam += 0.6 * wk; churn += wk
        if spec["boil"] > 0:
            boil = spec["boil"] * np.exp(-((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2) / (2 * 180.0 ** 2))
            foam += 0.6 * boil; churn += boil
    if splash and spec["impact"] > 0:
        c = np.asarray(splash["centre_cm"][:2], float); r = max(float(splash["radius_cm"]), 150.0)
        imp = spec["impact"] * np.exp(-((gx - c[0]) ** 2 + (gy - c[1]) ** 2) / (2 * (1.1 * r) ** 2))
        foam += imp; churn += 0.9 * imp
    # break-up: patches (isotropic noise) and streaks along the heading (stretched noise)
    n1 = value_noise(rng, gx, gy, 140.0); n2 = value_noise(rng, gx, gy, 45.0)
    sx, sy = gx * hx + gy * hy, -gx * hy + gy * hx
    streak = value_noise(rng, sx, sy, 90.0, stretch=(6.0, 1.0))
    pattern = 0.15 + 0.85 * (0.45 * n1 + 0.25 * n2 + 0.3 * streak)
    foam = np.clip(foam * pattern * 1.8, 0, 1)
    foam = np.clip((foam - 0.15) / 0.45, 0, 1) ** 1.2                     # soft threshold: real holes between patches
    churn = np.clip(churn * (0.4 + 0.6 * n1) * 0.9, 0, 1)
    keep = (foam > 0.02) | (churn > 0.04)
    # vertices on the wave surface (the rendered mesh follows the same displacement). The froth has
    # thickness: a lumpy layer up to ~10 cm where the foam is dense, so at a grazing angle it reads as
    # a raised, self-shadowing rim instead of a flat white line.
    lump = value_noise(rng, gx, gy, 30.0)
    V = gerstner.surface_points(waves, gx, gy, t); V[:, 2] += 1.5 + foam * (2.0 + 8.0 * lump)
    W, H = len(xs), len(ys)
    i = np.arange(H - 1)[:, None] * W + np.arange(W - 1)[None, :]
    i = i.ravel()
    faces = np.concatenate([np.stack([i, i + 1, i + W + 1], 1), np.stack([i, i + W + 1, i + W], 1)])
    return V, faces, np.stack([foam, churn], 1), keep


def _write(path: Path, V: np.ndarray, faces: np.ndarray, uv: np.ndarray, keep: np.ndarray, comment: str) -> dict:
    """OBJ of the faces touching a kept vertex (Y mirrored and V flipped for Unreal's importer, like factory.splash)."""
    fk = faces[keep[faces].any(1)]
    used = np.unique(fk)
    remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    lines = [f"# {comment}"]
    lines.extend(f"v {x:.1f} {-y:.1f} {z:.1f}" for x, y, z in V[used])
    lines.extend(f"vt {u:.3f} {1.0 - v:.3f}" for u, v in uv[used])
    f = remap[fk] + 1
    lines.extend(f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f)
    path.write_text("\n".join(lines) + "\n")
    return {"vertices": int(len(used)), "faces": int(len(f))}


def ensure_obj(spec: dict | None, directory, name: str, world_cm: np.ndarray, waves: list[dict], splash: dict | None) -> dict | None:
    """Write the foam carpet (<name>_foam.obj) and the aerated-water cloud (<name>_churn.obj) for a shot."""
    if not spec:
        return spec
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    V, faces, uv, keep = build(spec, world_cm, waves, splash)
    foam_path = directory / f"{name}_foam.obj"
    spec["mesh"] = _write(foam_path, V, faces, uv, keep & (uv[:, 0] > 0.0), f"white water {spec['category']} foam, generated by factory.whitewater")
    spec["mesh_obj"] = str(foam_path.resolve()) if spec["mesh"]["faces"] else None     # an empty OBJ would fail the import
    Vc = V.copy(); Vc[:, 2] -= 20.0 + 70.0 * uv[:, 1]                    # the aerated cloud hangs below the surface
    churn_path = directory / f"{name}_churn.obj"
    spec["churn_mesh"] = _write(churn_path, Vc, faces, np.stack([uv[:, 1], uv[:, 0]], 1), keep & (uv[:, 1] > 0.0), f"white water {spec['category']} aerated water")
    spec["churn_obj"] = str(churn_path.resolve()) if spec["churn_mesh"]["faces"] else None
    if not spec["mesh_obj"] and spec["churn_obj"]:
        spec["mesh_obj"] = None                                          # wf_shot.apply_whitewater keys on the foam; no foam, no white water
    spec["foam_area_m2"] = round(float((uv[:, 0] > 0.1).sum() * (GRID_CM / 100.0) ** 2), 1)
    return spec
