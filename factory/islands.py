"""Distant land for the backgrounds: procedural island and headland silhouettes (ridges of random
height and extent) placed kilometres away on the horizon, so the segmentation sees land, haze and
hard horizon features it never saw in run01-run03 (where it fired on islands). Meshes are generated
once per run (a small library), placed per batch at random bearings and distances; the label pass
treats them as background (no stencil). Aerial perspective comes from the height fog.

    islands = make_library(run_dir / "islands", rng, count=6)       # OBJ files
    placement = place(rng, config, n_islands)                      # per batch: name, bearing, distance, yaw, scale
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from factory.whitewater import value_noise


def ridge_mesh(rng: np.random.Generator, length_m: float, width_m: float, height_m: float, peaks: int = 3, res_m: float = 25.0):
    """A heightfield island: a few Gaussian peaks along the length, eroded by noise, with a shore ring.
    Returns vertices (N, 3) cm, faces (M, 3), uv (N, 2) = (height fraction, slope)."""
    nx, ny = int(length_m / res_m) + 1, int(width_m / res_m) + 1
    xs = np.linspace(-length_m / 2, length_m / 2, nx); ys = np.linspace(-width_m / 2, width_m / 2, ny)
    gx, gy = np.meshgrid(xs, ys); gx = gx.ravel(); gy = gy.ravel()
    z = np.zeros_like(gx)
    for _ in range(peaks):
        cx = rng.uniform(-0.35, 0.35) * length_m; cy = rng.uniform(-0.25, 0.25) * width_m
        sx = rng.uniform(0.18, 0.4) * length_m; sy = rng.uniform(0.25, 0.5) * width_m
        z += rng.uniform(0.5, 1.0) * height_m * np.exp(-((gx - cx) / sx) ** 2 - ((gy - cy) / sy) ** 2)
    n = value_noise(rng, gx * 100.0, gy * 100.0, 18000.0) * 0.6 + value_noise(rng, gx * 100.0, gy * 100.0, 6000.0) * 0.4
    z *= 0.65 + 0.7 * n
    edge = np.exp(-((2 * gx / length_m) ** 6 + (2 * gy / width_m) ** 6))     # fall to the sea at the ends
    z = z * edge - 2.0
    W, H = nx, ny
    i = (np.arange(H - 1)[:, None] * W + np.arange(W - 1)[None, :]).ravel()
    faces = np.concatenate([np.stack([i, i + 1, i + W + 1], 1), np.stack([i, i + W + 1, i + W], 1)])
    zz = z.reshape(H, W); gzy, gzx = np.gradient(zz, res_m)
    slope = np.clip(np.sqrt(gzx ** 2 + gzy ** 2).ravel() / 0.8, 0, 1)
    V = np.stack([gx, gy, z], 1) * 100.0
    uv = np.stack([np.clip(z / max(z.max(), 1.0), 0, 1), slope], 1)
    return V, faces, uv


def write_obj(path: Path, V, faces, uv, comment: str) -> dict:
    lines = [f"# {comment}"]
    lines.extend(f"v {x:.0f} {-y:.0f} {z:.0f}" for x, y, z in V)              # Unreal's OBJ import mirrors Y
    lines.extend(f"vt {u:.3f} {1.0 - v:.3f}" for u, v in uv)
    lines.extend(f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces + 1)
    path.write_text("\n".join(lines) + "\n")
    return {"vertices": int(len(V)), "faces": int(len(faces))}


def make_library(directory: Path, rng: np.random.Generator, count: int = 6) -> list[dict]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(count):
        length = float(rng.uniform(600.0, 5000.0)); width = float(rng.uniform(0.3, 0.7)) * length
        height = float(rng.uniform(60.0, 420.0)) * (0.6 + 0.4 * length / 5000.0)
        V, faces, uv = ridge_mesh(rng, length, width, height, peaks=int(rng.integers(2, 5)))
        path = directory / f"island_{k:02d}.obj"
        spec = {"name": f"island_{k:02d}", "mesh_obj": str(path.resolve()), "length_m": round(length), "height_m": round(height),
                "mesh": write_obj(path, V, faces, uv, f"island {k}, {length:.0f} m long, {height:.0f} m high (factory.islands)")}
        out.append(spec)
    return out


def place(rng: np.random.Generator, library: list[dict], config: dict) -> list[dict]:
    """Per batch: which islands sit where (bearing and distance from the whale, yaw, scale)."""
    cfg = config.get("islands", {})
    if not library or rng.uniform() > cfg.get("fraction", 0.5):
        return []
    n = int(rng.integers(1, cfg.get("max_count", 3) + 1))
    out = []
    for _ in range(n):
        isl = library[int(rng.integers(len(library)))]
        bearing = float(rng.uniform(0, 360)); distance = float(np.exp(rng.uniform(np.log(cfg.get("distance_m", [2500, 18000])[0]), np.log(cfg.get("distance_m", [2500, 18000])[1]))))
        out.append({"name": isl["name"], "mesh_obj": isl["mesh_obj"], "bearing_deg": round(bearing, 2), "distance_m": round(distance),
                    "yaw_deg": round(float(rng.uniform(0, 360)), 2), "scale": round(float(rng.uniform(0.7, 1.5)), 3),
                    "location_cm": [round(100.0 * distance * np.cos(np.radians(bearing)), 1), round(100.0 * distance * np.sin(np.radians(bearing)), 1), 0.0]})
    return out
