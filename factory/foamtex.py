"""A tileable sea-foam texture, generated: bubble cells (Worley noise, several scales) with bright
rims and a large-scale coverage mask, so foam breaks into lacy patches and holes like real wash.

    python3 -m factory.foamtex third_party_assets/generated/T_Foam.png [--size 2048 --seed 1]

Channels: R = foam density (0..1), G = rim highlight, B = bubble-interior darkening, A = coverage
(large-scale patchiness). The white-water materials tile it in world space (about 1.5 m per repeat).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def worley(rng: np.random.Generator, size: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Tileable F1 and F2 distances (in cell units) of a jittered grid of `cells` x `cells` feature points."""
    pts = (np.indices((cells, cells)).transpose(1, 2, 0) + rng.uniform(0, 1, (cells, cells, 2))) / cells
    ys, xs = np.mgrid[0:size, 0:size] / size
    f1 = np.full((size, size), np.inf); f2 = np.full((size, size), np.inf)
    for i in range(cells):
        for j in range(cells):
            px, py = pts[i, j]
            dx = np.abs(xs - px); dx = np.minimum(dx, 1 - dx)            # wrap-around: tileable
            dy = np.abs(ys - py); dy = np.minimum(dy, 1 - dy)
            d = np.sqrt(dx * dx + dy * dy) * cells
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d)); f1 = np.where(closer, d, f1)
    return f1, f2


def value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    g = rng.uniform(0, 1, (cells + 1, cells + 1)); g[-1, :] = g[0, :]; g[:, -1] = g[:, 0]
    ys, xs = np.mgrid[0:size, 0:size] / size * cells
    i0, j0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - i0, xs - j0; fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
    i1, j1 = np.minimum(i0 + 1, cells), np.minimum(j0 + 1, cells)
    return (g[i0, j0] * (1 - fx) + g[i0, j1] * fx) * (1 - fy) + (g[i1, j0] * (1 - fx) + g[i1, j1] * fx) * fy


def make(size: int = 2048, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    density = np.zeros((size, size)); rim = np.zeros((size, size)); dark = np.zeros((size, size))
    for cells, weight in ((9, 0.45), (19, 0.35), (41, 0.2)):
        f1, f2 = worley(rng, size, cells)
        edge = np.clip((f2 - f1) * 2.2, 0, 1)                           # 1 inside a cell, 0 on the cell walls
        density += weight * (1 - np.clip(f1 * 1.4, 0, 1) ** 1.5)        # foam concentrated around feature points
        rim += weight * np.exp(-((f2 - f1) / 0.12) ** 2)                # bright bubble walls
        dark += weight * edge * np.clip(1 - f1 * 1.2, 0, 1)             # bubble interiors a touch darker
    cover = 0.55 * value_noise(rng, size, 3) + 0.3 * value_noise(rng, size, 7) + 0.15 * value_noise(rng, size, 17)
    cover = np.clip((cover - 0.3) / 0.45, 0, 1)
    density = np.clip(density * (0.6 + 0.6 * value_noise(rng, size, 29)), 0, 1)
    tex = np.stack([density, np.clip(rim, 0, 1), np.clip(dark, 0, 1), cover], -1)
    return (tex * 255).astype(np.uint8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path); p.add_argument("--size", type=int, default=2048); p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    from PIL import Image

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make(a.size, a.seed), "RGBA").save(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
