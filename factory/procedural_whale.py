"""Procedural stand-in humpback whale mesh (our own geometry, safe for the public repo).

This is a placeholder for building and validating the scene and label pipeline until a
realistic rigged humpback asset is sourced. Proportions loosely follow an adult
humpback: about 14 m long, pectoral fins about a third of body length, flukes about a
third of body length in span, and a small dorsal fin about two-thirds of the way back.

Units are centimetres. Axes match Unreal: +X forward (snout), +Y right, +Z up. The
origin is at the body's mid-length on the centre line.

    python3 -m factory.procedural_whale unreal/WhaleFactory/SourceAssets/procedural_whale.obj
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LENGTH_CM = 1400.0

# Body profile: (position from snout 0..1, half-height m, half-width m).
_BODY_PROFILE = np.array([
    (0.00, 0.06, 0.06),
    (0.03, 0.42, 0.52),
    (0.10, 0.80, 0.98),
    (0.20, 1.15, 1.32),
    (0.33, 1.42, 1.52),
    (0.45, 1.42, 1.46),
    (0.60, 1.18, 1.12),
    (0.72, 0.84, 0.74),
    (0.82, 0.52, 0.40),
    (0.90, 0.32, 0.22),
    (0.96, 0.22, 0.14),
    (1.00, 0.08, 0.06),
])


class MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.faces: list[tuple[int, int, int]] = []

    def _add_vertices(self, points: np.ndarray) -> int:
        start = sum(len(v) for v in self.vertices)
        self.vertices.append(np.asarray(points, dtype=float))
        return start

    def add_loft(self, rings: np.ndarray, cap_start: bool = True, cap_end: bool = True) -> None:
        """Skin a (n_rings, n_around, 3) array of closed cross-section rings."""
        n_rings, n_around, _ = rings.shape
        base = self._add_vertices(rings.reshape(-1, 3))
        idx = lambda r, a: base + r * n_around + (a % n_around)  # noqa: E731
        for r in range(n_rings - 1):
            for a in range(n_around):
                v00, v01 = idx(r, a), idx(r, a + 1)
                v10, v11 = idx(r + 1, a), idx(r + 1, a + 1)
                self.faces += [(v00, v10, v11), (v00, v11, v01)]
        for ring, flip, wanted in ((0, True, cap_start), (n_rings - 1, False, cap_end)):
            if not wanted:
                continue
            centre = self._add_vertices(rings[ring].mean(axis=0, keepdims=True))
            for a in range(n_around):
                tri = (centre, idx(ring, a + 1), idx(ring, a)) if not flip else (centre, idx(ring, a), idx(ring, a + 1))
                self.faces.append(tri)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.vertices), np.asarray(self.faces, dtype=np.int64)


def _smooth_profile(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, n)
    h = np.interp(s, _BODY_PROFILE[:, 0], _BODY_PROFILE[:, 1])
    w = np.interp(s, _BODY_PROFILE[:, 0], _BODY_PROFILE[:, 2])
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    h = np.convolve(np.pad(h, pad, mode="edge"), kernel, mode="valid")
    w = np.convolve(np.pad(w, pad, mode="edge"), kernel, mode="valid")
    return s, h * 100.0, w * 100.0


def _x_at(s: float | np.ndarray) -> float | np.ndarray:
    return (0.5 - s) * LENGTH_CM


def _body_rings(n_along: int = 120, n_around: int = 48) -> np.ndarray:
    s, h, w = _smooth_profile(n_along)
    theta = np.linspace(0.0, 2 * np.pi, n_around, endpoint=False)
    rings = np.empty((n_along, n_around, 3))
    for i in range(n_along):
        # Humpback head: flatter on top with a deep lower jaw, so shift the centre down.
        head = np.clip(1.0 - s[i] / 0.3, 0.0, 1.0)
        z_centre = -0.18 * h[i] * head
        top_scale = np.where(np.sin(theta) > 0, 1.0 - 0.25 * head, 1.0)
        rings[i, :, 0] = _x_at(s[i])
        rings[i, :, 1] = w[i] * np.cos(theta)
        rings[i, :, 2] = z_centre + h[i] * np.sin(theta) * top_scale
    return rings


def _profile_at(s: float) -> tuple[float, float]:
    grid, h, w = _smooth_profile(400)
    return float(np.interp(s, grid, h)), float(np.interp(s, grid, w))


def _foil_rings(
    root: np.ndarray,
    direction: np.ndarray,
    length: float,
    chord: tuple[float, float],
    thickness: tuple[float, float],
    chord_hint: np.ndarray,
    sweep: np.ndarray | None = None,
    n_along: int = 24,
    n_around: int = 20,
) -> np.ndarray:
    """Loft a tapered, flattened elliptical foil (fin or fluke) along a straight spine."""
    direction = direction / np.linalg.norm(direction)
    chord_axis = chord_hint - direction * np.dot(chord_hint, direction)
    chord_axis /= np.linalg.norm(chord_axis)
    thick_axis = np.cross(direction, chord_axis)
    t = np.linspace(0.0, 1.0, n_along)
    theta = np.linspace(0.0, 2 * np.pi, n_around, endpoint=False)
    # Rounded tip: taper chord and thickness with a quarter-ellipse toward the end.
    tip = np.sqrt(np.clip(1.0 - t**4, 0.05, 1.0))
    rings = np.empty((n_along, n_around, 3))
    for i, ti in enumerate(t):
        c = (chord[0] + (chord[1] - chord[0]) * ti) * tip[i] / 2
        k = (thickness[0] + (thickness[1] - thickness[0]) * ti) * tip[i] / 2
        centre = root + direction * length * ti
        if sweep is not None:
            centre = centre + sweep * ti**1.5
        # Thicker toward the leading edge (forward half of the chord), like a real fin.
        lead = np.where(np.cos(theta) > 0, 1.0, 0.6)
        rings[i] = (centre
                    + np.outer(c * np.cos(theta), chord_axis)
                    + np.outer(k * lead * np.sin(theta), thick_axis))
    return rings


def build_whale() -> tuple[np.ndarray, np.ndarray]:
    mesh = MeshBuilder()
    mesh.add_loft(_body_rings())

    # Pectoral fins: about 4.5 m, rooted low on the flank behind the head, swept back and down.
    s_root = 0.27
    h, w = _profile_at(s_root)
    for side in (1.0, -1.0):
        root = np.array([_x_at(s_root), side * w * 0.8, -0.45 * h])
        direction = np.array([-0.55, side * 0.75, -0.40])
        mesh.add_loft(_foil_rings(root, direction, 450.0, chord=(95.0, 30.0), thickness=(28.0, 6.0),
                                  chord_hint=np.array([1.0, 0.0, 0.0])))

    # Flukes: about 4.4 m span, horizontal, swept back toward the tips.
    s_root = 0.955
    h, _ = _profile_at(s_root)
    for side in (1.0, -1.0):
        root = np.array([_x_at(s_root), 0.0, 0.0])
        mesh.add_loft(_foil_rings(root, np.array([0.0, side, 0.0]), 220.0, chord=(110.0, 35.0),
                                  thickness=(18.0, 3.0), chord_hint=np.array([1.0, 0.0, 0.0]),
                                  sweep=np.array([-70.0, 0.0, 0.0])))

    # Small dorsal fin on a hump, about two-thirds of the way back.
    s_root = 0.64
    h, _ = _profile_at(s_root)
    root = np.array([_x_at(s_root), 0.0, 0.85 * h])
    mesh.add_loft(_foil_rings(root, np.array([-0.35, 0.0, 1.0]), 45.0, chord=(90.0, 20.0),
                              thickness=(22.0, 4.0), chord_hint=np.array([1.0, 0.0, 0.0]),
                              n_along=10))
    return mesh.arrays()


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Procedural stand-in humpback (Whale-Image-Data-Factory), centimetres, Z up, X forward\n")
        f.write("o procedural_whale\ns 1\n")
        for x, y, z in vertices:
            f.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
        # Side-projected UVs (length x height); Unreal's OBJ importer requires UVs.
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        for x, _y, z in vertices:
            f.write(f"vt {(x - lo[0]) / span[0]:.5f} {(z - lo[2]) / span[2]:.5f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "procedural_whale.obj"
    v, f = build_whale()
    write_obj(out, v, f)
    print(f"wrote {out}: {len(v)} vertices, {len(f)} triangles, "
          f"bounds min {v.min(axis=0).round(1)} max {v.max(axis=0).round(1)} cm")
