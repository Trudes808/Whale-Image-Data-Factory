"""Whale-water interaction on a heightfield: an iWave-style linear wave solver (Tessendorf, "Interactive Water
Surfaces", 2004) with the whale's waterline cross-section as a moving obstruction.

The surface h(x, y, t) around the whale obeys deep-water linear wave dynamics
    h_tt + alpha h_t = -g * sqrt(-laplacian) h
solved on a periodic grid with the vertical-derivative operator applied spectrally (|k| in Fourier space, exact on
the grid). The whale enters as an obstruction mask O (1 where the body crosses the water plane): the surface under
the body is pinned, and water is pushed outwards where the mask grows (the body advancing or rising) and pulled in
where it shrinks. A moving mask therefore produces the bow wave, the wash along the flanks and the wake; a rising
one the surfacing bulge; an impact (slap, breach) a ring wave. Whitecap foam is injected where the surface slope is
steep and along the moving waterline, and decays.

This is a heightfield model: it gives displacement, wake and ring waves consistent with the whale's motion, which
the labels can evaluate exactly. It does not produce sheets, droplets or plunging crests; those stay with the
Niagara splash meshes (factory.splash) placed on this surface.

    sim = IWave(size_m=64.0, res=512)
    for t in np.arange(0, 6, sim.dt):
        sim.step(obstruction_at(t))      # (res, res) float in [0, 1]
    h, foam = sim.h, sim.foam
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRAV = 9.81


@dataclass
class IWave:
    size_m: float = 64.0
    res: int = 512
    dt: float = 0.02
    alpha: float = 0.3            # damping (1/s)
    push_gain: float = 0.35       # metres of surface displacement per unit mask growth (volume pushed aside)
    foam_slope: float = 0.45      # slope magnitude where foam starts
    foam_decay: float = 0.12      # 1/s
    h: np.ndarray = field(default=None, repr=False)
    h_prev: np.ndarray = field(default=None, repr=False)
    foam: np.ndarray = field(default=None, repr=False)
    mask_prev: np.ndarray = field(default=None, repr=False)
    _kmag: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        n = self.res
        self.h = np.zeros((n, n)); self.h_prev = np.zeros((n, n)); self.foam = np.zeros((n, n)); self.mask_prev = np.zeros((n, n))
        k = 2.0 * math.pi * np.fft.fftfreq(n, d=self.size_m / n)
        kx, ky = np.meshgrid(k, k)
        self._kmag = np.hypot(kx, ky)
        self.dx = self.size_m / n

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        xs = (np.arange(self.res) + 0.5) * self.dx - self.size_m / 2
        return np.meshgrid(xs, xs)

    def vertical_derivative(self, h: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(self._kmag * np.fft.fft2(h)))

    def step(self, mask: np.ndarray, vertical_rate: float = 0.0) -> None:
        """Advance one time step with the body's waterline mask (1 inside). vertical_rate (m/s, positive up) is
        the body's vertical speed, which pushes the surface around the mask edge as it rises or sinks."""
        dt, a = self.dt, self.alpha
        g_term = GRAV * self.vertical_derivative(self.h)
        h_new = (2.0 * self.h - self.h_prev + 0.5 * a * dt * self.h_prev - dt * dt * g_term) / (1.0 + 0.5 * a * dt)
        # source: mask growth pushes water up around the body, shrinkage lets it fall back
        grow = np.clip(mask - self.mask_prev, 0.0, 1.0); shrink = np.clip(self.mask_prev - mask, 0.0, 1.0)
        push = (grow - 0.6 * shrink) * self.push_gain + mask * vertical_rate * dt * 0.8
        h_new = h_new + _blur(push, 2)
        # the surface is pinned under the body
        h_new = h_new * (1.0 - mask)
        self.h_prev, self.h = self.h, h_new
        # foam: steep slopes and the moving waterline
        gy, gx = np.gradient(self.h, self.dx)
        slope = np.hypot(gx, gy)
        inject = np.clip((slope - self.foam_slope) * 3.0, 0.0, 1.0) + _blur(grow + shrink, 1) * 1.5
        self.foam = np.minimum(np.maximum(self.foam * math.exp(-dt * self.foam_decay), inject), 1.5) * (1.0 - mask)
        self.mask_prev = mask


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a
    out = a.copy()
    for _ in range(r):
        out = 0.25 * out + 0.125 * (np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) + 0.0625 * (
            np.roll(np.roll(out, 1, 0), 1, 1) + np.roll(np.roll(out, 1, 0), -1, 1) + np.roll(np.roll(out, -1, 0), 1, 1) + np.roll(np.roll(out, -1, 0), -1, 1))
    return out


def waterline_mask(points_xy: np.ndarray, sim: IWave, centre_xy=(0.0, 0.0), dilate: int = 1) -> np.ndarray:
    """Rasterise the body's waterline cross-section (xy points of the body within a band of the water plane) into
    the simulation grid; the hull is the convex hull of the points, filled."""
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MPath
    n = sim.res
    if len(points_xy) < 4:
        return np.zeros((n, n))
    try:
        hull = ConvexHull(points_xy)
    except Exception:
        return np.zeros((n, n))
    poly = points_xy[hull.vertices]
    X, Y = sim.coords
    inside = MPath(poly).contains_points(np.stack([X.ravel() + centre_xy[0], Y.ravel() + centre_xy[1]], 1)).reshape(n, n).astype(float)
    return _blur(inside, dilate) if dilate else inside


def body_section(vertices_world_cm: np.ndarray, water_z_cm: float, band_cm: float = 35.0) -> np.ndarray:
    """xy (metres) of the body vertices within +-band of the water plane: the waterline cross-section."""
    v = vertices_world_cm
    sel = np.abs(v[:, 2] - water_z_cm) < band_cm
    return v[sel][:, :2] / 100.0
