"""The contact layer: the water a whale disturbs, on a heightfield grid around the animal, baked per shot.

Solver: eWave (Tessendorf 2014, the exact-exponential successor of iWave): linear deep-water waves with capillarity,
each Fourier mode advanced exactly over a step (no CFL limit, no numerical dispersion):
    dh/dt = |k| phi,    dphi/dt = -(omega^2 / |k|) h - g P,      omega^2 = g |k| + sigma |k|^3
driven by the whale as a moving surface pressure P (Havelock's pressure-patch model of a hull, in metres of water
head): the body's thickness in slabs below the mean surface (0-0.5, 0.5-1, 1-2, 2-4 m), each attenuated with depth
as exp(-|k| z). The baked height is the dynamic part h + P, so a motionless body leaves the surface flat while a
moving / rising / sinking one radiates its bow and flank waves, Kelvin wake, surfacing bulge or the collapsing hole of
a dive; an impact (slap, breach landing) starts a crater that rings out.
The whale's motion before the shot is a rigid replay of the shot's posed mesh: moving along its heading at a
category speed, rising / sinking over the last seconds, plus an optional impact (CATEGORY_MOTION, guesses).

Channels of the per-shot texture (RGBA half EXR, res x res over size_m, centred on the whale, world-aligned):
    R height (m, added to the sea's height at the vertex's rest position), G foam (0..1.5: contact line, breaking
    bow / impact water, wake), B damping (0..1: flukeprint / wake slicks, short waves suppressed by up to 90 %),
    A wash (0..1: water sheeting over the just-awash back, for the whale material).

    layer = run_shot(shot, posed_vertices_world_m, faces)          # dict with "tex", "origin_m", "size_m", ...
    write_exr(layer, path)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GRAV = 9.81
SURF = 7.4e-5

# category -> motion before the shot: speed along the heading (m/s), vertical rate (m/s, + up) over the last
# rise_window_s, history (s), impact (seconds before the shot, radius m, crater depth m, where: "fin" / "body")
CATEGORY_MOTION = {
    "logging":     dict(speed=0.3, rise=0.0),
    "surface":     dict(speed=1.6, rise=0.15),
    "blow":        dict(speed=1.8, rise=0.3),
    "blow_sliver": dict(speed=1.6, rise=0.2),
    "back_arch":   dict(speed=1.8, rise=-0.2),
    "arch":        dict(speed=1.8, rise=-0.2),
    "awash":       dict(speed=1.2, rise=0.0),
    "submerged":   dict(speed=1.2, rise=0.0),
    "fluke_down":  dict(speed=1.4, rise=-0.5),
    "peduncle":    dict(speed=1.4, rise=-0.5),
    "tail_stock":  dict(speed=1.4, rise=-0.5),
    "fluke_up":    dict(speed=1.2, rise=-0.7),
    "fluke_edge":  dict(speed=1.2, rise=-0.7),
    "fluke_rollout": dict(speed=1.4, rise=-0.5),
    "fluke_sinking": dict(speed=0.8, rise=-0.6),
    "breach":      dict(speed=1.0, rise=3.5, rise_window_s=1.2, impact=(-6.0, 4.0, 1.0, "body")),
    "pec_up":      dict(speed=0.3, rise=0.0, impact=(-1.6, 1.3, 0.35, "fin")),
    "fin_tip":     dict(speed=0.3, rise=0.0, impact=(-2.0, 1.0, 0.25, "fin")),
    "roll_belly":  dict(speed=0.4, rise=0.0, impact=(-2.5, 2.0, 0.4, "body")),
    "spyhop":      dict(speed=0.0, rise=0.6),
    "spyhop_low":  dict(speed=0.0, rise=0.3),
}
SLABS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]


@dataclass
class ContactParams:
    size_m: float = 72.0
    res: int = 576
    dt: float = 0.04
    history_s: float = 10.0
    rise_window_s: float = 2.5
    visc: float = 2e-4             # eddy viscosity on the waves (m^2/s): damps the shortest ripples
    sponge_m: float = 6.0
    foam_decay_s: float = 9.0
    damp_decay_s: float = 120.0
    break_acc_g: float = 0.5       # foam where the surface accelerates downward faster than this x g (breaking)
    edge_foam: float = 0.5         # contact-line foam per (m/s) of relative water speed
    wake_damp: float = 0.25        # slick added per (m/s * s) of the body's top slab passing
    upwell_damp: float = 1.5       # slick per (m/s) of the body leaving the top metre (flukeprint)


def body_columns(vertices_m: np.ndarray, faces: np.ndarray, origin, size_m: float, res: int, order: int = 8):
    """Top-view zmin / zmax (m) of the body per grid column (NaN outside), from barycentric samples of every face."""
    from scipy import ndimage
    V = vertices_m[faces]                                    # (F, 3, 3)
    ii = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    B = np.array([[i / order, j / order, 1 - (i + j) / order] for i, j in ii])            # (S, 3)
    P = np.einsum("sk,fkd->fsd", B, V).reshape(-1, 3)
    dx = size_m / res
    ci = np.floor((P[:, 0] - origin[0]) / dx).astype(int); cj = np.floor((P[:, 1] - origin[1]) / dx).astype(int)
    ok = (ci >= 0) & (ci < res) & (cj >= 0) & (cj < res)
    flat = cj[ok] * res + ci[ok]
    zmin = np.full(res * res, np.inf); zmax = np.full(res * res, -np.inf)
    np.minimum.at(zmin, flat, P[ok, 2]); np.maximum.at(zmax, flat, P[ok, 2])
    zmin = zmin.reshape(res, res); zmax = zmax.reshape(res, res)
    got = np.isfinite(zmin)
    filled = ndimage.binary_closing(got, iterations=2) | got
    filled = ndimage.binary_fill_holes(filled)
    if (filled & ~got).any():
        # fill gaps from the nearest sampled column
        idx = ndimage.distance_transform_edt(~got, return_distances=False, return_indices=True)
        zmin = np.where(got, zmin, zmin[idx[0], idx[1]]); zmax = np.where(got, zmax, zmax[idx[0], idx[1]])
    zmin[~filled] = np.nan; zmax[~filled] = np.nan
    return zmin, zmax


def _fbm(rng, res, size_m, scales=(0.4, 1.0, 2.5), weights=(0.3, 0.4, 0.3)):
    k = 2 * math.pi * np.fft.fftfreq(res, d=size_m / res)
    kx, ky = np.meshgrid(k, k); kk = np.hypot(kx, ky)
    out = np.zeros((res, res))
    for s, w in zip(scales, weights):
        spec = np.exp(-0.5 * (kk * s) ** 2) * (kk > 0)
        f = np.real(np.fft.ifft2(spec * (rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res)))))
        out += w * f / max(f.std(), 1e-9)
    return out


class EWave:
    def __init__(self, p: ContactParams):
        self.p = p
        n = p.res
        k1 = 2 * math.pi * np.fft.fftfreq(n, d=p.size_m / n)
        self.kx, self.ky = np.meshgrid(k1, k1)
        self.k = np.hypot(self.kx, self.ky)
        self.w = np.sqrt(GRAV * self.k + SURF * self.k ** 3)
        kz = np.where(self.k > 0, self.k, 1.0)
        self.cw = np.cos(self.w * p.dt); self.sw = np.sin(self.w * p.dt)
        self.k_over_w = np.where(self.k > 0, self.k / np.where(self.w > 0, self.w, 1.0), 0.0)
        self.w_over_k = np.where(self.k > 0, self.w / kz, 0.0)
        self.heq_gain = np.where(self.k > 0, -GRAV * self.k / np.where(self.w > 0, self.w, 1.0) ** 2, 0.0)   # h_eq = gain * P
        self.visc = np.exp(-2.0 * p.visc * self.k ** 2 * p.dt)
        self.H = np.zeros((n, n), complex); self.PHI = np.zeros((n, n), complex)
        x = (np.arange(n) + 0.5) / n * p.size_m
        e = np.minimum(x, p.size_m - x)
        s1 = np.clip(1.0 - e / p.sponge_m, 0, 1) ** 2
        self.sponge = 1.0 - 0.15 * np.maximum(s1[None, :], s1[:, None])

    def step(self, P_hat: np.ndarray):
        heq = self.heq_gain * P_hat
        u = self.H - heq
        u2 = u * self.cw + self.k_over_w * self.PHI * self.sw
        phi2 = self.PHI * self.cw - self.w_over_k * u * self.sw
        self.H = (u2 + heq) * self.visc; self.PHI = phi2 * self.visc

    def apply_sponge(self):
        h = _ifft(self.H) * self.sponge; ph = _ifft(self.PHI) * self.sponge
        self.H = _fft(h); self.PHI = _fft(ph)

    def height(self) -> np.ndarray:
        return _ifft(self.H)


def run(vertices_m: np.ndarray, faces: np.ndarray, category: str, heading_xy, centre_xy=None, seed: int = 0,
        params: ContactParams | None = None, motion: dict | None = None, water_z: float = 0.0) -> dict:
    """Simulate the history before the shot and return the contact texture (res, res, 4) and its placement.
    vertices_m: the posed whale at the shot, world metres (z relative to the local mean water level water_z)."""
    t0 = time.time()
    p = params or ContactParams()
    mot = dict(CATEGORY_MOTION.get(category, dict(speed=1.0, rise=0.0)))
    if motion:
        mot.update(motion)
    rise_window = mot.get("rise_window_s", p.rise_window_s)
    V = np.asarray(vertices_m, float).copy(); V[:, 2] -= water_z
    if centre_xy is None:
        sub = V[V[:, 2] > -1.0]
        centre_xy = (sub[:, :2].mean(0) if len(sub) else V[:, :2].mean(0))
    origin = (float(centre_xy[0]) - p.size_m / 2, float(centre_xy[1]) - p.size_m / 2)
    n = p.res; dx = p.size_m / n
    hd = np.asarray(heading_xy, float); hd = hd / max(np.linalg.norm(hd), 1e-9)
    speed = float(mot["speed"]); rise = float(mot["rise"])
    zmin, zmax = body_columns(V, faces, origin, p.size_m, n)
    body = np.isfinite(zmin)
    zmin0 = np.where(body, zmin, 0.0); zmax0 = np.where(body, zmax, 0.0)
    ew = EWave(p)
    kx, ky, kk = ew.kx, ew.ky, ew.k
    att = [np.exp(-kk * 0.5 * (a + b)) for a, b in SLABS]
    rng = np.random.default_rng(seed)
    lace = np.clip(0.65 + 0.35 * _fbm(rng, n, p.size_m), 0.1, 1.3)
    foam = np.zeros((n, n)); damp = np.zeros((n, n)); wash = np.zeros((n, n))
    xs = origin[0] + (np.arange(n) + 0.5) * dx; X, Y = np.meshgrid(xs, origin[1] + (np.arange(n) + 0.5) * dx)
    steps = int(round(p.history_s / p.dt))
    h_prev = h_prev2 = None
    P_prev_top = None
    imp = mot.get("impact")
    imp_step = None if imp is None else steps - 1 + int(round(imp[0] / p.dt))
    fd = math.exp(-p.dt / p.foam_decay_s); dd = math.exp(-p.dt / p.damp_decay_s)
    from scipy import ndimage

    def slabs_at(dz):
        lo = zmin0 + dz; hi = zmax0 + dz
        out = []
        for a, b in SLABS:
            # thickness of the body within depth [a, b] below the surface (z in [-b, -a])
            th = np.clip(np.minimum(hi, -a) - np.maximum(lo, -b), 0, None) * body
            out.append(th)
        return out, lo, hi

    spec_cache: dict = {}
    for s in range(steps):
        tau = (s - (steps - 1)) * p.dt                  # seconds relative to the shot (<= 0)
        # the body was at shift(tau) relative to its pose at the shot; it starts moving smoothly (3 s ramp from the
        # start of the history) and its rise / sink eases in over 0.7 s, so neither switches on a ring wave
        along = -_eased_path(-tau, p.history_s - p.dt, 3.0) * speed
        shift = np.array([along * hd[0], along * hd[1]])
        dz = -rise * _eased_path(-tau, rise_window, 0.7)  # earlier = displaced backwards along the rise
        key = round(dz / 0.02) * 0.02                    # the body's spectra only change with its height (2 cm steps)
        if key not in spec_cache:
            spec_cache.clear()
            th, lo, hi = slabs_at(key)
            F = [_fft(t) for t in th]
            spec_cache[key] = (th, lo, hi, F, sum(f * a for f, a in zip(F, att)), _fft(((hi > 0) & (lo < 0) & body).astype(float)), _fft(hi))
        th, lo, hi, Fth, Psum, Fpierce, Fhi = spec_cache[key]
        ramp = np.exp(-1j * (kx * shift[0] + ky * shift[1]))
        P_hat = Psum * ramp
        if s == 0:                                       # start from the body's static equilibrium: no switch-on ring
            ew.H = ew.heq_gain * P_hat; ew.PHI = np.zeros_like(P_hat)
        ew.step(P_hat)
        if s % 10 == 9:
            ew.apply_sponge()
        P_real = _ifft(P_hat)
        h = ew.height()
        # impact: a crater with a small rim at the impact point
        if imp_step is not None and s == imp_step:
            ip = _impact_xy(V, imp[3], centre_xy)
            r2 = (X - ip[0]) ** 2 + (Y - ip[1]) ** 2; R = imp[1]
            crater = -imp[2] * np.exp(-2.0 * r2 / R ** 2) + 0.35 * imp[2] * np.exp(-((np.sqrt(r2) - 1.2 * R) / (0.35 * R)) ** 2)
            ew.H = ew.H + np.fft.fft2(crater)
            disc = np.exp(-r2 / (1.3 * R) ** 2)
            foam = np.maximum(foam, np.clip(1.4 * disc * lace, 0, 1.5)); damp = np.maximum(damp, 0.8 * disc)
            h = ew.height()
        # top slab footprint (shifted body columns piercing or within 0.5 m of the surface)
        top = _ifft(Fth[0] * ramp)
        pierce_now = _ifft(Fpierce * ramp) > 0.5
        # contact-line foam: a ring just outside the piercing outline, stronger where the water meets it head-on
        if pierce_now.any():
            ring = ndimage.binary_dilation(pierce_now, iterations=3) & ~pierce_now
            soft = ndimage.gaussian_filter(pierce_now.astype(float), 2.0)
            gy, gx = np.gradient(-soft)                     # outward normal of the waterline
            nrm = np.hypot(gx, gy) + 1e-9
            head_on = np.clip((gx * hd[0] + gy * hd[1]) / nrm, 0, 1)
            rel = speed * (0.35 + 0.65 * head_on) + abs(rise) * (1.0 if tau > -rise_window else 0.0) + 0.15
            inj = ring * np.clip(p.edge_foam * rel, 0, 1.2) * lace
            foam = np.maximum(foam, inj)
            wash = np.where(pierce_now, np.maximum(wash * 0.98, np.clip((0.4 - np.clip(_ifft(Fhi * ramp), 0, None)) / 0.4, 0, 1)
                                                    * np.clip(0.4 * speed + abs(rise), 0, 1)), wash * 0.98)
        # breaking: downward acceleration of the dynamic surface
        if h_prev2 is not None:
            acc = (h - 2 * h_prev + h_prev2) / p.dt ** 2
            brk = np.clip((-acc / GRAV - p.break_acc_g) * 4.0, 0, 1) * lace * (ew.sponge > 0.999)   # not in the sponge rim
            foam = np.maximum(foam, brk)
        # slicks: the body's top slab passing (turbulent wake) and leaving the top metre (upwelling, flukeprint)
        damp = damp * dd + np.clip(top, 0, 0.5) * speed * p.wake_damp * p.dt
        top2 = top + _ifft(Fth[1] * ramp)
        if P_prev_top is not None:
            leaving = np.clip((P_prev_top - top2) / p.dt, 0, None)
            damp = damp + np.clip(leaving * p.upwell_damp * p.dt, 0, 1)
        P_prev_top = top2
        foam = foam * fd
        if s % 25 == 24:
            foam = ndimage.gaussian_filter(foam, 0.5); damp = ndimage.gaussian_filter(damp, 0.7)
        h_prev2, h_prev = h_prev, h
    h_out = ew.height() + P_real
    tex = np.stack([h_out, np.clip(foam, 0, 1.5), np.clip(damp, 0, 1), np.clip(wash, 0, 1)], -1).astype(np.float32)
    pierce = (zmax0 > 0) & (zmin0 < 0) & body
    return {"tex": tex, "origin_m": origin, "size_m": p.size_m, "res": n, "category": category, "motion": mot,
            "heading": hd.tolist(), "params": asdict(p), "seconds": round(time.time() - t0, 2), "pierce_mask": pierce,
            "stats": {"h_min": float(h_out[~pierce].min()), "h_max": float(h_out[~pierce].max()), "foam_area_m2": float((foam > 0.3).sum() * dx * dx),
                      "damp_area_m2": float((damp > 0.3).sum() * dx * dx)}}


def _eased_path(back_s: float, window_s: float, ramp_s: float) -> float:
    """Distance (in units of the speed) travelled between back_s seconds before the shot and the shot, for a motion
    that starts window_s before the shot and eases in over ramp_s (constant speed afterwards)."""
    def pos(t):           # position at time t after the start, unit speed with a linear speed ramp
        t = min(max(t, 0.0), window_s)
        return t * t / (2 * ramp_s) if t < ramp_s else ramp_s / 2 + (t - ramp_s)
    return pos(window_s) - pos(window_s - back_s)


def _fft(a):
    from scipy import fft as sfft
    return sfft.fft2(a, workers=4)


def _ifft(A):
    from scipy import fft as sfft
    return np.real(sfft.ifft2(A, workers=4))


def _shift(a, shift, kx, ky):
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-1j * (kx * shift[0] + ky * shift[1]))))


def _impact_xy(V, where, centre_xy):
    if where == "fin":
        hi = V[V[:, 2] > np.percentile(V[:, 2], 97)]
        return hi[:, :2].mean(0)
    return np.asarray(centre_xy, float)


def write_exr(layer: dict, path: Path) -> Path:
    from factory import exr
    t = layer["tex"]
    p = exr.write(path, {"R": t[..., 0], "G": t[..., 1], "B": t[..., 2], "A": t[..., 3]}, half=True)
    meta = {k: v for k, v in layer.items() if k not in ("tex", "pierce_mask")}
    meta["texel_convention"] = "texel (i, j) centre at origin + ((i + 0.5), (j + 0.5)) * size / res; UV = (world_xy_m - origin) / size"
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return p


def read_exr(path: Path) -> dict:
    from factory import exr
    c = exr.read(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    meta["tex"] = np.stack([c["R"], c["G"], c["B"], c["A"]], -1)
    return meta


def height_at(layer: dict, x_m, y_m) -> np.ndarray:
    """Contact height (m) at world points (bilinear, zero outside) -- the label pipeline adds it to BakedSea.height
    evaluated at the same rest position (see water_engine/WIRING.md)."""
    n = layer["res"]; size = layer["size_m"]; o = layer["origin_m"]
    cu = (np.asarray(x_m, float) - o[0]) / size; cv = (np.asarray(y_m, float) - o[1]) / size
    inside = (cu > 0) & (cu < 1) & (cv > 0) & (cv < 1)          # as the material: UV in (0, 1), clamp addressing
    u = np.clip(cu * n - 0.5, 0, n - 1.001); v = np.clip(cv * n - 0.5, 0, n - 1.001)
    i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int); fu = u - i0; fv = v - j0
    h = layer["tex"][..., 0]
    out = h[j0, i0] * (1 - fu) * (1 - fv) + h[j0, i0 + 1] * fu * (1 - fv) + h[j0 + 1, i0] * (1 - fu) * fv + h[j0 + 1, i0 + 1] * fu * fv
    return np.where(inside, out, 0.0)
