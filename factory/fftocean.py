"""Tessendorf FFT ocean, a numpy port of web-ocean's simulation (js/wgsl/sim.js, MIT, Noniv/web-ocean).

Spectrum: fetch-limited JONSWAP wind sea with Donelan-Banner directional spreading plus a narrow long-crested
swell, deep-water dispersion with the capillary term, choppy horizontal displacement. Several square cascades
(patch sizes from ~1 km down to a few metres) tile the plane; each is an N x N FFT. Per cascade and time the
simulation returns the displacement (Dx, Dy, Dz), the slopes (sx, sy) and the horizontal Jacobian terms
(jxx, jyy, jxy); whitecap foam is injected where the total Jacobian folds and then decays and drifts.

    sim = FFTOcean.from_sea_state("moderate", seed=3)
    f = sim.fields(t=12.0)                      # dict of cascade arrays
    h = sim.height(x_m, y_m, t=12.0)            # surface height at world points (horizontal displacement inverted)
    foam = sim.foam_history(t_end=12.0, seconds=25.0)   # whitecap coverage map, after a stretch of simulated time

Axes: x, y horizontal (metres), z up; textures are indexed [row = y, col = x] with world x = col * L / N.
Meant for the run06 water: the cascades are baked to textures for the Unreal water material and evaluated here
for the labels (factory.postprocess, whitewater, shotlist placement) so that the render and the labels agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRAV = 9.81
SURF = 7.4e-5                 # surface tension / density (m^3/s^2)
LOOP = 600.0                  # s: frequencies quantised so the simulation loops

# web-ocean's cascades (m); we keep the four largest for the water mesh (the two finest only matter for sun sparkle
# at a few metres from the camera, which our photographs never are)
CASCADES_WEB = [1103.0, 211.3, 39.7, 7.43, 1.393, 0.2617]
CASCADES = [1103.0, 211.3, 39.7, 7.43, 1.393, 0.2617]     # all six: the two finest fold the Jacobian (whitecaps) and feed the normal map

# our sea states -> (U10 m/s, fetch km, swell Hs m, swell length m); Beaufort-ish, tuned so that the significant
# wave heights roughly match the Gerstner presets' intent (glassy ~0.1 m, calm ~0.3, light ~0.6, moderate ~1.2, rough ~2.5)
SEA_STATES = {
    "glassy":   dict(wind=1.5, fetch_km=20.0, swell_hs=0.25, swell_len=140.0, choppy=0.9),
    "calm":     dict(wind=3.5, fetch_km=40.0, swell_hs=0.5, swell_len=150.0, choppy=1.0),
    "light":    dict(wind=6.0, fetch_km=60.0, swell_hs=0.9, swell_len=160.0, choppy=1.1),
    "moderate": dict(wind=9.0, fetch_km=90.0, swell_hs=1.4, swell_len=180.0, choppy=1.2),
    "rough":    dict(wind=13.0, fetch_km=150.0, swell_hs=2.2, swell_len=200.0, choppy=1.25),
}


def omega_of(k: np.ndarray) -> np.ndarray:
    return np.sqrt(GRAV * k + SURF * k ** 3)


def d_omega_of(k: np.ndarray) -> np.ndarray:
    return (GRAV + 3.0 * SURF * k * k) / (2.0 * np.maximum(omega_of(k), 1e-9))


def jonswap(w, wp, alpha, gamma):
    sigma = np.where(w <= wp, 0.07, 0.09)
    d = (w - wp) / (sigma * wp)
    r = np.exp(-0.5 * d * d)
    q2 = (wp / np.maximum(w, 1e-9)) ** 2
    return alpha * GRAV * GRAV / np.maximum(w, 1e-9) ** 5 * np.exp(-1.25 * q2 * q2) * gamma ** r


def donelan_banner(theta, w, wp):
    ratio = w / wp
    beta = np.where(ratio < 0.95, 2.61 * ratio ** 1.3, np.where(ratio < 1.6, 2.28 * ratio ** -1.3, 10.0 ** (-0.4 + 0.8393 * np.exp(-0.567 * np.log(np.maximum(ratio * ratio, 1e-9))))))
    sh = 1.0 / np.cosh(np.clip(beta * theta, -20, 20))
    return beta / (2.0 * np.tanh(beta * np.pi)) * sh * sh


def cos_spread(theta, s):
    q = math.sqrt(s + 0.25) / (2.0 * math.sqrt(math.pi))
    return q * np.abs(np.cos(theta * 0.5)) ** (2.0 * s)


def wrap_angle(a):
    return a - 2.0 * np.pi * np.floor((a + np.pi) / (2.0 * np.pi))


@dataclass
class FFTOcean:
    wind: float = 7.0             # U10 m/s
    wind_dir_deg: float = 0.0     # waves travel towards this direction (x axis = 0)
    fetch_km: float = 90.0
    gamma: float = 3.3
    swell_hs: float = 1.8
    swell_len: float = 180.0
    swell_dir_deg: float = 25.0
    swell_spread: float = 30.0    # cos^2s exponent
    choppy: float = 1.2
    short_cut_m: float = 0.0012
    seed: int = 1
    n: int = 512
    cascades: list = field(default_factory=lambda: list(CASCADES))
    fine_gain: tuple = (1.0, 1.0, 1.0, 1.0, 0.85, 0.6)
    _h0: list = field(default_factory=list, repr=False)
    _k: list = field(default_factory=list, repr=False)

    @classmethod
    def from_sea_state(cls, name: str, seed: int = 1, wind_dir_deg: float = 0.0, swell_dir_deg: float | None = None, **kw) -> "FFTOcean":
        p = SEA_STATES[name]
        return cls(wind=p["wind"], fetch_km=p["fetch_km"], swell_hs=p["swell_hs"], swell_len=p["swell_len"], choppy=p["choppy"], seed=seed,
                   wind_dir_deg=wind_dir_deg, swell_dir_deg=wind_dir_deg + 25.0 if swell_dir_deg is None else swell_dir_deg, **kw)

    # ---- spectrum -------------------------------------------------------------------------------------------
    def spectrum(self, kx, ky):
        kl = np.hypot(kx, ky)
        w = omega_of(np.maximum(kl, 1e-9))
        theta = np.arctan2(ky, kx)
        S = np.zeros_like(kl)
        U, F = self.wind, self.fetch_km * 1000.0
        if U > 0.1:
            alpha = 0.076 * (U * U / (F * GRAV)) ** 0.22
            wp = 22.0 * (GRAV * GRAV / (U * F)) ** (1.0 / 3.0)
            S = jonswap(w, wp, alpha, self.gamma) * donelan_banner(wrap_angle(theta - math.radians(self.wind_dir_deg)), w, wp)
        if self.swell_hs > 1e-3:
            ks = 2.0 * math.pi / self.swell_len
            ws = math.sqrt(GRAV * ks)
            gs = 6.0
            hs4 = self.swell_hs * 0.25
            alpha_s = hs4 * hs4 * 5.0 * ws ** 4 / (GRAV * GRAV) * (1.0 - 0.287 * math.log(gs))
            S = S + jonswap(w, ws, alpha_s, gs) * cos_spread(wrap_angle(theta - math.radians(self.swell_dir_deg)), self.swell_spread)
        sc = self.short_cut_m
        S = S * np.exp(-kl * kl * sc * sc)
        return S * d_omega_of(kl) / np.maximum(kl, 1e-9)

    def _band(self, c: int) -> tuple[float, float]:
        L = self.cascades
        k_min = math.pi / L[0] if c == 0 else 2.0 * math.pi * 6.0 / L[c]
        k_max = 2.0 * math.pi * 6.0 / L[c + 1] if c < len(L) - 1 else (math.pi * self.n / L[c]) * 0.98
        if c > 0:
            k_min = self._band(c - 1)[1]
        return k_min, k_max

    def init(self) -> "FFTOcean":
        """h0(k) per cascade: Gaussian complex amplitudes shaped by the spectrum within the cascade's band."""
        rng = np.random.default_rng(self.seed)
        n = self.n
        idx = np.fft.fftfreq(n, d=1.0 / n)                  # -n/2 .. n/2-1 as integer wave numbers
        self._h0, self._k = [], []
        for c, L in enumerate(self.cascades):
            kx = (2.0 * math.pi / L) * idx[None, :]; ky = (2.0 * math.pi / L) * idx[:, None]
            kx = np.broadcast_to(kx, (n, n)).copy(); ky = np.broadcast_to(ky, (n, n)).copy()
            kl = np.hypot(kx, ky)
            k_min, k_max = self._band(c)
            dk = 2.0 * math.pi / L
            g = self.fine_gain[c] if c < len(self.fine_gain) else 1.0
            amp = np.sqrt(np.maximum(self.spectrum(kx, ky), 0.0)) * dk * 0.5 * g
            amp[(kl < 1e-5) | (kl < k_min) | (kl >= k_max)] = 0.0
            xi = rng.standard_normal((n, n, 2)); xi_m = rng.standard_normal((n, n, 2))
            a = (xi[..., 0] + 1j * xi[..., 1]) * amp
            amp_m = np.sqrt(np.maximum(self.spectrum(-kx, -ky), 0.0)) * dk * 0.5 * g
            amp_m[(kl < 1e-5) | (kl < k_min) | (kl >= k_max)] = 0.0
            b = (xi_m[..., 0] - 1j * xi_m[..., 1]) * amp_m           # conj(h0(-k)) with its own noise
            self._h0.append((a, b)); self._k.append((kx, ky, kl))
        return self

    # ---- evolution --------------------------------------------------------------------------------------------
    def fields(self, t: float) -> list[dict]:
        """Per cascade at time t: dx, dy (horizontal displacement, m), dz (height, m), sx, sy (slopes),
        jxx, jyy, jxy (horizontal Jacobian terms), all N x N real arrays, world x = col * L / N, y = row * L / N."""
        if not self._h0:
            self.init()
        out = []
        w0 = 2.0 * math.pi / LOOP
        for (a, b), (kx, ky, kl) in zip(self._h0, self._k):
            w = np.floor(omega_of(np.maximum(kl, 1e-9)) / w0) * w0
            e = np.exp(-1j * w * t)
            h = a * e + b * np.conj(e)                       # h(k, t)
            ik = 1.0 / np.maximum(kl, 1e-9)
            lam = self.choppy
            ih = 1j * h
            Dx = ih * (lam * kx * ik); Dy = ih * (lam * ky * ik)
            jxy = h * (-lam * kx * ky * ik)
            sx = ih * kx; sy = ih * ky
            jxx = h * (-lam * kx * kx * ik); jyy = h * (-lam * ky * ky * ik)
            n2 = self.n * self.n
            # inverse FFT with the forward-sign convention of the shader (sum over k of H e^{+i k x})
            def inv(F):
                return np.real(np.fft.ifft2(F)) * n2
            f = {k: inv(v) for k, v in dict(dz=h, dx=Dx, dy=Dy, sx=sx, sy=sy, jxx=jxx, jyy=jyy, jxy=jxy).items()}
            out.append(f)
        return out

    def sample(self, fields: list[dict], key: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear, periodic sample of one field summed over the cascades at world (x, y) metres."""
        x = np.asarray(x, float); y = np.asarray(y, float); total = np.zeros_like(x)
        n = self.n
        for f, L in zip(fields, self.cascades):
            u = (x / L) * n; v = (y / L) * n
            i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int); fu = u - i0; fv = v - j0
            i0 %= n; j0 %= n; i1 = (i0 + 1) % n; j1 = (j0 + 1) % n
            arr = f[key]
            total = total + (arr[j0, i0] * (1 - fu) * (1 - fv) + arr[j0, i1] * fu * (1 - fv) + arr[j1, i0] * (1 - fu) * fv + arr[j1, i1] * fu * fv)
        return total

    def height(self, x, y, t: float, fields: list[dict] | None = None, iterations: int = 6) -> np.ndarray:
        """Surface height above the mean plane at world (x, y): the horizontal displacement is inverted so the
        returned height is the one rendered above that point (same scheme as factory.gerstner.surface_height)."""
        f = fields if fields is not None else self.fields(t)
        x = np.asarray(x, float); y = np.asarray(y, float); qx, qy = x.copy(), y.copy()
        for _ in range(iterations):
            dx = self.sample(f, "dx", qx, qy); dy = self.sample(f, "dy", qx, qy)
            qx = x - dx; qy = y - dy
        return self.sample(f, "dz", qx, qy)

    def jacobian(self, fields: list[dict], x, y) -> np.ndarray:
        jxx = self.sample(fields, "jxx", x, y); jyy = self.sample(fields, "jyy", x, y); jxy = self.sample(fields, "jxy", x, y)
        return (1.0 + jxx) * (1.0 + jyy) - jxy * jxy

    def normal(self, fields: list[dict], x, y) -> np.ndarray:
        sx = self.sample(fields, "sx", x, y); sy = self.sample(fields, "sy", x, y)
        n = np.stack([-sx, -sy, np.ones_like(sx)], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # ---- whitecaps ---------------------------------------------------------------------------------------------
    def foam_history(self, t_end: float, seconds: float = 25.0, dt: float = 0.25, size_m: float = 256.0, res: int = 512,
                     origin=(0.0, 0.0), threshold: float = 0.68, gain: float = 2.0, growth: float = 1.0, decay: float = 0.22) -> np.ndarray:
        # defaults = web-ocean at its foam slider 1.0 (threshold 0.52 + 0.16, gain 2, decay 0.22 1/s)
        """Whitecap coverage on a size_m x size_m world patch at t_end after `seconds` of simulated history (the
        foam needs a past: it is injected where the Jacobian folds and decays over ~10 s, drifting downwind).
        Returns an array (res, res) in [0, 1.6]; world x = origin[0] + col * size_m / res."""
        xs = origin[0] + (np.arange(res) + 0.5) * size_m / res; ys = origin[1] + (np.arange(res) + 0.5) * size_m / res
        X, Y = np.meshgrid(xs, ys)
        foam = np.zeros((res, res)); drift = 0.03 * self.wind
        dvx, dvy = drift * math.cos(math.radians(self.wind_dir_deg)), drift * math.sin(math.radians(self.wind_dir_deg))
        steps = int(round(seconds / dt))
        for s in range(steps):
            t = t_end - (steps - 1 - s) * dt
            f = self.fields(t)
            J = self.jacobian(f, X, Y)
            inject = np.clip((threshold - J) * gain, 0.0, 1.0)
            # semi-Lagrangian drift
            if drift > 0:
                shift_x = dvx * dt / (size_m / res); shift_y = dvy * dt / (size_m / res)
                foam = np.roll(np.roll(foam, int(round(shift_y)), axis=0), int(round(shift_x)), axis=1)
            foam = foam * math.exp(-dt * decay)
            foam = np.maximum(foam, inject)
            foam = np.minimum(foam + inject * dt * growth, 1.6)
        return foam


def significant_wave_height(sim: FFTOcean, t: float = 0.0) -> float:
    """4 x the standard deviation of the surface height over the largest cascade (a quick check of a sea state)."""
    f = sim.fields(t)
    h = sum(np.std(c["dz"]) ** 2 for c in f) ** 0.5
    return 4.0 * h
