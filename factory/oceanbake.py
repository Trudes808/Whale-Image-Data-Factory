"""The run06 sea as textures: bake an FFT sea state (factory.fftocean) to world-periodic EXR textures for the
Unreal water material, and evaluate exactly that baked surface for the labels.

    spec = draw_state("tw_breeze", seed=4, rng=np.random.default_rng(4))      # or SeaSpec(...)
    layout = bake(spec, Path("/var/tmp/.../bakes/tw_breeze_s4"))              # writes the texture set + layout.json
    sea = BakedSea.load(Path("/var/tmp/.../bakes/tw_breeze_s4"))
    z_m = sea.height(x_m, y_m)                 # rendered surface height above the mean plane at world (x, y) metres
    z_cm = sea.height_cm(x_cm, y_cm)           # same in Unreal units

    python3 -m factory.oceanbake bake tw_breeze --seed 4 --out /var/tmp/sat3737-parts-cache/water_engine/bakes
    python3 -m factory.oceanbake calibrate      # cap_scale (Cox-Munk) and foam threshold (whitecap W(U)) per state

TEXTURE SET (all world-aligned, wrap addressing, linear (not sRGB), mips by 2x2 box average):
  T_Disp_c{0..5}.exr  512 x 512 RGBA half: R = Dx, G = Dy (horizontal displacement, m), B = Dz (height, m),
                      A = E[sx sy] (slope cross moment). Cascade c tiles every L_c metres (BAKE_CASCADES):
                      1056.5, 211.3, 42.26, 8.452, 1.6904, 0.33808 m.
  T_Mom_c{0..5}.exr   512 x 512 RGBA half: R = sx, G = sy (slopes dz/dx, dz/dy), B = E[sx^2], A = E[sy^2].
                      At mip 0 B = R^2, A = G^2; a box-filtered mip holds the footprint means, so
                      var = B - R^2 etc. is the unresolved slope variance of that cascade at that footprint (LEAN).
  T_Foam.exr          2048 x 2048 RGBA half over the 211.3 m tile of cascade 1: R = fresh whitecap foam (0..1.5),
                      G = aged foam (streaky, 0..~2), B = breaking now (0..1), A = 1.
  T_Detail.png        1024 x 1024 RGBA8, shared by all bakes, repeat 1.5 m: R = bubble density, G = bubble rims,
                      B = smooth isotropic noise (stretched downwind for foam streaks), A = smooth noise in [0, 1]
                      (texture bombing of the fine cascades and wind gusts).
  layout.json         everything the material and the labels need: tile sizes, which cascades displace the mesh
                      (geo_cascades = 3), units, value ranges, the calibrated capillary residual, foam constants.

Texel convention: texel (i = column, j = row) of cascade c sits at world (x, y) = (i L_c / N, j L_c / N) metres,
i.e. in the material UV = world_xy_m / L_c + 0.5 / N. Texture rows run along +y.
Only cascades 0-2 (waves longer than 1.4 m) displace the mesh (WPO); 3-5 are normal / moment maps only. The height
evaluator uses exactly cascades 0-2, read back from the EXR files (half precision), bilinear, mip 0.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from factory import exr, fftocean

CACHE = Path("/var/tmp/sat3737-parts-cache/water_engine")
N_TEX = 512
GEO_CASCADES = 3
FOAM_RES = 2048
FOAM_TILE_CASCADE = 1                   # the foam map tiles with cascade 1 (211.3 m); cascades 1-3 are periodic on it
DETAIL_REPEAT_M = 1.5
CALIBRATION = Path(__file__).with_name("oceanbake_calibration.json")

# Foam model constants (shared with the material: see layout["foam"]).
FOAM = dict(gain=2.0, tau_fresh_s=2.5, tau_aged_s=14.0, age_transfer=0.8, drift_frac=0.03,
            streak_diff_par=0.06, streak_diff_perp=0.004, windrow_spacing_m=6.0, windrow_rate=0.05,
            aged_gain=0.8, history_s=30.0, dt=0.2)


def _foam_opacity(fresh, aged, lace):
    """Foam opacity as the material computes it (see water_engine/M_OceanFFT.hlsl, FoamOpacity)."""
    cf = np.clip(fresh, 0.0, 1.0)
    ca = np.clip(FOAM["aged_gain"] * aged, 0.0, 1.0)
    return cf * (0.85 + 0.15 * lace) + (1.0 - cf) * ca * lace


@dataclass
class SeaSpec:
    state: str = "tw_breeze"
    wind: float = 7.0
    wind_dir_deg: float = 0.0
    fetch_km: float = 120.0
    swell_hs: float = 1.0
    swell_len: float = 160.0
    swell_dir_deg: float = 40.0
    choppy: float = 1.15
    seed: int = 1
    t: float = 0.0
    cap_scale: float | None = None     # None: from the calibration file
    foam_threshold: float | None = None
    short_spread_mix: float = 0.0      # isotropic share of the short wind sea (round 2: 0.5, short-crested chop)
    short_spread_lmax_m: float = 1.5   # ... for waves shorter than this (round 2: 6 m)

    def ocean(self) -> fftocean.FFTOcean:
        cal = load_calibration().get(self.state, {})
        cap = self.cap_scale if self.cap_scale is not None else cal.get("cap_scale", 0.0)
        return fftocean.FFTOcean(wind=self.wind, wind_dir_deg=self.wind_dir_deg, fetch_km=self.fetch_km, swell_hs=self.swell_hs,
                                 swell_len=self.swell_len, swell_dir_deg=self.swell_dir_deg, choppy=self.choppy, seed=self.seed,
                                 n=N_TEX, cascades=list(fftocean.BAKE_CASCADES), fine_gain=(1.0,) * 6, cap_scale=cap,
                                 short_spread_mix=self.short_spread_mix, short_spread_lmax_m=self.short_spread_lmax_m)


def draw_state(state: str, seed: int, rng: np.random.Generator | None = None, t: float = 0.0) -> SeaSpec:
    """A sea state with the wind direction drawn uniformly and the swell from 20-90 degrees off the wind."""
    rng = rng or np.random.default_rng(seed)
    p = fftocean.TRADE_STATES[state]
    wd = float(rng.uniform(0, 360)); sd = wd + float(rng.choice([-1, 1]) * rng.uniform(20, 90))
    return SeaSpec(state=state, wind=p["wind"], wind_dir_deg=wd, fetch_km=p["fetch_km"], swell_hs=p["swell_hs"], swell_len=p["swell_len"],
                   swell_dir_deg=sd % 360.0, choppy=p["choppy"], seed=seed, t=t)


def load_calibration() -> dict:
    return json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}


# ------------------------------------------------------------------------------------------------ helpers
def mip_chain(a: np.ndarray, quantize: bool = True) -> list[np.ndarray]:
    """2x2 box-filtered mips down to 1 x 1 (Unreal's SimpleAverage), each level rounded to half like the engine."""
    out = [a.astype(np.float32)]
    while out[-1].shape[0] > 1:
        b = out[-1]
        b = 0.25 * (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2])
        out.append(b.astype(np.float16).astype(np.float32) if quantize else b)
    return out


def bilinear_periodic(arr: np.ndarray, x, y, L: float) -> np.ndarray:
    """Periodic bilinear sample of a (N, N[, C]) texture whose texel (i, j) sits at world (i L / N, j L / N)."""
    n = arr.shape[0]
    u = np.asarray(x, np.float64) * (n / L); v = np.asarray(y, np.float64) * (n / L)
    i0 = np.floor(u); j0 = np.floor(v); fu = u - i0; fv = v - j0
    i0 = i0.astype(np.int64) % n; j0 = j0.astype(np.int64) % n; i1 = (i0 + 1) % n; j1 = (j0 + 1) % n
    if arr.ndim == 3:
        fu = fu[..., None]; fv = fv[..., None]
    return (arr[j0, i0] * (1 - fu) * (1 - fv) + arr[j0, i1] * fu * (1 - fv) + arr[j1, i0] * (1 - fu) * fv + arr[j1, i1] * fu * fv)


def slope_variance(fields: list[dict], wind_dir_deg: float) -> dict:
    """Up- / cross-wind slope variance of the baked field, per cascade and total."""
    a = math.radians(wind_dir_deg); c, s = math.cos(a), math.sin(a)
    up, cr = [], []
    for f in fields:
        su = f["sx"] * c + f["sy"] * s; sc = -f["sx"] * s + f["sy"] * c
        up.append(float(np.var(su))); cr.append(float(np.var(sc)))
    return {"upwind": up, "crosswind": cr, "total": sum(up) + sum(cr), "upwind_sum": sum(up), "crosswind_sum": sum(cr)}


def calibrate_cap(spec: SeaSpec) -> dict:
    """cap_scale such that the field's total mean-square slope equals Cox-Munk (it enters linearly)."""
    s0 = SeaSpec(**{**asdict(spec), "cap_scale": 0.0}); s1 = SeaSpec(**{**asdict(spec), "cap_scale": 1.0})
    v0 = slope_variance(s0.ocean().fields(spec.t), spec.wind_dir_deg)["total"]
    v1 = slope_variance(s1.ocean().fields(spec.t), spec.wind_dir_deg)["total"]
    target = fftocean.cox_munk_mss(spec.wind)["total"]
    scale = max(0.0, (target - v0) / max(v1 - v0, 1e-9))
    return {"cap_scale": round(scale, 4), "mss_without": v0, "mss_cap1": v1, "mss_target": target}


# ------------------------------------------------------------------------------------------------ detail texture
def detail_texture(size: int = 1024, seed: int = 11) -> np.ndarray:
    """RGBA8 (size, size, 4): bubble density, rims, streak noise, blend noise (tileable)."""
    from factory import foamtex
    rng = np.random.default_rng(seed)
    density = np.zeros((size, size)); rim = np.zeros((size, size))
    for cells, weight in ((9, 0.45), (19, 0.35), (41, 0.2)):
        f1, f2 = foamtex.worley(rng, size, cells)
        density += weight * (1 - np.clip(f1 * 1.4, 0, 1) ** 1.5)
        rim += weight * np.exp(-((f2 - f1) / 0.12) ** 2)
    density = np.clip(density * (0.6 + 0.6 * foamtex.value_noise(rng, size, 29)), 0, 1)

    def fbm(octaves):
        v = sum(w * foamtex.value_noise(rng, size, c) for c, w in octaves)
        v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
        # equalise to a near-uniform distribution (rank transform), so weights built from it have a known mean
        r = np.empty(v.size); r[np.argsort(v.ravel())] = (np.arange(v.size) + 0.5) / v.size
        return r.reshape(v.shape)
    streak = fbm(((4, 0.5), (8, 0.3), (16, 0.2)))
    blend = fbm(((3, 0.6), (6, 0.3), (12, 0.1)))
    tex = np.stack([density, np.clip(rim, 0, 1), streak, blend], -1)
    return (tex * 255 + 0.5).astype(np.uint8)


def ensure_detail(directory: Path) -> Path:
    p = directory / "T_Detail.png"
    if not p.exists():
        from PIL import Image
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(detail_texture(), "RGBA").save(p)
    return p


# ------------------------------------------------------------------------------------------------ foam
class FoamSim:
    """Whitecaps on the 211.3 m tile of cascade 1 from the exact spectra of cascades 1-3 (all periodic on it):
    injection where the Jacobian folds, fresh foam decaying in ~2.5 s into aged foam that lives ~14 s, drifts
    downwind (3 % of the wind), diffuses mostly along the wind and gathers in wind rows (Langmuir convergence)."""

    def __init__(self, ocean: fftocean.FFTOcean, res: int = FOAM_RES, cascades=(1, 2, 3)):
        if not ocean._h0:
            ocean.init()
        self.oc = ocean; self.res = res
        L = ocean.cascades[FOAM_TILE_CASCADE]; self.L = L
        A = np.zeros((res, res), complex); B = np.zeros((res, res), complex)
        n = ocean.n
        idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
        for c in cascades:
            ratio = L / ocean.cascades[c]
            m = int(round(ratio))
            if abs(ratio - m) > 1e-6:
                raise ValueError(f"cascade {c} is not periodic on the foam tile ({ratio})")
            a, b = ocean._h0[c]
            nz = (np.abs(a) > 0) | (np.abs(b) > 0)
            jj, ii = np.nonzero(nz)
            ty = (idx[jj] * m) % res; tx = (idx[ii] * m) % res
            if (np.abs(idx[jj] * m) >= res // 2).any() or (np.abs(idx[ii] * m) >= res // 2).any():
                raise ValueError(f"cascade {c} does not fit a {res} grid on the foam tile")
            np.add.at(A, (ty, tx), a[jj, ii]); np.add.at(B, (ty, tx), b[jj, ii])
        k = 2.0 * math.pi / L * np.fft.fftfreq(res, d=1.0 / res)
        self.kx = np.broadcast_to(k[None, :], (res, res)); self.ky = np.broadcast_to(k[:, None], (res, res))
        kl = np.hypot(self.kx, self.ky)
        w0 = 2.0 * math.pi / fftocean.LOOP
        self.w = np.floor(fftocean.omega_of(np.maximum(kl, 1e-9)) / w0) * w0
        self.A, self.B = A, B
        lam = ocean.choppy; ik = 1.0 / np.maximum(kl, 1e-9)
        self.cxx = -lam * self.kx * self.kx * ik; self.cyy = -lam * self.ky * self.ky * ik; self.cxy = -lam * self.kx * self.ky * ik

    def jacobian(self, t: float) -> np.ndarray:
        from scipy import fft as sfft
        e = np.exp(-1j * self.w * t)
        h = self.A * e + self.B * np.conj(e)
        n2 = self.res * self.res
        stack = np.stack([h * self.cxx, h * self.cyy, h * self.cxy])
        jxx, jyy, jxy = np.real(sfft.ifft2(stack, axes=(1, 2), workers=8)) * n2
        return (1.0 + jxx) * (1.0 + jyy) - jxy * jxy

    def jacobian_history(self, t_end: float, seconds: float | None = None, dt: float | None = None) -> list[np.ndarray]:
        seconds = seconds or FOAM["history_s"]; dt = dt or FOAM["dt"]
        steps = int(round(seconds / dt))
        return [self.jacobian(t_end - (steps - 1 - s) * dt).astype(np.float32) for s in range(steps)]

    def run(self, t_end: float, threshold: float, seconds: float | None = None, dt: float | None = None, seed: int = 0,
            jac: list | None = None) -> dict:
        from scipy import fft as sfft
        P = FOAM; seconds = seconds or P["history_s"]; dt = dt or P["dt"]
        res, L = self.res, self.L
        texel = L / res
        wd = math.radians(self.oc.wind_dir_deg); ux, uy = math.cos(wd), math.sin(wd)
        drift = P["drift_frac"] * self.oc.wind
        vx, vy = drift * ux, drift * uy
        # spectral operator for aged foam: drift + anisotropic diffusion (along / across the wind)
        kpar = self.kx * ux + self.ky * uy; kperp = -self.kx * uy + self.ky * ux
        op_aged = np.exp(-1j * (self.kx * vx + self.ky * vy) * dt - (P["streak_diff_par"] * kpar ** 2 + P["streak_diff_perp"] * kperp ** 2) * dt)
        op_fresh = np.exp(-1j * (self.kx * vx + self.ky * vy) * dt)
        # wind rows: a periodic pattern varying fast across the wind, slowly along it
        rng = np.random.default_rng(seed + 977)
        kk = np.hypot(self.kx, self.ky)
        spec = np.exp(-0.5 * ((kperp - 2 * math.pi / P["windrow_spacing_m"]) / (0.35 * 2 * math.pi / P["windrow_spacing_m"])) ** 2) \
            * np.exp(-0.5 * (kpar / (2 * math.pi / 60.0)) ** 2) * (kk > 0)
        rows = np.real(sfft.ifft2(spec * (rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res))), workers=8))
        rows = (rows - rows.mean()) / max(rows.std(), 1e-9)
        conv = np.exp(P["windrow_rate"] * dt * rows)
        fresh = np.zeros((res, res)); aged = np.zeros((res, res)); inject = np.zeros((res, res))
        steps = int(round(seconds / dt))
        df = math.exp(-dt / P["tau_fresh_s"]); da = math.exp(-dt / P["tau_aged_s"])
        for s in range(steps):
            t = t_end - (steps - 1 - s) * dt
            J = jac[s] if jac is not None else self.jacobian(t)
            inject = np.clip((threshold - J) * P["gain"], 0.0, 1.0)
            fresh = np.clip(np.real(sfft.ifft2(sfft.fft2(fresh, workers=8) * op_fresh, workers=8)), 0, None)
            lost = fresh * (1.0 - df)
            fresh = np.maximum(fresh * df, inject)
            fresh = np.minimum(fresh + inject * dt, 1.5)
            tot = aged.sum()
            aged = aged * conv
            if aged.sum() > 0:
                aged *= tot / aged.sum()
            aged = aged * da + lost * P["age_transfer"]
            aged = np.clip(np.real(sfft.ifft2(sfft.fft2(aged, workers=8) * op_aged, workers=8)), 0, None)
        return {"fresh": fresh, "aged": aged, "inject": inject, "texel_m": texel}


def lace_mean() -> float:
    """Mean of the detail texture's bubble density (the foam opacity's expectation over the lace)."""
    p = ensure_detail(CACHE / "bakes" / "common")
    from PIL import Image
    return float(np.asarray(Image.open(p))[..., 0].mean() / 255.0)


def foam_coverage(fresh, aged, lace_m: float) -> float:
    return float(_foam_opacity(fresh, aged, lace_m).mean())


def calibrate_foam(spec: SeaSpec, res: int = FOAM_RES, seconds: float | None = None, iters: int = 9) -> dict:
    """Foam threshold on the Jacobian so that the mean foam opacity equals W(U) (bisection; coverage grows with it)."""
    oc = spec.ocean()
    sim = FoamSim(oc, res=res)
    target = fftocean.whitecap_fraction(spec.wind)
    lm = lace_mean()
    lo, hi = -0.6, 1.0
    hist = []
    jac = sim.jacobian_history(spec.t, seconds)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = sim.run(spec.t, mid, seconds=seconds, seed=spec.seed, jac=jac)
        cov = foam_coverage(r["fresh"], r["aged"], lm)
        hist.append((round(mid, 4), cov))
        if cov > target:
            hi = mid
        else:
            lo = mid
    thr = 0.5 * (lo + hi)
    return {"foam_threshold": round(thr, 4), "whitecap_target": target, "history": hist}


# ------------------------------------------------------------------------------------------------ bake
def bake(spec: SeaSpec, out: Path, foam: bool = True, foam_res: int = FOAM_RES, write_preview: bool = True) -> dict:
    """Write the texture set for one sea state / seed / time to `out` and return the layout dict."""
    t0 = time.time()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    try:
        from factory import diskguard
        if diskguard.free_gib(out) < 20:
            raise RuntimeError(f"less than 20 GiB free at {out}")
    except ImportError:
        pass
    cal = load_calibration().get(spec.state, {})
    oc = spec.ocean()
    fields = oc.fields(spec.t)
    files = {}; ranges = {}
    for c, (f, L) in enumerate(zip(fields, oc.cascades)):
        disp = {"R": f["dx"], "G": f["dy"], "B": f["dz"], "A": f["sx"] * f["sy"]}
        mom = {"R": f["sx"], "G": f["sy"], "B": f["sx"] ** 2, "A": f["sy"] ** 2}
        for name, ch in (("Disp", disp), ("Mom", mom)):
            p = exr.write(out / f"T_{name}_c{c}.exr", ch, half=True)
            files[p.name] = p.stat().st_size
            ranges[p.name] = {k: [round(float(v.min()), 6), round(float(v.max()), 6)] for k, v in ch.items()}
    sv = slope_variance(fields, spec.wind_dir_deg)
    thr = spec.foam_threshold if spec.foam_threshold is not None else cal.get("foam_threshold", 0.3)
    foam_info = None
    if foam:
        sim = FoamSim(oc, res=foam_res)
        r = sim.run(spec.t, thr, seed=spec.seed)
        ch = {"R": r["fresh"], "G": r["aged"], "B": r["inject"], "A": np.ones_like(r["fresh"])}
        p = exr.write(out / "T_Foam.exr", ch, half=True)
        files[p.name] = p.stat().st_size
        ranges[p.name] = {k: [round(float(v.min()), 6), round(float(v.max()), 6)] for k, v in ch.items()}
        foam_info = {"coverage": foam_coverage(r["fresh"], r["aged"], lace_mean()), "whitecap_target": fftocean.whitecap_fraction(spec.wind),
                     "threshold": thr, "tile_m": sim.L, "res": foam_res, "texel_m": r["texel_m"]}
    detail = ensure_detail(out.parent / "common")
    cm = fftocean.cox_munk_mss(spec.wind)
    layout = {
        "spec": asdict(spec), "cap_scale": oc.cap_scale, "units": "metres (Unreal: multiply by 100 for cm)",
        "n": N_TEX, "cascades_m": list(oc.cascades), "geo_cascades": GEO_CASCADES,
        "texel_convention": "texel (i=col, j=row) at world (i L/N, j L/N); material UV = world_xy_m / L + 0.5 / N; rows along +y",
        "disp": "R Dx, G Dy, B Dz (m), A E[sx sy]", "mom": "R sx, G sy, B E[sx^2], A E[sy^2]",
        "foam": {"file": "T_Foam.exr", "channels": "R fresh, G aged, B breaking now, A 1", **(foam_info or {}), "model": FOAM},
        "detail": {"file": str(Path("..") / "common" / detail.name), "repeat_m": DETAIL_REPEAT_M,
                   "channels": "R bubble density, G rims, B streak noise, A blend noise (uniform in [0,1])", "lace_mean": lace_mean()},
        "slopes": {"field_upwind": sv["upwind_sum"], "field_crosswind": sv["crosswind_sum"], "field_total": sv["total"],
                   "per_cascade_upwind": sv["upwind"], "per_cascade_crosswind": sv["crosswind"],
                   "cox_munk": cm, "residual_upwind": max(cm["upwind"] - sv["upwind_sum"], 0.0),
                   "residual_crosswind": max(cm["crosswind"] - sv["crosswind_sum"], 0.0)},
        "hs_m": 4.0 * math.sqrt(sum(float(np.var(f["dz"])) for f in fields)),
        "hs_geo_m": 4.0 * math.sqrt(sum(float(np.var(f["dz"])) for f in fields[:GEO_CASCADES])),
        "files": files, "ranges": ranges, "bytes_total": sum(files.values()),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "layout.json").write_text(json.dumps(layout, indent=1))
    return layout


# ------------------------------------------------------------------------------------------------ evaluator
class BakedSea:
    """The baked surface, read back from the EXR files exactly as the engine samples them (half precision)."""

    def __init__(self, layout: dict, disp: list[np.ndarray], mom: list[np.ndarray] | None, foam: np.ndarray | None, directory: Path | None = None):
        self.layout = layout; self.disp = disp; self.mom = mom; self.foam = foam; self.dir = directory
        self.L = list(layout["cascades_m"]); self.geo = int(layout["geo_cascades"])

    @classmethod
    def load(cls, directory: Path, moments: bool = True, foam: bool = True) -> "BakedSea":
        d = Path(directory)
        layout = json.loads((d / "layout.json").read_text())
        nc = len(layout["cascades_m"])

        def rgba(p):
            c = exr.read(p)
            return np.stack([c["R"], c["G"], c["B"], c["A"]], -1)
        disp = [rgba(d / f"T_Disp_c{c}.exr") for c in range(nc)]
        mom = [rgba(d / f"T_Mom_c{c}.exr") for c in range(nc)] if moments else None
        fm = rgba(d / "T_Foam.exr") if foam and (d / "T_Foam.exr").exists() else None
        return cls(layout, disp, mom, fm, d)

    def displacement(self, x, y) -> np.ndarray:
        """(..., 3) Dx, Dy, Dz (m) of the mesh vertex whose rest position is (x, y) metres: geometric cascades only."""
        out = 0.0
        for c in range(self.geo):
            out = out + bilinear_periodic(self.disp[c][..., :3], x, y, self.L[c])
        return out

    def rest_position(self, x, y, iterations: int = 8):
        """Rest position q with q + D(q) = (x, y), by fixed-point iteration (converges: the summed horizontal
        Jacobian of the geometric cascades stays well below 1)."""
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); qx, qy = x.copy(), y.copy()
        for _ in range(iterations):
            d = self.displacement(qx, qy)
            qx = x - d[..., 0]; qy = y - d[..., 1]
        return qx, qy

    def height(self, x, y, iterations: int = 8) -> np.ndarray:
        """Height (m) of the rendered surface above the mean plane at world (x, y) metres."""
        qx, qy = self.rest_position(x, y, iterations)
        return self.displacement(qx, qy)[..., 2]

    def height_cm(self, x_cm, y_cm, iterations: int = 8) -> np.ndarray:
        return 100.0 * self.height(np.asarray(x_cm, float) / 100.0, np.asarray(y_cm, float) / 100.0, iterations)

    def surface_points_cm(self, x_cm, y_cm) -> np.ndarray:
        """Displaced mesh vertices (N, 3) cm for rest positions (x, y) cm (what factory.gerstner.surface_points gives)."""
        d = self.displacement(np.asarray(x_cm, float) / 100.0, np.asarray(y_cm, float) / 100.0) * 100.0
        return np.stack([np.asarray(x_cm, float) + d[..., 0], np.asarray(y_cm, float) + d[..., 1], d[..., 2]], -1)


class ShotSurface:
    """The surface one shot renders: its batch's bake plus (optionally) its contact layer (factory.contact), both
    sampled at the mesh vertex's rest position exactly like the material's World Position Offset. This is the
    drop-in for factory.gerstner in the label pipeline (units: cm, like the factory's world):
        gerstner.surface_height(waves, x, y, t)  ->  surf.height_cm(x, y)
        gerstner.surface_points(waves, x, y, t)  ->  surf.surface_points_cm(x, y)"""

    def __init__(self, sea: BakedSea, contact_layer: dict | None = None):
        self.sea = sea; self.contact = contact_layer

    def _contact_z(self, qx_m, qy_m):
        if self.contact is None:
            return 0.0
        from factory import contact as _c
        return _c.height_at(self.contact, qx_m, qy_m)

    def height_cm(self, x_cm, y_cm, iterations: int = 8) -> np.ndarray:
        qx, qy = self.sea.rest_position(np.asarray(x_cm, float) / 100.0, np.asarray(y_cm, float) / 100.0, iterations)
        return 100.0 * (self.sea.displacement(qx, qy)[..., 2] + self._contact_z(qx, qy))

    def surface_points_cm(self, x_cm, y_cm) -> np.ndarray:
        P = self.sea.surface_points_cm(x_cm, y_cm)
        P[..., 2] += 100.0 * self._contact_z(np.asarray(x_cm, float) / 100.0, np.asarray(y_cm, float) / 100.0)
        return P


_SEA_CACHE: dict = {}


def shot_surface(shot: dict) -> ShotSurface:
    """ShotSurface for a shot dict carrying "sea_bake" (bake directory) and optionally "contact_layer" (EXR path)."""
    d = str(shot["sea_bake"])
    if d not in _SEA_CACHE:
        _SEA_CACHE.clear()
        _SEA_CACHE[d] = BakedSea.load(Path(d), moments=False, foam=False)
    layer = None
    if shot.get("contact_layer"):
        from factory import contact as _c
        layer = _c.read_exr(Path(shot["contact_layer"]))
    return ShotSurface(_SEA_CACHE[d], layer)


# ------------------------------------------------------------------------------------------------ CLI
def _cmd_calibrate(a) -> int:
    cal = load_calibration()
    for state in a.states or list(fftocean.TRADE_STATES):
        spec = draw_state(state, seed=1, rng=np.random.default_rng(1))
        spec.wind_dir_deg = 0.0; spec.swell_dir_deg = 40.0
        c = calibrate_cap(spec)
        spec.cap_scale = c["cap_scale"]
        f = calibrate_foam(spec)
        cal[state] = {**c, **f}
        print(state, json.dumps(cal[state]), flush=True)
        CALIBRATION.write_text(json.dumps(cal, indent=1))
    return 0


def _cmd_bake(a) -> int:
    spec = draw_state(a.state, a.seed, np.random.default_rng(a.seed), t=a.time)
    spec.short_spread_mix = a.short_spread; spec.short_spread_lmax_m = a.short_spread_lmax
    out = Path(a.out) / (f"{a.state}_s{a.seed}" + (f"_ss{int(round(a.short_spread * 100))}" if a.short_spread > 0 else "")
                         + (f"_l{a.short_spread_lmax:g}" if a.short_spread > 0 and a.short_spread_lmax != 1.5 else ""))
    lay = bake(spec, out)
    print(json.dumps({k: lay[k] for k in ("hs_m", "hs_geo_m", "bytes_total", "seconds")}), lay["foam"].get("coverage"), "->", out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("states", nargs="*")
    b = sub.add_parser("bake"); b.add_argument("state"); b.add_argument("--seed", type=int, default=1); b.add_argument("--time", type=float, default=0.0)
    b.add_argument("--out", default=str(CACHE / "bakes")); b.add_argument("--short-spread", type=float, default=0.0)
    b.add_argument("--short-spread-lmax", type=float, default=1.5)
    a = p.parse_args(argv)
    return {"calibrate": _cmd_calibrate, "bake": _cmd_bake}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
