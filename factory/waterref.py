"""Reference water renderer without the engine: images a baked sea (factory.oceanbake) through a field camera, with
the shading the Unreal material is to reproduce (factory/water_engine/M_OceanFFT.hlsl), so the look can be tuned and
measured against real photographs cheaply. torch (GPU if free, else CPU) + cv2: run it with the Dinov3 venv
interpreter and the factory on PYTHONPATH.

    PYTHONPATH=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory .venv/bin/python -m factory.waterref --help

Geometry: the camera (height 2-4 m, focal 65-300 mm on a 22.3 mm wide sensor, 2560 x 1707) looks at the sea with
the earth's curvature; the geometric cascades 0-2 (and the contact layer's height) are the visible surface: per
bearing column the camera ray is intersected with the exact baked height h(r) (horizontal displacement inverted)
through the running minimum of (H - h(r) + r^2 / 2R) / r (exact occlusion of troughs by crests, no step artefacts).
Shading at the hit, from the moment maps of ALL cascades sampled over the pixel footprint with anisotropic
trilinear filtering (like a GPU sampler with max_aniso taps; LEAN): mean slope + slope covariance, plus a
calibrated capillary residual and the sun disc;
  - sun glitter: Gaussian slope density at the half vector, Smith masking, exact Fresnel, finite sun disc
    (so it clips like a photograph);
  - sky reflection: Gauss-Hermite 3 x 3 quadrature over the visible facets, HDRI pre-blurred to the lobe width;
  - water body: remote-sensing reflectance R_rs (RGB, sr^-1) x downwelling irradiance, transmitted (1 - F);
  - foam: baked whitecaps (fresh + aged, streaky lace) and the contact layer's foam, diffuse white;
  - aerial haze, then the camera: metering, tone curve, saturation, lens blur, noise / CA / vignette / JPEG
    (factory.camera_fx, the factory's own camera pipeline).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from factory import oceanbake, project

EARTH_R = 6.371e6
SUN_RADIUS = 0.004654            # rad
SKIES_DIR = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/third_party_assets/skies")
CLEAR_SKIES = ["kloofendal_43d_clear_puresky", "kloofendal_48d_partly_cloudy_puresky", "qwantani_mid_morning_puresky",
               "qwantani_noon_puresky", "sunflowers_puresky"]
SKY_CACHE = oceanbake.CACHE / "skies"
LUMA = (0.2126, 0.7152, 0.0722)


# ------------------------------------------------------------------------------------------------ parameters
@dataclass
class Look:
    """Shading and camera parameters (the material's scalar parameters carry the same names)."""
    rrs: tuple = (0.0014, 0.0030, 0.0140)     # water-leaving reflectance R_rs (sr^-1) in linear sRGB primaries (fitted)
    foam_albedo: float = 0.8
    haze_km: float = 25.0                      # meteorological visibility
    res_scale: float = 1.0                     # x the calibrated capillary residual (layout slopes.residual_*)
    gust_amp: float = 0.35                     # wind-gust patches on the fine cascades (0 = off)
    gust_scale_m: float = 70.0
    bomb: bool = True                          # texture bombing of cascades 3-5 against visible tiling
    max_aniso: int = 16                        # GPU-like anisotropic taps
    exposure_ev_sd: float = 0.35               # metering scatter
    exposure_bias_ev: float = 0.4
    contrast: float = 0.25                     # S-curve after sRGB
    saturation: float = 1.15                   # camera "picture style"
    lens_blur_px: float = 1.3                  # lens + OLPF + air shimmer (native px sigma), when no range is given
    lens_blur_range: tuple = (0.45, 1.4)       # per-frame sigma drawn log-uniformly (sharp to soft frames)
    sharpen: float = 0.15                      # in-camera unsharp mask
    sky_fold_dim: float = 0.6                  # radiance factor for reflections that point below the horizon
    tail_w: float = 0.05                       # slope-density mixture: weight of the wide component
    tail_s: float = 1.8                        # ... and its width factor (kurtosis ~3.6, Cox-Munk peakedness)
    glint_area_m2: float = 0.0                 # round 1: Poisson sun images of this patch area (made round "salt" discs; off)
    subsamples: int = 4                        # sub-footprints shaded separately along the range axis (1 = the material)
    steep_w: float = 0.0                       # weight of a steep-facet glint population (round 1: 0.12; off since the facets are resolved)
    steep_n: float = 2.0                       # its Blinn-Phong exponent (2: normals out to ~60 degrees)
    mod_strength: float = 0.8                  # hydrodynamic modulation of the short waves by the long-wave slope
    sky_sat: float = 1.4                       # saturation of the reflected sky (the HDRIs' skies are paler than the tropical sky)
    camera_fx: dict = field(default_factory=lambda: {"noise_sigma": [0.003, 0.010], "chroma_px": [0.0, 0.8], "vignette": [0.0, 0.15],
                                                     "jpeg_quality": [88, 96]})


@dataclass
class Shot:
    cam_height_m: float = 3.0
    focal_mm: float = 200.0
    sensor_w_mm: float = 22.3
    width: int = 2560
    height: int = 1707
    yaw_deg: float = 0.0
    pitch_deg: float | None = None             # None: aim at `distance_m` (frame centre on the sea at that range)
    distance_m: float = 150.0
    roll_deg: float = 0.0
    cam_xy_m: tuple = (0.0, 0.0)
    sky: str = "qwantani_noon_puresky"
    sun_azimuth_deg: float = 0.0               # world azimuth of the sun (the HDRI is turned to put it there)
    sun_elevation_deg: float | None = None     # None: the HDRI's own; else the analytic sun is placed there (sky unchanged)
    seed: int = 0


# ------------------------------------------------------------------------------------------------ torch sampling
class Pyramid:
    """Mip chain of a (H, W, C) periodic texture as one flat buffer, for per-pixel-lod gathers."""

    def __init__(self, arr: np.ndarray, device, quantize: bool = True):
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        C = arr.shape[2]
        levels = []
        for c in range(C):
            levels.append(oceanbake.mip_chain(arr[..., c], quantize=quantize))
        self.n = arr.shape[0]; self.C = C; self.nlev = len(levels[0])
        flat, offs, sizes = [], [], []
        off = 0
        for l in range(self.nlev):
            lv = np.stack([levels[c][l] for c in range(C)], 0).reshape(C, -1)
            flat.append(lv); offs.append(off); sizes.append(levels[0][l].shape[0]); off += lv.shape[1]
        self.buf = torch.from_numpy(np.concatenate(flat, 1)).to(device)
        self.offs = torch.tensor(offs, device=device, dtype=torch.int64)
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.int64)

    def bilinear(self, u, v, lev):
        """u, v: level-0 texel-centre coordinates (texel i at u = i); lev: int64 level per point. -> (C, P)."""
        s = self.sizes[lev]; o = self.offs[lev]
        scale = torch.exp2(-lev.to(u.dtype))
        ul = (u + 0.5) * scale - 0.5; vl = (v + 0.5) * scale - 0.5
        i0 = torch.floor(ul); j0 = torch.floor(vl); fu = ul - i0; fv = vl - j0
        i0 = i0.long(); j0 = j0.long()
        i0m = torch.remainder(i0, s); j0m = torch.remainder(j0, s); i1m = torch.remainder(i0 + 1, s); j1m = torch.remainder(j0 + 1, s)
        b = self.buf
        g = lambda jj, ii: b[:, o + jj * s + ii]
        return (g(j0m, i0m) * ((1 - fu) * (1 - fv)) + g(j0m, i1m) * (fu * (1 - fv)) + g(j1m, i0m) * ((1 - fu) * fv) + g(j1m, i1m) * (fu * fv))

    def sample(self, u, v, maj_u, maj_v, minor, major, max_aniso: int):
        """Anisotropic trilinear sample: footprint major axis (maj_u, maj_v) unit vector in texel space with length
        `major` texels, minor axis length `minor` texels. Taps along the major axis like a GPU sampler."""
        ratio = major / torch.clamp(minor, min=1e-6)
        ntap = torch.clamp(torch.ceil(ratio), 1, max_aniso)
        lodw = torch.maximum(minor, major / ntap)
        lod = torch.clamp(torch.log2(torch.clamp(lodw, min=1e-6)), 0, self.nlev - 1)
        l0 = torch.floor(lod).long(); l1 = torch.clamp(l0 + 1, max=self.nlev - 1); fl = lod - l0
        acc = 0.0
        nmax = int(ntap.max().item())
        for t in range(nmax):
            active = (t < ntap).to(u.dtype)
            off = ((t + 0.5) / ntap - 0.5) * major * active
            uu = u + maj_u * off; vv = v + maj_v * off
            s = self.bilinear(uu, vv, l0) * (1 - fl) + self.bilinear(uu, vv, l1) * fl
            acc = acc + s * active
        return acc / ntap


class Sky:
    """An HDRI with its sun taken out (replaced by an analytic disc of the measured or restored irradiance), as
    lat-long levels pre-blurred for glossy reflections. u = 0.5 at azimuth 0 (+x), v = 0 at the zenith."""
    BLUR_DEG = [0.0, 1.5, 4.0, 10.0, 22.0]
    SIZES = [2048, 1024, 512, 256, 128]

    def __init__(self, name: str, device):
        self.name = name
        d = self._prepare(name)
        self.sun_el = float(d["sun_el"]); self.sun_az_hdri = float(d["sun_az"])
        self.E_sun = torch.tensor(d["E_sun"], device=device, dtype=torch.float32)       # normal irradiance, RGB
        self.E_sky = torch.tensor(d["E_sky"], device=device, dtype=torch.float32)       # horizontal irradiance from the sky
        self.levels = [torch.from_numpy(d[f"lev{i}"]).to(device) for i in range(len(self.BLUR_DEG))]
        self.restored = bool(d["restored"])

    @classmethod
    def _prepare(cls, name: str) -> dict:
        SKY_CACHE.mkdir(parents=True, exist_ok=True)
        cp = SKY_CACHE / f"{name}.npz"
        if cp.exists():
            z = np.load(cp)
            return {k: z[k] for k in z.files}
        import cv2
        cat = json.loads((SKIES_DIR / "catalogue.json").read_text())[name]
        im = cv2.cvtColor(cv2.imread(str(SKIES_DIR / f"{name}.hdr"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32)
        H, W = im.shape[:2]
        el = 90.0 - 180.0 * (np.arange(H) + 0.5) / H
        az = 2 * np.pi * ((np.arange(W) + 0.5) / W - 0.5)
        dOm = (2 * np.pi / W) * (np.pi / H) * np.cos(np.radians(el))[:, None]
        se, sa = math.radians(cat["sun_elevation_deg"]), math.radians(cat["sun_azimuth_deg"])
        E = np.radians(el)[:, None]; A = az[None, :]
        ang = np.degrees(np.arccos(np.clip(np.sin(E) * math.sin(se) + np.cos(E) * math.cos(se) * np.cos(A - sa), -1, 1)))
        disc = ang < 3.0; ring = (ang > 4.0) & (ang < 8.0)
        ringmed = np.median(im[ring], 0)
        E_sun = ((im - ringmed).clip(0) * disc[..., None] * dOm[..., None]).sum((0, 1))
        im[disc] = ringmed
        up = (el > 0)[:, None]
        E_sky = (im * (dOm * np.sin(np.radians(el))[:, None] * up)[..., None]).sum((0, 1))
        ratio = float((E_sun * math.sin(se)).mean() / E_sky.mean())
        restored = ratio < 3.0
        if restored:        # clipped sun in the capture: restore a clear-sky direct / diffuse ratio of 6
            E_sun = np.array([1.0, 0.97, 0.92]) * 6.0 * E_sky.mean() / math.sin(se)
        out = {"sun_el": cat["sun_elevation_deg"], "sun_az": cat["sun_azimuth_deg"], "E_sun": E_sun.astype(np.float32),
               "E_sky": E_sky.astype(np.float32), "restored": restored}
        for i, (b, s) in enumerate(zip(cls.BLUR_DEG, cls.SIZES)):
            lv = cv2.resize(im, (s, s // 2), interpolation=cv2.INTER_AREA)
            if b > 0:
                sig = b / 360.0 * s
                pad = int(3 * sig) + 2
                lv = np.concatenate([lv[:, -pad:], lv, lv[:, :pad]], 1)
                lv = cv2.GaussianBlur(lv, (0, 0), sig)[:, pad:-pad]
            out[f"lev{i}"] = np.ascontiguousarray(lv.transpose(2, 0, 1)).astype(np.float32)
        np.savez(cp, **out)
        return out

    def lookup(self, d, rot_deg: float, blur_deg=None):
        """Radiance (3, P) in world directions d (P, 3); the HDRI azimuth is world azimuth - rot_deg."""
        az = torch.atan2(d[:, 1], d[:, 0]) - math.radians(rot_deg)
        el = torch.asin(torch.clamp(d[:, 2], -1, 1))
        u = torch.remainder(az / (2 * math.pi) + 0.5, 1.0); v = torch.clamp(0.5 - el / math.pi, 0, 1)
        if blur_deg is None:
            return self._bilin(self.levels[0], u, v)
        if not hasattr(self, "_bl"):
            self._bl = torch.tensor(self.BLUR_DEG, device=d.device)
        bl = self._bl
        x = torch.clamp(blur_deg, 0, bl[-1])
        hi = torch.clamp(torch.searchsorted(bl, x.contiguous()), 1, len(self.BLUR_DEG) - 1)
        lo = hi - 1
        f = (x - bl[lo]) / (bl[hi] - bl[lo])
        out = 0.0
        for i in range(len(self.BLUR_DEG)):
            w = torch.where(lo == i, 1 - f, torch.zeros_like(f)) + torch.where(hi == i, f, torch.zeros_like(f))
            if bool((w > 0).any()):
                out = out + self._bilin(self.levels[i], u, v) * w
        return out

    @staticmethod
    def _bilin(lv, u, v):
        C, H, W = lv.shape
        x = u * W - 0.5; y = torch.clamp(v * H - 0.5, 0, H - 1)
        x0 = torch.floor(x); y0 = torch.floor(y); fx = x - x0; fy = y - y0
        x0 = x0.long(); y0 = y0.long()
        x0m = torch.remainder(x0, W); x1m = torch.remainder(x0 + 1, W); y1 = torch.clamp(y0 + 1, max=H - 1)
        f = lv.reshape(C, -1)
        g = lambda yy, xx: f[:, yy * W + xx]
        return g(y0, x0m) * ((1 - fx) * (1 - fy)) + g(y0, x1m) * (fx * (1 - fy)) + g(y1, x0m) * ((1 - fx) * fy) + g(y1, x1m) * (fx * fy)


def fresnel(c):
    """Unpolarised Fresnel reflectance air -> water (n = 1.333) at cos(incidence) c."""
    n = 1.333
    c = torch.clamp(c, 1e-4, 1.0)
    s2 = 1 - c * c
    ct = torch.sqrt(torch.clamp(1 - s2 / (n * n), min=0))
    rs = (c - n * ct) / (c + n * ct); rp = (n * c - ct) / (n * c + ct)
    return 0.5 * (rs * rs + rp * rp)


def smith_lambda(cot, sig2):
    """Smith Lambda for a Gaussian slope distribution (Walter's rational approximation)."""
    a = torch.clamp(cot, min=1e-3) / torch.sqrt(2 * torch.clamp(sig2, min=1e-8))
    lam = (1 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a)
    return torch.where(a >= 1.6, torch.zeros_like(a), lam)


# ------------------------------------------------------------------------------------------------ the renderer
class Renderer:
    def __init__(self, sea: oceanbake.BakedSea, device=None, look: Look | None = None, geo: int | None = None):
        """geo: number of cascades treated as visible geometry (default: the bake's geo_cascades = what the engine
        displaces); 4 adds the 0.28-1.4 m waves of cascade 3 as geometry (needs parallax / a finer mesh in the engine)."""
        self.dev = device or torch.device("cpu")
        self.sea = sea; self.look = look or Look()
        lay = sea.layout
        self.L = [float(x) for x in lay["cascades_m"]]; self.N = int(lay["n"]); self.geo = int(geo or lay["geo_cascades"])
        sp = lay["spec"]; self.wind_dir = math.radians(sp["wind_dir_deg"]); self.wind = sp["wind"]
        # moment pyramids per cascade: (sx, sy, sxx, syy, sxy)
        self.mom = [Pyramid(np.concatenate([sea.mom[c], sea.disp[c][..., 3:4]], -1), self.dev) for c in range(len(self.L))]
        self.disp0 = [torch.from_numpy(np.ascontiguousarray(sea.disp[c][..., :3].transpose(2, 0, 1).reshape(3, -1))).to(self.dev) for c in range(self.geo)]
        self.hmax = float(sum(np.abs(sea.disp[c][..., 2]).max() for c in range(self.geo)))
        self.foam = Pyramid(sea.foam[..., :2], self.dev) if sea.foam is not None else None
        self.foam_L = float(lay["foam"].get("tile_m", self.L[1]))
        from PIL import Image
        det = np.asarray(Image.open(sea.dir / lay["detail"]["file"]).convert("RGBA")).astype(np.float32) / 255.0
        self.detail = Pyramid(det, self.dev, quantize=False)
        self.detail_L = float(lay["detail"]["repeat_m"])
        sl = lay["slopes"]
        c, s = math.cos(self.wind_dir), math.sin(self.wind_dir)
        ru, rc = sl["residual_upwind"], sl["residual_crosswind"]
        # residual covariance in world x, y
        self.res_cov = (ru * c * c + rc * s * s, ru * s * s + rc * c * c, (ru - rc) * c * s)
        self.sigma_long_up = math.sqrt(max(sum(sl["per_cascade_upwind"][1:4]), 1e-8))
        # texture bombing transforms of cascades 3-5: mirror about the wind axis, 180 deg turn, mirror about crosswind
        m = np.array([[math.cos(2 * self.wind_dir), math.sin(2 * self.wind_dir)], [math.sin(2 * self.wind_dir), -math.cos(2 * self.wind_dir)]])
        self.bomb_T = {3: m, 4: -np.eye(2), 5: -m}
        self.bomb_off = {3: (0.371, 0.613), 4: (0.519, 0.173), 5: (0.283, 0.797)}
        self._bomb_t = {c: torch.tensor(T, device=self.dev, dtype=torch.float32) for c, T in self.bomb_T.items()}
        self._luma = torch.tensor(LUMA, device=self.dev)
        self.contact = None

    # -------------------------------------------------------------------------------- surface geometry
    def _disp(self, x, y):
        out = torch.zeros((3,) + x.shape, device=self.dev, dtype=x.dtype)
        for c in range(self.geo):
            L = self.L[c]; n = self.N
            u = x * (n / L); v = y * (n / L)
            i0 = torch.floor(u); j0 = torch.floor(v); fu = u - i0; fv = v - j0
            i0 = torch.remainder(i0.long(), n); j0 = torch.remainder(j0.long(), n); i1 = torch.remainder(i0 + 1, n); j1 = torch.remainder(j0 + 1, n)
            b = self.disp0[c]
            out += (b[:, j0 * n + i0] * ((1 - fu) * (1 - fv)) + b[:, j0 * n + i1] * (fu * (1 - fv)) + b[:, j1 * n + i0] * ((1 - fu) * fv)
                    + b[:, j1 * n + i1] * (fu * fv)).to(x.dtype)
        return out

    def _contact_sample(self, x, y, ch):
        """Bilinear sample of the contact texture channel(s) at world x, y (zero outside)."""
        C = self.contact
        if C is None:
            return None
        n = C["res"]; size = C["size_m"]
        cu = (x - C["origin"][0]) / size; cv = (y - C["origin"][1]) / size
        inside = (cu > 0) & (cu < 1) & (cv > 0) & (cv < 1)
        u = torch.clamp(cu * n - 0.5, 0, n - 1.001); v = torch.clamp(cv * n - 0.5, 0, n - 1.001)
        i0 = torch.floor(u).long(); j0 = torch.floor(v).long(); fu = u - i0; fv = v - j0
        t = C["tex"][ch]
        g = lambda jj, ii: t[:, jj * n + ii]
        out = g(j0, i0) * ((1 - fu) * (1 - fv)) + g(j0, i0 + 1) * (fu * (1 - fv)) + g(j0 + 1, i0) * ((1 - fu) * fv) + g(j0 + 1, i0 + 1) * (fu * fv)
        return out * inside

    def height_at(self, x, y, iters: int = 5):
        qx, qy = x.clone(), y.clone()
        for _ in range(iters):
            d = self._disp(qx, qy)
            qx = x - d[0]; qy = y - d[1]
        d = self._disp(qx, qy)
        h = d[2]
        if self.contact is not None:
            h = h + self._contact_sample(qx, qy, [0])[0]
        return h, qx, qy

    def set_contact(self, tex: np.ndarray | None, origin_xy, size_m: float):
        """Contact layer (res, res, 4: height m, foam, damping, wash) on a square of size_m whose corner is origin."""
        if tex is None:
            self.contact = None; return
        n = tex.shape[0]
        t = torch.from_numpy(np.ascontiguousarray(tex.transpose(2, 0, 1).reshape(4, -1).astype(np.float32))).to(self.dev)
        # slopes of the contact height by central differences
        h = tex[..., 0].astype(np.float32); dx = size_m / n
        sx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * dx); sy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * dx)
        s = torch.from_numpy(np.stack([sx, sy], 0).reshape(2, -1)).to(self.dev)
        self.contact = {"tex": torch.cat([t, s], 0), "res": n, "size_m": size_m, "origin": (float(origin_xy[0]), float(origin_xy[1]))}

    # -------------------------------------------------------------------------------- camera
    def camera(self, shot: Shot) -> project.Camera:
        pitch = shot.pitch_deg
        if pitch is None:
            pitch = -math.degrees(math.atan2(shot.cam_height_m, shot.distance_m) + shot.distance_m / (2 * EARTH_R))
        loc = [shot.cam_xy_m[0] * 100.0, shot.cam_xy_m[1] * 100.0, shot.cam_height_m * 100.0]
        return project.Camera(loc, {"roll": shot.roll_deg, "pitch": pitch, "yaw": shot.yaw_deg}, shot.focal_mm, shot.width, shot.height,
                              sensor_width_mm=shot.sensor_w_mm, sensor_height_mm=shot.sensor_w_mm * shot.height / shot.width)

    def rays(self, cam: project.Camera):
        W, H = cam.width, cam.height
        basis = torch.tensor(cam.basis, device=self.dev, dtype=torch.float64)
        u = torch.arange(W, device=self.dev, dtype=torch.float64) + 0.5
        v = torch.arange(H, device=self.dev, dtype=torch.float64) + 0.5
        cy = -(v[:, None] - cam.cy) / cam.fy
        cx = (u[None, :] - cam.cx) / cam.fx
        d = torch.stack([torch.ones(H, W, device=self.dev, dtype=torch.float64), cx.expand(H, W), cy.expand(H, W)], -1)
        d = d @ basis
        return d / torch.linalg.norm(d, dim=-1, keepdim=True)

    # -------------------------------------------------------------------------------- render
    @torch.no_grad()
    def render(self, shot: Shot, sky: Sky, whale_mesh_m: np.ndarray | None = None, whale_faces: np.ndarray | None = None,
               rng_seed: int | None = None) -> dict:
        look = self.look; dev = self.dev
        cam = self.camera(shot)
        H0 = shot.cam_height_m; cxy = shot.cam_xy_m
        d = self.rays(cam)                                       # (H, W, 3) float64
        Hpx, Wpx = d.shape[:2]
        hor = torch.hypot(d[..., 0], d[..., 1])
        theta = torch.atan2(d[..., 1], d[..., 0])
        tand = -d[..., 2] / hor                                  # tan of the depression (positive down)
        pix = 1.0 / cam.fx                                       # pixel angle (rad)
        # ---- polar grid of bearings and ranges
        th0, th1 = float(theta.min()) - 4 * pix, float(theta.max()) + 4 * pix
        nb = int(math.ceil((th1 - th0) / (0.5 * pix))) + 1
        bear = th0 + torch.arange(nb, device=dev, dtype=torch.float64) * 0.5 * pix
        tmax = float(tand.max()); tmin = float(tand.min())
        r = []
        rr = max(0.5, (H0 - self.hmax) / max(tmax, 1e-4)) * 0.8
        r_far = 2.5 * math.sqrt(2 * EARTH_R * (H0 + self.hmax)) if tmin < (H0 + self.hmax) / 4000.0 else min(3.0 * (H0 + self.hmax) / tmin, 20000.0)
        while rr < r_far:
            r.append(rr)
            rr += min(max(rr * rr * pix / (H0 * 2.0), 0.004), 40.0)
        r = torch.tensor(r, device=dev, dtype=torch.float64)
        J = len(r)
        a = torch.empty((nb, J), device=dev, dtype=torch.float32)          # float32 is ample: a pixel row is 3e-5 in tan
        chunk = max(1, int(6e6 // J))
        for b0 in range(0, nb, chunk):
            bb = bear[b0:b0 + chunk]
            X = cxy[0] + r[None, :] * torch.cos(bb)[:, None]; Y = cxy[1] + r[None, :] * torch.sin(bb)[:, None]
            h, _, _ = self.height_at(X.float(), Y.float())
            a[b0:b0 + chunk] = ((H0 - h.double() + r[None, :] ** 2 / (2 * EARTH_R)) / r[None, :]).float()
        m = torch.cummin(a, dim=1).values
        BIG = 64.0
        keys = torch.clamp(-m, -30, 30).double(); del m
        keys += torch.arange(nb, device=dev, dtype=torch.float64)[:, None] * BIG
        keys = keys.reshape(-1)
        bidx = torch.clamp(torch.round((theta - th0) / (0.5 * pix)).long(), 0, nb - 1)
        q = bidx.double() * BIG + torch.clamp(-tand, -30, 30)
        pos = torch.searchsorted(keys, q.reshape(-1)).reshape(Hpx, Wpx) - bidx * J
        hit = pos < J
        j = torch.clamp(pos, 0, J - 1); jm = torch.clamp(j - 1, 0, J - 1)
        a_j = a.reshape(-1)[bidx * J + j].double(); a_m = a.reshape(-1)[bidx * J + jm].double()
        frac = torch.where(j > 0, torch.clamp((a_m - tand) / torch.clamp(a_m - a_j, min=1e-12), 0, 1), torch.zeros_like(a_j))
        rh = r[jm] + frac * (r[j] - r[jm])
        del a, keys
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        out_lin = torch.zeros((3, Hpx, Wpx), device=dev, dtype=torch.float32)
        rot = shot.sun_azimuth_deg - sky.sun_az_hdri
        se = math.radians(sky.sun_el if shot.sun_elevation_deg is None else shot.sun_elevation_deg)
        sa = math.radians(shot.sun_azimuth_deg)
        Ls = torch.tensor([math.cos(se) * math.cos(sa), math.cos(se) * math.sin(sa), math.sin(se)], device=dev, dtype=torch.float32)
        E_sun = sky.E_sun[:, None]; E_sky = sky.E_sky[:, None]
        E_d = sky.E_sun * math.sin(se) + sky.E_sky
        beta = 3.912 / (look.haze_km * 1000.0)
        aux = {}
        # ---- sky pixels
        sk = ~hit
        if bool(sk.any()):
            ds = d[sk].float()
            Lsky = sky.lookup(ds, rot)
            # the sun disc itself
            cosang = ds @ Ls
            sun_r = sky.E_sun[:, None] / (math.pi * SUN_RADIUS ** 2)
            Lsky = Lsky + sun_r * (cosang > math.cos(SUN_RADIUS)).float()
            out_lin[:, sk] = Lsky
        # ---- water pixels, in chunks
        hp = hit.reshape(-1).nonzero().squeeze(1)
        depth_fwd = torch.full((Hpx * Wpx,), float("inf"), device=dev, dtype=torch.float64)
        stats = {"foam": torch.zeros(Hpx * Wpx, device=dev), "glint": torch.zeros(Hpx * Wpx, device=dev)}
        flat = out_lin.reshape(3, -1)
        fwd = torch.tensor(cam.basis[0], device=dev, dtype=torch.float64)
        CH = 300_000
        for k0 in range(0, len(hp), CH):
            idx = hp[k0:k0 + CH]
            res = self._shade(idx, d.reshape(-1, 3)[idx], theta.reshape(-1)[idx], tand.reshape(-1)[idx], rh.reshape(-1)[idx], shot, sky, rot,
                              Ls, E_sun, E_sky, E_d, beta, pix)
            flat[:, idx] = res["L"]
            stats["foam"][idx] = res["foam"]; stats["glint"][idx] = res["glint"]
            psum = res["parts"].sum(1) + (psum if k0 > 0 else 0.0)
            depth_fwd[idx] = res["dist"] * (d.reshape(-1, 3)[idx] @ fwd)
        out_lin = flat.reshape(3, Hpx, Wpx)
        # ---- whale: dark matte shape where it is nearer than the water
        whale_vis = None
        if whale_mesh_m is not None:
            uv, depth = cam.project(whale_mesh_m * 100.0)
            img, zbuf, _ = project.rasterize(uv, depth, whale_faces, Wpx, Hpx, return_extra=True)
            zb = torch.from_numpy(zbuf / 100.0).to(dev)
            whale_vis = (torch.from_numpy(img > 0).to(dev)) & (zb < depth_fwd.reshape(Hpx, Wpx))
            alb = 0.035
            out_lin[:, whale_vis] = (alb / math.pi * E_d.float())[:, None] * (1.0 if True else 0)
            aux["whale_amodal"] = (img > 0)
        if len(hp):
            aux["parts_mean"] = dict(zip(["refl", "body", "glint", "F_eff", "sigma_slope"], (psum / len(hp)).cpu().numpy().round(5).tolist()))
        aux.update({"hit": hit.cpu().numpy(), "horizon_rows": self._horizon_rows(hit), "foam": stats["foam"].reshape(Hpx, Wpx).cpu().numpy(),
                    "glint": stats["glint"].reshape(Hpx, Wpx).cpu().numpy(), "whale": None if whale_vis is None else whale_vis.cpu().numpy(),
                    "pitch_deg": float(np.degrees(np.arcsin(cam.basis[0][2]))), "E_d": E_d.cpu().numpy()})
        return {"linear": out_lin.permute(1, 2, 0).cpu().numpy(), "aux": aux}

    @staticmethod
    def _horizon_rows(hit):
        """First water row per column (None-like -1 where the column is all water)."""
        h = hit.cpu().numpy()
        first = np.argmax(h, axis=0).astype(np.float32)
        allw = h.all(axis=0)
        first[allw] = -1
        return first

    def _sample_moments(self, c, qx, qy, dirx, diry, w_minor, w_major, bomb_w=None):
        L = self.L[c]; n = self.N
        k = n / L
        u = qx * k; v = qy * k
        taps = max(4, self.look.max_aniso // max(1, int(self.look.subsamples)))
        s = self.mom[c].sample(u, v, dirx, diry, w_minor * k, w_major * k, taps)
        if bomb_w is None:
            return s
        T = self.bomb_T[c]; o = self.bomb_off[c]
        T_t = self._bomb_t[c]
        q2x = T_t[0, 0] * qx + T_t[0, 1] * qy + o[0] * L; q2y = T_t[1, 0] * qx + T_t[1, 1] * qy + o[1] * L
        d2x = T_t[0, 0] * dirx + T_t[0, 1] * diry; d2y = T_t[1, 0] * dirx + T_t[1, 1] * diry
        s2 = self.mom[c].sample(q2x * k, q2y * k, d2x, d2y, w_minor * k, w_major * k, taps)
        # slopes of g(q) = f(T q + o): grad = T^T grad f ; T symmetric orthogonal here
        a, b, cc, dd = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
        sx2 = a * s2[0] + cc * s2[1]; sy2 = b * s2[0] + dd * s2[1]
        mxx = a * a * s2[2] + cc * cc * s2[3] + 2 * a * cc * s2[4]
        myy = b * b * s2[2] + dd * dd * s2[3] + 2 * b * dd * s2[4]
        mxy = a * b * s2[2] + cc * dd * s2[3] + (a * dd + b * cc) * s2[4]
        s2 = torch.stack([sx2, sy2, mxx, myy, mxy])
        wa = torch.cos(0.5 * math.pi * bomb_w); wb = torch.sin(0.5 * math.pi * bomb_w)
        sx = wa * s[0] + wb * s2[0]; sy = wa * s[1] + wb * s2[1]
        vxx = wa * wa * (s[2] - s[0] ** 2) + wb * wb * (s2[2] - s2[0] ** 2)
        vyy = wa * wa * (s[3] - s[1] ** 2) + wb * wb * (s2[3] - s2[1] ** 2)
        vxy = wa * wa * (s[4] - s[0] * s[1]) + wb * wb * (s2[4] - s2[0] * s2[1])
        return torch.stack([sx, sy, vxx + sx * sx, vyy + sy * sy, vxy + sx * sy])

    def _facets(self, qx, qy, dirx, diry, w_minor, w_major, g, damp, csl, V, Ls, E_sun, E_d, sky, rot, shot, idx, dist, pix):
        """Moments over one (sub-)footprint and the shading of that patch of facets: glitter, sky, water body."""
        look = self.look; dev = self.dev
        mu = torch.zeros((2,) + qx.shape, device=dev); cov = torch.zeros((3,) + qx.shape, device=dev)
        mu_geo = torch.zeros_like(mu)
        samples = []
        for c in range(len(self.L)):
            bw = None
            if look.bomb and c >= 3:
                kb = self.detail.n / (2.37 * self.L[c])
                bw = self.detail.sample(qx * kb + 3.1 * c, qy * kb + 7.3 * c, dirx, diry, w_minor * kb, w_major * kb, 4)[3]
            samples.append(self._sample_moments(c, qx, qy, dirx, diry, w_minor, w_major, bw))
        # hydrodynamic modulation: the short waves (cascades 4-5 and the residual) are rougher on the downwind
        # faces of the longer waves (1-3) and smoother on their backs; gain exp(-M z - M^2) keeps E[gain^2] = 1
        gm = torch.ones_like(qx)
        if look.mod_strength > 0:
            cw_, sw_ = math.cos(self.wind_dir), math.sin(self.wind_dir)
            s_up = sum(samples[c][0] * cw_ + samples[c][1] * sw_ for c in (1, 2, 3))
            z = torch.clamp(s_up / self.sigma_long_up, -3, 3)
            gm = torch.exp(-look.mod_strength * z - look.mod_strength ** 2)
        for c in range(len(self.L)):
            s = samples[c]
            gain = g if c >= 3 else torch.ones_like(g)
            if c >= 4:
                gain = gain * gm
            if damp is not None and c >= 3:
                gain = gain * (1 - (0.9 if c >= 4 else 0.5) * damp)
            mx, my = s[0] * gain, s[1] * gain
            vx = torch.clamp(s[2] - s[0] ** 2, min=0) * gain ** 2; vy = torch.clamp(s[3] - s[1] ** 2, min=0) * gain ** 2
            vxy = (s[4] - s[0] * s[1]) * gain ** 2
            mu[0] += mx; mu[1] += my; cov[0] += vx; cov[1] += vy; cov[2] += vxy
            if c < self.geo:
                mu_geo[0] += mx; mu_geo[1] += my
        if csl is not None:
            mu[0] += csl[0]; mu[1] += csl[1]; mu_geo[0] += csl[0]; mu_geo[1] += csl[1]
        rs = look.res_scale * (g ** 2) * gm ** 2 * (1 - 0.9 * damp if damp is not None else 1.0)
        cov[0] += self.res_cov[0] * rs; cov[1] += self.res_cov[1] * rs; cov[2] += self.res_cov[2] * rs
        # mesh normal (filtered geometric slope) for projected area / masking
        nmesh = torch.stack([-mu_geo[0], -mu_geo[1], torch.ones_like(qx)], 0); nmesh = nmesh / torch.linalg.norm(nmesh, dim=0, keepdim=True)
        vnm = torch.clamp((V * nmesh).sum(0), min=1e-3)
        # ---- sun glitter
        Hv = V + Ls[:, None]; Hv = Hv / torch.linalg.norm(Hv, dim=0, keepdim=True)
        hz = torch.clamp(Hv[2], min=1e-3)
        zx = -Hv[0] / hz - mu[0]; zy = -Hv[1] / hz - mu[1]
        sun_var = (SUN_RADIUS / 2) ** 2 / 4 / hz ** 4
        # slope density: a two-Gaussian mixture of the same covariance (weight tail_w at tail_s x the width)
        # giving the Cox-Munk peakedness (kurtosis ~3.6) instead of a pure Gaussian's thin tails
        tw, ts = look.tail_w, look.tail_s
        base = 1.0 / (1 - tw + tw * ts * ts)
        pdf = 0.0
        for wgt, sc in ((1 - tw, base), (tw, base * ts * ts)):
            cxx = cov[0] * sc + sun_var; cyy = cov[1] * sc + sun_var; cxy = cov[2] * sc
            det = torch.clamp(cxx * cyy - cxy * cxy, min=1e-12)
            ee = (zx * zx * cyy + zy * zy * cxx - 2 * zx * zy * cxy) / det
            pdf = pdf + wgt * torch.exp(-0.5 * ee) / (2 * math.pi * torch.sqrt(det))
        Dm = pdf / hz ** 4
        if look.steep_w > 0:
            # a sparse population of steep facets (micro-breaking, parasitic capillaries on crests): a broad
            # Blinn-Phong normal distribution of weight steep_w, so glints also come from far off the specular
            # direction, as in photographs taken across or down sun
            ns = look.steep_n
            sw = torch.clamp(look.steep_w * gm ** 2 * g ** 2, max=0.5)   # on the rough (modulated, gusty) patches
            Dm = (1 - sw) * Dm + sw * (ns + 2) / (2 * math.pi) * hz ** ns

        def lam(w):
            wn = torch.clamp((w * nmesh).sum(0), -1, 1)
            hx, hy = w[0] - wn * nmesh[0], w[1] - wn * nmesh[1]
            hl = torch.clamp(torch.hypot(hx, hy), min=1e-6)
            ax, ay = hx / hl, hy / hl
            sig2 = ax * ax * cov[0] + ay * ay * cov[1] + 2 * ax * ay * cov[2]
            return smith_lambda(wn / torch.sqrt(torch.clamp(1 - wn * wn, min=1e-8)), sig2)
        Ls_b = Ls[:, None].expand_as(V)
        G = 1.0 / (1.0 + lam(V) + lam(Ls_b))
        Fh = fresnel((V * Hv).sum(0))
        sun_up = ((Ls_b * nmesh).sum(0) > 0).float()
        glint = Fh * Dm * G / (4 * vnm) * sun_up
        L_glint = E_sun * glint[None]
        if look.glint_area_m2 > 0:
            # discrete glints: the expected glitter radiance arrives as N ~ Poisson(lambda) sun images, each of a
            # flat facet patch of area glint_area_m2 seen in the pixel's cross-section (d * pixel angle)^2:
            # one glint = L_sun * F * min(1, a cos / A_pix). Mean preserved; near pixels sparkle and clip.
            Lsun = E_sun.mean() / (math.pi * SUN_RADIUS ** 2)
            A_pix = (dist.float() * pix) ** 2
            one = Lsun * Fh * torch.clamp(look.glint_area_m2 * (V * Hv).sum(0).clamp(min=0.05) / A_pix, max=1.0)
            lam = E_sun.mean() * glint / torch.clamp(one, min=1e-12)
            gen = torch.Generator(device=dev); gen.manual_seed(int(shot.seed) * 7919 + int(idx[0]))
            lam = torch.nan_to_num(torch.clamp(lam, min=0.0, max=1e4), nan=0.0)
            cnt = torch.poisson(lam, generator=gen)
            disc = torch.where(lam < 50, cnt * one / torch.clamp(E_sun.mean() * glint, min=1e-20), torch.ones_like(lam))
            L_glint = L_glint * disc[None]
        # ---- sky reflection: Gauss-Hermite 3x3 over the slope distribution, visible-facet weights
        gh_x = [-math.sqrt(3.0), 0.0, math.sqrt(3.0)]; gh_w = [1 / 6, 2 / 3, 1 / 6]
        # keep the quadrature on visible facets: if the mean facet faces away from the camera by more than one
        # sigma (back of a crest at grazing view), shift the lobe's centre towards the camera
        vxy2 = torch.clamp(V[0] ** 2 + V[1] ** 2, min=1e-8)
        ux_, uy_ = V[0] / torch.sqrt(vxy2), V[1] / torch.sqrt(vxy2)
        sig_v = torch.sqrt(torch.clamp(ux_ * ux_ * cov[0] + uy_ * uy_ * cov[1] + 2 * ux_ * uy_ * cov[2], min=1e-8))
        a_ = mu[0] * V[0] + mu[1] * V[1]
        lim = V[2] + sig_v * torch.sqrt(vxy2)
        excess = torch.clamp(a_ - lim, min=0) / vxy2
        mu_r = torch.stack([mu[0] - excess * V[0], mu[1] - excess * V[1]])
        a11 = torch.sqrt(torch.clamp(cov[0], min=1e-10)); a21 = cov[2] / a11
        a22 = torch.sqrt(torch.clamp(cov[1] - a21 * a21, min=1e-10))
        # one sky lookup per (sub-)footprint: at the Fresnel-weighted mean reflection direction of the 9 nodes, blurred
        # to the nodes' angular spread; the share of nodes reflecting below the horizon is dimmed (sky_fold_dim)
        Rm = torch.zeros((3,) + qx.shape, device=dev); Fsum = torch.zeros_like(qx); wsum = torch.zeros_like(qx); fb = torch.zeros_like(qx)
        R2 = torch.zeros_like(qx)
        for i, xi in enumerate(gh_x):
            for j2, yj in enumerate(gh_x):
                zx_i = mu_r[0] + a11 * xi; zy_i = mu_r[1] + a21 * xi + a22 * yj
                n = torch.stack([-zx_i, -zy_i, torch.ones_like(zx_i)], 0); n = n / torch.linalg.norm(n, dim=0, keepdim=True)
                vdn = (V * n).sum(0)
                w = gh_w[i] * gh_w[j2] * torch.clamp(vdn, min=0) / n[2]
                R = 2 * vdn[None] * n - V
                below = (R[2] < 0.0).float()
                # at least 3 deg up: the HDRIs' lowest degrees are land-tinted haze (the photographs' horizon is the sea)
                R = torch.stack([R[0], R[1], torch.clamp(torch.where(R[2] < 0, -R[2] * 0.3, R[2]), min=0.052)], 0)
                Fi = fresnel(torch.clamp(vdn, min=1e-4))
                wf = w * Fi
                Rm += R * wf[None]; R2 += wf * (R * R).sum(0); fb += wf * below; Fsum += wf; wsum += w
        Fs = torch.clamp(Fsum, min=1e-12)
        Rd = Rm / Fs[None]
        spread = torch.sqrt(torch.clamp(R2 / Fs - (Rd * Rd).sum(0), min=0.0))          # rad, rms angular spread
        Rd = Rd / torch.clamp(torch.linalg.norm(Rd, dim=0, keepdim=True), min=1e-6)
        Lr = sky.lookup(Rd.T, rot, torch.clamp(torch.rad2deg(spread) * 0.8, 0, 22))
        if look.sky_sat != 1.0:
            # saturate only blue sky (blue above red): warm haze and clouds keep their colour, no khaki casts
            lumk = (Lr * self._luma[:, None]).sum(0, keepdim=True)
            blue = torch.clamp(3.0 * (Lr[2:3] - Lr[0:1]) / torch.clamp(Lr[2:3] + Lr[0:1], min=1e-6), 0, 1)
            Lr = torch.clamp(lumk + (Lr - lumk) * (1 + (look.sky_sat - 1) * blue), min=0)
        Lr = Lr * (1 - (1 - look.sky_fold_dim) * fb / Fs)[None]
        wsum = torch.clamp(wsum, min=1e-9)
        Feff = Fsum / wsum
        Lr = Lr * Feff[None]
        # ---- water body
        rrs = torch.tensor(look.rrs, device=dev, dtype=torch.float32)[:, None]
        L_body = rrs * E_d.float()[:, None] * ((1 - Feff) / 0.98)[None]
        Lw = Lr + L_body + L_glint
        return {"Lw": Lw, "glint": glint, "Feff": Feff, "Lr": Lr, "L_body": L_body, "L_glint": L_glint,
                "sigma": torch.sqrt(0.5 * (cov[0] + cov[1])), "nmesh": nmesh}

    def _shade(self, idx, d, theta, tand, rh, shot, sky, rot, Ls, E_sun, E_sky, E_d, beta, pix):
        look = self.look; dev = self.dev
        H0 = shot.cam_height_m
        ct, st = torch.cos(theta), torch.sin(theta)
        x = shot.cam_xy_m[0] + rh * ct; y = shot.cam_xy_m[1] + rh * st
        zr = H0 - rh * tand - rh * rh / (2 * EARTH_R)
        dist = torch.sqrt(rh * rh + (H0 - zr) ** 2)
        xf, yf = x.float(), y.float()
        _, qx, qy = self.height_at(xf, yf)
        # view vector in the local frame (earth curvature tilts the local vertical by r / R)
        e = torch.atan(tand) + rh / EARTH_R
        V = torch.stack([-torch.cos(e) * ct, -torch.cos(e) * st, torch.sin(e)], 0).float()        # towards the camera
        # geometric (mesh) slope at mip 0 for the footprint
        gx = torch.zeros_like(xf); gy = torch.zeros_like(xf)
        for c in range(self.geo):
            k = self.N / self.L[c]
            s = self.mom[c].bilinear(qx * k, qy * k, torch.zeros_like(qx, dtype=torch.long))
            gx = gx + s[0]; gy = gy + s[1]
        csl = None
        if self.contact is not None:
            csl = self._contact_sample(qx, qy, [4, 5])
            gx = gx + csl[0]; gy = gy + csl[1]
        nm = torch.stack([-gx, -gy, torch.ones_like(gx)], 0); nm = nm / torch.linalg.norm(nm, dim=0, keepdim=True)
        vn = torch.clamp((V * nm).sum(0), min=0.004)
        w_minor = dist.float() * pix
        w_major = w_minor / vn
        dirx = -V[0] / torch.clamp(torch.hypot(V[0], V[1]), min=1e-6); diry = -V[1] / torch.clamp(torch.hypot(V[0], V[1]), min=1e-6)
        # ---- moments over the footprint
        k_det = self.detail.n / self.detail_L
        det_minor = w_minor * k_det; det_major = w_major * k_det
        # wind gusts (cat's paws) from the detail texture's blend channel at a large scale
        g = torch.ones_like(qx)
        if look.gust_amp > 0:
            kg = self.detail.n / look.gust_scale_m
            gn = self.detail.sample(qx * kg + 17.0, qy * kg + 41.0, dirx, diry, w_minor * kg, w_major * kg, 4)[3]
            g = (1 + look.gust_amp * (2 * gn - 1)) / math.sqrt(1 + look.gust_amp ** 2 / 3)
        damp = None
        if self.contact is not None:
            damp = torch.clamp(self._contact_sample(qx, qy, [2])[0], 0, 1)
        # ---- shading of the facets: K sub-footprints along the footprint's long (range) axis, each filtered over
        # its own 1/K of the footprint and shaded separately, so waves longer than a sub-footprint are shaded as the
        # actual facets they are (no filtered-statistics "long exposure"); K = 1 is the material's single sample
        K = max(1, int(look.subsamples))
        Lw = 0.0; glint = 0.0; Feff = 0.0; Lr = 0.0; L_body = 0.0; L_glint = 0.0; sig_acc = 0.0; nmesh = None
        for kk in range(K):
            off = ((kk + 0.5) / K - 0.5) * w_major
            fr = self._facets(qx + dirx * off, qy + diry * off, dirx, diry, w_minor, w_major / K, g, damp, csl, V, Ls, E_sun, E_d,
                              sky, rot, shot, idx, dist, pix)
            Lw = Lw + fr["Lw"] / K; glint = glint + fr["glint"] / K; Feff = Feff + fr["Feff"] / K
            Lr = Lr + fr["Lr"] / K; L_body = L_body + fr["L_body"] / K; L_glint = L_glint + fr["L_glint"] / K
            sig_acc = sig_acc + fr["sigma"] / K
            if nmesh is None:
                nmesh = fr["nmesh"]
        cov = None
        # ---- foam
        alpha = torch.zeros_like(qx)
        lace = self.detail.sample(qx * k_det, qy * k_det, dirx, diry, det_minor, det_major, look.max_aniso)[0]
        if self.foam is not None:
            kf = self.foam.n / self.foam_L
            fs = self.foam.sample(qx * kf, qy * kf, dirx, diry, w_minor * kf, w_major * kf, look.max_aniso)
            # aged foam lace stretched down-wind into streaks
            wdx, wdy = math.cos(self.wind_dir), math.sin(self.wind_dir)
            sp = qx * wdx + qy * wdy; sc_ = -qx * wdy + qy * wdx
            ks = self.detail.n / 9.0
            streak = self.detail.sample(sp * ks / 6.0, sc_ * ks, dirx, diry, w_minor * ks, w_major * ks, 4)[2]
            fresh = fs[0]; aged = fs[1] * (0.4 + 1.2 * streak)
            cf = torch.clamp(fresh, 0, 1); ca = torch.clamp(oceanbake.FOAM["aged_gain"] * aged, 0, 1)
            # a coarser lace (6 m repeat: bubble cells of 0.2-0.7 m) so whitecaps keep visible texture at the pixel scale
            k6 = self.detail.n / 6.0
            lc = self.detail.sample(qx * k6 + 11.0, qy * k6 + 5.0, dirx, diry, w_minor * k6, w_major * k6, 8)
            lace6 = torch.clamp((lc[0] - 0.25) / 0.5, 0, 1)
            alpha = cf * (0.45 + 0.55 * lace6) * (0.85 + 0.15 * lace) + (1 - cf) * ca * lace * (0.3 + 0.7 * lace6)
            foam_tex = 0.72 + 0.4 * lc[1]                  # bright bubble rims, darker cell interiors
        if self.contact is not None:
            cfo = torch.clamp(self._contact_sample(qx, qy, [1])[0], 0, 1.5)
            # broken white water: the foam amount sets a threshold on a two-scale bubble pattern (0.6 m and 1.5 m
            # cells), so thin foam is a lace of patches and holes and only dense foam closes up; soft but crisp edges
            k2 = self.detail.n / 2.0
            lp = self.detail.sample(qx * k2 + 23.0, qy * k2 + 3.0, dirx, diry, w_minor * k2, w_major * k2, 8)
            pat = 0.55 * lp[0] + 0.25 * lace + 0.2 * lp[2]
            edge = 0.06 + 0.5 * torch.clamp(w_major * k2 / self.detail.n * 4.0, max=0.3)        # wider where the pixel is coarse
            ac = torch.clamp((pat - (1.0 - torch.clamp(cfo * 1.4, 0, 1.2)) + edge) / (2 * edge), 0, 1)
            if self.foam is None:
                foam_tex = torch.ones_like(qx)
            foam_tex = torch.where(ac > alpha, 0.65 + 0.45 * lp[1], foam_tex)
            alpha = 1 - (1 - alpha) * (1 - ac)
        L_foam = look.foam_albedo / math.pi * (E_sun * torch.clamp((Ls[:, None] * nmesh).sum(0), min=0)[None] * 0.6 + E_sky)
        if self.foam is not None or self.contact is not None:
            L_foam = L_foam * foam_tex[None]
        Ls_ = Lw * (1 - alpha)[None] + L_foam * alpha[None]
        # ---- haze
        T = torch.exp(-beta * dist.float())[None]
        hzdir = torch.stack([-V[0], -V[1], torch.full_like(V[0], 0.035)], 0); hzdir = hzdir / torch.linalg.norm(hzdir, dim=0, keepdim=True)
        L_h = sky.lookup(hzdir.T, rot, torch.full_like(qx, 10.0))
        L = Ls_ * T + L_h * (1 - T)
        lw = self._luma[:, None]
        parts = torch.stack([(Lr * lw).sum(0), (L_body * lw).sum(0), (L_glint * lw).sum(0), Feff, sig_acc])
        return {"L": L, "foam": alpha, "glint": glint, "dist": dist, "parts": parts}


# ------------------------------------------------------------------------------------------------ camera pipeline
def develop(linear: np.ndarray, look: Look, seed: int, meter_mask: np.ndarray | None = None, ev: float | None = None) -> tuple[np.ndarray, dict]:
    """Linear radiance -> 8-bit photograph: metering, tone curve, saturation, lens blur + sharpening, then
    factory.camera_fx (noise, CA, vignette, JPEG)."""
    import cv2
    from factory import camera_fx
    rng = np.random.default_rng(seed)
    Y = linear @ np.array(LUMA, np.float32)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0)
    sel = Y if meter_mask is None else Y[meter_mask]
    med = float(np.median(sel))
    clipY = np.minimum(sel, 4.0 * med)
    m = float(clipY.mean())
    ev = float(rng.normal(look.exposure_bias_ev, look.exposure_ev_sd)) if ev is None else ev
    k = 0.18 * 2.0 ** ev / max(m, 1e-9)
    x = np.nan_to_num(linear * k, nan=0.0, posinf=50.0, neginf=0.0)      # a stray NaN would spread over the blur kernel
    blur = look.lens_blur_px
    if look.lens_blur_range:
        lo, hi = look.lens_blur_range
        blur = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
    if blur > 0:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    x = np.clip(x, 0, None)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055)
    y = np.clip(y, 0, 1)
    if look.saturation != 1.0:
        lum = (y @ np.array(LUMA, np.float32))[..., None]
        y = np.clip(lum + (y - lum) * look.saturation, 0, 1)
    y = (1 - look.contrast) * y + look.contrast * (y * y * (3 - 2 * y))
    if look.sharpen > 0:
        bl = cv2.GaussianBlur(y, (0, 0), 1.0)
        y = np.clip(y + look.sharpen * (y - bl), 0, 1)
    img = (y * 255 + 0.5).astype(np.uint8)
    img, fx = camera_fx.apply(img, seed, look.camera_fx)
    return img, {"ev": ev, "k": k, "lens_blur_px": round(blur, 3), **fx}


def export_engine_skies(out_dir: Path, names=None) -> dict:
    """The material's inputs per sky: the HDRI lat-long with the sun taken out (2048 x 1024 16-bit PNG holding
    radiance / 8, like factory.skies.PNG_SCALE) and the sun / sky irradiance in the same radiance units."""
    import cv2
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cat = json.loads((SKIES_DIR / "catalogue.json").read_text())
    info = {}
    for name in names or list(cat):
        if cat[name].get("overcast"):
            continue
        d = Sky._prepare(name)
        lv = d["lev0"].transpose(1, 2, 0)
        cv2.imwrite(str(out_dir / f"{name}_sunless.png"), np.clip(lv[..., ::-1] / 8.0 * 65535.0, 0, 65535).astype(np.uint16))
        info[name] = {"png": f"{name}_sunless.png", "png_scale": 8.0, "sun_elevation_deg": float(d["sun_el"]), "sun_azimuth_deg": float(d["sun_az"]),
                      "sun_normal_irradiance_rgb": [round(float(x), 5) for x in d["E_sun"]],
                      "sky_horizontal_irradiance_rgb": [round(float(x), 5) for x in d["E_sky"]], "sun_restored": bool(d["restored"])}
    (out_dir / "sky_irradiance.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------------------------------------ CLI
def pick_device(max_gb: float = 4.0, wait_s: float = 1800.0):
    """The shared GPU, capped at max_gb; waits up to wait_s for that much to be free. CPU (8 threads) only when no
    GPU exists or WATERREF_ALLOW_CPU=1."""
    import os
    import sys
    import time
    if torch.cuda.is_available():
        t0 = time.time()
        while True:
            free, total = torch.cuda.mem_get_info()
            if free / 2 ** 30 > max_gb + 1.0:
                torch.cuda.set_per_process_memory_fraction(min(max_gb * 2 ** 30 / total, 0.95))
                print(f"waterref: cuda, {free / 2 ** 30:.1f} GB free, capped at {max_gb} GB", file=sys.stderr)
                return torch.device("cuda")
            if os.environ.get("WATERREF_ALLOW_CPU") == "1" or time.time() - t0 > wait_s:
                break
            time.sleep(30)
    torch.set_num_threads(8)
    print("waterref: cpu (8 threads)", file=sys.stderr)
    return torch.device("cpu")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bake", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--focal", type=float, default=200.0); ap.add_argument("--distance", type=float, default=150.0)
    ap.add_argument("--height", type=float, default=3.0); ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--sun-az", type=float, default=60.0); ap.add_argument("--sky", default="qwantani_noon_puresky")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = pick_device()
    sea = oceanbake.BakedSea.load(a.bake)
    R = Renderer(sea, dev)
    sky = Sky(a.sky, dev)
    shot = Shot(cam_height_m=a.height, focal_mm=a.focal, distance_m=a.distance, yaw_deg=a.yaw, sky=a.sky, sun_azimuth_deg=a.sun_az, seed=a.seed)
    res = R.render(shot, sky)
    img, info = develop(res["linear"], R.look, a.seed)
    from PIL import Image
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(a.out, quality=95)
    print(json.dumps(info), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
