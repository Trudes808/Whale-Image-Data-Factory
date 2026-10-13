"""Camera realism applied to the rendered RGB (labels untouched): photon/read noise like a DSLR at
the shot's ISO, a touch of lateral chromatic aberration, vignetting, and JPEG re-encoding, so the
dataset carries the statistics of field photos (Canon 7D II / R5 at long focal lengths) rather than
of a clean render. Deterministic per shot (seeded by its id).

    config "camera_fx": {"noise_sigma": [0.002, 0.012], "chroma_px": [0.0, 1.2], "vignette": [0.0, 0.25], "jpeg_quality": [82, 96]}
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

DEFAULTS = {"noise_sigma": [0.002, 0.012], "chroma_px": [0.0, 1.2], "vignette": [0.0, 0.25], "jpeg_quality": [82, 96]}


def apply(image: np.ndarray, seed: int, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """uint8 RGB (H, W, 3) -> uint8 RGB with the effects, plus the parameters used."""
    c = dict(DEFAULTS); c.update(cfg or {})
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32) / 255.0
    H, W = img.shape[:2]
    params = {}
    # lateral chromatic aberration: red and blue scaled about the centre by a fraction of a pixel at the edge
    chroma = float(rng.uniform(*c["chroma_px"])); params["chroma_px"] = round(chroma, 3)
    if chroma > 0.05:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        r = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
        out = img.copy()
        for ch, sign in ((0, 1.0), (2, -1.0)):
            k = sign * chroma * r / max(cx, cy)                      # pixels of displacement, growing with radius
            sx = np.clip(xs + (xs - cx) * k, 0, W - 1); sy = np.clip(ys + (ys - cy) * k, 0, H - 1)
            x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int); x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
            fx = sx - x0; fy = sy - y0
            p = img[..., ch]
            out[..., ch] = (p[y0, x0] * (1 - fx) + p[y0, x1] * fx) * (1 - fy) + (p[y1, x0] * (1 - fx) + p[y1, x1] * fx) * fy
        img = out
    # vignetting
    vig = float(rng.uniform(*c["vignette"])); params["vignette"] = round(vig, 3)
    if vig > 0.01:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xs - (W - 1) / 2) / (W / 2)) ** 2 + ((ys - (H - 1) / 2) / (H / 2)) ** 2
        img = img * (1.0 - vig * np.clip(r2, 0, 1.5) ** 1.2)[..., None]
    # sensor noise: shot noise grows with brightness, plus read noise; a little stronger in blue
    sigma = float(rng.uniform(*c["noise_sigma"])); params["noise_sigma"] = round(sigma, 4)
    lin = np.clip(img, 0, 1) ** 2.2
    shot = rng.normal(0, 1, img.shape).astype(np.float32) * np.sqrt(lin + 1e-4) * sigma * 2.5
    read = rng.normal(0, 1, img.shape).astype(np.float32) * sigma * np.array([1.0, 0.9, 1.3], np.float32)
    lin = np.clip(lin + shot + read, 0, 1)
    img = lin ** (1 / 2.2)
    # JPEG
    q = int(rng.integers(c["jpeg_quality"][0], c["jpeg_quality"][1] + 1)); params["jpeg_quality"] = q
    buf = io.BytesIO(); Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=q, subsampling=0)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out, params
