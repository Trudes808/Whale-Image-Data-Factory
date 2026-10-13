"""Real sky HDRIs (Poly Haven, CC0) as the sky and the light of a shot: the catalogue of the
downloaded skies with the sun direction and brightness read from each image, so a shot's sun
light can follow its sky, and the per-batch / per-shot sampling.

    python3 -m factory.skies third_party_assets/skies        # (re)build skies/catalogue.json

Lat-long convention: u = 0..1 maps to azimuth 0..360 deg (u = 0.5 straight ahead, +X in the
sky material at rotation 0), v = 0 at the zenith. The sky sphere is rotated per shot about +Z
(the HDRI yaw); the sun light's yaw follows: yaw = sun_azimuth + rotation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CATALOGUE = "catalogue.json"


def read_hdr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)


def analyse(path: Path) -> dict:
    """Sun direction (elevation and azimuth in degrees, azimuth 0 = +X at rotation 0), the sun's
    brightness relative to the sky, the mean sky luminance and an overcast flag."""
    img = read_hdr(path)
    H, W = img.shape[:2]
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    v = (np.arange(H) + 0.5) / H; elev = 90.0 - 180.0 * v                      # row -> elevation
    sky = lum[elev > 2.0]                                                        # above the horizon
    k = max(int(0.0005 * lum.size), 50)
    idx = np.argpartition(lum.ravel(), -k)[-k:]
    ys, xs = np.unravel_index(idx, lum.shape)
    w = lum.ravel()[idx]
    # mean direction of the brightest pixels (circular mean for the azimuth)
    az = 2 * math.pi * ((xs + 0.5) / W - 0.5); el = np.radians(90.0 - 180.0 * (ys + 0.5) / H)
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    m = (d * w[:, None]).sum(0) / w.sum(); m /= np.linalg.norm(m)
    sun_elev = math.degrees(math.asin(np.clip(m[2], -1, 1))); sun_az = math.degrees(math.atan2(m[1], m[0]))
    sun_peak = float(np.percentile(w, 90)); sky_median = float(np.median(sky))
    ratio = sun_peak / max(sky_median, 1e-6)
    return {"file": path.name, "width": int(W), "height": int(H), "sun_elevation_deg": round(sun_elev, 2), "sun_azimuth_deg": round(sun_az, 2),
            "sky_median_luminance": round(sky_median, 4), "sun_to_sky_ratio": round(ratio, 1), "overcast": bool(ratio < 25.0),
            "horizon_luminance": round(float(np.median(lum[(elev > 1.0) & (elev < 6.0)])), 4), "zenith_luminance": round(float(np.median(lum[elev > 60.0])), 4)}


PNG_SCALE = 8.0     # the 16-bit PNG stores radiance / PNG_SCALE (the material multiplies back; the sun disc is clamped anyway)


def write_png16(path: Path) -> Path:
    """The HDRI as a 16-bit linear PNG (radiance / PNG_SCALE, clamped): Unreal imports .hdr files as
    cubemaps, a 2D lat-long texture needs a 2D source format."""
    import cv2

    out = path.with_suffix(".png")
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)        # BGR
    cv2.imwrite(str(out), np.clip(img / PNG_SCALE * 65535.0, 0, 65535).astype(np.uint16))
    return out


def build_catalogue(directory: Path) -> dict:
    out = {}
    for p in sorted(directory.glob("*.hdr")) + sorted(directory.glob("*.exr")):
        out[p.stem] = analyse(p)
        out[p.stem]["png"] = write_png16(p).name
        out[p.stem]["png_scale"] = PNG_SCALE
        print(p.stem, out[p.stem])
    (directory / CATALOGUE).write_text(json.dumps(out, indent=1))
    return out


def load_catalogue(directory: Path) -> dict:
    p = Path(directory) / CATALOGUE
    return json.loads(p.read_text()) if p.exists() else {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    a = p.parse_args(argv)
    cat = build_catalogue(a.directory)
    print(f"{len(cat)} skies -> {a.directory / CATALOGUE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
