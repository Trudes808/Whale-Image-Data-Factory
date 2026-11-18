"""run06 realism round 3: darker, field-like base-colour maps for the 4k humpback individuals (new files with the suffix
"d": T_Humpback_BaseColor_<id>d.png; the existing maps and every older config stay as they are).

The shipped atlas (and every individual derived from it) has the same large paper-white areas on the fluke underside,
pectorals, throat pleats and belly, and a navy / violet dark skin. North Pacific humpbacks seen from a boat are near-black
on top, dark grey with narrow wet highlights, and their white is a pattern that varies by animal (many flukes are mostly
black), never paper white. Per map:
  * the dark skin is desaturated toward a neutral grey-black (chroma x DARK_CHROMA) and its albedo scaled by DARK_SCALE;
  * the white areas (texture L* above about 60, soft edge) keep white only where a per-animal low-frequency pattern is
    below the animal's white fraction (drawn per individual: mostly black / mixed / mostly white flukes), elsewhere they
    become the animal's dark skin; the kept white is an off-white (WHITE_MAX albedo) with its texture.
Deterministic per individual id.

    python3 -m factory.skin_run06 [--size 4096] [IDS ...]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SKINS = REPO / "third_party_assets" / "humpback_4k" / "skins"
SUFFIX = "d"
DARK_CHROMA = 0.2       # chroma kept in the dark skin (the shipped dark is navy / violet)
DARK_SCALE = 1.8         # dark skin albedo (round 4: 0.85 left the dark skin at L* 14 against real 24, 1.6 made backs light grey; the per-actor BaseColorScale key had no effect)
ALL_CHROMA = 0.5         # round 4: chroma kept everywhere after the mix (whites and transitions too)
WHITE_MAX = 0.7         # off-white: the brightest white albedo (linear), never paper white
# fraction of each animal's white areas that stays white: mostly black / mixed / mostly white (field photo fluke types)
# (measured on training-day photos: whale pixels lighter than L* 75 are a median 1 % of a fluke view, 2 % of a fin view;
# 7 % of fluke views have more than 10 % white: Hawaii humpbacks' flukes are mostly dark underneath)
WHITE_KEEP = ((0.85, (0.0, 0.03)), (0.12, (0.03, 0.15)), (0.03, (0.2, 0.4)))     # round 9 (round 4: 0.80 / 0.15 / 0.05 up to 0.6): r9 rendered 56 % of shots over 10 % white


def _lin(s):
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def _noise(rng, size, scales=(24, 64, 160)):
    """Low-frequency value noise in 0..1 (bilinear upsampling of random grids)."""
    out = np.zeros((size, size), np.float32)
    for k, n in enumerate(scales):
        g = rng.random((n + 1, n + 1)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += np.asarray(im, np.float32) / 255 * (0.55 if k == 0 else 0.3 if k == 1 else 0.15)
    out -= out.min(); out /= max(out.max(), 1e-6)
    return out


_DARK_UV: dict = {}


def dark_uv_mask(size: int) -> np.ndarray:
    """Round 5: the UV texels of the dorsal fin and the upper back / flanks (rest-pose faces whose outward normal points
    up; the mesh's winding makes the computed face normal point inward, hence the sign), where a humpback is never white:
    the white pattern is not kept there. 1 = dark only. Cached per size."""
    if size in _DARK_UV:
        return _DARK_UV[size]
    import json
    from PIL import ImageDraw
    from factory.whalemodel import WhaleModel
    cfg = json.loads((REPO / "factory" / "config" / "run06_realism.json").read_text())
    model = WhaleModel.from_config(cfg)
    R, F, uv, ids = model.rest, model.faces, model._uv(), model.vertex_part_ids()
    fn = np.cross(R[F[:, 1]] - R[F[:, 0]], R[F[:, 2]] - R[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9
    up = -fn[:, 2]                                                   # outward normal's z (winding is inward)
    pid = ids[F[:, 0]]
    sel = (pid == 2) | (((pid == 5) | (pid == 4)) & (up > 0.25)) | ((pid == 3) & (up > 0.0))   # round 9: + the pectorals' top side
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    for f in F[sel]:
        d.polygon([(float(uv[v, 0] % 1) * size, float(1 - uv[v, 1] % 1) * size) for v in f], fill=255)
    m = np.asarray(im, np.float32) / 255
    _DARK_UV[size] = m
    return m


def make(key: str, size: int = 4096, out_dir: Path | None = None) -> Path:
    src = SKINS / f"T_Humpback_BaseColor_{key}.png"
    out = (out_dir or SKINS) / f"T_Humpback_BaseColor_{key}{SUFFIX}.png"
    seed = int(hashlib.sha1(f"skin_run06:{key}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    s = np.asarray(Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS), np.float32) / 255
    lin = _lin(s)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
    # round 4: everything lighter than the dark skin takes the pattern path (the atlas's light-grey fluke and fin areas,
    # L* 55-70, had been scaled with the dark skin and rendered as light grey flukes)
    white = np.clip((L - 38.0) / 17.0, 0, 1)                       # 0 below L* 38, 1 above L* 55
    # dark skin: neutral grey-black
    dark = (y[..., None] + (lin - y[..., None]) * DARK_CHROMA) * DARK_SCALE
    dark_ref = np.median(dark[L < 40], axis=0) if (L < 40).any() else np.array([0.02, 0.02, 0.022])
    dark = np.minimum(dark, dark_ref[None, None, :] * 1.3)        # no light texels inside the dark skin (round 9: 1.6 -> 1.3, pale grey bellies and flanks in r9)
    # the animal's white pattern
    u = rng.random(); acc = 0.0
    for p, (lo, hi) in WHITE_KEEP:
        acc += p
        if u <= acc:
            keep = float(rng.uniform(lo, hi)); break
    n = _noise(rng, size)
    thr = np.quantile(n, keep)
    soft = np.clip((thr - n) / 0.04 + 0.5, 0, 1)                    # 1 = stays white, soft 4 % edge
    soft = soft * (1.0 - dark_uv_mask(size))                         # never white on the fin and the upper back
    # mottling of the darkened white: the animal's dark skin with a little of the old structure
    # round 7: the darkened white is as dark as the animal's skin, with only a faint trace of the old structure (0.8 +
    # 0.4 x left sunlit fluke undersides beige-grey and blotchy)
    gone = dark_ref[None, None, :] * (0.7 + 0.15 * np.clip(y / max(np.percentile(y, 99), 1e-6), 0, 1))[..., None]
    kept = np.minimum(lin * (WHITE_MAX / max(float(np.percentile(lin[white > 0.9], 99)) if (white > 0.9).any() else 1.0, 1e-6)), WHITE_MAX)
    kept = y[..., None] * 0 + kept                                   # keep the white's own texture, scaled to off-white
    w = white[..., None]
    k = soft[..., None]
    res = dark * (1 - w) + w * (k * kept + (1 - k) * gone)
    # round 4: no pink / red anywhere (the atlas's mouth, pleat and rim tints showed as pink patches on heads)
    yr = res @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    res = yr[..., None] + (res - yr[..., None]) * ALL_CHROMA
    img = (_srgb(res) * 255 + 0.5).astype(np.uint8)
    tmp = out.with_suffix(".png.tmp")
    Image.fromarray(img).save(tmp, format="PNG")
    tmp.replace(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path, default=None, help="write here instead of the skins folder (preview)")
    a = ap.parse_args(argv)
    keys = a.ids or sorted(p.stem.rsplit("_", 1)[1] for p in SKINS.glob("T_Humpback_BaseColor_*.png")
                           if p.stem.rsplit("_", 1)[1][0] in "ic" and not p.stem.endswith(SUFFIX))
    for key in keys:
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
        print(key, "->", make(key, a.size, a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
