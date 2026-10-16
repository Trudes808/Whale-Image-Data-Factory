"""Skin variants: different individuals' pigmentation, scars and barnacles on the base
colour texture, so neither the segmenter nor a later feature-matching model keys on one
whale's skin.

    python3 -m factory.skin third_party_assets/humpback_k9239/converted/T_Humpback_BaseColor.png OUT_DIR --count 8 --seed 1

Each variant is a seeded edit of the 4K base colour map:
    darkness and hue     overall body tone from near-black to dark brown-grey
    mottling             low-frequency lightening in patches (pigment variation)
    rake marks           sets of 3-6 parallel thin pale lines (orca tooth rakes)
    round scars          small pale discs (cookie-cutter shark bites, healed wounds)
    barnacle clusters    clusters of small grey-white rings (acorn barnacles)
Edits are applied in texture space; they land wherever the UV islands put them, which is
fine for variety (exact anatomical placement needs the UV layout, a later refinement).
Variant 0 is the untouched base texture. Each variant's recipe is saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _noise(rng: np.random.Generator, shape, scale: int) -> np.ndarray:
    small = rng.uniform(0, 1, size=(shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    up = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BICUBIC)) / 255.0
    return up


def make_variant(base: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    h, w = base.shape[:2]
    img = base.astype(np.float32) / 255.0
    recipe = {"seed": seed}
    # Tone: darkness multiplier and a slight hue shift (brownish or bluish black).
    darkness = float(rng.uniform(0.7, 1.35))
    tint = np.array([rng.uniform(0.92, 1.1), rng.uniform(0.95, 1.05), rng.uniform(0.92, 1.12)], np.float32)
    recipe.update(darkness=round(darkness, 3), tint=[round(float(t), 3) for t in tint])
    dark = img.mean(axis=2, keepdims=True) < 0.35          # only the dark skin, keep the white areas white
    img = np.where(dark, np.clip(img * darkness * tint, 0, 1), img)
    # Mottling: patches of lighter pigment.
    amount = float(rng.uniform(0.0, 0.35))
    mottle = _noise(rng, (h, w), scale=int(rng.integers(120, 400)))
    mottle = np.clip((mottle - 0.55) * 3.0, 0, 1)[..., None] * amount
    img = np.clip(img + mottle * np.array([0.45, 0.45, 0.47], np.float32) * dark, 0, 1)
    recipe["mottling"] = round(amount, 3)
    # Scars and barnacles drawn on an overlay layer.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n_rakes = int(rng.integers(0, 5))
    for _ in range(n_rakes):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.6) * h
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(0.02, 0.08) * w
        spacing = rng.uniform(6, 16)
        for k in range(int(rng.integers(3, 7))):
            off = (k - 2) * spacing
            ox, oy = -np.sin(angle) * off, np.cos(angle) * off
            x0, y0 = cx + ox - np.cos(angle) * length / 2, cy + oy - np.sin(angle) * length / 2
            x1, y1 = cx + ox + np.cos(angle) * length / 2, cy + oy + np.sin(angle) * length / 2
            draw.line([(x0, y0), (x1, y1)], fill=(235, 230, 225, int(rng.uniform(120, 220))), width=int(rng.uniform(2, 5)))
    n_round = int(rng.integers(0, 12))
    for _ in range(n_round):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.6) * h
        r = rng.uniform(6, 22)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 215, 210, int(rng.uniform(100, 200))))
    n_barnacle_clusters = int(rng.integers(0, 6))
    for _ in range(n_barnacle_clusters):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h
        for _ in range(int(rng.integers(6, 30))):
            px, py = cx + rng.normal(0, 40), cy + rng.normal(0, 40)
            r = int(rng.integers(3, 10))
            # a healed round scar: soft-edged, mottled, slightly off-white with a darker rim, never a solid disc
            scar = Image.new("RGBA", (4 * r + 2, 4 * r + 2), (0, 0, 0, 0))
            sd = ImageDraw.Draw(scar)
            sd.ellipse([r, r, 3 * r, 3 * r], fill=(205, 200, 192, 200), outline=(95, 88, 82, 160), width=max(1, r // 6))
            scar = scar.filter(ImageFilter.GaussianBlur(max(1.0, r / 3)))
            sa = np.asarray(scar).astype(np.float32)
            grain = rng.uniform(0.55, 1.0, size=sa.shape[:2])[..., None]
            sa[..., 3] *= grain[..., 0]
            scar = Image.fromarray(np.clip(sa, 0, 255).astype(np.uint8))
            overlay.alpha_composite(scar, (int(px - 2 * r - 1), int(py - 2 * r - 1)))
    recipe.update(rake_marks=n_rakes, round_scars=n_round, barnacle_clusters=n_barnacle_clusters)
    overlay = overlay.filter(ImageFilter.GaussianBlur(0.8))
    ov = np.asarray(overlay).astype(np.float32) / 255.0
    alpha = ov[..., 3:4] * dark  # marks only show on dark skin
    img = img * (1 - alpha) + ov[..., :3] * alpha
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_texture", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--count", type=int, default=8, help="number of variants including the untouched base")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(Image.open(args.base_texture).convert("RGB"))
    recipes = {}
    for k in range(args.count):
        if k == 0:
            image, recipe = base, {"base": True}
        else:
            image, recipe = make_variant(base, args.seed * 1000 + k)
        path = args.out_dir / f"T_Humpback_BaseColor_{k:02d}.png"
        tmp = path.with_suffix(".tmp.png"); Image.fromarray(image).save(tmp); tmp.replace(path)   # atomic for concurrent imports
        recipes[f"{k:02d}"] = recipe
        print(path, recipe)
    (args.out_dir / "skins.json").write_text(json.dumps(recipes, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
