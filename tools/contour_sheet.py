#!/usr/bin/env python3
"""Contour review sheet: every processed shot of a run, cropped to the whale, with the body hidden by water /
foam tinted red and the visible mask's boundary coloured by contour class (green = open-air outline,
orange = seam into water or foam, magenta = edge against the calf).

    python3 tools/contour_sheet.py data/renders/RUN out.png [--max 30] [--width 960]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--max", type=int, default=30); ap.add_argument("--width", type=int, default=960); ap.add_argument("--min-pixels", type=int, default=800)
    args = ap.parse_args()
    D = args.run_dir / "dataset"; rows = [json.loads(l) for l in open(D / "manifest.jsonl")]
    tiles = []
    for r in rows:
        if len(tiles) >= args.max:
            break
        meta = json.load(open(D / "meta" / f"{r['id']}.json"))
        mask = np.asarray(Image.open(D / "masks" / f"{r['id']}.png")) == 1
        if mask.sum() < args.min_pixels:
            continue
        amodal = np.asarray(Image.open(D / "amodal" / f"{r['id']}.png")) > 0
        c = np.asarray(Image.open(D / "contours" / f"{r['id']}.png"))
        a = np.asarray(Image.open(D / "images" / f"{r['id']}.png").convert("RGB")).astype(np.float32)
        hidden = amodal & ~mask
        a[hidden] = a[hidden] * 0.45 + np.array([255, 40, 40]) * 0.55
        a = a.astype(np.uint8)
        for cls, col in ((1, (0, 255, 0)), (2, (255, 160, 0)), (3, (255, 0, 255))):
            a[ndimage.binary_dilation(c == cls, iterations=2)] = col
        ys, xs = np.nonzero(amodal | mask)
        x0, x1 = max(0, xs.min() - 40), min(a.shape[1], xs.max() + 40); y0, y1 = max(0, ys.min() - 40), min(a.shape[0], ys.max() + 40)
        t = Image.fromarray(a[y0:y1, x0:x1]); t = t.resize((args.width, max(60, int(args.width * t.height / t.width))))
        ImageDraw.Draw(t).text((6, 4), f"{r['id']} {meta['category']} {meta['sea_state']}  open-air fraction {meta['visible'].get('contour_true_fraction')}", fill=(255, 255, 0))
        tiles.append(t)
    H = sum(t.height + 8 for t in tiles); sheet = Image.new("RGB", (args.width, max(H, 1))); y = 0
    for t in tiles:
        sheet.paste(t, (0, y)); y += t.height + 8
    sheet.save(args.out); print(len(tiles), "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
