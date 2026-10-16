#!/usr/bin/env python3
"""Water consistency check: does the factory's frozen Gerstner surface agree with what the render showed?
For every processed shot, predict the visible whale from the geometry alone (whole-whale projection, hidden
where the whale point is under the local surface or where the camera ray to it passes under a crest, marched
through the band near the whale) and compare with the rendered visible mask. Agreement (IoU >= ~0.85 for
whales of a few thousand pixels) means lift, white water, through-water labels and contour classes are
computed on the water the render used; disagreement means the waves or the wave time differ.

    python3 tools/water_check.py data/renders/RUN [--stride 3] [--steps 48]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from factory import gerstner, project  # noqa: E402
from factory.whalemodel import WhaleModel  # noqa: E402


def visible_by_raymarch(points_world: np.ndarray, cam_loc, waves, t: float, steps: int, band_cm: float) -> np.ndarray:
    C = np.asarray(cam_loc, float); D = points_world - C; L = np.linalg.norm(D, axis=1); dirn = D / np.maximum(L, 1e-6)[:, None]
    hidden = points_world[:, 2] < gerstner.surface_height(waves, points_world[:, 0], points_world[:, 1], t)
    for i in range(1, steps + 1):
        S = points_world - dirn * (np.minimum(L, band_cm) * i / steps)[:, None]
        hidden |= S[:, 2] < gerstner.surface_height(waves, S[:, 0], S[:, 1], t)
    return ~hidden


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path); ap.add_argument("--stride", type=int, default=3); ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--band-cm", type=float, default=2500.0); ap.add_argument("--min-pixels", type=int, default=1500)
    args = ap.parse_args()
    D = args.run_dir / "dataset"; run_config = json.loads((args.run_dir / "run_config.json").read_text())
    m = WhaleModel.from_config(run_config.get("config", {}))
    rows = [json.loads(l) for l in open(D / "manifest.jsonl")]
    ious = []
    print("shot    category     sea       IoU   render-only  factory-only")
    for r in rows:
        meta = json.load(open(D / "meta" / f"{r['id']}.json")); mask = np.asarray(Image.open(D / "masks" / f"{r['id']}.png")) == 1
        if mask.sum() < args.min_pixels or not meta.get("waves"):
            continue
        amodal = np.asarray(Image.open(D / "amodal" / f"{r['id']}.png")) > 0
        G = m.posed(meta); cam = project.Camera(meta["camera"]["location"], meta["camera"]["rotation"], meta["camera"]["focal_mm"], mask.shape[1], mask.shape[0])
        _, _, _, _, _, cd = project.amodal_and_parts(G, m.faces, meta["whale"], cam, extra=True, canonical_cm=m.canonical, world_z=True, canonical_xyz=True, deformed_xyz=True)
        s = args.stride
        sub = np.zeros(mask.shape, bool); sub[::s, ::s] = True
        sel = amodal & sub & ~np.isnan(cd[..., 0])
        pts = project.transform_mesh(cd[sel].reshape(-1, 3).astype(np.float64), meta["whale"])
        vis = visible_by_raymarch(pts, meta["camera"]["location"], gerstner.generate_waves(meta["waves"]), float(meta["wave_time_s"]), args.steps, args.band_cm)
        pred = np.zeros(mask.shape, bool); pred[sel] = vis
        rend = mask & sub
        iou = (pred & rend).sum() / max((pred | rend).sum(), 1); ious.append(iou)
        print(f"{r['id']}  {meta['category']:11s} {meta['sea_state']:9s} {iou:.2f}   {(rend & ~pred).sum() / max(rend.sum(), 1):.2f}         {(pred & ~rend).sum() / max(pred.sum(), 1):.2f}")
    if ious:
        ious = np.array(ious); print(f"\n{len(ious)} shots: IoU median {np.median(ious):.2f}, p10 {np.percentile(ious, 10):.2f}, fraction >= 0.85: {(ious >= 0.85).mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
