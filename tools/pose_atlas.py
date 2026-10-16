#!/usr/bin/env python3
"""Pose atlas: one row per pose category of a run config, sampled shots drawn as side silhouettes at the
waterline (black = above water, blue = under), with the category's weight and share, plus a table of the
mean above-water fraction. Edit "pose_weights" / "pose_jitter" in the config and re-run to tune the mix.

    python3 tools/pose_atlas.py factory/config/run04_realism.json data/renders/pose_atlas_run04.png --per-category 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory import shotlist  # noqa: E402
from factory.whalemodel import WhaleModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--per-category", type=int, default=8); ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cfg = json.loads(args.config.read_text()); m = WhaleModel.from_config(cfg)
    weights = cfg["pose_weights"]; total = sum(weights.values()); cats = sorted(weights, key=lambda k: -weights[k])
    n = args.per_category; rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(cats), n, figsize=(2.3 * n, 1.75 * len(cats)))
    print(f"{'category':12s} weight  share  mean above-water fraction")
    for r, cat in enumerate(cats):
        c = dict(cfg); c["pose_weights"] = {cat: 1}; c["companion"] = dict(cfg.get("companion", {}), fraction=0.0); c["no_whale_in_frame_fraction"] = 0.0
        ups = []
        for k in range(n):
            s = shotlist.make_shot(k, rng, c, m.canonical, None, m if m.kind != "k9239" else None)
            G = m.posed(s); W = shotlist.transform_mesh(G, s["whale"]); z0 = s["water_surface_z_cm"]["whale"]
            yaw = np.radians(s["whale"]["rotation"]["yaw"]); fwd = np.array([np.cos(yaw), np.sin(yaw)]); along = (W[:, :2] - W[:, :2].mean(0)) @ fwd
            above = W[:, 2] > z0; ups.append(above.mean())
            ax = axes[r, k]; ax.scatter(along[::5], W[::5, 2] - z0, s=0.25, c=np.where(above[::5], "k", "#9ec5e8")); ax.axhline(0, color="b", lw=0.5)
            ax.set_aspect("equal"); ax.set_xlim(-1000, 1000); ax.set_ylim(-900, 700); ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f"{cat}\nweight {weights[cat]} ({100 * weights[cat] / total:.0f}%)", fontsize=8)
            ax.set_title(f"t{s['pose'].get('clip_t')} p{s['pose']['pitch_deg']:.0f} r{s['pose']['roll_deg']:.0f} up {above.mean():.2f}", fontsize=6)
        print(f"{cat:12s} {weights[cat]:6d}  {100 * weights[cat] / total:4.0f}%  {np.mean(ups):.2f}")
    plt.tight_layout(); plt.savefig(args.out, dpi=55); print("atlas ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
