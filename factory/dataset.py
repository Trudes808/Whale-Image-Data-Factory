"""Train/val split and summary for a processed run.

    python3 -m factory.dataset data/renders/run01 [--val-fraction 0.15] [--seed 1]

Splits by *batch* (sea state + water look group), so near-duplicate conditions do not
leak between train and val, while keeping every pose category in both. Writes
RUN_DIR/dataset/splits.json and prints a distribution summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_manifest(run_dir: Path) -> list[dict]:
    with open(run_dir / "dataset" / "manifest.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split(rows: list[dict], val_fraction: float, seed: int) -> dict:
    """Group ids by (sea_state, look); whole groups go to val until the fraction is met."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        groups[(row["sea_state"], row["look"])].append(row["id"])
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    target = val_fraction * len(rows)
    val, count = [], 0
    for key in keys:
        if count >= target:
            break
        val.extend(groups[key])
        count += len(groups[key])
    val_set = set(val)
    return {"train": sorted(r["id"] for r in rows if r["id"] not in val_set), "val": sorted(val)}


def summary(rows: list[dict]) -> dict:
    def hist(values, edges):
        counts, _ = np.histogram(values, bins=edges)
        return {f"{edges[i]:g}-{edges[i + 1]:g}": int(c) for i, c in enumerate(counts)}

    return {
        "shots": len(rows),
        "negatives": sum(r["negative"] for r in rows),
        "flagged": sum(bool(r["qa_flags"]) for r in rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "sea_states": dict(Counter(r["sea_state"] for r in rows)),
        "looks": dict(Counter(r["look"] for r in rows)),
        "visible_fraction_of_whale": hist([r["visible_fraction_of_whale"] for r in rows], [0, 0.001, 0.05, 0.2, 0.5, 0.8, 1.01]),
        "distance_m": hist([r["orientation"]["distance_m"] for r in rows], [0, 25, 50, 100, 200]),
        "relative_bearing_deg": hist([r["orientation"]["relative_bearing_deg"] for r in rows], [0, 90, 180, 270, 360]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    rows = load_manifest(args.run_dir)
    splits = split(rows, args.val_fraction, args.seed)
    (args.run_dir / "dataset" / "splits.json").write_text(json.dumps(splits, indent=1))
    report = summary(rows)
    report["train"], report["val"] = len(splits["train"]), len(splits["val"])
    (args.run_dir / "dataset" / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
