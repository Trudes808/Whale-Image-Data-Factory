"""Paired render test: copy the shot definitions of an existing run (shot list, run config, batches and the OBJ meshes
the shot list wrote: splash, white water, islands) into a new run folder, keeping only the first N shot ids, so the new
folder renders exactly the same shots (same whales, poses, cameras, sun, sea, white water) under different render
settings (for example r.Water.SingleLayer.Reflection=1 vs 2). Nothing is re-sampled: the JSON is copied with the old run
folder's absolute path replaced by the new one, and batches keep their names (batch_007 here = batch_007 there, minus the
shots past N). The source run is only read; its batches need not be rendered yet.

    python3 tools/paired_subset.py SRC_RUN DST_RUN --first 1000
then render DST_RUN with tools/build_dataset.sh DST_RUN (it keeps an existing shotlist.jsonl and skips sampling).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--first", type=int, required=True)
    a = ap.parse_args(argv)
    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    if (dst / "shotlist.jsonl").exists():
        raise SystemExit(f"{dst} already has a shot list")
    lines = [l for l in (src / "shotlist.jsonl").read_text().splitlines() if l.strip()]
    keep = [json.loads(l)["id"] for l in lines[: a.first]]
    want = set(keep)
    s_src, s_dst = str(src), str(dst)
    dst.mkdir(parents=True, exist_ok=True); (dst / "batches").mkdir(exist_ok=True)
    (dst / "shotlist.jsonl").write_text("".join(l.replace(s_src, s_dst) + "\n" for l in lines[: a.first]))
    (dst / "run_config.json").write_text((src / "run_config.json").read_text().replace(s_src, s_dst))
    n_b = n_s = 0
    for bp in sorted((src / "batches").glob("batch_*.json")):
        b = json.loads(bp.read_text().replace(s_src, s_dst))
        b["shots"] = [s for s in b["shots"] if s["id"] in want]
        if b["shots"]:
            (dst / "batches" / bp.name).write_text(json.dumps(b))
            n_b += 1; n_s += len(b["shots"])
    n_f = 0
    for sub in ("splash", "whitewater", "islands"):
        if not (src / sub).is_dir():
            continue
        (dst / sub).mkdir(exist_ok=True)
        for f in (src / sub).iterdir():
            # per-shot meshes are named after the shot id; shared ones (islands library) are copied whole
            sid = f.name[:6]
            if f.is_file() and (not sid.isdigit() or sid in want):
                shutil.copy2(f, dst / sub / f.name); n_f += 1
    print(f"{len(keep)} shots ({keep[0]}..{keep[-1]}) in {n_b} batches ({n_s} shots) and {n_f} mesh files -> {dst}")
    return 0 if n_s == len(keep) else 1


if __name__ == "__main__":
    raise SystemExit(main())
