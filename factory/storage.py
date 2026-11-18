"""Frugal storage switches (run06 realism; config "storage", defaults keep the old behaviour):

    "storage": {"raw_frames": "last",          # "all" (default): keep every rendered frame; "last": after a batch is
                                               # rendered and verified, delete the warm-up frames (beauty and label pass)
                "raw_after_postprocess": "delete",   # "keep" (default); "delete": tools/build_dataset.sh removes the raw frames
                                               # of every fully post-processed batch (no re-labelling from raw afterwards)
                "dense_body_maps": false}      # true (default): canon/, canon_deformed/ and the per-animal canon / depth
                                               # crops in animals/; false: not written (shot list, meta and meshes suffice
                                               # to compute them later: factory.postprocess with the switch on)

    python3 -m factory.storage prune-raw RUN_DIR BATCH_NAME     # called by tools/render_batch.sh after each batch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PASSES = ("FinalImage", "FinalImageM_WhaleVisibleMask", "FinalImageM_WhaleInstanceMask")


def settings(run_dir: Path) -> dict:
    try:
        return (json.loads((Path(run_dir) / "run_config.json").read_text()).get("config") or {}).get("storage") or {}
    except FileNotFoundError:
        return {}


def prune_raw(run_dir: Path, batch_name: str) -> dict:
    """Delete the warm-up frames of a rendered batch when the config asks for it, after checking that the kept frame of
    every shot exists for the beauty pass and a label pass. Returns counts; deletes nothing on any doubt."""
    st = settings(run_dir)
    if st.get("raw_frames", "all") != "last":
        return {"pruned": 0, "reason": "storage.raw_frames is not 'last'"}
    raw = Path(run_dir) / "raw" / batch_name
    if not (raw / "done").exists():
        return {"pruned": 0, "reason": "batch not done"}
    batch = json.loads((raw / "batch.json").read_text())
    per = int(batch.get("frames_per_shot", 3))
    keep = {i * per + per - 1 for i in range(len(batch["shots"]))}
    for f in sorted(keep):
        beauty = raw / f"FinalImage.{f:04d}.png"
        label = [raw / f"{p}.{f:04d}.png" for p in PASSES[1:]]
        if not beauty.exists() or beauty.stat().st_size == 0 or not any(l.exists() and l.stat().st_size > 0 for l in label):
            return {"pruned": 0, "reason": f"frame {f} incomplete: nothing deleted"}
    n = 0
    for path in raw.glob("*.png"):
        stem, _, num = path.stem.rpartition(".")
        if stem in PASSES and num.isdigit() and int(num) not in keep:
            path.unlink(); n += 1
    return {"pruned": n, "kept_frames": len(keep)}


def drop_raw(run_dir: Path) -> dict:
    """storage.raw_after_postprocess = "delete": remove the rendered frames of every batch whose shots all have their
    dataset image, mask and meta (batch.json and the done marker stay, so a re-run does not render them again)."""
    st = settings(run_dir)
    if st.get("raw_after_postprocess", "keep") != "delete":
        return {"dropped_batches": 0, "reason": "storage.raw_after_postprocess is not 'delete'"}
    ds = Path(run_dir) / "dataset"
    n_b = n_f = 0
    for raw in sorted((Path(run_dir) / "raw").glob("batch_*")):
        if not (raw / "done").exists():
            continue
        ids = [s["id"] for s in json.loads((raw / "batch.json").read_text())["shots"]]
        if not all((ds / "meta" / f"{i}.json").exists() and (ds / "images" / f"{i}.png").exists() and (ds / "masks" / f"{i}.png").exists() for i in ids):
            continue
        for path in raw.glob("*.png"):
            path.unlink(); n_f += 1
        n_b += 1
    return {"dropped_batches": n_b, "deleted_frames": n_f}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prune-raw":
        print(prune_raw(Path(sys.argv[2]), sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "drop-raw":
        print(drop_raw(Path(sys.argv[2])))
    else:
        sys.exit(__doc__)
