#!/usr/bin/env bash
# run04 end to end, sequential and resumable, meant to be launched detached so that it survives
# a lost session:
#
#   nohup setsid tools/run04_pipeline.sh > data/renders/run04_pipeline.log 2>&1 &
#
# Stages (each leaves a marker in RUN_DIR/pipeline/, re-running skips finished stages):
#   render    tools/build_dataset.sh (itself resumable: rendered batches are skipped)
#   qa        contact sheet + label statistics for a look at the finished dataset
#   train     the Dinov3-WhaleID fine-tunes on run04 (scripts/run04_train.sh in that repo)
#   evaluate  synthetic validation + real-photo metrics (scripts/run04_evaluate.sh there)
# The train/evaluate scripts live in the sibling repo; if they are not there yet when their
# stage comes up, the pipeline waits for them (polling) instead of failing.
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/ue_env.sh

RUN_DIR="${WF_RUN04_DIR:-data/renders/run04}"
COUNT="${WF_RUN04_COUNT:-10000}"
SEED="${WF_RUN04_SEED:-5}"
CONFIG="${WF_RUN04_CONFIG:-factory/config/run04_realism.json}"
# 2x supersampling by default: run04 frames whales down to 8 % of the frame (field distances to 400 m), where
# thin flukes and the waterline alias at one sample per pixel. MRQ_Batch_AA4 is the slower, cleaner option.
export WF_RENDER_CONFIG="${WF_RENDER_CONFIG:-MRQ_Batch_AA2}"
DINO="${WF_DINO_REPO:-$WF_REPO_ROOT/../Dinov3-WhaleID}"
STAGES="$RUN_DIR/pipeline"
mkdir -p "$STAGES"

log() { echo "[run04_pipeline $(date -u +%FT%TZ)] $*"; }

stage() {
    local name="$1"; shift
    if [ -f "$STAGES/$name.done" ]; then
        log "$name: already done ($(cat "$STAGES/$name.done"))"
        return 0
    fi
    log "$name: start"
    if "$@"; then
        date -u +%FT%TZ > "$STAGES/$name.done"
        log "$name: done"
    else
        log "$name: FAILED (exit $?); fix and re-run this script to resume"
        exit 1
    fi
}

wait_for_file() {
    local path="$1" waited=0
    while [ ! -x "$path" ]; do
        if [ "$waited" -eq 0 ]; then log "waiting for $path to appear"; fi
        sleep 300; waited=$((waited + 300))
    done
}

render() {
    tools/build_dataset.sh "$RUN_DIR" "$COUNT" "$SEED" "$CONFIG"
}

qa() {
    wf_python -m factory.qa_sheet "$RUN_DIR" --count 64 --out "$RUN_DIR/dataset/qa_sheet.png"
    wf_python - "$RUN_DIR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
run = Path(sys.argv[1]); rows = [json.loads(l) for l in open(run / "dataset" / "manifest.jsonl")]
flags = Counter(f for r in rows for f in r.get("qa_flags", []))
print(f"run04: {len(rows)} shots, negatives {sum(r['negative'] for r in rows)}, flags {dict(flags)}")
cats = Counter(r["category"] for r in rows); print("categories", dict(cats))
PY
}

train() {
    wait_for_file "$DINO/scripts/run04_train.sh"
    "$DINO/scripts/run04_train.sh" "$WF_REPO_ROOT/$RUN_DIR/dataset"
}

evaluate() {
    wait_for_file "$DINO/scripts/run04_evaluate.sh"
    "$DINO/scripts/run04_evaluate.sh" "$WF_REPO_ROOT/$RUN_DIR/dataset"
}

stage render render
stage qa qa
stage train train
stage evaluate evaluate
log "all stages done"
