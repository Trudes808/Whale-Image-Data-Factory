#!/usr/bin/env bash
# run05 overnight, detached and resumable:
#   1. review render r05_individuals4 (108 frames, 36 adults x 3 poses) -> postprocess -> docs/run05_review pages
#   2. run05 full dataset: tools/build_dataset.sh (render -> postprocess -> dataset), 10 000 shots, seed 6, AA2
#   3. QA sheet
# A janitor clears the engine's Zen DDC (/var/tmp/sat3737-unreal/ZenData, ~0.5 GB per batch) between editor sessions
# when the root disk runs low, so the run does not stop on the diskguard floor.
#   nohup setsid tools/run05_overnight.sh > data/renders/run05_overnight.log 2>&1 &
set -uo pipefail
FACT=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory
DINO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$FACT"
. tools/ue_env.sh
export WF_RENDER_CONFIG="${WF_RENDER_CONFIG:-MRQ_Batch_AA2}"
log() { echo "[run05_overnight $(date -u +%FT%TZ)] $*"; }

janitor() {
    while true; do
        sleep 20
        free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
        if [ "${free_gb:-999}" -lt 45 ] && ! pgrep -f "^[^ ]*UnrealEditor" >/dev/null && [ -d /var/tmp/sat3737-unreal/ZenData ]; then
            rm -rf /var/tmp/sat3737-unreal/ZenData && log "janitor: cleared ZenData (root had ${free_gb} GB free)"
        fi
    done
}
janitor &
JANITOR=$!
trap 'kill $JANITOR 2>/dev/null' EXIT

# --- 1. review render with the corrected dorsal-fin landmarks
REVIEW=$FACT/data/renders/r05_individuals4
if [ ! -f "$REVIEW/individuals_sheet_p06.png" ]; then
    log "review render: $REVIEW"
    mkdir -p "$REVIEW"
    [ -f "$REVIEW/shotlist.jsonl" ] || wf_python -m factory.shotlist factory/config/run05_individuals_test.json third_party_assets/humpback_k9239/converted/SM_Humpback.obj -o "$REVIEW" --count 108 --seed 7 2>&1 | grep -v -i warn | tail -1
    tools/render_batch.sh "$REVIEW" 2>&1 | grep -E "^===|FAILED|Error" | tail -20
    wf_python -m factory.postprocess "$REVIEW" --workers 8 2>&1 | grep -v -i warn | tail -1
    wf_python -m factory.dataset "$REVIEW" 2>&1 | grep -v -i warn | tail -1
    "$DINO/.venv/bin/python" tools/individuals_sheet.py "$REVIEW" --cols 6 --tile 420 --per-page 6 --library third_party_assets/humpback_4k/individuals.json 2>&1 | grep -v -i warn | tail -1
    mkdir -p docs/run05_review && cp "$REVIEW"/individuals_sheet_p0*.png docs/run05_review/ && log "review pages -> docs/run05_review/"
fi

# --- 2. run05 full dataset (resumable: rendered batches are skipped)
# the run03 / run04 raw frames are being moved to the root disk to make room on /home: wait for that to finish
while [ -f /var/tmp/sat3737-whale-raw/move.log ] && ! grep -q "run03 raw moved" /var/tmp/sat3737-whale-raw/move.log; do sleep 60; done
log "/home free: $(df -h --output=avail /home | tail -1)"
RUN=$FACT/data/renders/run05
log "run05 build: $RUN (10000 shots, seed 6, factory/config/run05_realism.json, $WF_RENDER_CONFIG)"
WF_POSTPROCESS_WORKERS=16 tools/build_dataset.sh "$RUN" 10000 6 "$FACT/factory/config/run05_realism.json" 2>&1 | grep -vE "LogPython|^\s*$" | grep -E "^===|FAILED|Error|build_dataset|manifest|diskguard|retry" | tail -400
status=$?
log "build_dataset exit $status"

# --- 3. QA
if [ -f "$RUN/dataset/manifest.jsonl" ]; then
    wf_python -m factory.qa_sheet "$RUN" --count 64 --out "$RUN/dataset/qa_sheet.png" 2>&1 | grep -v -i warn | tail -1
    log "qa sheet: $RUN/dataset/qa_sheet.png"
fi
log "done"
