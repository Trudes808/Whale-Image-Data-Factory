#!/usr/bin/env bash
# Finish run05 after the disk-floor stop: wait for the space-making move, render the remaining batches (done ones are
# skipped), postprocess, split, QA sheet. Same Zen DDC janitor as run05_overnight.sh. Detached + resumable:
#   nohup setsid tools/run05_finish.sh > data/renders/run05_finish.log 2>&1 &
set -uo pipefail
FACT=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory
cd "$FACT"; . tools/ue_env.sh
export WF_RENDER_CONFIG="${WF_RENDER_CONFIG:-MRQ_Batch_AA2}"
RUN=$FACT/data/renders/run05
log() { echo "[run05_finish $(date -u +%FT%TZ)] $*"; }
janitor() {
    while true; do
        sleep 20
        free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
        if [ "${free_gb:-999}" -lt 45 ] && ! pgrep -f "^[^ ]*UnrealEditor" >/dev/null && [ -d /var/tmp/sat3737-unreal/ZenData ]; then
            rm -rf /var/tmp/sat3737-unreal/ZenData && log "janitor: cleared ZenData (root had ${free_gb} GB free)"
        fi
    done
}
janitor & JANITOR=$!; trap 'kill $JANITOR 2>/dev/null' EXIT
while ! grep -q "ALL MOVED" /var/tmp/sat3737-whale-raw/move2.log 2>/dev/null; do sleep 30; done
log "space: /home $(df -h --output=avail /home | tail -1) free, root $(df -h --output=avail / | tail -1) free"
log "render: remaining batches"
tools/render_batch.sh "$RUN" 2>&1 | grep -E "^===|FAILED|REFUSED|retry" | tail -80
n_done=$(ls "$RUN"/raw/*/done 2>/dev/null | wc -l); n_all=$(ls "$RUN"/batches/batch_*.json | wc -l)
log "rendered $n_done / $n_all batches"
if [ "$n_done" -lt "$n_all" ]; then log "not all batches rendered; stopping (re-run to resume)"; exit 1; fi
log "postprocess"
wf_python -m factory.postprocess "$RUN" --workers "${WF_POSTPROCESS_WORKERS:-16}" 2>&1 | grep -v -i warn | tail -3
wf_python -m factory.dataset "$RUN" 2>&1 | grep -v -i warn | tail -2
wf_python -m factory.qa_sheet "$RUN" --count 64 --out "$RUN/dataset/qa_sheet.png" 2>&1 | grep -v -i warn | tail -1
log "done: $(ls "$RUN"/dataset/meta | wc -l) shots in $RUN/dataset"
touch "$RUN/dataset/.finished"
