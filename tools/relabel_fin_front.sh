#!/usr/bin/env bash
# Refresh the dorsal_fin_front keypoint of the processed runs in place (postprocess --keypoints-only, revision
# 2026-11-02-fin-front: the front of the whole fin / hump instead of the blade's root; see WhaleModel.DORSAL_FRONT_RATIO).
# Saves the previous entries to dataset/keypoint_prev_dorsal_fin_front.json first. run04 is labelled with its own
# individual library and fin code (WF_INDIVIDUALS, WF_LEGACY_DORSAL), as for its 2026-10-21 refresh.
#   nohup setsid tools/relabel_fin_front.sh > data/renders/relabel_fin_front.log 2>&1 &
set -uo pipefail
FACT=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory
cd "$FACT"; export PYTHONPATH="$FACT${PYTHONPATH:+:$PYTHONPATH}"
WORKERS="${WF_RELABEL_WORKERS:-16}"
log() { echo "[relabel_fin_front $(date -u +%FT%TZ)] $*"; }
save_prev() {
    python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
ds = Path(sys.argv[1]) / "dataset"; out = ds / "keypoint_prev_dorsal_fin_front.json"
if out.exists():
    print("previous entries already saved:", out); sys.exit(0)
prev = {}
for p in sorted((ds / "meta").glob("*.json")):
    m = json.loads(p.read_text())
    if m.get("keypoint_revision"):
        continue
    k = (m.get("keypoints") or {}).get("dorsal_fin_front")
    if k:
        prev[p.stem] = k
out.write_text(json.dumps({"note": "dorsal_fin_front before revision 2026-11-02-fin-front (the blade's root)", "entries": prev}))
print("saved", len(prev), "previous entries ->", out)
PY
}
for RUN in run05 run04; do
    DIR=$FACT/data/renders/$RUN
    if [ -f "$DIR/dataset/keypoint_revision.json" ] && grep -q '"dry_run": false' "$DIR/dataset/keypoint_revision.json"; then
        log "$RUN already relabelled"; continue
    fi
    log "$RUN: saving previous entries"; save_prev "$DIR"
    log "$RUN: relabel ($WORKERS workers)"
    if [ "$RUN" = run04 ]; then
        export WF_INDIVIDUALS=$FACT/third_party_assets/humpback_4k/individuals_run04.json WF_LEGACY_DORSAL=1
    else
        unset WF_INDIVIDUALS WF_LEGACY_DORSAL
    fi
    python3 -m factory.postprocess "$DIR" --keypoints-only --workers "$WORKERS" 2>&1 | grep -v -i warn | grep -v '^\s*\[$\|^\s*\]' | tail -30
    log "$RUN: exit ${PIPESTATUS[0]}"
done
log "done"
