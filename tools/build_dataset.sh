#!/usr/bin/env bash
# End to end: sample a shot list, render every batch, post-process, split.
#
#   tools/build_dataset.sh RUN_DIR [COUNT] [SEED] [CONFIG]
#   nohup setsid tools/build_dataset.sh data/renders/run01 2000 1 > data/renders/run01.log 2>&1 &
#
# Re-running resumes: an existing shot list is kept and rendered batches are skipped.
# Needs the scene built (tools/run_editor_script.sh build_scene) and the render configs
# (tools/run_editor_script.sh make_render_config). Don't open the project in the editor
# while this runs.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

run_dir="$(realpath -m "${1:?usage: $0 RUN_DIR [COUNT] [SEED] [CONFIG]}")"
count="${2:-2000}"
seed="${3:-1}"
config="${4:-$WF_REPO_ROOT/factory/config/poc_boat.json}"
mesh="$WF_REPO_ROOT/third_party_assets/humpback_k9239/converted/SM_Humpback.obj"

echo "[$(date -u +%FT%TZ)] build_dataset: $run_dir count=$count seed=$seed"
if [ ! -f "$run_dir/shotlist.jsonl" ]; then
    wf_python -m factory.shotlist "$config" "$mesh" -o "$run_dir" --count "$count" --seed "$seed"
fi
# Upper bound on disk: 3 frames x (RGB ~1 MiB + mask) per shot raw, plus the dataset copy.
wf_python -m factory.diskguard check --path "$run_dir" --need-gib "$(( count * 5 / 1000 + 2 ))"
"$WF_REPO_ROOT/tools/render_batch.sh" "$run_dir"
wf_python -m factory.postprocess "$run_dir" --workers "${WF_POSTPROCESS_WORKERS:-16}"
wf_python -m factory.dataset "$run_dir"
echo "[$(date -u +%FT%TZ)] build_dataset: done -> $run_dir/dataset"
