#!/usr/bin/env bash
# Render one or more batches from factory/shotlist.py: each batch is keyed into LS_Batch
# by wf_build_batch.py (one editor session) and rendered by Movie Render Queue in one
# game session. Frames land in RUN_DIR/raw/batch_NNN/; factory/postprocess.py turns them
# into the dataset.
#
#   tools/render_batch.sh RUN_DIR [BATCH.json ...]      # default: every batch in RUN_DIR/batches
#
# Skips batches whose raw directory already has a "done" marker, so it can be re-run.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

run_dir="$(realpath -m "${1:?usage: $0 RUN_DIR [BATCH.json ...]}")"
shift
if [ "$#" -eq 0 ]; then
    set -- "$run_dir"/batches/batch_*.json
fi
config="${WF_RENDER_CONFIG:-MRQ_Batch}"
for batch in "$@"; do
    batch="$(realpath -m "$batch")"          # the editor runs with another cwd; batch paths must be absolute
    name="$(basename "${batch%.json}")"
    out_dir="$run_dir/raw/$name"
    if [ -f "$out_dir/done" ]; then
        echo "=== $name already rendered, skipping"
        continue
    fi
    echo "=== $name ==="
    # Deformed whale meshes for this batch (factory/deform.py), kept only while it renders.
    mesh_dir="$run_dir/whales/$name"
    rm -rf -- "$mesh_dir"
    wf_python -m factory.deform "$batch" "$WF_REPO_ROOT/third_party_assets/humpback_k9239/converted/SM_Humpback.obj" "$mesh_dir"
    rm -rf -- "$out_dir"
    mkdir -p -- "$out_dir"
    # build_batch writes the batch as rendered (whale, camera and splash on the wave surface) to raw/<batch>/batch.json.
    WF_BATCH_JSON="$mesh_dir/batch.json" WF_BATCH_SOURCE="$batch" WF_BATCH_ADJUSTED="$out_dir/batch.json" \
        "$WF_REPO_ROOT/tools/run_editor_script.sh" build_batch
    [ -f "$out_dir/batch.json" ] || { echo "render_batch: build_batch wrote no adjusted batch for $name" >&2; exit 1; }
    WF_RENDER_CONFIG="$config" WF_RENDER_TIMEOUT_S="${WF_RENDER_TIMEOUT_S:-7200}" \
        "$WF_REPO_ROOT/tools/render_sequence.sh" /Game/WhaleFactory/Sequences/LS_Batch "$out_dir" | tail -1
    rm -rf -- "$mesh_dir"
    date -u +%FT%TZ > "$out_dir/done"
done
