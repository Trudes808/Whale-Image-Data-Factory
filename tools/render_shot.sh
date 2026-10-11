#!/usr/bin/env bash
# Apply one or more shot JSON files (see factory/poses.py) to the scene and render each
# with Movie Render Queue: RGB + label pass from the same frame, plus an overlay PNG.
#
#   tools/render_shot.sh OUT_DIR SHOT.json [SHOT.json ...]
#   tools/render_shot.sh data/shots/renders data/shots/poses/*.json
#
# Each shot renders to OUT_DIR/<shot name>/. Set WF_RENDER_CONFIG=MRQ_RgbOnly to skip the
# label pass. Don't run this while the project is open in the editor in FastX.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

out_root="${1:?usage: $0 OUT_DIR SHOT.json [SHOT.json ...]}"
shift
[ "$#" -gt 0 ] || { echo "no shot files given" >&2; exit 1; }
mkdir -p "$out_root"

for shot in "$@"; do
    name="$(basename "${shot%.json}")"
    out_dir="$out_root/$name"
    echo "=== shot $name ==="
    WF_SHOT_JSON="$(realpath "$shot")" "$WF_REPO_ROOT/tools/run_editor_script.sh" apply_shot
    "$WF_REPO_ROOT/tools/render_sequence.sh" /Game/WhaleFactory/Sequences/LS_BoatPreview "$out_dir"
    cp -f -- "$shot" "$out_dir/shot.json"
    last_rgb="$(ls "$out_dir"/FinalImage.*.png | tail -1)"
    last_mask="$(ls "$out_dir"/FinalImageM_WhaleVisibleMask.*.png 2>/dev/null | tail -1 || true)"
    if [ -n "$last_mask" ]; then
        wf_python -m factory.qa overlay "$last_rgb" "$last_mask" "$out_dir/overlay.png"
    fi
done
