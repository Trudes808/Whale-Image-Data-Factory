#!/usr/bin/env bash
# Render a level sequence with Movie Render Queue in game mode, headless on the GPU.
#
#   tools/render_sequence.sh [SEQUENCE] [OUTPUT_DIR]
#
# Defaults: /Game/WhaleFactory/Sequences/LS_BoatPreview -> unreal/WhaleFactory/Saved/Previews/mrq
# Uses the config asset made by `tools/run_editor_script.sh make_render_config`.
# Don't run this while the project is open in the editor in FastX.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

map="/Game/WhaleFactory/Maps/OceanPOC"
sequence="${1:-/Game/WhaleFactory/Sequences/LS_BoatPreview}"
out_dir="$(realpath -m "${2:-$WF_PROJECT_DIR/Saved/Previews/mrq}")"
config="/Game/WhaleFactory/Render/${WF_RENDER_CONFIG:-MRQ_RgbAndLabel}"
log="$WF_PROJECT_DIR/Saved/Logs/wf_render_sequence.log"
mkdir -p "$out_dir" "$(dirname "$log")"

render_dir="$WF_PROJECT_DIR/Saved/MovieRenders/wf"   # set in the config asset
rm -rf -- "$render_dir"
wf_python -m factory.diskguard check --path "$out_dir"
echo "Rendering $sequence -> $out_dir (log: $log) ..."
wf_guarded timeout --signal=TERM "${WF_RENDER_TIMEOUT_S:-3600}" \
    "$WF_REPO_ROOT/tools/ue_exec.sh" "$WF_UE_EDITOR" "$WF_UPROJECT" "$map" -game \
    -LevelSequence="$sequence" -MoviePipelineConfig="$config" \
    -windowed -resx=1280 -resy=720 -RenderOffscreen -unattended -nosplash -nosound \
    -abslog="$log" >/dev/null 2>&1 || true

grep -E "LogMovieRenderPipeline: (Error|Warning)|Movie Pipeline completed|Fatal" "$log" | sed -E 's/^\[[^]]*\]\[[^]]*\]//' || true
shopt -s nullglob
written=("$render_dir"/*.png)
if [ "${#written[@]}" -eq 0 ]; then
    echo "render_sequence: no images written (see $log)" >&2
    exit 1
fi
mv -f -- "${written[@]}" "$out_dir"/
rm -rf -- "$render_dir"
ls -1 "$out_dir"
