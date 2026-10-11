#!/usr/bin/env bash
# Run Content/Python/wf_<name>.py in the full Unreal Editor, headless on the GPU
# (-RenderOffscreen: no display needed). The script must quit the editor itself when
# WF_QUIT_WHEN_DONE is set (-ExecutePythonScript is avoided: it exits after one frame).
#
#   tools/run_editor_script.sh build_scene
#   tools/run_editor_script.sh preview
#
# Don't run this while the same project is open in the editor in FastX.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

name="${1:?usage: $0 <script name, e.g. build_scene>}"
shift
script="$WF_PROJECT_DIR/Content/Python/wf_$name.py"
[ -f "$script" ] || { echo "no such script: $script" >&2; exit 1; }
log="$WF_PROJECT_DIR/Saved/Logs/wf_$name.log"
mkdir -p "$(dirname "$log")"

wf_python -m factory.diskguard check --path "$HOME"
echo "Running $script headless (log: $log) ..."
WF_QUIT_WHEN_DONE=1 wf_guarded timeout --signal=TERM "${WF_SCRIPT_TIMEOUT_S:-3600}" \
    "$WF_REPO_ROOT/tools/ue_exec.sh" "$WF_UE_EDITOR" "$WF_UPROJECT" \
    -RenderOffscreen -unattended -nosplash -nosound -abslog="$log" \
    -ExecCmds="py $script" "$@" >/dev/null 2>&1 || true

grep -E "\[wf_$name\]|LogPython: Error|Traceback|Error: " "$log" | sed -E 's/^\[[^]]*\]\[[^]]*\]//' || true
if grep -q "\[wf_$name\] OK" "$log"; then
    echo "wf_$name: OK"
else
    echo "wf_$name: FAILED (see $log)" >&2
    exit 1
fi
