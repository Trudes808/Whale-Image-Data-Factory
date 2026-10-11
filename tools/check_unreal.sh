#!/usr/bin/env bash
# Headless sanity check: open the project with the Python commandlet (no GPU needed)
# and run Content/Python/wf_env_check.py (engine version, plugins, custom depth, disk).
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

wf_python -m factory.diskguard check --path "$HOME"
mkdir -p "$WF_PROJECT_DIR/Saved"
log="$WF_PROJECT_DIR/Saved/wf_env_check.log"

wf_guarded_ue "$WF_UE_EDITOR_CMD" "$WF_UPROJECT" \
    -run=pythonscript -script="$WF_PROJECT_DIR/Content/Python/wf_env_check.py" \
    -unattended -nullrhi -nosplash -stdout -FullStdOutLogOutput 2>&1 | tee "$log" \
    | grep -E '\[wf_env_check\]|Error:' || true

if grep -q '\[wf_env_check\] OK' "$log"; then
    echo "Unreal check passed (full log: $log)"
else
    echo "Unreal check FAILED (full log: $log)" >&2
    exit 1
fi
