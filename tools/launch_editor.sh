#!/usr/bin/env bash
# Launch the Unreal Editor on the WhaleFactory project (run inside the FastX desktop).
# Refuses to start below WF_MIN_FREE_GIB free on /home; while running, warns below
# WF_WARN_FREE_GIB and terminates the editor below WF_KILL_FREE_GIB, because a
# 100%-full /home breaks every session on this machine and can corrupt saved assets.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_require_engine

wf_python -m factory.diskguard check --path "$HOME"
wf_guarded_ue "$WF_UE_EDITOR" "$WF_UPROJECT" "$@"
