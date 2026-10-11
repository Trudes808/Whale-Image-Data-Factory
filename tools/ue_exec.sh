#!/usr/bin/env bash
# exec an Unreal binary with the factory's environment: no core dumps (multi-GiB on a
# crash), the local derived-data cache pinned inside the project, and the Zen cache
# server's data on the root disk (see WF_ZEN_DATA_DIR in ue_env.sh). Used by the other
# tools via wf_guarded; the DDC variable name contains '-', so it must go through `env`.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
ulimit -c 0
mkdir -p "$WF_DDC_DIR" "$WF_ZEN_DATA_DIR"
# -ZenDataPath is the override UE 5.8 verifiably honours (the UE-ZenDataPath env var
# alone was not picked up in testing).
exec env "UE-LocalDataCachePath=$WF_DDC_DIR" "UE-ZenDataPath=$WF_ZEN_DATA_DIR" \
    "$@" -ZenDataPath="$WF_ZEN_DATA_DIR"
