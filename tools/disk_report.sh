#!/usr/bin/env bash
# Show free space and how much each Unreal/factory directory is using.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"
wf_python -m factory.diskguard report
