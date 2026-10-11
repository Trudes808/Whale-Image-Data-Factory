#!/usr/bin/env bash
# Install Epic's precompiled Linux Unreal Engine zip without overfilling /home.
#
#   tools/install_unreal.sh ~/Downloads/Linux_Unreal_Engine_5.x.y.zip [--delete-zip]
#
# 1. Measures the zip's uncompressed size and refuses if extraction would leave
#    less than WF_MIN_FREE_GIB free on the install filesystem.
# 2. Extracts into $WF_UE_INSTALL_BASE/<name>.partial under the disk watchdog.
# 3. Renames it to the engine version from Build.version and points `current` at it.
set -euo pipefail
. "$(dirname "$0")/ue_env.sh"

zip_path="${1:?usage: $0 <Linux_Unreal_Engine_*.zip> [--delete-zip]}"
delete_zip="${2:-}"
[ -f "$zip_path" ] || { echo "no such file: $zip_path" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }

mkdir -p "$WF_UE_INSTALL_BASE"
chmod 700 "$WF_UE_INSTALL_BASE"   # /var/tmp is world-writable; keep the engine private
echo "Checking space for extraction into $WF_UE_INSTALL_BASE ..."
need_gib="$(wf_python -m factory.diskguard zip-check --zip "$zip_path" --dest "$WF_UE_INSTALL_BASE")"

stem="$(basename "$zip_path" .zip)"
partial="$WF_UE_INSTALL_BASE/$stem.partial"
if [ -e "$partial" ]; then
    echo "Removing leftover partial extraction: $partial"
    rm -rf -- "$partial"
fi
mkdir -p "$partial"

echo "Extracting ~${need_gib} GiB (this takes a while) ..."
# unzip (not Python's zipfile) so executable bits on engine binaries are preserved.
if ! wf_guarded unzip -q "$zip_path" -d "$partial"; then
    echo "Extraction failed or was stopped for low disk; removing $partial" >&2
    rm -rf -- "$partial"
    exit 1
fi

# The engine root is the directory that contains Engine/Binaries/Linux/UnrealEditor.
editor="$(find "$partial" -maxdepth 5 -path '*/Engine/Binaries/Linux/UnrealEditor' -type f | head -n 1)"
[ -n "$editor" ] || { echo "UnrealEditor binary not found in the zip" >&2; exit 1; }
engine_root="$(dirname "$(dirname "$(dirname "$(dirname "$editor")")")")"

version="$(python3 - "$engine_root/Engine/Build/Build.version" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
print(f"{v['MajorVersion']}.{v['MinorVersion']}.{v['PatchVersion']}")
EOF
)"
final="$WF_UE_INSTALL_BASE/UE_$version"
[ -e "$final" ] && { echo "$final already exists; leaving $partial for you to inspect" >&2; exit 1; }

if [ "$engine_root" = "$partial" ]; then
    mv -- "$partial" "$final"
else
    mv -- "$engine_root" "$final"
    rm -rf -- "$partial"
fi
ln -sfn "UE_$version" "$WF_UE_INSTALL_BASE/current"
echo "Installed Unreal Engine $version at $final (current -> UE_$version)"

if [ "$delete_zip" = "--delete-zip" ]; then
    rm -f -- "$zip_path"
    echo "Deleted $zip_path"
else
    echo "The zip is still at $zip_path ($(du -h "$zip_path" | cut -f1)); once the editor opens, free that space: rm '$zip_path'"
fi
wf_python -m factory.diskguard report
