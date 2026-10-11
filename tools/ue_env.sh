# Shared environment for tools/*.sh. Source it; don't execute it.
# Override any value in tools/ue_env.local.sh (gitignored), e.g. WF_UE_ROOT=...

WF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$WF_REPO_ROOT/tools/ue_env.local.sh" ] && . "$WF_REPO_ROOT/tools/ue_env.local.sh"

# Installed engines live in $WF_UE_INSTALL_BASE/UE_<version>; "current" is a symlink.
# Default is on the root disk (/var/tmp: ~300 GiB free, no age-based cleanup on this
# machine) to keep the ~50 GiB engine off the shared, nearly full /home. It is not
# backed up; the engine can always be re-downloaded.
: "${WF_UE_INSTALL_BASE:=/var/tmp/${USER%%@*}-unreal}"
: "${WF_UE_ROOT:=$WF_UE_INSTALL_BASE/current}"

WF_PROJECT_DIR="$WF_REPO_ROOT/unreal/WhaleFactory"
WF_UPROJECT="$WF_PROJECT_DIR/WhaleFactory.uproject"
# Keep the editor's local derived-data cache inside the project (gitignored) so it is
# visible in `tools/disk_report.sh` and can be deleted safely; UE rebuilds it.
WF_DDC_DIR="$WF_PROJECT_DIR/DerivedDataCache"
# UE 5.4+ caches derived data (compiled shaders etc.) in a local Zen server instead,
# which defaults to ~/.config/Epic/UnrealEngine/Common/Zen/Data on /home and ignores
# WF_DDC_DIR. Put its data next to the engine on the root disk; it is regenerable.
: "${WF_ZEN_DATA_DIR:=$WF_UE_INSTALL_BASE/ZenData}"

WF_UE_EDITOR="$WF_UE_ROOT/Engine/Binaries/Linux/UnrealEditor"
WF_UE_EDITOR_CMD="$WF_UE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd"

export WF_REPO_ROOT WF_UE_INSTALL_BASE WF_UE_ROOT \
       WF_PROJECT_DIR WF_UPROJECT WF_DDC_DIR WF_ZEN_DATA_DIR WF_UE_EDITOR WF_UE_EDITOR_CMD

wf_python() {
    PYTHONPATH="$WF_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
}

# Run a command under the disk watchdog, watching /home and the engine's filesystem
# (the engine writes some caches into its own tree).
wf_guarded() {
    wf_python -m factory.diskguard run --path "$HOME" --path "$WF_UE_INSTALL_BASE" -- "$@"
}

# Run an Unreal binary under the disk watchdog with the factory's UE environment.
wf_guarded_ue() {
    wf_guarded "$WF_REPO_ROOT/tools/ue_exec.sh" "$@"
}

wf_require_engine() {
    if [ ! -x "$WF_UE_EDITOR" ]; then
        echo "Unreal Editor not found at $WF_UE_EDITOR" >&2
        echo "Install it first: tools/install_unreal.sh <Linux_Unreal_Engine_*.zip>" >&2
        return 1
    fi
}
