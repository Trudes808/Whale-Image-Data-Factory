#!/usr/bin/env bash
# Add "WhaleFactory Unreal Editor" to the GNOME app menu, so it can be started from a
# FastX "Ubuntu Desktop"/"Gnome" session (Activities -> search "WhaleFactory").
# It opens a terminal running tools/launch_editor.sh, so disk-watchdog warnings stay visible.
# Remove with: rm ~/.local/share/applications/whalefactory-unreal.desktop
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.local/share/applications/whalefactory-unreal.desktop"
mkdir -p "$(dirname "$dest")"
cat > "$dest" <<DESKTOP
[Desktop Entry]
Type=Application
Name=WhaleFactory Unreal Editor
Comment=Unreal Editor on the WhaleFactory project, with disk guards
Icon=applications-games
Terminal=false
Exec=gnome-terminal --title=WhaleFactory-Unreal -- $repo/tools/launch_editor_terminal.sh
Categories=Development;
DESKTOP
chmod 644 "$dest"
command -v desktop-file-validate >/dev/null && desktop-file-validate "$dest"
echo "Installed $dest"
