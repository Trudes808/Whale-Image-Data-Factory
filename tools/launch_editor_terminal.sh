#!/usr/bin/env bash
# Terminal wrapper for the GNOME launcher: runs launch_editor.sh and keeps the window
# open afterwards so disk-watchdog messages and exit codes can be read.
"$(dirname "$0")/launch_editor.sh" "$@"
code=$?
echo
read -rp "Editor exited (code $code). Press Enter to close."
exit "$code"
