"""Sanity-check the WhaleFactory project from inside Unreal.

Run headless with ``tools/check_unreal.sh``. Prints ``[wf_env_check] OK`` when the
engine has the plugins and renderer settings the label pipeline depends on.
"""

import sys
from pathlib import Path

import unreal

PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()))
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factory import diskguard  # noqa: E402  (needs the repo root on sys.path)

REQUIRED_PLUGINS = (
    "Water",
    "PythonScriptPlugin",
    "EditorScriptingUtilities",
    "SequencerScripting",
    "MovieRenderPipeline",
)


def _log(message: str) -> None:
    unreal.log(f"[wf_env_check] {message}")


def main() -> None:
    problems = []
    _log(f"engine {unreal.SystemLibrary.get_engine_version()}")
    _log(f"project {PROJECT_DIR}")

    try:
        enabled = set(unreal.PluginBlueprintLibrary.get_enabled_plugin_names())
    except AttributeError:
        enabled = None
        unreal.log_warning("[wf_env_check] cannot list plugins in this engine version; skipped")
    if enabled is not None:
        missing = [name for name in REQUIRED_PLUGINS if name not in enabled]
        if missing:
            problems.append(f"plugins not enabled: {', '.join(missing)}")

    custom_depth = unreal.SystemLibrary.get_console_variable_int_value("r.CustomDepth")
    _log(f"r.CustomDepth={custom_depth}")
    if custom_depth != 3:
        problems.append(f"r.CustomDepth is {custom_depth}, need 3 (enabled with stencil)")

    free = diskguard.free_gib(PROJECT_DIR)
    _log(f"{free:.1f} GiB free on the project filesystem")
    if free < diskguard.DEFAULT_MIN_FREE_GIB:
        problems.append(f"only {free:.1f} GiB free (< {diskguard.DEFAULT_MIN_FREE_GIB:g} GiB floor)")

    for problem in problems:
        unreal.log_error(f"[wf_env_check] {problem}")
    _log("OK" if not problems else "FAILED")


main()
