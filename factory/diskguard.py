"""Disk-space guards for everything in the factory that can write many GiB.

/home is shared by every project on this machine, so the Unreal install, the
editor's shader/derived-data caches and render batches all go through these
checks instead of being allowed to run until the disk is full.

Thresholds (GiB of free space left on the filesystem), overridable via env:

- ``WF_MIN_FREE_GIB``  (30): refuse to *start* a job that would leave less.
- ``WF_WARN_FREE_GIB`` (40): a guarded process logs warnings below this.
- ``WF_KILL_FREE_GIB`` (15): a guarded process is terminated below this.

CLI (run from the repo root, or via the tools/*.sh wrappers):

    python3 -m factory.diskguard check --path ~ --need-gib 12
    python3 -m factory.diskguard estimate --shots 2000 --width 1536 --height 1024
    python3 -m factory.diskguard zip-check --zip UE.zip --dest /var/tmp/<user>-unreal
    python3 -m factory.diskguard report
    python3 -m factory.diskguard run --path ~ --path /var/tmp -- <command> [args...]
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Sequence

GIB = 1024**3

DEFAULT_MIN_FREE_GIB = float(os.environ.get("WF_MIN_FREE_GIB", 30))
DEFAULT_WARN_FREE_GIB = float(os.environ.get("WF_WARN_FREE_GIB", 40))
DEFAULT_KILL_FREE_GIB = float(os.environ.get("WF_KILL_FREE_GIB", 15))

# Exit code used when a guarded process is killed for low disk (EX_TEMPFAIL).
EXIT_LOW_DISK = 75

# Per-shot metadata JSON plus manifest line, generous.
_META_BYTES_PER_SHOT = 32 * 1024


class InsufficientDiskSpace(RuntimeError):
    """Raised when a job would leave less than the allowed free space."""


def _log(message: str) -> None:
    print(f"[diskguard] {message}", file=sys.stderr, flush=True)


def _existing_ancestor(path: str | Path) -> Path:
    """Return ``path`` or its nearest existing parent (for not-yet-created outputs)."""
    candidate = Path(path).expanduser().absolute()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def free_bytes(path: str | Path) -> int:
    return shutil.disk_usage(_existing_ancestor(path)).free


def free_gib(path: str | Path) -> float:
    return free_bytes(path) / GIB


def ensure_free(
    path: str | Path,
    *,
    need_gib: float = 0.0,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> float:
    """Raise unless writing ``need_gib`` under ``path`` still leaves ``min_free_gib`` free.

    Returns the free GiB that would remain afterwards.
    """
    free = free_gib(path)
    remaining = free - need_gib
    if remaining < min_free_gib:
        raise InsufficientDiskSpace(
            f"{_existing_ancestor(path)}: {free:.1f} GiB free, job needs {need_gib:.1f} GiB, "
            f"which would leave {remaining:.1f} GiB (< {min_free_gib:.1f} GiB floor)"
        )
    return remaining


def estimate_render_bytes(
    n_shots: int,
    width: int,
    height: int,
    *,
    rgb_channels: int = 3,
    n_label_passes: int = 2,
) -> int:
    """Upper-bound bytes for a render batch (uncompressed RGB + uint8 label passes).

    PNG compression usually lands well below this, so it is a safe budget, not a
    prediction.
    """
    pixels = width * height
    per_shot = pixels * rgb_channels + pixels * n_label_passes + _META_BYTES_PER_SHOT
    return n_shots * per_shot


def zip_uncompressed_bytes(zip_path: str | Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        return sum(info.file_size for info in archive.infolist())


def dir_size_bytes(path: str | Path) -> int:
    """Apparent size of a directory tree (``du -sb``), 0 if it does not exist.

    Symlinks (e.g. the engine's ``current`` link) are resolved first, since ``du``
    would otherwise report the size of the link itself.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return 0
    try:
        out = subprocess.run(
            ["du", "-sb", str(path)], capture_output=True, text=True, check=False
        ).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError):
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total


def default_report_paths() -> list[tuple[str, Path]]:
    """Everything the Unreal workflow is known (or likely) to grow on disk.

    Paths come from the env exported by tools/ue_env.sh; unset ones are skipped.
    """
    home = Path.home()
    entries: list[tuple[str, str | None]] = [
        ("UE engine install", os.environ.get("WF_UE_ROOT")),
        ("UE engine installs (all)", os.environ.get("WF_UE_INSTALL_BASE")),
        ("~/Downloads", str(home / "Downloads")),
        ("project Saved/", _join_env("WF_PROJECT_DIR", "Saved")),
        ("project Intermediate/", _join_env("WF_PROJECT_DIR", "Intermediate")),
        ("project DerivedDataCache/", os.environ.get("WF_DDC_DIR")),
        ("Zen cache data", os.environ.get("WF_ZEN_DATA_DIR")),
        ("project Content/", _join_env("WF_PROJECT_DIR", "Content")),
        ("rendered data/", _join_env("WF_REPO_ROOT", "data")),
        ("repo .git/ (incl. LFS)", _join_env("WF_REPO_ROOT", ".git")),
        ("~/.config/Epic (Zen default)", str(home / ".config/Epic")),
        ("~/.local/share/Epic", str(home / ".local/share/Epic")),
        ("~/.local/share/UnrealEngine", str(home / ".local/share/UnrealEngine")),
        ("~/Documents/Unreal Projects", str(home / "Documents/Unreal Projects")),
    ]
    return [(label, Path(p)) for label, p in entries if p]


def _join_env(var: str, *parts: str) -> str | None:
    base = os.environ.get(var)
    return str(Path(base, *parts)) if base else None


def report(paths: Sequence[tuple[str, Path]] | None = None) -> str:
    paths = default_report_paths() if paths is None else paths
    lines = ["Filesystems:"]
    seen_devices: set[int] = set()
    for probe in [Path.home(), Path("/var/tmp")] + [p for _, p in paths]:
        anchor = _existing_ancestor(probe)
        device = anchor.stat().st_dev
        if device in seen_devices:
            continue
        seen_devices.add(device)
        usage = shutil.disk_usage(anchor)
        lines.append(
            f"  {str(anchor):<40} {usage.free / GIB:7.1f} GiB free of {usage.total / GIB:7.1f} GiB "
            f"({100 * usage.used / usage.total:.0f}% used)"
        )
    lines.append("Factory-related directories:")
    for label, path in paths:
        size = dir_size_bytes(path)
        shown = f"{size / GIB:8.2f} GiB" if path.exists() else "       (none)"
        lines.append(f"  {label:<30} {shown}  {path}")
    lines.append(
        f"Thresholds: refuse-start < {DEFAULT_MIN_FREE_GIB:g} GiB, warn < {DEFAULT_WARN_FREE_GIB:g} GiB, "
        f"kill < {DEFAULT_KILL_FREE_GIB:g} GiB free"
    )
    return "\n".join(lines)


def run_guarded(
    cmd: Sequence[str],
    *,
    paths: Sequence[str | Path],
    warn_below_gib: float = DEFAULT_WARN_FREE_GIB,
    kill_below_gib: float = DEFAULT_KILL_FREE_GIB,
    interval_s: float = 5.0,
    grace_s: float = 20.0,
    free_fn: Callable[[str | Path], float] = free_gib,
) -> int:
    """Run ``cmd`` and terminate its whole process group if free space drops too low
    on any of ``paths`` (one path per filesystem the job writes to).

    Unreal spawns helpers (ShaderCompileWorker, zen server, crash reporter), so the
    child gets its own session and the group is signalled, not just the parent.
    Returns the child's exit code, or ``EXIT_LOW_DISK`` if it was killed for space.
    """
    proc = subprocess.Popen(list(cmd), start_new_session=True)

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def _forward(signum, _frame):  # Ctrl-C / SIGTERM to the wrapper reaches the child.
        _signal_group(signum)

    previous = {s: signal.signal(s, _forward) for s in (signal.SIGINT, signal.SIGTERM)}
    last_warning = 0.0
    try:
        while True:
            try:
                return proc.wait(timeout=interval_s)
            except subprocess.TimeoutExpired:
                pass
            free, path = min((free_fn(p), p) for p in paths)
            if free < kill_below_gib:
                _log(
                    f"only {free:.1f} GiB free on {_existing_ancestor(path)} "
                    f"(< {kill_below_gib:g} GiB): terminating {cmd[0]}"
                )
                _signal_group(signal.SIGTERM)
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    _signal_group(signal.SIGKILL)
                    proc.wait()
                return EXIT_LOW_DISK
            if free < warn_below_gib and time.monotonic() - last_warning > 60:
                _log(
                    f"WARNING: {free:.1f} GiB free on {_existing_ancestor(path)}; "
                    f"{cmd[0]} will be terminated below {kill_below_gib:g} GiB"
                )
                last_warning = time.monotonic()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m factory.diskguard", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="fail if a job of --need-gib would breach the floor")
    p_check.add_argument("--path", default=str(Path.home()))
    p_check.add_argument("--need-gib", type=float, default=0.0)
    p_check.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)

    p_est = sub.add_parser("estimate", help="upper-bound size of a render batch")
    p_est.add_argument("--shots", type=int, required=True)
    p_est.add_argument("--width", type=int, required=True)
    p_est.add_argument("--height", type=int, required=True)
    p_est.add_argument("--label-passes", type=int, default=2)
    p_est.add_argument("--path", help="also check this output location has room")

    p_zip = sub.add_parser("zip-check", help="check a zip can be extracted into --dest")
    p_zip.add_argument("--zip", required=True)
    p_zip.add_argument("--dest", required=True)
    p_zip.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)

    sub.add_parser("report", help="free space and sizes of factory-related directories")

    p_run = sub.add_parser("run", help="run a command, killing it if free space gets too low")
    p_run.add_argument("--path", action="append", dest="paths",
                       help="filesystem to watch (repeatable; default: home)")
    p_run.add_argument("--warn-below-gib", type=float, default=DEFAULT_WARN_FREE_GIB)
    p_run.add_argument("--kill-below-gib", type=float, default=DEFAULT_KILL_FREE_GIB)
    p_run.add_argument("--interval", type=float, default=5.0)
    p_run.add_argument("cmd", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            remaining = ensure_free(args.path, need_gib=args.need_gib, min_free_gib=args.min_free_gib)
            _log(f"ok: {remaining:.1f} GiB would remain on {_existing_ancestor(args.path)}")
        elif args.command == "estimate":
            need = estimate_render_bytes(args.shots, args.width, args.height,
                                         n_label_passes=args.label_passes) / GIB
            print(f"{need:.2f}")
            if args.path:
                remaining = ensure_free(args.path, need_gib=need)
                _log(f"ok: batch needs <= {need:.1f} GiB, {remaining:.1f} GiB would remain")
        elif args.command == "zip-check":
            need = zip_uncompressed_bytes(args.zip) / GIB * 1.02  # filesystem block overhead
            remaining = ensure_free(args.dest, need_gib=need, min_free_gib=args.min_free_gib)
            _log(f"ok: extraction needs ~{need:.1f} GiB, {remaining:.1f} GiB would remain")
            print(f"{need:.2f}")
        elif args.command == "report":
            print(report())
        elif args.command == "run":
            cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
            if not cmd:
                parser.error("run needs a command after --")
            paths = args.paths or [str(Path.home())]
            for path in paths:
                ensure_free(path, min_free_gib=max(args.kill_below_gib, 0.0))
            return run_guarded(cmd, paths=paths, warn_below_gib=args.warn_below_gib,
                               kill_below_gib=args.kill_below_gib, interval_s=args.interval)
    except InsufficientDiskSpace as exc:
        _log(f"REFUSED: {exc}")
        return EXIT_LOW_DISK
    return 0


if __name__ == "__main__":
    sys.exit(_main())
