#!/usr/bin/env python3
"""Take off this machine what the launcher put on it, asking for each.

    uninstall.bat / uninstall.sh            (a release's, beside ets2ad.bat)
    python tools/uninstall.py               # asks, one by one
    python tools/uninstall.py --dry-run     # says what there is, takes nothing

Each is asked in turn and is kept unless the answer is yes:

1. **plugin** -- in the game's plugins folder. With it gone the game no
   longer warns about SDK features. The game must be closed.
2. **downloads** -- uv's cached copies of this project's packages, about
   2.5 GB with the GPU build. Only this project's: other projects' are left.
   On the same disk uv links them into this folder's packages, so they take
   their space once; it comes back when both are gone.
3. **Python** -- the one uv fetched for this folder, if uv fetched it.
   Another project of uv's may use it too.
4. **uv** -- if the launcher installed it, in `~/.local/bin`
   (`%USERPROFILE%\\.local\\bin` on Windows). The PATH entry its installer
   added is left: it does no harm.
5. **this folder** -- its packages, the map read out of the game, and the
   drives recorded in it.

Then what only the driver can do: the barrier mod is unsubscribed in Steam.

Python and this folder are in use while this runs, and uv when Python is
removed with it, so those three are removed by a small job that starts when
this ends, a few seconds later.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT = "ets2-ad"
"""This project's own name in uv.lock: not a download."""

GAME_PROCESS = "eurotrucks2"
BARRIER_MOD = "https://steamcommunity.com/sharedfiles/filedetails/?id=2131396358"
WAIT_S = 3
"""How long the job that finishes waits for this process, and the window it
ran in, to let go of the folder."""


def plugin_name(platform: str = sys.platform) -> str:
    return "ets2ad_telemetry.dll" if platform == "win32" else "ets2ad_telemetry.so"


def installed_plugin(game: Path | None = None) -> tuple[Path | None, str]:
    """The plugin in the game's plugins folder, and where it was looked for;
    `game` is the game's folder, found through Steam when not given."""
    if game is not None:
        binary = "win_x64" if sys.platform == "win32" else "linux_x64"
        folder = game / "bin" / binary / "plugins"
    else:
        try:
            from pal import PalError, get_backend

            folder = get_backend().game_paths().plugins_dir
        except (ImportError, PalError, OSError) as exc:
            return None, f"the game was not found: {exc}"
    plugin = folder / plugin_name()
    return (plugin if plugin.is_file() else None), str(plugin)


def game_running(platform: str = sys.platform, proc: Path = Path("/proc")) -> bool:
    """Whether the game is running: it holds the plugin while it is."""
    if platform == "win32":
        listing = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS}.exe", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return f"{GAME_PROCESS}.exe" in listing.lower()
    for comm in proc.glob("[0-9]*/comm"):
        try:
            if comm.read_text(encoding="utf-8").strip().startswith(GAME_PROCESS):
                return True
        except OSError:
            continue
    return False


def project_packages(lock: Path = REPO_ROOT / "uv.lock") -> list[str]:
    """Every package uv.lock names, but the project itself."""
    if not lock.is_file():
        return []
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    return sorted({p["name"] for p in packages if p.get("name") and p["name"] != PROJECT})


def uv_install_dir(environment=os.environ, home: Path | None = None) -> Path:
    """Where uv's installer puts uv: as its own variables say, else
    `~/.local/bin`, which is where the launcher's run put it."""
    for name in ("UV_INSTALL_DIR", "XDG_BIN_HOME"):
        if environment.get(name):
            return Path(environment[name])
    return (home or Path.home()) / ".local" / "bin"


def uv_binaries(folder: Path, platform: str = sys.platform) -> list[Path]:
    names = ("uv.exe", "uvx.exe", "uvw.exe") if platform == "win32" else ("uv", "uvx")
    return [folder / name for name in names if (folder / name).is_file()]


def venv_python(venv: Path, uv: Path | None, run=subprocess.run) -> tuple[str | None, str]:
    """The version of the Python the folder's packages run on, if uv fetched
    it (it lives under `uv python dir`); and where it is."""
    config = venv / "pyvenv.cfg"
    if not config.is_file() or uv is None:
        return None, ""
    values = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    home = values.get("home", "")
    version = values.get("version_info") or values.get("version")
    managed = run([str(uv), "python", "dir"], capture_output=True, text=True, check=False)
    root = managed.stdout.strip()
    if not home or not version or not root:
        return None, home
    try:
        Path(home).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return None, home  # a Python of the system's, or of another tool's
    return ".".join(version.split(".")[:3]), home


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def ask(question: str, answer: Callable[[str], str] = input) -> bool:
    """Yes only for yes: anything else, an empty line included, keeps it."""
    try:
        said = answer(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return said in ("y", "yes", "e", "evet")


@dataclass
class Plan:
    """What was chosen to go after this process ends: these are in use now."""

    python: str | None = None
    folder: Path | None = None
    uv_files: list[Path] = field(default_factory=list)
    uv: Path | None = None

    def empty(self) -> bool:
        return self.python is None and self.folder is None and not self.uv_files


def finishing_job(plan: Plan, platform: str = sys.platform) -> list[str]:
    """The commands that remove what is in use, once it is not: Python (by
    uv, before uv goes), the folder, then uv."""
    lines: list[str] = []
    if platform == "win32":
        lines.append(f"timeout /t {WAIT_S} /nobreak >nul")
        if plan.python and plan.uv:
            lines.append(f'"{plan.uv}" python uninstall {plan.python}')
        if plan.folder:
            lines.append(f'rd /s /q "{plan.folder}"')
        lines.extend(f'del /f /q "{path}"' for path in plan.uv_files)
        lines.append('del /f /q "%~f0"')
    else:
        lines.append(f"sleep {WAIT_S}")
        if plan.python and plan.uv:
            lines.append(f"{shlex.quote(str(plan.uv))} python uninstall {plan.python}")
        if plan.folder:
            lines.append(f"rm -rf {shlex.quote(str(plan.folder))}")
        lines.extend(f"rm -f {shlex.quote(str(path))}" for path in plan.uv_files)
        lines.append('rm -f "$0"')
    return lines


def start_finishing_job(plan: Plan, platform: str = sys.platform) -> Path:
    """Write the job to a file outside the folder and start it detached."""
    suffix = ".cmd" if platform == "win32" else ".sh"
    handle, name = tempfile.mkstemp(prefix="ets2ad-uninstall-", suffix=suffix)
    script = Path(name)
    header = ["@echo off"] if platform == "win32" else ["#!/bin/sh"]
    newline = "\r\n" if platform == "win32" else "\n"
    with os.fdopen(handle, "w", encoding="utf-8", newline=newline) as f:
        f.write("\n".join(header + finishing_job(plan, platform)) + "\n")
    nothing = subprocess.DEVNULL
    quiet = {"stdin": nothing, "stdout": nothing, "stderr": nothing, "cwd": tempfile.gettempdir()}
    if platform == "win32":
        detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        subprocess.Popen(["cmd", "/c", str(script)], creationflags=detached, **quiet)
    else:
        subprocess.Popen(["sh", str(script)], start_new_session=True, **quiet)
    return script


def main(argv: list[str] | None = None, answer: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="say what there is, take nothing")
    parser.add_argument("--game", type=Path, help="the game's folder, if Steam does not say")
    parser.add_argument("--folder", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    folder = args.folder.resolve()
    dry = args.dry_run

    def offer(question: str) -> bool:
        if dry:
            print(f"{question} -- would ask")
            return False
        return ask(question, answer)

    print("Each is kept unless you answer y.\n")
    plan = Plan()

    # 1. The plugin.
    plugin, where = installed_plugin(args.game)
    if plugin is None:
        print(f"1. plugin: not installed ({where})")
    elif offer(f"1. Remove the plugin from the game?\n   {plugin}\n  "):
        if game_running():
            print("   The game is running: close it, then run this again.")
        else:
            try:
                plugin.unlink()
                print("   removed")
            except OSError as exc:
                print(f"   could not remove it ({exc}); delete it by hand:\n   {plugin}")

    # 2. The downloads in uv's cache.
    uv_folder = uv_install_dir()
    uv = shutil.which("uv") or next(iter(uv_binaries(uv_folder)), None)
    uv = Path(uv) if uv else None
    packages = project_packages(folder / "uv.lock")
    if uv is None or not packages:
        print("2. downloads: no uv here, so none")
    elif offer(
        f"2. Remove uv's downloaded copies of this project's {len(packages)} packages?"
        "\n   About 2.5 GB with the GPU build. On the same disk they share their space"
        "\n   with this folder's: it all comes back once this folder is deleted too.\n  "
    ):
        # uv's own count ("404 KiB" for 280 MB) would only mislead: kept quiet.
        done = subprocess.run(
            [str(uv), "cache", "clean", *packages], check=False, capture_output=True, text=True
        )
        if done.returncode != 0:
            print(f"   uv could not clean its cache: {done.stderr.strip()}")
        else:
            print("   removed")

    # 3. Python, if uv fetched it.
    version, home = venv_python(folder / ".venv", uv)
    if version is None:
        print("3. Python: not one uv fetched" + (f" ({home})" if home else ""))
    elif offer(f"3. Remove Python {version}, which uv fetched? Other uv projects may use it.\n  "):
        plan.python, plan.uv = version, uv

    # 4. uv itself, where the launcher put it.
    binaries = uv_binaries(uv_folder)
    if not binaries:
        print(f"4. uv: not in {uv_folder}")
    elif offer(f"4. Remove uv from {uv_folder}? Other projects may use it.\n  "):
        plan.uv_files = binaries
        plan.uv = plan.uv or binaries[0]

    # 5. This folder.
    recordings = folder / "data" / "missions"
    size = size_of(folder)
    kept = (
        f", {human(size_of(recordings))} of it your recorded drives" if recordings.is_dir() else ""
    )
    if offer(f"5. Delete this folder, {human(size)}{kept}?\n   {folder}\n  "):
        plan.folder = folder

    if not dry and not plan.empty():
        start_finishing_job(plan)
        going = [
            *([f"Python {plan.python}"] if plan.python else []),
            *(["this folder"] if plan.folder else []),
            *(["uv"] if plan.uv_files else []),
        ]
        print(f"\n{', '.join(going)}: removed in a few seconds, once this has ended.")
    print(f"\nBy hand, if you wish: unsubscribe from the barrier mod in Steam:\n{BARRIER_MOD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
