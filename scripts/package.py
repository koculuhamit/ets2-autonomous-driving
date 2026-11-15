#!/usr/bin/env python3
"""Build a release: everything a driver needs, in one folder and one zip.

    python scripts/package.py                          # for this platform
    python scripts/package.py --rid win-x64 --plugin path/to/ets2ad_telemetry.dll

The release holds the sources the stack runs from (what git tracks, less the
tests), and in `bin/` the plugin and the map extractor built for the target
-- the extractor self-contained, so the driver needs no .NET -- with the
vehicle detector under `data/models/` when it is here to copy. A driver
unzips it and runs `ets2ad.sh` or `ets2ad.bat`, nothing installed first: the
launcher fetches uv if it is missing (and uv Python), tools/setup.py installs
the plugin -- built with its C++ runtime inside -- and builds the map, and
tools/mission.py drives.

The plugin for Windows is built by MSVC, not here: CI uploads it
(`plugin-windows-latest`) and `--plugin` takes it.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIPPED = ("tests/", ".github/", ".pre-commit-config.yaml")
"""Tracked, but not for a driver: the tests and CI."""
EXCLUDE_FILE = REPO_ROOT / "scripts" / "export_exclude.txt"
"""Paths kept out of a release as well, one a line (a directory ends in /),
where the file is there."""
MODEL = Path("data/models/dfine-nano-ets2")

LAUNCH_SH = """#!/usr/bin/env sh
# Get ready (plugin, map), then drive: the truck's beacon (O) in the game.
cd "$(dirname "$0")" || exit 1
# uv brings Python and everything else; it is fetched once if it is not here.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1
fi
# With an NVIDIA GPU, the GPU build of ONNX Runtime (2.5 GB with CUDA);
# without one, the small CPU build, and perception three times slower.
extra=perception-cpu
cpu=--allow-cpu
if command -v nvidia-smi >/dev/null 2>&1; then
    extra=perception
    cpu=
fi
uv sync --extra "$extra" || exit 1
uv run python tools/setup.py || exit 1
exec uv run python tools/mission.py $cpu "$@"
"""

LAUNCH_BAT = """@echo off
rem Get ready (plugin, map), then drive: the truck's beacon (O) in the game.
cd /d "%~dp0"
rem uv brings Python and everything else; it is fetched once if it is not here.
set "PATH=%USERPROFILE%\\.local\\bin;%PATH%"
where uv >nul 2>&1 || powershell -NoProfile -ExecutionPolicy ByPass ^
    -Command "irm https://astral.sh/uv/install.ps1 | iex" || exit /b 1
rem With an NVIDIA GPU, the GPU build of ONNX Runtime (2.5 GB with CUDA);
rem without one, the small CPU build, and perception three times slower.
set "EXTRA=perception-cpu"
set "CPU=--allow-cpu"
where nvidia-smi >nul 2>&1 && (set "EXTRA=perception" & set "CPU=")
uv sync --extra %EXTRA% || exit /b 1
uv run python tools/setup.py || exit /b 1
uv run python tools/mission.py %CPU% %*
"""


def host_rid() -> str:
    system = "win" if sys.platform == "win32" else "linux"
    machine = platform.machine().lower()
    return f"{system}-{'arm64' if machine in ('arm64', 'aarch64') else 'x64'}"


def left_out(exclude_file: Path = EXCLUDE_FILE) -> tuple[str, ...]:
    """`SKIPPED` and the paths in `exclude_file`, if it is there."""
    if not exclude_file.is_file():
        return SKIPPED
    lines = exclude_file.read_text(encoding="utf-8").splitlines()
    return SKIPPED + tuple(
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    )


def tracked(root: Path, skipped: tuple[str, ...] | None = None) -> list[Path]:
    """The files git tracks, less what a release has no use for."""
    skipped = left_out() if skipped is None else skipped
    listing = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [Path(name) for name in listing if not name.startswith(skipped)]


def dotnet() -> str:
    found = shutil.which("dotnet")
    if found:
        return found
    for candidate in (Path.home() / ".dotnet", Path.home() / ".local" / "share" / "dotnet"):
        if (candidate / "dotnet").is_file():
            return str(candidate / "dotnet")
    raise SystemExit("the .NET SDK is needed to build the map extractor")


def publish_extractor(rid: str, into: Path) -> Path:
    """The map extractor for `rid`, one self-contained file."""
    staging = REPO_ROOT / "dist" / f"map_extract-{rid}"
    subprocess.run(
        [
            dotnet(),
            "publish",
            str(REPO_ROOT / "tools" / "map_extract"),
            "-c",
            "Release",
            "-r",
            rid,
            "--self-contained",
            "true",
            "-p:PublishSingleFile=true",
            "-p:IncludeNativeLibrariesForSelfExtract=true",
            "-o",
            str(staging),
            "--nologo",
        ],
        check=True,
    )
    name = "MapExtract.exe" if rid.startswith("win") else "MapExtract"
    target = into / name
    shutil.copy2(staging / name, target)
    return target


def build_plugin() -> Path:
    build = REPO_ROOT / "plugin" / "build"
    subprocess.run(
        ["cmake", "-S", "plugin", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True)
    found = sorted(build.rglob("ets2ad_telemetry.so")) + sorted(build.rglob("ets2ad_telemetry.dll"))
    if not found:
        raise SystemExit("the plugin build produced no library")
    return found[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rid", default=host_rid(), help="linux-x64, win-x64, ...")
    parser.add_argument("--plugin", type=Path, help="the plugin built for --rid")
    parser.add_argument("--no-model", action="store_true", help="leave the detector out")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args()

    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    name = f"ets2ad-{version}-{args.rid}"
    release = args.output / name
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    for path in tracked(REPO_ROOT):
        target = release / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / path, target)

    bin_dir = release / "bin"
    bin_dir.mkdir()
    windows = args.rid.startswith("win")
    if args.plugin is not None:
        plugin = args.plugin
    elif windows != (sys.platform == "win32"):
        raise SystemExit(f"--plugin is needed: this machine cannot build the {args.rid} plugin")
    else:
        plugin = build_plugin()
    shutil.copy2(plugin, bin_dir / ("ets2ad_telemetry.dll" if windows else "ets2ad_telemetry.so"))
    publish_extractor(args.rid, bin_dir)

    if not args.no_model and (REPO_ROOT / MODEL).is_dir():
        shutil.copytree(REPO_ROOT / MODEL, release / MODEL)
    elif not args.no_model:
        print(f"no detector at {MODEL}: the release goes without it", file=sys.stderr)

    launcher = release / ("ets2ad.bat" if windows else "ets2ad.sh")
    launcher.write_text(LAUNCH_BAT if windows else LAUNCH_SH, encoding="utf-8", newline="")
    launcher.chmod(0o755)

    archive = shutil.make_archive(str(args.output / name), "zip", args.output, name)
    size = Path(archive).stat().st_size / 1e6
    print(f"{release}\n{archive} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
