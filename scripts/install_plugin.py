#!/usr/bin/env python3
"""Build the telemetry plugin and install it into the game.

Run from the repo root:

    python scripts/install_plugin.py [--no-build]

The plugin directory is resolved through the PAL rather than configured, so
this works wherever Steam put the game. `game.log.txt` is where the game
reports whether it accepted the plugin; the path is printed at the end.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SOURCE = REPO_ROOT / "plugin"
BUILD_DIR = PLUGIN_SOURCE / "build"
LIBRARY_NAME = "ets2ad_telemetry.so" if sys.platform != "win32" else "ets2ad_telemetry.dll"


def build() -> Path:
    """Configure and build the plugin, returning the built library."""
    subprocess.run(
        ["cmake", "-S", str(PLUGIN_SOURCE), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    # Multi-config generators (MSVC) ignore CMAKE_BUILD_TYPE and build Debug
    # unless told: a Debug DLL needs the debug runtime, which only Build Tools has.
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "-j"], check=True)
    return built_library(BUILD_DIR)


def built_library(build_dir: Path) -> Path:
    """The Release library: at the top for single-config generators, under
    `Release/` for multi-config ones, never a stale Debug build beside it."""
    for library in (build_dir / LIBRARY_NAME, build_dir / "Release" / LIBRARY_NAME):
        if library.is_file():
            return library
    raise SystemExit(f"build produced no Release {LIBRARY_NAME} under {build_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-build", action="store_true", help="install the existing build output")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from pal import PalError, get_backend

    try:
        paths = get_backend().game_paths()
        plugins_dir = paths.plugins_dir
    except PalError as exc:
        print(f"could not locate the game: {exc}", file=sys.stderr)
        return 1

    try:
        library = built_library(BUILD_DIR) if args.no_build else build()
    except SystemExit as exc:
        print(f"{exc}; drop --no-build", file=sys.stderr)
        return 1

    plugins_dir.mkdir(parents=True, exist_ok=True)
    destination = plugins_dir / LIBRARY_NAME
    shutil.copy2(library, destination)

    print(f"installed {destination} ({destination.stat().st_size} bytes)")
    print(f"game log:  {paths.user_dir / 'game.log.txt'}")
    print("\nStart the game and look for '<ets2ad>' in the log.")
    print("While the game runs, 'sdk reload' in its console reloads the plugin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
