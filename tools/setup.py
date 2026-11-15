#!/usr/bin/env python3
"""Get a machine ready for its first drive: what is missing, and do it.

    python tools/setup.py            # check each step, and do what is missing
    python tools/setup.py --check    # only say what is there and what is not

In order, each step checked first and done only if it is missing:

1. **game** -- found through Steam (pal: the registry on Windows).
2. **plugin** -- the telemetry and input plugin in the game's plugins
   folder: from `bin/` in a release, or built (scripts/install_plugin.py).
   The game loads it at start: restart the game after it is installed.
3. **map** -- the road network of the installed game and DLCs, read out of
   the game's archives by the map extractor (tools/map_extract: `bin/` in a
   release, or built with the .NET SDK) and compiled (tools/build_map.py)
   into `data/map/installed`. About ten minutes; again after a map update.
4. **model** -- the vehicle detector, `data/models/dfine-nano-ets2`: it is
   not in the repository; copy it in from a release.
5. **truck** -- what is calibrated for the truck in the game
   (tools/calibrate.py): a truck with nothing measured drives on the
   Actros's numbers.

Then: `python tools/mission.py`, and the truck's beacon (O) in the game.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAP_CACHE = REPO_ROOT / "data" / "map" / "installed"
EXTRACTION = REPO_ROOT / "data" / "map" / "extraction" / "roads.json"
MODEL = REPO_ROOT / "data" / "models" / "dfine-nano-ets2"
EXTRACTOR_ENV = "ETS2AD_MAP_EXTRACT"


def plugin_name(platform: str = sys.platform) -> str:
    return "ets2ad_telemetry.dll" if platform == "win32" else "ets2ad_telemetry.so"


def extractor_command(
    root: Path = REPO_ROOT,
    environment: Mapping[str, str] = os.environ,
    platform: str = sys.platform,
    which=shutil.which,
) -> list[str] | None:
    """How to run the map extractor here: the one `ETS2AD_MAP_EXTRACT`
    names, a release's in `bin/`, or a build of tools/map_extract run by
    dotnet; None for none."""
    named = environment.get(EXTRACTOR_ENV)
    if named:
        return [named]
    bundled = root / "bin" / ("MapExtract.exe" if platform == "win32" else "MapExtract")
    if bundled.is_file():
        return [str(bundled)]
    dotnet = which("dotnet") or _home_dotnet(platform)
    if dotnet is None:
        return None
    built = sorted((root / "tools" / "map_extract" / "bin").glob("*/net*/MapExtract.dll"))
    if built:
        return [dotnet, str(built[-1])]
    return [dotnet, "run", "--project", str(root / "tools" / "map_extract"), "--"]


def _home_dotnet(platform: str) -> str | None:
    """dotnet installed for the user only (dotnet-install.sh's default)."""
    name = "dotnet.exe" if platform == "win32" else "dotnet"
    for candidate in (
        Path.home() / ".dotnet" / name,
        Path.home() / ".local" / "share" / "dotnet" / name,
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def run(command: list[str]) -> bool:
    print("  $ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode == 0


def game(do: bool) -> tuple[bool, str]:
    from pal import PalError, get_backend

    try:
        root = get_backend().game_paths().game_root
    except PalError as exc:
        return False, f"not found: {exc}"
    return True, str(root)


def plugin(do: bool) -> tuple[bool, str]:
    from pal import PalError, get_backend

    try:
        installed = get_backend().game_paths().plugins_dir / plugin_name()
    except PalError as exc:
        return False, f"no game to install it in: {exc}"
    bundled = REPO_ROOT / "bin" / plugin_name()
    stale = (
        bundled.is_file()
        and installed.is_file()
        and (bundled.read_bytes() != installed.read_bytes())
    )
    if installed.is_file() and not stale:
        return True, str(installed)
    if not do:
        return False, "an older build installed" if stale else f"not in {installed.parent}"
    if bundled.is_file():
        installed.parent.mkdir(parents=True, exist_ok=True)
        # Beside it, then over it in one step: the game may have the old one loaded.
        fresh = installed.with_name(installed.name + ".new")
        shutil.copy2(bundled, fresh)
        fresh.replace(installed)
    elif not run([sys.executable, "scripts/install_plugin.py"]):
        return False, "the build failed: CMake and a C++ compiler are needed"
    return installed.is_file(), f"{installed} -- restart the game to load it"


def map_cache(do: bool) -> tuple[bool, str]:
    if (MAP_CACHE / "manifest.json").is_file():
        return True, str(MAP_CACHE)
    if not do:
        return False, "not built"
    ok, root = game(do)
    if not ok:
        return False, f"no game to read it from: {root}"
    if not EXTRACTION.is_file():
        command = extractor_command()
        if command is None:
            return False, "no map extractor: a release's bin/, or the .NET SDK"
        EXTRACTION.parent.mkdir(parents=True, exist_ok=True)
        if not run([*command, "roads", root, str(EXTRACTION), "all"]):
            return False, "the extraction failed"
    built = run(
        [
            sys.executable,
            "tools/build_map.py",
            str(EXTRACTION),
            str(MAP_CACHE),
            "--dlc",
            "installed",
            "--game",
            root,
        ]
    )
    return built and (MAP_CACHE / "manifest.json").is_file(), str(MAP_CACHE)


def model(do: bool) -> tuple[bool, str]:
    if (MODEL / "model.onnx").is_file() or any(MODEL.glob("*.onnx")):
        return True, str(MODEL)
    return False, f"copy the detector from a release into {MODEL}"


def truck(do: bool) -> tuple[bool, str]:
    from core.bridge.reader import TelemetryError, TelemetryReader
    from core.calibration import load_profile, truck_id
    from pal import PalError

    try:
        with TelemetryReader() as reader:
            profile = load_profile(truck_id(reader.read().block))
    except (PalError, TelemetryError):
        return True, "the game is not running: tools/calibrate.py status, later"
    missing = profile.missing()
    if missing:
        return True, f"{profile.truck}: the Actros's {', '.join(missing)} (tools/calibrate.py)"
    return True, f"{profile.truck}: calibrated"


STEPS = (("game", game), ("plugin", plugin), ("map", map_cache), ("model", model), ("truck", truck))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only check")
    args = parser.parse_args()
    ready = True
    for name, step in STEPS:
        ok, detail = step(not args.check)
        ready &= ok
        print(f"[{'ok' if ok else '--'}] {name:7s} {detail}", flush=True)
        if not ok and name == "game":
            break
    if ready:
        print("ready: python tools/mission.py, then the truck's beacon (O) in the game")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
