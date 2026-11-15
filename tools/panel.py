#!/usr/bin/env python3
"""A status panel for the driver, in a terminal beside the game.

    python tools/panel.py

What the mission is doing and why (tools/mission.py leaves it in
data/status.json), and what the truck reports: speed against the limit,
gear, the beacon that engages the autopilot, the parking brake, fuel range,
the GPS's distance to go, damage, and what is calibrated for this truck.
Nothing is drawn over the game, and the game window is never touched: put
the terminal on another screen, or beside the game's window. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Before numpy: on Windows OpenBLAS commits a buffer for each CPU thread as it
# is imported -- 360 MB in each of the drive's processes, where Linux commits
# nothing until it is used -- and one thread is as quick for what is done here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge import convert  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from core.calibration import load_profile, truck_id  # noqa: E402
from core.control.limits import DamageWatch  # noqa: E402
from core.mission.status import STATUS_FILE, read_status  # noqa: E402
from pal.base import PalError  # noqa: E402

REFRESH_S = 0.5
STALE_S = 30.0
"""A status this old with the mission not saying more: is it still running?"""

CLEAR = "\x1b[H\x1b[2J"


def render(status: dict | None, truck: dict | None, now: float) -> str:
    """The panel's text, from the mission's status and the truck's fields."""
    lines = ["ETS2 autopilot", ""]
    if status is None:
        lines.append("mission   -- not running (python tools/mission.py)")
    else:
        age = now - float(status.get("at", now))
        state = str(status.get("state", "?")).upper()
        lines.append(f"mission   {state}: {status.get('message', '')}")
        job = status.get("job")
        if job:
            lines.append(f"job       to {job[0]} in {job[1]}")
        if age > STALE_S and state != "DRIVING":
            lines.append(f"          (said {age / 60:.0f} min ago)")
    lines.append("")
    if truck is None:
        lines.append("truck     -- no telemetry: is the game running with the plugin?")
        return "\n".join(lines) + "\n"
    limit = truck["limit_kmh"]
    over = limit is not None and truck["speed_kmh"] > limit + 3
    lines += [
        f"speed     {truck['speed_kmh']:5.0f} km/h"
        + (f"   limit {limit:.0f}" if limit is not None else "")
        + ("   OVER" if over else ""),
        f"gear      {truck['gear']}",
        f"autopilot {'ENGAGED (beacon on)' if truck['beacon'] else 'off (beacon O to engage)'}",
        f"brake     {'parking brake set' if truck['parking_brake'] else 'released'}",
        f"fuel      {truck['fuel_range_km']:.0f} km",
        f"to go     {truck['nav_km']:.1f} km" if truck["nav_km"] > 0 else "to go     --",
        f"damage    {truck['damage_pct']:.1f} %",
        f"truck     {truck['id']}: {truck['calibration']}",
    ]
    if truck["paused"]:
        lines.append("          PAUSED")
    return "\n".join(lines) + "\n"


def truck_fields(snapshot, calibration: str) -> dict:
    block = snapshot.block
    truck = block["truck"]
    return {
        "speed_kmh": convert.speed_kmh(block),
        "limit_kmh": convert.road_speed_limit(block),
        "gear": int(truck["gear"]),
        "beacon": bool(int(truck["light_beacon"])),
        "parking_brake": bool(int(truck["parking_brake"])),
        "fuel_range_km": float(truck["fuel_range"]),
        "nav_km": float(block["navigation"]["distance"]) / 1000.0,
        "damage_pct": DamageWatch.damage(block) * 100.0,
        "id": truck_id(block),
        "calibration": calibration,
        "paused": bool(snapshot.is_paused),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", type=Path, default=STATUS_FILE)
    args = parser.parse_args()
    reader: TelemetryReader | None = None
    calibrations: dict[str, str] = {}
    try:
        while True:
            fields = None
            try:
                if reader is None:
                    reader = TelemetryReader()
                    reader.open()
                snapshot = reader.read()
                truck = truck_id(snapshot.block)
                if truck not in calibrations:
                    missing = load_profile(truck).missing()
                    calibrations[truck] = (
                        f"the Actros's {', '.join(missing)}" if missing else "calibrated"
                    )
                fields = truck_fields(snapshot, calibrations[truck])
            except (PalError, TelemetryError):
                if reader is not None:
                    reader.close()
                reader = None
            sys.stdout.write(CLEAR + render(read_status(args.status), fields, time.time()))
            sys.stdout.flush()
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        return 0
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    sys.exit(main())
