#!/usr/bin/env python3
"""Press one of the plugin's buttons in the game, to see that it arrives.

    python tools/press.py hazard          # the hazard lights (flasher4way)
    python tools/press.py activate        # Enter: at a pump, fills the tank
    python tools/press.py trailer_attach  # T: hitch the trailer under the fifth wheel
    python tools/press.py horn --hold 1.0

For checking the plugin's control mixes after a game update or a new
plugin build (docs/compat.md). Only with the truck standing on its parking
brake: while the button is held the plugin drives, and every axis is held
at zero -- the steering does not move, the pedals are up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.command import BUTTON_BITS, Commander, ControlCommand  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from core.common.controls import Button  # noqa: E402
from pal.base import PalError  # noqa: E402

LOOP_S = 1.0 / 60.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("button", choices=sorted(b.value for b in BUTTON_BITS))
    parser.add_argument("--hold", type=float, default=0.3, help="seconds held")
    args = parser.parse_args()
    try:
        with TelemetryReader() as reader, Commander() as commander:
            truck = reader.read().block["truck"]
            if abs(float(truck["speed"])) > 0.1 or not int(truck["parking_brake"]):
                print("stop the truck and set its parking brake first", file=sys.stderr)
                return 1
            command = ControlCommand(buttons=frozenset({Button(args.button)}))
            until = time.perf_counter() + args.hold
            while time.perf_counter() < until:
                commander.send(command)
                time.sleep(LOOP_S)
            commander.disengage()
    except (PalError, TelemetryError) as exc:
        print(f"no game: {exc}", file=sys.stderr)
        return 1
    print(f"pressed {args.button} for {args.hold:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
