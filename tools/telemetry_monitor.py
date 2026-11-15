#!/usr/bin/env python3
"""Live view of the telemetry block.

The smoke test for the whole bridge: start the game, run this, and watch the
numbers move.

    python tools/telemetry_monitor.py
    python tools/telemetry_monitor.py --once --json    # one snapshot, scriptable

It also reports what the stack cares about beyond the values themselves: the
publish rate the plugin is actually achieving, and how often a read had to be
retried because it caught the writer mid-frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.reader import (  # noqa: E402
    StaleTelemetryError,
    TelemetryError,
    TelemetryReader,
    TelemetrySnapshot,
)
from pal import PalError  # noqa: E402

#: Orientation arrives in rotations; degrees are what a human can sanity-check.
ROTATIONS_TO_DEGREES = 360.0
MS_PER_SECOND = 1000.0


def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def summarise(snapshot: TelemetrySnapshot) -> dict[str, Any]:
    """Pull the fields worth looking at out of the raw block."""
    block = snapshot.block
    truck = block["truck"]
    trailer = block["trailers"][0]
    navigation = block["navigation"]

    return {
        "write_count": snapshot.write_count,
        "write_time": round(snapshot.write_time, 3),
        "sequence": snapshot.sequence,
        "attempts": snapshot.attempts,
        "paused": snapshot.is_paused,
        "truck": {
            "model": f"{_text(truck['brand_id'])}/{_text(truck['model_id'])}",
            "speed_kmh": round(float(truck["speed"]) * 3.6, 2),
            "position": [round(float(truck["placement"]["position"][a]), 2) for a in "xyz"],
            "heading_deg": round(
                float(truck["placement"]["orientation"]["heading"]) * ROTATIONS_TO_DEGREES, 2
            ),
            "pitch_deg": round(
                float(truck["placement"]["orientation"]["pitch"]) * ROTATIONS_TO_DEGREES, 2
            ),
            "steering_input": round(float(truck["input_steering"]), 4),
            "steering_effective": round(float(truck["effective_steering"]), 4),
            "throttle": round(float(truck["effective_throttle"]), 3),
            "brake": round(float(truck["effective_brake"]), 3),
            "gear": int(truck["gear"]),
            "rpm": round(float(truck["engine_rpm"]), 0),
            "engine_on": bool(truck["engine_enabled"]),
            "parking_brake": bool(truck["parking_brake"]),
            "wheel_count": int(truck["wheels"]["count"]),
            "wheelbase_m": round(_wheelbase(truck), 3),
        },
        "trailer": {
            "connected": bool(trailer["connected"]),
            "id": _text(trailer["id"]),
            "hitch_angle_deg": round(_hitch_angle_degrees(truck, trailer), 2),
        },
        "navigation": {
            "distance_m": round(float(navigation["distance"]), 1),
            "speed_limit_kmh": round(float(navigation["speed_limit"]) * 3.6, 1),
        },
    }


def _wheelbase(truck: Any) -> float:
    """Distance from the front axle to the mean driven axle, in metres.

    Measured from the wheel positions the game reports rather than assumed, so
    it stays correct across truck models. This is the `L` of the bicycle model
    the lateral controllers are built on.
    """
    count = int(truck["wheels"]["count"])
    if count == 0:
        return 0.0
    positions = truck["wheels"]["position"][:count]
    steerable = truck["wheels"]["steerable"][:count]
    powered = truck["wheels"]["powered"][:count]

    front = [float(p["z"]) for p, s in zip(positions, steerable, strict=True) if s]
    rear = [float(p["z"]) for p, d in zip(positions, powered, strict=True) if d]
    if not front or not rear:
        return 0.0
    return abs(sum(front) / len(front) - sum(rear) / len(rear))


def _hitch_angle_degrees(truck: Any, trailer: Any) -> float:
    """Truck heading minus trailer heading, wrapped to (-180, 180].

    Zero when the combination is straight. This is the quantity reverse parking
    lives or dies on, and it is worth eyeballing early: the sign convention and
    the wrap-around are both easy to get wrong and hard to notice later.
    """
    if not bool(trailer["connected"]):
        return 0.0
    truck_deg = float(truck["placement"]["orientation"]["heading"]) * ROTATIONS_TO_DEGREES
    trailer_deg = float(trailer["placement"]["orientation"]["heading"]) * ROTATIONS_TO_DEGREES
    return (truck_deg - trailer_deg + 180.0) % 360.0 - 180.0


def format_line(summary: dict[str, Any], publish_hz: float) -> str:
    truck = summary["truck"]
    trailer = summary["trailer"]
    flags = []
    if summary["paused"]:
        flags.append("PAUSED")
    if truck["parking_brake"]:
        flags.append("PBRAKE")
    if not truck["engine_on"]:
        flags.append("ENGINE-OFF")
    if trailer["connected"]:
        flags.append(f"HITCH{trailer['hitch_angle_deg']:+6.1f}°")

    return (
        f"{truck['speed_kmh']:6.1f} km/h  "
        f"gear {truck['gear']:+3d}  "
        f"rpm {truck['rpm']:5.0f}  "
        f"steer {truck['steering_effective']:+.3f}  "
        f"thr {truck['throttle']:.2f} brk {truck['brake']:.2f}  "
        f"hdg {truck['heading_deg']:6.1f}°  "
        f"{publish_hz:5.1f} Hz  "
        f"{' '.join(flags)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a status line")
    parser.add_argument("--rate", type=float, default=10.0, help="samples per second (default 10)")
    args = parser.parse_args()

    try:
        reader = TelemetryReader()
        reader.open()
        # Prove the game is actually publishing before reporting anything. A
        # lone read cannot distinguish live telemetry from a block left behind
        # by a game that has already exited.
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot read telemetry: {exc}", file=sys.stderr)
        return 1

    interval = 1.0 / max(args.rate, 0.1)
    previous_count: int | None = None
    previous_time: float | None = None
    publish_hz = 0.0
    retried = 0
    samples = 0
    stale = False

    try:
        while True:
            try:
                snapshot = reader.read()
            except StaleTelemetryError as exc:
                # Not fatal for a monitor: a paused game and a closed one both
                # look like this, and both are worth sitting through rather
                # than making the user restart the tool.
                if not stale:
                    print(f"[stale] {exc}", file=sys.stderr)
                    stale = True
                if args.once:
                    return 1
                time.sleep(interval)
                continue

            if stale:
                print("[live] publishing resumed", file=sys.stderr)
                stale = False

            summary = summarise(snapshot)
            samples += 1
            if snapshot.attempts > 1:
                retried += 1

            if previous_count is not None and previous_time is not None:
                published = snapshot.write_count - previous_count
                elapsed = snapshot.write_time - previous_time
                if published > 0 and elapsed > 0:
                    publish_hz = published / elapsed
            previous_count, previous_time = snapshot.write_count, snapshot.write_time

            if args.json:
                summary["publish_hz"] = round(publish_hz, 2)
                print(json.dumps(summary, indent=2))
            else:
                print(format_line(summary, publish_hz))

            if args.once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
    except (PalError, TelemetryError) as exc:
        print(f"\ntelemetry lost: {exc}", file=sys.stderr)
        return 1
    finally:
        reader.close()
        if samples and not args.once:
            print(f"{samples} samples, {retried} needed a retry ({100.0 * retried / samples:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
