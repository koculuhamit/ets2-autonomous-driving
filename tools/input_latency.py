#!/usr/bin/env python3
"""Measure command-to-game latency, and answer ADR-0002.

Applies a steering step through the plugin's input device and watches
telemetry for the response. What comes back is the number the whole control
design rests on: a controller cannot be tuned tighter than its actuator delay,
and guessing it is how oscillation gets designed in.

    python tools/input_latency.py                 # steering step, 20 repeats
    python tools/input_latency.py --axis throttle

Park the truck somewhere flat with the parking brake on before running the
steering test. The wheels turn; the truck should not.

Three latencies are reported, and they are not the same thing:

* **command to effective** -- how long until the game acknowledges the input in
  `effective_steering`. This is the input pipeline alone.
* **command to wheel** -- how long until the front wheels actually move. This
  includes the game's steering actuator, and it is what the lateral controller
  is really fighting.
* **publish interval** -- how long the plugin's own frames take, so the other
  two can be read as multiples of a frame rather than as absolute numbers that
  say nothing about where the delay lives.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.command import Commander, ControlCommand  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from pal import PalError  # noqa: E402

#: How long to wait for a response before calling the step a miss.
RESPONSE_TIMEOUT = 1.0

#: Smallest change that counts as a response.
#:
#: Latency is the delay before a command has *any* effect, so the threshold
#: belongs at the noise floor, not at some fraction of the commanded value. The
#: telemetry values are exact, and the steering integrator moves about 0.006 per
#: frame at half command, so this triggers on the first frame that carries the
#: command and on no frame before it.
#:
#: Getting this wrong is easy and quiet: a threshold set to a fraction of the
#: step measures how long the plant took to travel that far, which for an
#: integrating plant is a slew time wearing a latency's clothes.
RESPONSE_EPSILON = 1e-4

#: Commands must keep flowing or the plugin's watchdog releases the controls,
#: so every wait loop publishes rather than sleeping quietly.
COMMAND_INTERVAL = 0.005


def _effective(block: object, axis: str) -> float:
    field = "steering" if axis == "steering" else axis
    return float(block["truck"][f"effective_{field}"])  # type: ignore[index]


def _wheel_angle(block: object) -> float:
    return float(block["truck"]["wheels"]["steering"][0])  # type: ignore[index]


def settle(
    commander: Commander, reader: TelemetryReader, command: ControlCommand, seconds: float
) -> None:
    """Hold a command steadily for `seconds`, keeping the watchdog fed."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        commander.send(command)
        reader.read()
        time.sleep(COMMAND_INTERVAL)


def measure_step(
    commander: Commander,
    reader: TelemetryReader,
    axis: str,
    magnitude: float,
) -> tuple[float, float] | None:
    """Apply one step and time the first response.

    Returns (command_to_effective, command_to_wheel) in seconds, with the
    second NaN for non-steering axes, or None if nothing moved in time.
    """
    settle(commander, reader, ControlCommand.neutral(), 0.25)

    baseline = reader.read().block
    baseline_effective = _effective(baseline, axis)
    baseline_wheel = _wheel_angle(baseline)

    step = ControlCommand(**{"steer" if axis == "steering" else axis: magnitude})

    commander.send(step)
    sent_at = time.perf_counter()

    effective_at: float | None = None
    wheel_at: float | None = None

    deadline = sent_at + RESPONSE_TIMEOUT
    while time.perf_counter() < deadline:
        commander.send(step)  # keep the watchdog fed
        snapshot = reader.read()
        now = time.perf_counter()

        moved = abs(_effective(snapshot.block, axis) - baseline_effective) > RESPONSE_EPSILON
        if effective_at is None and moved:
            effective_at = now

        wheel_moved = abs(_wheel_angle(snapshot.block) - baseline_wheel) > RESPONSE_EPSILON
        if axis == "steering" and wheel_at is None and wheel_moved:
            wheel_at = now
        if effective_at is not None and (wheel_at is not None or axis != "steering"):
            break
        time.sleep(0.001)

    settle(commander, reader, ControlCommand.neutral(), 0.15)

    if effective_at is None:
        return None
    return (
        effective_at - sent_at,
        (wheel_at - sent_at) if wheel_at is not None else float("nan"),
    )


def report(name: str, samples: list[float]) -> None:
    usable = [s for s in samples if s == s]  # drop NaN
    if not usable:
        print(f"{name:<24} no response")
        return
    print(
        f"{name:<24} median {statistics.median(usable) * 1000:6.1f} ms   "
        f"min {min(usable) * 1000:6.1f}   max {max(usable) * 1000:6.1f}   "
        f"n={len(usable)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        default="steering",
        choices=["steering", "throttle", "brake"],
        help="which axis to step (default: steering)",
    )
    parser.add_argument("--magnitude", type=float, default=0.5, help="step size (default 0.5)")
    parser.add_argument("--repeats", type=int, default=20, help="steps to average (default 20)")
    args = parser.parse_args()

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"telemetry unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        commander = Commander()
        commander.open()
    except PalError as exc:
        print(f"command block unavailable: {exc}", file=sys.stderr)
        print("Is this plugin build the one with input support?", file=sys.stderr)
        reader.close()
        return 1

    print(f"stepping {args.axis} to {args.magnitude} x{args.repeats}\n")

    effective_samples: list[float] = []
    wheel_samples: list[float] = []
    publish_intervals: list[float] = []
    misses = 0

    try:
        previous = reader.read()
        for index in range(args.repeats):
            # Steering integrates and holds, so stepping the same way every
            # time would walk it into the end stop. Alternating keeps it near
            # centre without needing a recentring loop between samples.
            magnitude = args.magnitude * (1 if index % 2 == 0 else -1)
            result = measure_step(commander, reader, args.axis, magnitude)
            if result is None:
                misses += 1
                print(f"  step {index + 1:2d}: no response")
                continue
            to_effective, to_wheel = result
            effective_samples.append(to_effective)
            wheel_samples.append(to_wheel)

            current = reader.read()
            published = current.write_count - previous.write_count
            elapsed = current.write_time - previous.write_time
            if published > 0 and elapsed > 0:
                publish_intervals.append(elapsed / published)
            previous = current

            wheel_text = f"{to_wheel * 1000:6.1f}" if to_wheel == to_wheel else "     -"
            print(
                f"  step {index + 1:2d}: effective {to_effective * 1000:6.1f} ms   "
                f"wheel {wheel_text} ms"
            )
    except (PalError, TelemetryError) as exc:
        print(f"\nlost the game mid-measurement: {exc}", file=sys.stderr)
        return 1
    finally:
        commander.close()
        reader.close()

    print()
    report("command to effective", effective_samples)
    if args.axis == "steering":
        report("command to wheel", wheel_samples)
    report("publish interval", publish_intervals)
    if misses:
        print(f"\n{misses} of {args.repeats} steps drew no response")
    return 0 if effective_samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
