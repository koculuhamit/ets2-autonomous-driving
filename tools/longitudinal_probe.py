#!/usr/bin/env python3
"""Drive the truck briefly to characterise the longitudinal actuators.

The first experiment in which the stack actually moves the vehicle, and the
seed of the Phase 3 system identification: what a throttle command does to
acceleration, what `abackward` does when the truck is already moving, and how
long a stop takes.

    python tools/longitudinal_probe.py --target-speed 15

**Clear space is required ahead of the truck.** The probe accelerates to the
target speed, brakes to a stop, and disengages. Nothing steers.

Safety, in the order it applies:

* A speed ceiling. Exceeding it aborts into full braking and disengages.
* A wall-clock deadline for the whole run, so a probe that fails to stop is
  still over within seconds.
* Every exit path -- success, exception, Ctrl-C -- disengages before returning,
  and the plugin's own watchdog covers the case where this process dies
  outright.

These exist because "it only runs for a moment" is how a test ends up driving
into something.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.command import Commander, ControlCommand  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from pal import PalError  # noqa: E402

MS_TO_KMH = 3.6
COMMAND_INTERVAL = 0.005

#: Hard ceiling regardless of the requested target. A probe is not a drive.
ABSOLUTE_SPEED_LIMIT_KMH = 40.0

#: Whole-run deadline, including the stop.
RUN_DEADLINE_SECONDS = 75.0


@dataclass(slots=True)
class Sample:
    t: float
    speed: float  # m/s
    throttle_cmd: float
    brake_cmd: float
    effective_throttle: float
    effective_brake: float
    gear: int
    acceleration: float  # m/s^2, from telemetry


class SpeedLimitExceeded(RuntimeError):
    """The truck went faster than the probe is allowed to make it go."""


class Probe:
    """Runs one throttle-and-brake cycle, keeping the watchdog fed throughout."""

    def __init__(self, commander: Commander, reader: TelemetryReader, limit_kmh: float) -> None:
        self._commander = commander
        self._reader = reader
        self._limit = limit_kmh / MS_TO_KMH
        self._started = time.perf_counter()
        self.samples: list[Sample] = []

    def step(self, command: ControlCommand) -> Sample:
        """Publish one command, read one snapshot, and check the limits."""
        now = time.perf_counter()
        if now - self._started > RUN_DEADLINE_SECONDS:
            raise TimeoutError("probe exceeded its run deadline")

        self._commander.send(command)
        block = self._reader.read().block
        truck = block["truck"]
        speed = float(truck["speed"])

        sample = Sample(
            t=now - self._started,
            speed=speed,
            throttle_cmd=command.throttle,
            brake_cmd=command.brake,
            effective_throttle=float(truck["effective_throttle"]),
            effective_brake=float(truck["effective_brake"]),
            gear=int(truck["gear"]),
            acceleration=float(truck["linear_acceleration"]["z"]),
        )
        self.samples.append(sample)

        if abs(speed) > self._limit:
            raise SpeedLimitExceeded(f"{speed * MS_TO_KMH:.1f} km/h exceeds the probe limit")
        return sample

    def hold(self, command: ControlCommand, seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.step(command)
            time.sleep(COMMAND_INTERVAL)

    def hold_until(self, command: ControlCommand, predicate, timeout: float) -> bool:
        """Hold a command until `predicate(sample)` or `timeout`."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate(self.step(command)):
                return True
            time.sleep(COMMAND_INTERVAL)
        return False


def summarise(samples: list[Sample], label: str, lo: float, hi: float) -> None:
    window = [s for s in samples if lo <= s.t <= hi]
    if len(window) < 2:
        print(f"{label}: too few samples")
        return
    dv = window[-1].speed - window[0].speed
    dt = window[-1].t - window[0].t
    print(
        f"{label}: {window[0].speed * MS_TO_KMH:5.1f} -> {window[-1].speed * MS_TO_KMH:5.1f} km/h "
        f"in {dt:4.1f}s  mean {dv / dt:+5.2f} m/s^2  "
        f"eff_thr {max(s.effective_throttle for s in window):.2f} "
        f"eff_brk {max(s.effective_brake for s in window):.2f}  "
        f"gear {window[-1].gear}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-speed", type=float, default=15.0, help="km/h (default 15)")
    parser.add_argument(
        "--throttle", type=float, default=0.4, help="throttle command (default 0.4)"
    )
    parser.add_argument("--brake", type=float, default=0.4, help="brake command (default 0.4)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    if args.target_speed > ABSOLUTE_SPEED_LIMIT_KMH:
        print(f"target above the {ABSOLUTE_SPEED_LIMIT_KMH:.0f} km/h ceiling", file=sys.stderr)
        return 2

    if not args.yes:
        print(f"This will drive the truck to {args.target_speed:.0f} km/h and brake to a stop.")
        print("Make sure there is clear space ahead. Ctrl-C at any point disengages.")
        try:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                return 1
        except EOFError:
            return 1

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
        commander = Commander()
        commander.open()
    except (PalError, TelemetryError) as exc:
        print(f"cannot reach the game: {exc}", file=sys.stderr)
        return 1

    probe = Probe(commander, reader, ABSOLUTE_SPEED_LIMIT_KMH)
    target = args.target_speed / MS_TO_KMH
    marks: dict[str, float] = {}

    try:
        probe.hold(ControlCommand(), 0.5)
        marks["accelerate"] = probe.samples[-1].t

        reached = probe.hold_until(
            ControlCommand(throttle=args.throttle), lambda s: s.speed >= target, timeout=30.0
        )
        marks["coast"] = probe.samples[-1].t
        if not reached:
            print("never reached the target speed; is the parking brake still on?")

        probe.hold(ControlCommand(), 1.5)
        marks["brake"] = probe.samples[-1].t

        stopped = probe.hold_until(
            ControlCommand(brake=args.brake), lambda s: abs(s.speed) < 0.15, timeout=25.0
        )
        marks["stopped"] = probe.samples[-1].t
        if not stopped:
            print("did not come to a stop within the timeout")

        probe.hold(ControlCommand(brake=1.0), 1.0)
    except SpeedLimitExceeded as exc:
        print(f"\nABORT: {exc} -- full braking", file=sys.stderr)
        # Already aborting; a further failure here changes nothing and must
        # not stop the disengage in `finally` from running.
        with contextlib.suppress(Exception):
            probe.hold(ControlCommand(brake=1.0), 4.0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    except (PalError, TelemetryError, TimeoutError) as exc:
        print(f"\nprobe stopped: {exc}", file=sys.stderr)
    finally:
        commander.disengage()
        commander.close()
        reader.close()

    print()
    if "coast" in marks:
        summarise(probe.samples, "accelerate", marks["accelerate"], marks["coast"])
    if "brake" in marks:
        summarise(probe.samples, "coast     ", marks["coast"], marks["brake"])
    if "stopped" in marks:
        summarise(probe.samples, "brake     ", marks["brake"], marks["stopped"])

    peak = max((s.speed for s in probe.samples), default=0.0) * MS_TO_KMH
    print(f"\n{len(probe.samples)} samples, peak {peak:.1f} km/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
