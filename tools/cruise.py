#!/usr/bin/env python3
"""Hold a target speed. The first thing the stack does on its own.

    python tools/cruise.py --speed 40 --distance 400

Longitudinal control only. It needs a straight road and it will not stay in a
lane -- the driver steers.

It sends **nothing** on the steering axis, deliberately. The game's steering
integrates the commanded rate and holds the resulting angle, so commanding zero
leaves the wheel exactly where the driver puts it. An earlier version ran a
position loop holding the angle at zero, and it undid every correction the
driver made to stay on the road. A speed controller has no business on the
steering axis.

Everything it does is recorded, so the run can be replayed and the controller
re-tuned without driving again -- which matters here, because the gains are a
starting point taken from a simulation that has no engine lag, no gearbox and
no noise.

Safety is `core.control.limits`: a speed ceiling, the distance budget the
operator supplies, a stuck detector and a direction check, with the whole lot
suspended while braking to rest so the guards cannot block their own stop.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge import convert  # noqa: E402
from core.bridge.command import Commander, ControlCommand  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from core.control.limits import SafetyMonitor, UnsafeCondition  # noqa: E402
from core.control.longitudinal import SpeedController  # noqa: E402
from core.recording import Recorder  # noqa: E402
from pal import PalError  # noqa: E402

MS_TO_KMH = 3.6
LOOP_INTERVAL = 0.005

#: Hard ceiling whatever is asked for.
ABSOLUTE_SPEED_LIMIT_KMH = 75.0

#: Below this there is not enough road to reach a speed and stop again.
MIN_DISTANCE_M = 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, required=True, help="target speed, km/h")
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="metres of clear straight road ahead *now*",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="seconds before stopping")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if args.speed > ABSOLUTE_SPEED_LIMIT_KMH:
        print(f"target above the {ABSOLUTE_SPEED_LIMIT_KMH:.0f} km/h ceiling", file=sys.stderr)
        return 2
    if args.distance < MIN_DISTANCE_M:
        print(f"--distance below the {MIN_DISTANCE_M:.0f} m minimum", file=sys.stderr)
        return 2

    if not args.yes:
        print(f"This drives the truck at {args.speed:.0f} km/h in a straight line")
        print(f"for up to {args.duration:.0f} s or {args.distance:.0f} m, then stops.")
        print("It does not steer. Ctrl-C disengages.")
        try:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                return 1
        except EOFError:
            return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    output = args.output or REPO_ROOT / "data" / "cruise" / f"{stamp}_cruise"

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot reach the game: {exc}", file=sys.stderr)
        return 1

    if reader.read().is_paused:
        print(
            "the game is paused or showing a menu -- close the map and try again",
            file=sys.stderr,
        )
        reader.close()
        return 1

    recorder = Recorder(
        output,
        notes=f"cruise at {args.speed:.0f} km/h",
        metadata={"target_speed_kmh": args.speed},
    )
    commander = Commander(recorder=recorder)
    try:
        recorder.open()
        commander.open()
    except (OSError, PalError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        recorder.close()
        reader.close()
        return 1

    speed_controller = SpeedController()
    guard = SafetyMonitor(
        speed_limit_ms=(args.speed + 15.0) / MS_TO_KMH,
        distance_budget_m=args.distance,
    )
    target = args.speed / MS_TO_KMH

    print(f"\nholding {args.speed:.0f} km/h -> {output}\n")
    outcome = 0
    last_write_count = -1
    paused_reported = False
    previous = time.perf_counter()
    started = previous
    last_print = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - previous
            previous = now
            if now - started > args.duration:
                break

            snapshot = reader.read()
            block = snapshot.block
            truck = block["truck"]

            # Paused means the map or a menu is up. The game ignores commands
            # while it is, and applies whatever is held the instant it
            # resumes -- so holding throttle here is a lurch waiting for the
            # player to close the map. Send neutral, freeze the loop, and do
            # not let the guards interpret a stationary truck as an obstacle.
            if snapshot.is_paused:
                commander.send(ControlCommand(), engaged=False)
                speed_controller.reset()
                previous = time.perf_counter()
                if not paused_reported:
                    print("\n[paused -- controls released, waiting]", file=sys.stderr)
                    paused_reported = True
                time.sleep(0.05)
                continue
            if paused_reported:
                print("[resumed]", file=sys.stderr)
                paused_reported = False

            command = speed_controller.step(
                target_speed=target,
                speed=float(truck["speed"]),
                grade=convert.grade_acceleration(block),
                dt=max(dt, 1e-4),
            )
            # No steering command: zero means "do not turn", and the wheel
            # holds whatever angle the driver has given it.
            commander.send(ControlCommand(throttle=command.throttle, brake=command.brake))

            if snapshot.write_count != last_write_count:
                last_write_count = snapshot.write_count
                recorder.write_telemetry(snapshot)

            position = truck["placement"]["position"]
            guard.check(
                speed=float(truck["speed"]),
                throttle=command.throttle,
                position=(float(position["x"]), float(position["z"])),
                expect_forward=True,
            )

            if now - last_print > 0.5:
                last_print = now
                error = (float(truck["speed"]) - target) * MS_TO_KMH
                print(
                    f"\r{convert.speed_kmh(block):6.2f} km/h  err {error:+6.2f}  "
                    f"req {speed_controller.last_request:+5.2f} m/s2  "
                    f"thr {command.throttle:.3f} brk {command.brake:.3f}  "
                    f"grade {convert.grade_acceleration(block):+5.2f}  "
                    f"{guard.distance_travelled:5.0f} m  {command.reason}",
                    end="",
                    flush=True,
                )
            time.sleep(LOOP_INTERVAL)
    except UnsafeCondition as exc:
        print(f"\n\nSTOP: {exc}", file=sys.stderr)
        outcome = 1
    except KeyboardInterrupt:
        print("\n\ninterrupted", file=sys.stderr)
    except (PalError, TelemetryError) as exc:
        print(f"\n\nlost the game: {exc}", file=sys.stderr)
        outcome = 1
    finally:
        with contextlib.suppress(Exception):
            _brake_to_rest(commander, reader, guard)
        with contextlib.suppress(Exception):
            commander.disengage()
        commander.close()
        recorder.close()
        reader.close()

    print(
        f"\n\n{recorder.telemetry_count} frames, {recorder.duration:.1f}s, "
        f"{guard.distance_travelled:.0f} m"
    )
    print(f"recording: {output}")
    return outcome


def _brake_to_rest(
    commander: Commander,
    reader: TelemetryReader,
    guard: SafetyMonitor,
    timeout: float = 25.0,
) -> None:
    """Bring the truck to a standstill, with the guards suspended.

    Suspended because the budget is very likely already overspent and the truck
    is about to be stationary: both limits trip on states a stop must pass
    through, and a guard that blocks its own stop leaves the vehicle rolling.
    """
    deadline = time.perf_counter() + timeout
    with guard.suspend():
        while time.perf_counter() < deadline:
            block = reader.read().block
            if abs(float(block["truck"]["speed"])) < 0.15:
                break
            commander.send(ControlCommand(brake=0.45))
            time.sleep(LOOP_INTERVAL)
    # Release: holding the brake at a standstill is a request to reverse.
    for _ in range(60):
        commander.send(ControlCommand())
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
