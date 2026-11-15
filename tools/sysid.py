#!/usr/bin/env python3
"""Run system-identification manoeuvres and record them.

Phase 3. The controllers that come later are only as good as the model they are
designed against, and the way to get that model is to drive the vehicle
deliberately and measure what it does.

    python tools/sysid.py steering-steps        # stationary, no space needed
    python tools/sysid.py throttle-sweep        # needs a long straight road
    python tools/sysid.py brake-sweep
    python tools/sysid.py coastdown

Every run writes a recording under `data/sysid/`, so the fitting happens
offline and can be redone without driving again. The analysis lives in
`tools/analyse_sysid.py`; this tool only collects.

Safety is the same shape as `longitudinal_probe.py` and for the same reason:

* a speed ceiling that aborts into full braking,
* a whole-run deadline,
* steering held closed-loop at zero, because releasing the command does not
  straighten the wheel -- it only stops turning it,
* a disengage on every exit path, with the plugin's 200 ms watchdog behind it.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.command import Commander, ControlCommand  # noqa: E402
from core.bridge.reader import TelemetryError, TelemetryReader, TelemetrySnapshot  # noqa: E402
from core.common.controls import Button  # noqa: E402
from core.control.limits import (  # noqa: E402
    SafetyMonitor,
    UnsafeCondition,
    WrongDirection,
)
from core.control.steering import SteeringPositionController  # noqa: E402
from core.recording import Recorder  # noqa: E402
from pal import PalError  # noqa: E402

MS_TO_KMH = 3.6
COMMAND_INTERVAL = 0.005

#: Hard ceiling for every manoeuvre here, whatever it asks for.
ABSOLUTE_SPEED_LIMIT_KMH = 75.0

#: Whole-run deadline. Generous, because a sweep is many cycles; the speed
#: ceiling and the distance budget are what actually bound the risk.
RUN_DEADLINE_SECONDS = 900.0

#: Manoeuvres that never move the truck, and so need no road.
STATIONARY = frozenset({"steering-steps"})

#: Below this there is not enough road for even one accelerate-and-stop cycle,
#: so the run would abort part-way through its first measurement.
MIN_DISTANCE_M = 80.0


class Paused(Exception):
    """The game is paused. The step did nothing and should be retried.

    Not an error: the player opening the map is a normal thing to happen
    mid-run, and a manoeuvre should wait it out rather than abort.
    """


@dataclass(slots=True)
class Phase:
    """A labelled stretch of a run, written into the recording's manifest."""

    name: str
    start: float
    end: float = 0.0
    detail: str = ""


class Session:
    """Drives one manoeuvre while recording, and keeps the truck inside limits.

    Steering is held closed-loop at zero throughout. That is not tidiness: the
    game integrates steering commands and holds the result, so a manoeuvre that
    simply stops steering drives off on whatever angle it inherited.
    """

    def __init__(
        self,
        commander: Commander,
        reader: TelemetryReader,
        recorder: Recorder,
        *,
        speed_limit_kmh: float = ABSOLUTE_SPEED_LIMIT_KMH,
        distance_budget_m: float | None = None,
    ) -> None:
        self._commander = commander
        self._reader = reader
        self._recorder = recorder
        self._guard = SafetyMonitor(
            speed_limit_ms=min(speed_limit_kmh, ABSOLUTE_SPEED_LIMIT_KMH) / MS_TO_KMH,
            distance_budget_m=distance_budget_m,
        )
        self._steering = SteeringPositionController()
        self._started = time.perf_counter()
        self.phases: list[Phase] = []
        self._phase: Phase | None = None
        self._last_write_count = -1
        self.duplicate_reads = 0
        self.expect_forward = False
        self._paused_reported = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    @contextlib.contextmanager
    def phase(self, name: str, detail: str = "") -> Iterator[Phase]:
        """Label the next stretch of the run."""
        entry = Phase(name=name, start=self.elapsed, detail=detail)
        self.phases.append(entry)
        self._phase = entry
        try:
            yield entry
        finally:
            entry.end = self.elapsed
            self._phase = None

    def step(
        self,
        *,
        throttle: float = 0.0,
        brake: float = 0.0,
        steer_target: float = 0.0,
        buttons: frozenset[Button] = frozenset(),
    ) -> TelemetrySnapshot:
        """Publish one command and read one frame, recording both.

        `steer_target` is an *angle*, not a rate: the inner loop converts it.
        It defaults to zero because a manoeuvre that stops commanding steering
        keeps whatever angle it had, which is how a straight-line test ends up
        in a ditch.
        """
        if self.elapsed > RUN_DEADLINE_SECONDS:
            raise TimeoutError("run exceeded its deadline")

        snapshot = self._reader.read()

        # A paused game ignores commands and applies whatever is held the
        # instant it resumes, so holding throttle here is a lurch waiting for
        # the player to close the map. It also looks exactly like an obstacle
        # to the stuck detector, which is how a paused game once got reported
        # as "something is in the way".
        if snapshot.is_paused:
            self._commander.send(ControlCommand(), engaged=False)
            if not self._paused_reported:
                print("\n[game paused -- controls released, waiting]", file=sys.stderr)
                self._paused_reported = True
            raise Paused
        if self._paused_reported:
            print("[resumed]", file=sys.stderr)
            self._paused_reported = False

        angle = float(snapshot.block["truck"]["effective_steering"])
        steer = self._steering.command(target=steer_target, measured=angle)

        self._commander.send(
            ControlCommand(steer=steer, throttle=throttle, brake=brake, buttons=buttons)
        )

        # The loop runs faster than the game publishes, so the same frame comes
        # back several times. Recording duplicates inflates the file and, worse,
        # makes the samples look uniformly spaced -- which any differentiation
        # in the analysis would quietly believe.
        if snapshot.write_count != self._last_write_count:
            self._last_write_count = snapshot.write_count
            self._recorder.write_telemetry(snapshot)
        else:
            self.duplicate_reads += 1

        position = snapshot.block["truck"]["placement"]["position"]
        self._guard.check(
            speed=float(snapshot.block["truck"]["speed"]),
            throttle=throttle,
            position=(float(position["x"]), float(position["z"])),
            expect_forward=self.expect_forward,
        )
        return snapshot

    @property
    def distance_travelled(self) -> float:
        return self._guard.distance_travelled

    def hold(
        self,
        seconds: float,
        *,
        throttle: float = 0.0,
        brake: float = 0.0,
        steer_target: float = 0.0,
        buttons: frozenset[Button] = frozenset(),
    ) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.step(throttle=throttle, brake=brake, steer_target=steer_target, buttons=buttons)
            time.sleep(COMMAND_INTERVAL)

    def hold_until(
        self,
        predicate: Callable[[TelemetrySnapshot], bool],
        timeout: float,
        *,
        throttle: float = 0.0,
        brake: float = 0.0,
    ) -> bool:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                if predicate(self.step(throttle=throttle, brake=brake)):
                    return True
            except Paused:
                # Paused time does not count against the timeout: a manoeuvre
                # interrupted by the map should resume, not expire.
                deadline += COMMAND_INTERVAL
            time.sleep(COMMAND_INTERVAL)
        return False

    def stop(self, *, brake: float = 0.45, timeout: float = 30.0) -> None:
        """Brake to a standstill, then release everything.

        Releasing matters. Holding the brake after stopping means holding
        `abackward` at a standstill, which the game reads as a request to
        reverse -- so the command meant to keep the truck still is the one that
        drives it away. With nothing commanded the game applies its own hold
        brake, which is what actually keeps it put.
        """
        was_forward, self.expect_forward = self.expect_forward, False
        try:
            # Suspended for the duration: the budget is already overspent and
            # the truck is about to be stationary with the brake on, so both
            # limits would fire and make the stop unrunnable.
            with self._guard.suspend():
                self.hold_until(
                    lambda s: abs(float(s.block["truck"]["speed"])) < 0.15, timeout, brake=brake
                )
                self.hold(1.0)
        finally:
            self.expect_forward = was_forward

    def settle(self, seconds: float = 1.5) -> None:
        """Sit still with nothing commanded, letting the drivetrain come to rest."""
        was_forward, self.expect_forward = self.expect_forward, False
        try:
            with self._guard.suspend():
                self.hold(seconds)
        finally:
            self.expect_forward = was_forward

    def accelerate_to(self, speed_kmh: float, *, throttle: float, timeout: float = 90.0) -> bool:
        target = speed_kmh / MS_TO_KMH
        return self.hold_until(
            lambda s: float(s.block["truck"]["speed"]) >= target, timeout, throttle=throttle
        )

    def straighten(self, timeout: float = 4.0) -> None:
        """Bring the wheel back to centre before moving off."""
        self.hold_until(
            lambda s: abs(float(s.block["truck"]["effective_steering"])) < 0.002, timeout
        )


# --------------------------------------------------------------------------- #
# Manoeuvres
# --------------------------------------------------------------------------- #


def steering_steps(session: Session, args: argparse.Namespace) -> None:
    """Static steering map: command to applied angle to road wheel angle.

    Stationary, so it needs no space. Each step returns to centre first, which
    matters because the steering integrates and holds -- without it, every
    measurement would start from wherever the last one finished.
    """
    targets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    for target in targets:
        for sign in (1.0, -1.0):
            with session.phase("centre"):
                session.straighten()
            with session.phase("steer", detail=f"{sign * target:+.2f}"):
                session.hold(2.5, steer_target=sign * target)
    with session.phase("centre"):
        session.straighten()


def throttle_sweep(session: Session, args: argparse.Namespace) -> None:
    """Acceleration at a fixed throttle, for a fixed time from rest.

    Repeated per throttle level. Fixed *time* rather than a target speed,
    because a low throttle takes a long time to reach any given speed and the
    road is finite: an eight-second pull is roughly fifty metres whatever the
    throttle, while "accelerate to 45 km/h at 20 %" is several hundred.

    The gearbox dominates a pull from rest, so the analysis segments by gear
    rather than averaging across the run -- which is exactly what made the
    first-drive figure unusable as a map.
    """
    for throttle in args.levels or (0.2, 0.35, 0.5, 0.7, 1.0):
        with session.phase("settle"):
            session.settle()
            session.straighten()
        with session.phase("accelerate", detail=f"throttle={throttle:.2f}"):
            session.expect_forward = True
            # Whichever comes first: the time budget or the speed target.
            session.hold_until(
                lambda s: float(s.block["truck"]["speed"]) >= args.speed / MS_TO_KMH,
                args.burst,
                throttle=throttle,
            )
        session.expect_forward = False
        with session.phase("brake", detail="return to rest"):
            session.stop()


def brake_sweep(session: Session, args: argparse.Namespace) -> None:
    """Deceleration at a fixed brake command, from the target speed.

    The comfort limits in Phase 6 need this: braking authority turned out to be
    far higher than a comfortable ACC should ever use, and the shape of the
    command-to-deceleration curve is what decides where to cap it.
    """
    for brake in args.levels or (0.15, 0.25, 0.4, 0.6, 0.85):
        with session.phase("settle"):
            session.settle()
            session.straighten()
        with session.phase("accelerate", detail=f"to {args.speed:.0f} km/h"):
            session.expect_forward = True
            # Time-capped like the throttle sweep: an unbounded run-up is how a
            # five-cycle experiment quietly needs a kilometre of road.
            session.accelerate_to(args.speed, throttle=0.7, timeout=args.burst * 2.0)
        with session.phase("coast", detail="let the drivetrain settle"):
            session.hold(1.0)
        with session.phase("brake", detail=f"brake={brake:.2f}"):
            if args.brake_hold:
                # A fixed pulse rather than a stop. Braking to rest at a light
                # command takes a minute and several hundred metres, and the
                # light end is exactly the part worth mapping: measuring the
                # slope over three seconds says the same thing in twenty
                # metres.
                session.hold(args.brake_hold, brake=brake)
            else:
                session.hold_until(
                    lambda s: abs(float(s.block["truck"]["speed"])) < 0.2, 60.0, brake=brake
                )
        with session.phase("settle"):
            session.settle(1.0)


def coastdown(session: Session, args: argparse.Namespace) -> None:
    """Deceleration with the throttle released, against speed.

    Not a true coast-down. The automatic stays in gear, so what is measured is
    lift-off deceleration: aerodynamic drag, rolling resistance *and* engine
    braking together. Separating them would need neutral, which the input
    device cannot select.

    That is the more useful quantity anyway -- it is what happens when a
    longitudinal controller stops asking for throttle -- but it must not be
    fitted as pure drag. The aerodynamic term is the part that grows with the
    square of speed, so the fit needs the widest speed range the road allows.

    The coast is time-bounded rather than run to a standstill: at these speeds
    the deceleration is small, and coasting to rest would take several minutes
    and most of a kilometre.
    """
    # Stop coasting once most of the speed has bled off; there is nothing left
    # to learn below a crawl. Bound outside the loop so the closure below binds
    # a constant rather than a loop variable.
    floor = max(args.speed * 0.25, 8.0) / MS_TO_KMH

    def below_floor(snapshot: TelemetrySnapshot) -> bool:
        return float(snapshot.block["truck"]["speed"]) < floor

    for repeat in range(args.repeats):
        with session.phase("settle"):
            session.settle()
            session.straighten()
        with session.phase("accelerate", detail=f"to {args.speed:.0f} km/h"):
            session.expect_forward = True
            if not session.accelerate_to(args.speed, throttle=1.0, timeout=args.burst * 3.0):
                print(f"  run {repeat + 1}: never reached {args.speed:.0f} km/h")
        session.expect_forward = False
        with session.phase("coast", detail=f"run={repeat + 1}"):
            session.hold_until(below_floor, args.coast_hold)
        with session.phase("brake", detail="brake=0.45"):
            session.stop()


def engine_brake(session: Session, args: argparse.Namespace) -> None:
    """Deceleration from engine braking, against lift-off alone.

    This exists because the service brake cannot decelerate gently. It does
    nothing below roughly 0.055 of command and then gives 1.8 m/s^2, and this
    truck has no retarder, so if a comfortable ACC is to hold a gap without
    lurching, engine braking is the remaining candidate.

    Each run coasts first and then engages, so the comparison is against this
    road and this speed rather than against a figure from another day. The
    difference between the two is the quantity of interest; the absolute
    deceleration includes drag, rolling resistance and whatever gradient the
    road carries.
    """
    for repeat in range(args.repeats):
        with session.phase("settle"):
            session.settle()
            session.straighten()
        with session.phase("accelerate", detail=f"to {args.speed:.0f} km/h"):
            session.expect_forward = True
            session.accelerate_to(args.speed, throttle=1.0, timeout=args.burst * 3.0)
        session.expect_forward = False

        with session.phase("coast", detail=f"run={repeat + 1}"):
            session.hold(args.coast_hold / 2.0)
        with session.phase("engine-brake", detail=f"run={repeat + 1}"):
            session.hold(args.coast_hold / 2.0, buttons=frozenset({Button.MOTOR_BRAKE}))

        with session.phase("brake", detail="brake=0.45"):
            session.stop()


def throttle_taper(session: Session, args: argparse.Namespace) -> None:
    """Deceleration at partial throttle, stepping down from a held speed.

    The question this answers: can the range between lift-off (0.22 m/s^2) and
    the brake's floor (1.8) be commanded at all? Neither the service brake nor
    the engine brake reaches it, and it is where an ACC holding a gap would
    spend most of its time.

    At any speed and gear there is a throttle that exactly holds the speed.
    Below it the truck decelerates, continuously and by as little as you like
    -- if the relationship is smooth enough to command.

    Descending rather than one level per run, because a run-up per level would
    need a kilometre. The cost is that speed decays through the sweep, so
    throttle and speed are confounded; the analysis records both and the result
    is a first look rather than a clean map. Gear changes are the other
    confound and are why the steps are short.
    """
    levels = args.levels or (0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0)

    with session.phase("settle"):
        session.settle()
        session.straighten()
    with session.phase("accelerate", detail=f"to {args.speed:.0f} km/h"):
        session.expect_forward = True
        session.accelerate_to(args.speed, throttle=1.0, timeout=args.burst * 3.0)
    session.expect_forward = False

    for level in levels:
        with session.phase("taper", detail=f"throttle={level:.2f}"):
            session.hold(args.step_hold, throttle=level)

    with session.phase("brake", detail="brake=0.45"):
        session.stop()


MANOEUVRES: dict[str, Callable[[Session, argparse.Namespace], None]] = {
    "steering-steps": steering_steps,
    "throttle-sweep": throttle_sweep,
    "brake-sweep": brake_sweep,
    "coastdown": coastdown,
    "engine-brake": engine_brake,
    "throttle-taper": throttle_taper,
}


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manoeuvre", choices=sorted(MANOEUVRES))
    parser.add_argument("--speed", type=float, default=45.0, help="target speed, km/h")
    parser.add_argument("--repeats", type=int, default=3, help="coastdown runs")
    parser.add_argument(
        "--burst",
        type=float,
        default=8.0,
        help="seconds of acceleration per throttle level (bounds the distance used)",
    )
    parser.add_argument(
        "--coast-hold",
        type=float,
        default=20.0,
        help="seconds to coast per run (bounds the distance used)",
    )
    parser.add_argument(
        "--brake-hold",
        type=float,
        default=0.0,
        help="brake for this many seconds instead of to a standstill",
    )
    parser.add_argument(
        "--step-hold",
        type=float,
        default=2.5,
        help="seconds per level in throttle-taper",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=None,
        help="command levels to sweep, overriding the manoeuvre's defaults",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="metres of clear road ahead (required for manoeuvres that drive)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if args.speed > ABSOLUTE_SPEED_LIMIT_KMH:
        print(f"target above the {ABSOLUTE_SPEED_LIMIT_KMH:.0f} km/h ceiling", file=sys.stderr)
        return 2

    # Required, not optional, for anything that moves. Leaving it optional
    # meant it could be omitted or filled in from memory, and twice it was
    # filled in from a figure that had gone stale -- once after the truck had
    # been repositioned, at a cost of ten points of chassis damage. The harness
    # cannot see the road; the operator can; so the operator has to say.
    if args.manoeuvre not in STATIONARY and args.distance is None:
        print(
            f"'{args.manoeuvre}' drives the truck, so --distance is required.\n"
            "Give the metres of clear road actually ahead of it *now* -- not what\n"
            "was there before the last run, a reposition or a restart.",
            file=sys.stderr,
        )
        return 2
    if args.distance is not None and args.distance < MIN_DISTANCE_M:
        print(
            f"--distance {args.distance:.0f} m is below the {MIN_DISTANCE_M:.0f} m "
            "a single accelerate-and-stop cycle needs",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    output = args.output or REPO_ROOT / "data" / "sysid" / f"{stamp}_{args.manoeuvre}"

    if not args.yes and args.manoeuvre not in STATIONARY:
        print(f"'{args.manoeuvre}' drives the truck up to {args.speed:.0f} km/h and back to rest,")
        print("repeatedly. Make sure the road ahead is long, straight and clear.")
        print("Ctrl-C at any point disengages.")
        try:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                return 1
        except EOFError:
            return 1

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot reach the game: {exc}", file=sys.stderr)
        return 1

    block = reader.read().block
    metadata = {
        "manoeuvre": args.manoeuvre,
        "target_speed_kmh": args.speed,
        "truck": f"{_text(block['truck']['brand_id'])}/{_text(block['truck']['model_id'])}",
        "trailer": _text(block["trailers"][0]["id"]),
        "trailer_connected": bool(block["trailers"][0]["connected"]),
        "cargo": _text(block["job"]["cargo"]),
        "cargo_mass_kg": float(block["job"]["cargo_mass"]),
        "cargo_loaded": bool(block["job"]["cargo_loaded"]),
        "game_version": int(block["header"]["game_version"]),
    }

    recorder = Recorder(output, notes=f"sysid: {args.manoeuvre}", metadata=metadata)
    commander = Commander(recorder=recorder)
    try:
        recorder.open()
        commander.open()
    except (OSError, PalError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        recorder.close()
        reader.close()
        return 1

    print(f"\n{args.manoeuvre} -> {output}")
    print(
        f"truck {metadata['truck']}, trailer {metadata['trailer']}, "
        f"cargo {metadata['cargo_mass_kg']:.0f} kg\n"
    )

    session = Session(
        commander,
        reader,
        recorder,
        speed_limit_kmh=args.speed + 15.0,
        distance_budget_m=args.distance,
    )
    outcome = 0
    try:
        MANOEUVRES[args.manoeuvre](session, args)
    except KeyboardInterrupt:
        print("\ninterrupted -- braking to a stop", file=sys.stderr)
        with contextlib.suppress(Exception):
            session.stop(timeout=15.0)
        outcome = 1
    except UnsafeCondition as exc:
        print(f"\nSTOP: {exc} -- braking to rest", file=sys.stderr)
        if isinstance(exc, WrongDirection):
            print(
                "The truck is in reverse. Something held `brake` at a standstill.",
                file=sys.stderr,
            )
        with contextlib.suppress(Exception):
            session.stop(timeout=20.0)
        outcome = 1
    except (PalError, TelemetryError, TimeoutError) as exc:
        print(f"\nrun stopped: {exc}", file=sys.stderr)
        outcome = 1
    finally:
        with contextlib.suppress(Exception):
            commander.disengage()
        commander.close()
        recorder.close()
        reader.close()

    _write_phases(output, session)
    print(
        f"\n{recorder.telemetry_count} telemetry frames, "
        f"{recorder.command_count} commands, {recorder.duration:.1f}s"
    )
    print(f"{session.distance_travelled:.0f} m of road used")
    print(f"recording: {output}")
    return outcome


def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _write_phases(output: Path, session: Session) -> None:
    """Append the phase log to the manifest, so the analysis knows what is what.

    Written after `close()` rather than through the recorder: phases are this
    tool's idea, not part of the recording format, and a manoeuvre that aborted
    still has phases worth keeping.
    """
    import json

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phases"] = [
        {"name": p.name, "start": round(p.start, 3), "end": round(p.end, 3), "detail": p.detail}
        for p in session.phases
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
