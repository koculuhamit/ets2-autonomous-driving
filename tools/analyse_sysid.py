#!/usr/bin/env python3
"""Fit actuator models from system-identification recordings.

    python tools/analyse_sysid.py data/sysid/*_brake-sweep
    python tools/analyse_sysid.py data/sysid            # every run in there

Reads recordings, not the game, so a result can be re-derived, corrected and
argued with long after the truck has been parked. That separation is the point
of recording at all: the driving is expensive and cannot be repeated exactly,
and the analysis is cheap and will be wrong at least once.

Runs are grouped by manoeuvre and pooled, so a sweep split across several runs
-- which is what happens when the road runs out -- reads as one curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.recording import Recording, RecordingFormatError, open_recording  # noqa: E402

MS_TO_KMH = 3.6
G = 9.80665

#: Samples needed before a segment is worth fitting a slope to. At 60 Hz this
#: is about a sixth of a second.
MIN_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class Segment:
    """One labelled stretch of a run, with the command that produced it."""

    manoeuvre: str
    phase: str
    command: float
    t: np.ndarray
    block: np.ndarray
    source: str


def load(paths: list[Path]) -> list[tuple[Recording, dict]]:
    """Open every recording under the given paths."""
    found: list[tuple[Recording, dict]] = []
    for path in paths:
        candidates = [path] if (path / "manifest.json").is_file() else sorted(path.glob("*"))
        for candidate in candidates:
            if not (candidate / "manifest.json").is_file():
                continue
            try:
                recording = open_recording(candidate)
            except RecordingFormatError as exc:
                print(f"skipping {candidate.name}: {exc}", file=sys.stderr)
                continue
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            found.append((recording, manifest))
    return found


def segments(recordings: list[tuple[Recording, dict]]) -> list[Segment]:
    """Split recordings into labelled phases with their command level."""
    out: list[Segment] = []
    for recording, manifest in recordings:
        manoeuvre = manifest.get("metadata", {}).get("manoeuvre", "unknown")
        t = recording.telemetry["t"]
        for phase in manifest.get("phases", []):
            command = _command_from(phase.get("detail", ""))
            if command is None:
                continue
            mask = (t >= phase["start"]) & (t <= phase["end"])
            if mask.sum() < MIN_SAMPLES:
                continue
            out.append(
                Segment(
                    manoeuvre=manoeuvre,
                    phase=phase["name"],
                    command=command,
                    t=t[mask],
                    block=recording.telemetry["block"][mask],
                    source=recording.path.name,
                )
            )
    return out


def _command_from(detail: str) -> float | None:
    """Pull the command level out of a phase label.

    Manoeuvres label their phases either `throttle=0.35` or bare, as the
    steering sweep does with `+0.20`. Both carry the same thing.
    """
    text = detail.split("=", 1)[1] if "=" in detail else detail
    try:
        return float(text)
    except ValueError:
        return None


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    return float(np.polyfit(t, v, 1)[0])


def analyse_brake(segs: list[Segment]) -> None:
    """Brake command to deceleration."""
    rows = []
    for seg in (s for s in segs if s.phase == "brake"):
        truck = seg.block["truck"]
        applied, speed = truck["effective_brake"], truck["speed"]
        # Only while the brake is actually applied and the truck still moving:
        # the approach and the final creep are not part of the response.
        band = (applied > seg.command * 0.8) & (speed > 0.5)
        if band.sum() < MIN_SAMPLES:
            continue
        rows.append((seg.command, -_slope(seg.t[band], speed[band]), float(speed[band].max())))

    if not rows:
        print("no usable brake segments")
        return

    rows.sort()
    print("\nBRAKE COMMAND -> DECELERATION")
    print(f"{'command':>9} {'m/s^2':>9} {'g':>7} {'from km/h':>10}")
    print("-" * 40)
    for command, decel, v0 in rows:
        print(f"{command:>9.2f} {decel:>9.2f} {decel / G:>7.2f} {v0 * MS_TO_KMH:>10.1f}")

    data = np.array([(c, d) for c, d, _ in rows])
    dead = data[data[:, 1] < 0.2]
    live = data[data[:, 1] >= 0.2]
    if len(dead):
        print(f"\ndead zone: commands up to {dead[:, 0].max():.2f} produce no braking")
    if len(live) >= 2:
        print(f"first effective command: {live[0, 0]:.2f} -> {live[0, 1]:.2f} m/s^2")
        for target in (1.5, 2.0, 2.5):
            below = live[live[:, 1] <= target]
            above = live[live[:, 1] > target]
            if len(below) and len(above):
                (x0, y0), (x1, y1) = below[-1], above[0]
                command = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
                print(f"  {target:.1f} m/s^2 needs command {command:.3f}")
        comfort = live[(live[:, 1] >= 1.5) & (live[:, 1] <= 2.5)]
        if len(comfort):
            width = comfort[:, 0].max() - comfort[:, 0].min()
            print(
                f"\nthe comfortable band (1.5-2.5 m/s^2) spans about {width:.3f} of command "
                "range -- narrow enough that command resolution is a real constraint"
            )


def analyse_throttle(segs: list[Segment]) -> None:
    """Throttle command to acceleration, split by gear."""
    rows = []
    for seg in (s for s in segs if s.phase == "accelerate"):
        truck = seg.block["truck"]
        gear, speed, rpm = truck["gear"], truck["speed"], truck["engine_rpm"]
        for value in sorted({int(g) for g in gear}):
            if value <= 0:
                continue  # neutral or reverse: not a throttle response
            mask = gear == value
            if mask.sum() < 3 * MIN_SAMPLES:
                continue
            t, v = seg.t[mask], speed[mask]
            # Trim both ends: a gear change is a transient, not a response.
            trim = max(len(t) // 8, 3)
            t, v = t[trim:-trim], v[trim:-trim]
            if len(t) < MIN_SAMPLES:
                continue
            rows.append(
                (seg.command, value, _slope(t, v), float(v.mean()), float(rpm[mask].mean()))
            )

    if not rows:
        print("no usable throttle segments")
        return

    rows.sort()
    print("\nTHROTTLE COMMAND -> ACCELERATION, BY GEAR")
    print(f"{'throttle':>9} {'gear':>5} {'m/s^2':>8} {'km/h':>8} {'rpm':>7}")
    print("-" * 42)
    for command, gear, accel, speed, rpm in rows:
        print(f"{command:>9.2f} {gear:>5d} {accel:>8.3f} {speed * MS_TO_KMH:>8.1f} {rpm:>7.0f}")
    print(
        "\nAcceleration falls with gear at every throttle level, as torque "
        "multiplication drops. A single throttle-to-acceleration number, "
        "averaged across a pull, would be a statement about the gearbox."
    )


def analyse_steering(segs: list[Segment]) -> None:
    """Steering position to road wheel angle."""
    rows = []
    for seg in (s for s in segs if s.phase == "steer"):
        truck = seg.block["truck"]
        settled = seg.t >= seg.t[-1] - 0.8
        if settled.sum() < MIN_SAMPLES:
            continue
        applied = float(truck["effective_steering"][settled].mean())
        wheels = truck["wheels"]["steering"][settled]
        inner = float(np.abs(wheels[:, 0]).mean())
        outer = float(np.abs(wheels[:, 1]).mean())
        rows.append((seg.command, applied, inner, outer))

    if not rows:
        print("no usable steering segments")
        return

    rows.sort()
    print("\nSTEERING POSITION -> ROAD WHEEL ANGLE")
    print(f"{'target':>8} {'applied':>9} {'wheel0 deg':>11} {'wheel1 deg':>11} {'mean deg':>9}")
    print("-" * 52)
    for target, applied, inner, outer in rows:
        print(
            f"{target:>8.2f} {applied:>9.4f} {inner * 360:>11.2f} {outer * 360:>11.2f} "
            f"{(inner + outer) / 2 * 360:>9.2f}"
        )

    data = np.array(rows)
    tracking = np.abs(data[:, 0] - data[:, 1])
    print(f"\nposition loop tracking error: max {tracking.max():.4f}")
    biggest = data[np.abs(data[:, 0]).argmax()]
    print(
        f"at the largest target ({biggest[0]:+.2f}, applied {biggest[1]:+.4f}) the mean wheel "
        f"angle is {(biggest[2] + biggest[3]) / 2 * 360:.1f} deg"
    )
    spread = np.abs(data[:, 2] - data[:, 3])
    print(
        f"inner and outer wheels differ by up to {spread.max() * 360:.2f} deg -- Ackermann "
        "geometry, so the bicycle model's single wheel is an average, not either wheel"
    )


def analyse_coastdown(segs: list[Segment]) -> None:
    """Lift-off deceleration against speed -- if the road allows it.

    Two things ruin this measurement, and both are easy to miss in the numbers:
    a gradient, whose contribution is several times the drag being measured,
    and too narrow a speed range to separate a constant term from one that
    grows with the square of speed. So both are checked before any fit is
    reported, and a failed check produces the reason rather than a coefficient.
    """
    for seg in (s for s in segs if s.phase == "coast"):
        truck = seg.block["truck"]
        speed = truck["speed"].astype(float)
        pitch = truck["placement"]["orientation"]["pitch"].astype(float) * 2 * np.pi
        t = seg.t.astype(float)

        print(f"\nCOAST {seg.source}")
        print(
            f"  {speed.max() * MS_TO_KMH:.1f} -> {speed.min() * MS_TO_KMH:.1f} km/h "
            f"over {t[-1] - t[0]:.1f} s"
        )

        gravity = G * np.sin(pitch)
        grade_deg = np.degrees(pitch)
        print(
            f"  pitch {grade_deg.min():+.2f} to {grade_deg.max():+.2f} deg, "
            f"gravity term up to {np.abs(gravity).max():.3f} m/s^2"
        )

        mean_decel = (speed[0] - speed[-1]) / (t[-1] - t[0])
        if np.abs(gravity).max() > abs(mean_decel):
            print(
                f"  UNUSABLE: the gradient contributes up to "
                f"{np.abs(gravity).max() / max(abs(mean_decel), 1e-6):.0f}x the "
                f"deceleration being measured ({mean_decel:.3f} m/s^2). "
                "A coast-down needs a verified flat road."
            )
            continue

        span = (speed.max() - speed.min()) * MS_TO_KMH
        if span < 40.0:
            print(
                f"  UNUSABLE: only {span:.0f} km/h of speed range. Separating a "
                "constant term from a v^2 term needs roughly twice that."
            )
            continue

        window = min(30, max(len(speed) // 10, 3))
        kernel = np.ones(window) / window
        v = np.convolve(speed, kernel, mode="valid")
        ts = np.convolve(t, kernel, mode="valid")
        decel = -np.gradient(v, ts)
        design = np.c_[np.ones(len(v)), v**2]
        coefficients, *_ = np.linalg.lstsq(design, decel, rcond=None)
        residual = decel - design @ coefficients
        r_squared = 1 - (residual**2).sum() / ((decel - decel.mean()) ** 2).sum()
        print(f"  a(v) = {coefficients[0]:+.4f} {coefficients[1]:+.6f} v^2   R^2 = {r_squared:.3f}")
        if coefficients[0] < 0 or coefficients[1] < 0:
            print("  (a negative coefficient is not physical; treat this as a failed fit)")
        print(
            "  Note: the gearbox stays engaged, so this is drag plus rolling "
            "resistance plus engine braking, not drag alone."
        )


def analyse_engine_brake(segs: list[Segment]) -> None:
    """Engine braking against lift-off alone, per run.

    Reported as a pair from the same run at the same speed on the same road, so
    the difference is the engine brake's contribution and not a comparison with
    someone else's afternoon.
    """
    runs: dict[float, dict[str, Segment]] = defaultdict(dict)
    for seg in segs:
        if seg.phase in {"coast", "engine-brake"}:
            runs[seg.command][seg.phase] = seg

    if not runs:
        print("no usable engine-brake segments")
        return

    print("\nENGINE BRAKE vs LIFT-OFF")
    print(f"{'run':>4} {'phase':>13} {'km/h':>13} {'m/s^2':>9} {'pitch deg':>11}")
    print("-" * 56)
    for run in sorted(runs):
        pair = runs[run]
        results = {}
        for phase in ("coast", "engine-brake"):
            seg = pair.get(phase)
            if seg is None:
                continue
            truck = seg.block["truck"]
            speed = truck["speed"].astype(float)
            pitch = np.degrees(truck["placement"]["orientation"]["pitch"].astype(float) * 2 * np.pi)
            if len(speed) < MIN_SAMPLES:
                continue
            decel = -_slope(seg.t, speed)
            results[phase] = decel
            print(
                f"{int(run):>4} {phase:>13} "
                f"{speed.max() * MS_TO_KMH:>6.1f}->{speed.min() * MS_TO_KMH:<6.1f} "
                f"{decel:>9.3f} {pitch.mean():>11.2f}"
            )
        if len(results) == 2:
            gain = results["engine-brake"] - results["coast"]
            print(f"{'':>4} {'difference':>13} {'':>13} {gain:>+9.3f}")

    print(
        "\nThe difference is what the engine brake adds. Whether it is enough "
        "depends on the gap: comfortable deceleration is 1.5-2.5 m/s^2, and the "
        "service brake's smallest useful command already gives 1.8."
    )


ANALYSES = {
    "brake-sweep": analyse_brake,
    "throttle-sweep": analyse_throttle,
    "steering-steps": analyse_steering,
    "coastdown": analyse_coastdown,
    "engine-brake": analyse_engine_brake,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    recordings = load(args.paths)
    if not recordings:
        print("no recordings found", file=sys.stderr)
        return 1

    grouped: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments(recordings):
        grouped[segment.manoeuvre].append(segment)

    print(
        f"{len(recordings)} recording(s), {sum(len(v) for v in grouped.values())} labelled segments"
    )
    for manoeuvre, segs in sorted(grouped.items()):
        analysis = ANALYSES.get(manoeuvre)
        sources = {s.source for s in segs}
        print(f"\n{'=' * 70}\n{manoeuvre}  ({len(segs)} segments from {len(sources)} run(s))")
        if analysis is None:
            print("no analysis for this manoeuvre yet")
            continue
        analysis(segs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
