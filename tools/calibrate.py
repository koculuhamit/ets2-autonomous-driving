#!/usr/bin/env python3
"""Calibrate the truck being driven, step by step, into its profile.

    python tools/calibrate.py status                 # what this truck has, and lacks
    python tools/calibrate.py steering               # stationary: the steering map
    python tools/calibrate.py brake --distance 600   # a clear straight road: the brakes
    python tools/calibrate.py camera RUN [RUN ...]   # drives recorded with --images
    python tools/calibrate.py cab FOLDER [...]       # frames from the cab
    python tools/calibrate.py view [--keep]          # windscreen and what covers it, looked at

Each step measures one thing for the truck the telemetry reports and writes
it into `configs/trucks/<brand>.<model>.json` (core.calibration), with any
file it makes beside it; the drives (tools/follow.py) read it
for that truck and use the Actros's numbers for what it still lacks.

`steering` and `brake` drive the truck through tools/sysid.py -- read its
safety notes: `steering` turns the wheels standing still; `brake` speeds up
to 40 km/h and brakes, again and again, so it needs --distance, the metres
of clear straight road ahead of the truck now. `camera` measures the
interior camera from recordings of drives on the compiled map
(tools/calibrate_camera.py). `cab` keeps what the dashboard looks like
(tools/calibration/cab_reference.py). `view` grabs the game window and
draws on it, into data/calibration/, the windscreen the detector looks
through and the mirrors and HUD it leaves out: `--keep` takes them as right
for this truck, `--windscreen` and `--covered` set them (no window is
opened: OpenCV here has no GUI).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.reader import TelemetryError, TelemetryReader  # noqa: E402
from core.calibration import (  # noqa: E402
    PROFILES_DIR,
    TruckProfile,
    fit_brake,
    fit_steering_map,
    load_profile,
    save_profile,
    truck_id,
)
from core.calibration.profile import degrees_map  # noqa: E402
from core.calibration.samples import brake_samples, steering_samples  # noqa: E402
from pal.base import PalError  # noqa: E402

BRAKE_LEVELS = ("0.04", "0.05", "0.06", "0.08", "0.10", "0.12", "0.15", "0.25", "0.40", "0.60")
BRAKE_SPEED_KMH = "40"
BRAKE_HOLD_S = "3"
"""Brake pulses rather than stops: the light end, where the dead zone ends,
is the part worth mapping, and three seconds of it say as much as a stop."""

HUD = (
    # Money, navigation, clock, the job; and the speedometer (detection.DEFAULT_CAB_VIEW).
    [1560 / 1920, 560 / 1080, 1.0, 1.0],
    [0.0, 890 / 1080, 205 / 1920, 1.0],
)


def run(command: list[str]) -> int:
    return subprocess.run([sys.executable, *command], cwd=REPO_ROOT, check=False).returncode


def the_truck() -> tuple[str, object]:
    with TelemetryReader() as reader:
        block = reader.read().block
    return truck_id(block), block


def status(args, profile: TruckProfile, block) -> int:
    geometry = profile.geometry(block)
    print(f"truck: {profile.truck}   wheelbase {geometry.wheelbase:.2f} m (telemetry)")
    print(f"profile: {PROFILES_DIR / (profile.truck + '.json')}")
    shown = {
        "steering": degrees_map(profile.steering_map) if profile.steering_map else None,
        "brake": (
            f"dead to {profile.brake_dead_zone}, {len(profile.brake_curve)} points"
            if profile.brake_curve
            else None
        ),
        "camera": profile.camera,
        "cab": profile.cab_reference,
        "view": "drawn" if profile.view else None,
    }
    for name, value in shown.items():
        if value:
            print(f"  {name:9s} {value}  ({profile.measured.get(name, '')})")
        else:
            print(f"  {name:9s} -- the Actros's")
    missing = profile.missing()
    print("to do: " + ", ".join(missing) if missing else "all measured")
    return 0


def steering(args, profile: TruckProfile, block) -> int:
    output = _sysid_output(profile, "steering")
    if run(["tools/sysid.py", "steering-steps", "--yes", "--output", str(output)]) != 0:
        return 1
    try:
        fitted = fit_steering_map(steering_samples(_segments(output)))
    except ValueError as exc:
        print(f"no map: {exc}", file=sys.stderr)
        return 1
    print(f"steering map: {degrees_map(fitted)}")
    return _save(replace(profile, steering_map=fitted), "steering", output)


def brake(args, profile: TruckProfile, block) -> int:
    output = _sysid_output(profile, "brake")
    command = ["tools/sysid.py", "brake-sweep", "--yes", "--output", str(output)]
    command += ["--speed", BRAKE_SPEED_KMH, "--brake-hold", BRAKE_HOLD_S]
    command += ["--distance", str(args.distance), "--levels", *BRAKE_LEVELS]
    if run(command) != 0:
        return 1
    try:
        dead_zone, curve = fit_brake(brake_samples(_segments(output)))
    except ValueError as exc:
        print(f"no brake curve: {exc}", file=sys.stderr)
        return 1
    print(f"brakes: nothing up to {dead_zone}; " + ", ".join(f"{c}->{d}" for c, d in curve))
    return _save(replace(profile, brake_dead_zone=dead_zone, brake_curve=curve), "brake", output)


def camera(args, profile: TruckProfile, block) -> int:
    name = f"{profile.truck}.camera.json"
    out = PROFILES_DIR / name
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    recordings = [str(r) for r in args.recordings]
    if run(["tools/calibrate_camera.py", str(args.cache), *recordings, "--out", str(out)]) != 0:
        return 1
    return _save(replace(profile, camera=name), "camera", " ".join(recordings))


def cab(args, profile: TruckProfile, block) -> int:
    name = f"{profile.truck}.cab.npy"
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    folders = [str(f) for f in args.folders]
    command = ["tools/calibration/cab_reference.py", *folders, "--output", str(PROFILES_DIR / name)]
    if run(command) != 0:
        return 1
    return _save(replace(profile, cab_reference=name), "cab", " ".join(folders))


def view(args, profile: TruckProfile, block) -> int:
    """Grab the game window and draw the cab view on it, to look at; with
    `--keep`, take the view drawn as this truck's; with `--windscreen` and
    `--covered`, set it."""
    import cv2

    from pal import GAME_WINDOW_TITLE, get_backend

    with get_backend().screen_capture(GAME_WINDOW_TITLE) as capture:
        image = capture.grab().image
    height, width = image.shape[:2]
    changed = None
    if args.windscreen is not None:
        changed = view_from(args.windscreen, args.covered or [])
    elif args.keep:
        changed = view_as_dict(profile.cab_view())
    if changed is not None:
        profile = replace(profile, view=changed)
    picture = REPO_ROOT / "data" / "calibration" / f"{profile.truck}.view.png"
    picture.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(picture), drawn_view(image, profile.cab_view()))
    print(f"the view on the game's window: {picture}")
    if changed is None:
        print(
            "green the windscreen, red what covers it (mirrors, HUD), the grid in tenths.\n"
            "Right as drawn: --keep. Otherwise: --windscreen L,T,R,B --covered L,T,R,B ..."
        )
        return 0
    return _save(profile, "view", f"{width}x{height} frame")


def view_as_dict(view) -> dict:
    """A detection.CabView as a profile keeps it."""

    def box(rect) -> list[float]:
        return [rect.left, rect.top, rect.right, rect.bottom]

    return {
        "windscreen": box(view.windscreen),
        "covered": [box(r) for r in view.covered],
    }


def parse_box(text: str) -> list[float]:
    """`L,T,R,B` in fractions of the frame."""
    values = [float(v) for v in text.split(",")]
    if len(values) != 4 or not (
        0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1
    ):
        raise ValueError(f"not a box in fractions of the frame: {text!r}")
    return values


def view_from(windscreen: str, covered: list[str]) -> dict:
    """A view from `L,T,R,B` for the windscreen and for each mirror covering
    it; those and the HUD are what the detector leaves out."""
    return {
        "windscreen": parse_box(windscreen),
        "covered": [parse_box(box) for box in covered] + [list(r) for r in HUD],
    }


def drawn_view(image, view):
    """`image` with the view's regions and a grid of tenths drawn on it."""
    import cv2

    out = image.copy()
    height, width = out.shape[:2]

    def rectangle(rect, colour, label: str) -> None:
        a = (round(rect.left * width), round(rect.top * height))
        b = (round(rect.right * width) - 1, round(rect.bottom * height) - 1)
        cv2.rectangle(out, a, b, colour, 3)
        cv2.putText(out, label, (a[0] + 5, a[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    for k in range(1, 10):
        x, y = round(k * width / 10), round(k * height / 10)
        cv2.line(out, (x, 0), (x, height), (255, 255, 0), 1)
        cv2.line(out, (0, y), (width, y), (255, 255, 0), 1)
        cv2.putText(out, f"{k / 10:.1f}", (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        cv2.putText(out, f"{k / 10:.1f}", (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
    rectangle(view.windscreen, (0, 255, 0), "windscreen")
    for rect in view.covered:
        rectangle(rect, (0, 0, 255), "covered")
    return out


def _segments(output: Path):
    from tools.analyse_sysid import load, segments

    return segments(load([output]))


def _sysid_output(profile: TruckProfile, what: str) -> Path:
    return REPO_ROOT / "data" / "sysid" / f"{date.today().isoformat()}_{profile.truck}_{what}"


def _save(profile: TruckProfile, what: str, source) -> int:
    measured = {**profile.measured, what: f"{date.today().isoformat()} from {source}"}
    path = save_profile(replace(profile, measured=measured))
    print(f"saved: {path}")
    return 0


STEPS = {
    "status": status,
    "steering": steering,
    "brake": brake,
    "camera": camera,
    "cab": cab,
    "view": view,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("status")
    steps.add_parser("steering")
    braking = steps.add_parser("brake")
    braking.add_argument("--distance", type=float, required=True, help="metres of clear road")
    cam = steps.add_parser("camera")
    cam.add_argument("recordings", type=Path, nargs="+")
    cam.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "map" / "installed")
    cabin = steps.add_parser("cab")
    cabin.add_argument("folders", type=Path, nargs="+")
    looking = steps.add_parser("view")
    looking.add_argument("--keep", action="store_true", help="the view drawn is right")
    looking.add_argument("--windscreen", help="L,T,R,B in fractions of the frame")
    looking.add_argument(
        "--covered", action="append", help="L,T,R,B of a mirror the detector leaves out"
    )
    args = parser.parse_args()
    try:
        truck, block = the_truck()
    except (PalError, TelemetryError) as exc:
        print(f"no game: {exc}", file=sys.stderr)
        return 1
    return STEPS[args.step](args, load_profile(truck), block)


if __name__ == "__main__":
    sys.exit(main())
