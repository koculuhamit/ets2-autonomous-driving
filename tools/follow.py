#!/usr/bin/env python3
"""Follow a recorded route: steering and speed together.

    python tools/follow.py data/routes/teach-01 --speed 40 --distance 600

Teach-and-repeat. The path comes from a drive somebody did by hand, so the
truck drives the line that was taught it -- including any wandering. There is
no map and no perception; it knows where the road is only because it was shown.

`--speed` is a ceiling, not a target. The corners decide: a curvature-based
profile caps the speed at every point and brakes for it in advance, because a
40 km/h wish on a 10 m radius asks for 11.9 m/s^2 of lateral acceleration and
the truck would lie down before delivering it.

Can be rehearsed offline first, against the recording it came from:

    python tools/follow.py data/routes/teach-01 --replay

which runs the same controllers over the recorded frames, reports what the
tracking error would have been, and touches neither the game nor the truck.
Tuning belongs there; driving is for confirming.

Safety is `core.control.limits` plus one addition that matters for a lateral
controller: a lateral-error ceiling. Straying further than that from the path
means the controller has lost it, and continuing to steer on a stale idea of
where the road is, is worse than stopping.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path

# Before numpy: on Windows OpenBLAS commits a buffer for each CPU thread as it
# is imported -- 360 MB in each of the drive's processes, where Linux commits
# nothing until it is used -- and one thread is as quick for what is done here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.bridge import convert  # noqa: E402
from core.bridge.command import SAFETY_STOP_S, Commander, ControlCommand  # noqa: E402
from core.bridge.reader import StaleTelemetryError, TelemetryError, TelemetryReader  # noqa: E402
from core.calibration import TruckProfile, load_profile, save_profile, truck_id  # noqa: E402
from core.common.controls import Button  # noqa: E402
from core.control.acc import (  # noqa: E402
    AdaptiveCruise,
    OutOfCab,
    PerceptionLost,
    crossing_spans,
    junction_spans,
    lane_ahead,
    out_of_a_ring,
    prefab_spans,
    ring_entry_spans,
    ring_way_spans,
    turn_across_spans,
)
from core.control.end_stop import (  # noqa: E402
    reached,
    speed_to_stop_in,
)
from core.control.following import GapPolicy  # noqa: E402
from core.control.handover import HANDOVER_S, HazardsOff, handover_buttons  # noqa: E402
from core.control.indicators import Indicators, route_signals  # noqa: E402
from core.control.lane_change import LaneChange, LaneChangeGate, lane_changes  # noqa: E402
from core.control.lane_guard import GOING_AT_M, LaneGuard  # noqa: E402
from core.control.lateral import (  # noqa: E402
    ACTROS_GEOMETRY,
    PurePursuit,
    TrailerGeometry,
)
from core.control.limits import (  # noqa: E402
    Collided,
    DamageWatch,
    Impacts,
    SafetyMonitor,
    Standstill,
    UnsafeCondition,
)
from core.control.longitudinal import (  # noqa: E402
    HOLD_BELOW,
    STOP_TARGET,
    ActuatorMap,
    BrakeEstimate,
    LongitudinalCommand,
    SpeedController,
    held_still,
    mass_ratio,
)
from core.control.override import OverrideDetector, RecentPedals  # noqa: E402
from core.control.parking import MOVE_OFF_S, ParkingBrake, StopLatch  # noqa: E402
from core.control.retarder import RetarderEstimate, RetarderLever  # noqa: E402
from core.control.steering import (  # noqa: E402
    STEERING_CALIBRATION,
    SteeringPositionController,
    centring_at,
    rate_gain_at,
)
from core.control.traffic_light import (  # noqa: E402
    STOP_SHORT_M,
    LightLine,
    TrafficLights,
    ViewBounds,
    lines_on_path,
    stop_short_from_sightings,
    stop_short_to_see,
)
from core.control.turn_line import WideTurnChoice  # noqa: E402
from core.evaluation.health import HealthSampler  # noqa: E402
from core.localization.path import (  # noqa: E402
    ReferencePath,
    TrailerTracker,
    widen_for_trailer,
)
from core.map import MapCacheError, Places, RoadNetwork  # noqa: E402
from core.map.fuel import load_pumps  # noqa: E402
from core.perception.lanes import LANE_CHECK_M  # noqa: E402
from core.perception.lights import LightFilter, LightState, is_daylight  # noqa: E402
from core.planning.destination import GPS_COMPANY, place_from  # noqa: E402
from core.planning.game_route import GameRouter, depot_goals  # noqa: E402
from core.planning.gps import REGION as GPS_REGION  # noqa: E402
from core.planning.gps import GpsRouter, unpack_mask  # noqa: E402
from core.planning.gps_follow import GpsFollower  # noqa: E402
from core.planning.nav_field import nav_length  # noqa: E402
from core.planning.route import (  # noqa: E402
    CHANGE,
    Route,
    RouteError,
    load_points,
    load_stop_lines,
    service_road_stop,
)
from core.planning.speed_profile import DEFAULT_MAX_LATERAL, SpeedProfile  # noqa: E402
from core.planning.tight_turns import (  # noqa: E402
    Stretch,
    WideTurn,
    between,
    shaped_turn,
    splice,
    stretches,
    tight_turn_spans,
    wide_spans,
    wide_turn,
)
from core.recording import Recorder, open_recording  # noqa: E402
from core.recording.messages import MessageLog, keep_messages  # noqa: E402
from pal import PalError  # noqa: E402

MS_TO_KMH = 3.6
LOOP_INTERVAL = 0.005

#: Beyond this the controller has lost the path rather than being off it.
MAX_LATERAL_ERROR_M = 6.0

#: Off the path by more than this in one step, from on it the step before, is
#: no truck's doing: its place on the path was lost. Out of Göteborg, the
#: route new, the drive stopped at 80 km/h "5046 m from the path", the truck
#: on its lane. The place is found again over the whole path; the path is
#: kept beside the recording, to see how.
PLACE_LOST_M = 30.0

#: Steering doublets for `--excite`: every period, at this offset, one
#: amplitude up for half a doublet and down for the other half. Symmetric, so
#: the net steering is unchanged and the controller keeps the lane; the input
#: is known exactly, which closed-loop driving data never gives. Written
#: because the simulation under-predicts the lateral error two to three times
#: and the steering actuator is the suspect.
DOUBLET_PERIOD_S = 5.0
DOUBLET_AT_S = 2.0
DOUBLET_HALF_S = 0.3


#: Through a toll gate, km/h, from `BARRIER_APPROACH_M` before it to
#: `BARRIER_AFTER_M` past it: a toll lane is a booth's width, and the game
#: gives the motorway's limit through it.
BARRIER_SPEED_KMH = 30.0
BARRIER_APPROACH_M = 300.0
BARRIER_AFTER_M = 40.0

#: A pedal this far, in pedal units, from what was sent, for this long, is
#: the driver's foot on it: the driver has the truck. Up to this long between
#: turns of the loop, the pedals read are those last sent.
PEDAL_TAKEOVER = 0.2
PEDAL_TAKEOVER_S = 0.2
PEDAL_TAKEOVER_MAX_DT = 0.1

#: Following the red line to a place marked on the map: the game's distance
#: under this many of its metres, for this long, is the place come up to,
#: and the truck stops on it.
PLACE_ARRIVED_NAV_M = 150.0
PLACE_NAV_SETTLE_S = 0.3


@dataclass
class PlaceArrival:
    """When a place marked on the map is come up to, by the game's distance."""

    _near_since: float | None = None

    def to_go(self, nav_distance: float, now: float, game_scale: float) -> float | None:
        """The path's metres to the place, once the game's distance has been
        under `PLACE_ARRIVED_NAV_M` for `PLACE_NAV_SETTLE_S`; else None. A
        moment of it -- 0 while the game plans anew -- is not arriving."""
        if nav_distance >= PLACE_ARRIVED_NAV_M:
            self._near_since = None
            return None
        if self._near_since is None:
            self._near_since = now
        if now - self._near_since < PLACE_NAV_SETTLE_S:
            return None
        return max(nav_distance, 0.0) / max(game_scale, 1.0)


#: How far before its stop line a light's pole may stand.
LIGHT_POLE_BEFORE_M = 2.0

DEFAULT_DETECTOR = REPO_ROOT / "data" / "models" / "dfine-nano-ets2"
DEFAULT_CAMERA = REPO_ROOT / "configs" / "camera.json"


def game_window():
    """The capture the perception child opens; module level so a child can find it."""
    from pal import GAME_WINDOW_TITLE, get_backend

    return get_backend().screen_capture(GAME_WINDOW_TITLE)


class _DetectorFactory:
    """Opens the detector inside the perception child, not in this process."""

    def __init__(self, model: Path, view=None) -> None:
        self.model = model
        self.view = view

    def __call__(self):
        from core.perception.detection import DEFAULT_CAB_VIEW, Detector

        return Detector.open(self.model, view=self.view or DEFAULT_CAB_VIEW)


def start_perception(
    model: Path,
    camera_file: Path,
    allow_cpu: bool = False,
    black_box: Path | None = None,
    truck: TruckProfile | None = None,
    view: bool = True,
):
    """The perception child, started and warmed up; with the camera it uses,
    and the truck's dashboard and view of the cab (core.calibration). With
    `view`, not before it has seen the road (`sees_the_road`).

    Refuses a model that ended up on the CPU unless told otherwise. After a
    driver fault left CUDA dead, ONNX Runtime fell back to the CPU without a
    word, perception went from 66 to 186 ms old, and the drive went ahead on
    stopping margins worked out for 0.1 s.
    """
    import json

    from core.perception.camera import GroundCamera
    from core.perception.service import PerceptionService

    camera = GroundCamera.from_json(json.loads(camera_file.read_text(encoding="utf-8")))
    truck = truck or TruckProfile("unknown.unknown")
    service = PerceptionService(
        game_window,
        _DetectorFactory(model, truck.cab_view()),
        camera,
        black_box=black_box,
        cab_reference=truck.cab_file(),
        learn_cab=truck.learns_cab(),
        learn_roof=truck.learns_roof(),
    )
    service.start()
    if not allow_cpu and service.provider != "CUDAExecutionProvider":
        service.stop()
        raise RuntimeError(
            f"the detector runs on {service.provider or 'an unknown device'}, not the GPU; "
            "perception would be three times slower. Fix CUDA, or pass --allow-cpu"
        )
    # Ready means the model is loaded, not that a frame has been seen; the
    # first drive stopped on the spot waiting for one. Wait for a real view.
    if not view or sees_the_road(service):
        return service, camera
    why = no_view(service)
    service.stop()
    raise RuntimeError(f"{why} within 5 s")


BRAKE_TRUSTED_AFTER_S = 1.0
"""Braking seen this long, the brake's own strength is gone by, not the
load's (core.control.longitudinal.BrakeEstimate)."""

TURNING_LATERAL_MPS2 = DEFAULT_MAX_LATERAL
TURNING_FROM_RADIUS_M = 40.0
"""While the wheel is turned for a bend tighter than `TURNING_FROM_RADIUS_M`,
the truck gathers no speed beyond what that bend takes at
`TURNING_LATERAL_MPS2`: round a corner in Berlin the planned speed went up
with the wheel near lock and the truck met a lamp post at 20 km/h. A
motorway's bends are wider and are left alone. At the speed profile's own
lateral acceleration, no lower (the driver's word, 16 November: 1.5 had
the truck crawl round corners at 12 km/h)."""


def turning_cap(geometry, steering: float) -> float | None:
    """The speed the bend the wheel is turned for takes, m/s; None for a
    bend wider than `TURNING_FROM_RADIUS_M`."""
    angle = abs(geometry.angle_for_command(steering))
    if angle <= 1e-6:
        return None
    radius = geometry.wheelbase / math.tan(angle)
    if radius >= TURNING_FROM_RADIUS_M:
        return None
    return math.sqrt(TURNING_LATERAL_MPS2 * radius)


LIGHT_STOP_FROM_M = 40.0
"""How near a light at red or amber its lamps are measured from for where to
stop (core.control.traffic_light.stop_short_to_see)."""


LIGHT_STOP_MOVE_DECELERATION = 1.5
"""A light's stop is moved back, to keep its lamp in view, no further than
the truck can still stop short of at this: moved 10 m back 32 m short of
the line at 38 km/h, a red was braked for at 4 to 5.4 m/s^2, three times on
one drive (18 November)."""


def moved_stop(before: float, wanted: float, to_line: float, speed: float) -> float:
    """Where before a light's line to stop, metres, from `before` towards
    `wanted`: on towards the line at once, back from it only as far as is
    stopped short of at `LIGHT_STOP_MOVE_DECELERATION` from `speed`, `to_line`
    metres short of it."""
    if wanted <= before:
        return wanted
    reachable = to_line - max(speed, 0.0) ** 2 / (2.0 * LIGHT_STOP_MOVE_DECELERATION)
    return max(before, min(wanted, reachable))


LAMP_FOLLOWED_PX = 120.0
"""A lamp seen within this of the one followed, a view later, is it."""


def follow_lamp(
    sightings: list[tuple[float, float, float]],
    to_line: float,
    points: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """`sightings` of one lamp with this view's added: the lamp nearest the
    last sighting, or, with none near it, the one furthest out from the
    middle of the view -- the first to go out of it -- begun afresh."""
    if sightings:
        _, x, y = sightings[-1]
        near = min(points, key=lambda p: math.hypot(p[0] - x, p[1] - y))
        if math.hypot(near[0] - x, near[1] - y) <= LAMP_FOLLOWED_PX:
            return [*sightings, (to_line, *near)]
    first = max(points, key=lambda p: math.hypot(p[0] - 960.0, p[1] - 540.0))
    return [(to_line, *first)]


def view_bounds(profile) -> ViewBounds:
    """What of a 1920 x 1080 frame shows the road through `profile`'s cab:
    its windscreen, less its mirrors and HUD."""
    view = profile.cab_view()
    return ViewBounds(
        *view.windscreen.pixels(1920, 1080),
        covered=tuple(rect.pixels(1920, 1080) for rect in view.covered),
    )


def eye_ahead_m(block) -> float:
    """How far ahead of the truck's reference point its driver's eye is:
    over the front axle, as in a cab-over truck (the steered wheels', in
    the vehicle's space, z backwards)."""
    wheels = block["truck"]["wheels"]
    count = int(wheels["count"])
    ahead = [
        -float(wheels["position"][i]["z"]) for i in range(count) if int(wheels["steerable"][i])
    ]
    return max(ahead) if ahead else 0.0


def pedals_taken(truck) -> float:
    """How much of a pedal the game has from the stack: the larger of what it
    applies and what its input reads. An electric DAF XF brakes with its
    motor -- 0.2 to 0.3 of brake read, the truck slowing at 1.9 m/s^2 -- and
    its applied brake stayed under 0.07, so two drives were handed over as if
    the game took no input (core.control.limits, `InputsNotApplied`)."""
    return max(
        float(truck["effective_throttle"]),
        float(truck["effective_brake"]),
        float(truck["input_throttle"]),
        float(truck["input_brake"]),
    )


def uncalibrated(truck: TruckProfile) -> str:
    """What a truck drives on that is not its own, in words for the driver;
    empty for one calibrated."""
    missing = truck.missing()
    if not missing:
        return ""
    said = f"not calibrated: the Actros's {', '.join(missing)} (tools/calibrate.py)"
    if "camera" in missing:
        said += "; distances to the vehicles ahead may be off"
    if truck.learns_cab() is not None:
        said += "; its dashboard is learned in the drive's first 20 s: drive from the cab view (1)"
    return said


def no_view(service) -> str:
    """What to say for no view of the road: the capture's word where it had
    one -- the game window off the screen, another window in front of it --
    or the question to ask. With a 1920x1080 window run off a 1920x1080
    screen the launcher asked "is the game in front?" of a game in front."""
    report = service.latest()
    why = getattr(report, "covered_why", "") if report is not None else ""
    return f"no view of the road: {why}" if why else "no view of the road: is the game in front?"


def sees_the_road(service, since: float = -math.inf, within: float = 5.0) -> bool:
    """Whether a report of the road, not of a window in front of it, grabbed
    after `since`, comes within `within` seconds."""
    deadline = time.perf_counter() + within
    while time.perf_counter() < deadline:
        report = service.latest()
        if report is not None and not report.covered and report.grabbed_at >= since:
            return True
        time.sleep(0.02)
    return False


STANDBY_RETRY_S = 10.0
"""Waiting for the beacon, how often to try again what could not be made ready."""

STANDBY_LOOK_S = 0.05
"""Waiting for the beacon, how often to look for word from tools/mission.py."""

YARD = "in a yard: drive out onto the road, then switch the beacon on"


def take_word(word_file: Path, parent=None, *, only: str | None = None) -> str | None:
    """The word tools/mission.py left in `word_file`, taken (the file
    removed); None for none yet. With `only`, any other word is left in the
    file for later. A `parent` process gone is the word "stop".

    Waiting out a game that stood still, every word but "stop" was taken and
    dropped: the game started anew, the beacon's "go" was lost, and the
    launcher said driving while the truck stood."""
    if parent is not None and not parent.is_running():
        return "stop"
    try:
        said = word_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if only is not None and said != only:
        return None
    word_file.unlink(missing_ok=True)
    return said


def reattached(reader: TelemetryReader, opener=TelemetryReader) -> TelemetryReader | None:
    """A reader of the block the game publishes now, where `reader`'s stands
    still: on Linux a game started anew makes a new block, and the one mapped
    is the old game's, unlinked, its counter stuck for good. None where there
    is no other -- no block at all (the game closed), or the same counter
    (paused, or the old block left behind)."""
    fresh = opener()
    try:
        fresh.open()
        count = fresh.read().write_count
    except (PalError, TelemetryError):
        fresh.close()
        return None
    if count == reader.last_write_count:
        fresh.close()
        return None
    return fresh


@dataclass
class Standing:
    """What a drive made ready before the beacon was on (`standby_until_go`)."""

    reader: TelemetryReader
    perception: object
    camera: object
    went_at: float
    truck: str = ""
    """The truck perception was started for (`truck_id`): its camera, view
    and dashboard."""
    """`time.perf_counter()` when it was told to go."""

    took: str = ""
    """What the time from the word to go went on before the drive's own start."""


def plan_now(cache: Path, route: Path, to_gps: bool, *, quiet: bool = False) -> tuple[int, str]:
    """Plan from where the truck is, in this process (tools/plan_route.py):
    its return code, and what it said was wrong. What it says otherwise is
    printed, indented, as tools/mission.py printed it; `quiet`, not."""
    import contextlib
    import io

    from tools import plan_route

    argv = [str(cache), str(route), "--live", "--to-gps" if to_gps else "--to-job", "--game"]
    said, wrong = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(wrong):
            code = plan_route.main(argv)
    except SystemExit as exc:  # its parser's error: no job, no route on the GPS
        code = exc.code if isinstance(exc.code, int) else 2
    if not quiet:
        for line in said.getvalue().splitlines():
            print(f"  {line}")
    return code, wrong.getvalue()


def marked_place(route: Path) -> bool:
    """Whether a route from plan_route.py --to-gps goes to a place marked on
    the map, not a depot."""
    with np.load(route, allow_pickle=False) as data:
        destination = json.loads(str(data["metadata"])).get("destination") or {}
    return destination.get("company") == GPS_COMPANY


def warm_depot_goals(cache: Path) -> None:
    """Every depot's gate lanes, found once in the standby: where the GPS
    goes is looked for among them all (core.planning.destination.gps_places),
    1.5 s at the start of a drive to a place marked on the map."""
    from core.planning.game_route import depot_goals
    from tools.plan_route import open_network

    network = open_network(str(Path(cache).resolve()))
    for depot in Places.open(cache).companies:
        with contextlib.suppress(RouteError):
            depot_goals(network, depot)


def standby_until_go(args) -> Standing | int:
    """Make ready for a drive before the beacon is on, then wait for word
    from tools/mission.py in the file `--standby` names: `go job <folder>`
    or `go gps <folder>` to plan from where the truck is and drive, the
    drive's files in `<folder>`; `stop`, or the mission gone, to give up.
    A file, not stdin: on Windows a thread blocked reading stdin held up the
    start of the perception child, which never got going. The model is loaded and held
    (`PerceptionService.hold`), the depot's ways weighed by a plan made and
    thrown away (core.planning.game_route.depot_fields), and again when
    the job changes.

    Switched on, the beacon took 17 s to start the truck: 5 s of it loading
    the model, 3.3 s weighing the depot's ways, the rest the path of the
    whole route (now begun on its first part, GpsWorker `whole`).

    The drive's `Standing`; or the exit code, having given up.
    """
    import psutil

    from tools.plan_route import GPS_ELSEWHERE

    word_file: Path = args.standby
    parent = psutil.Process().parent()

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot reach the game: {exc}", file=sys.stderr)
        return 1
    truck = load_profile(truck_id(reader.read().block))
    told = uncalibrated(truck)
    if told:
        print(f"standby: truck {truck.truck}: {told}", flush=True)
    camera_file = truck.camera_file() if args.camera == DEFAULT_CAMERA else args.camera
    perception = camera = None
    tried_at = -math.inf
    weighed_for: object = ()
    # Beside the word file, in the mission's private directory, not under a
    # fixed name in the shared /tmp (tools/mission.py, private_dir).
    thrown_away = word_file.parent / "thrown-away.npz"

    def give_up(code: int) -> int:
        if perception is not None:
            perception.stop()
        reader.close()
        return code

    while True:
        now = time.perf_counter()
        if perception is not None and not perception.alive:
            print("standby: perception stopped: starting it again", file=sys.stderr)
            perception.stop()
            perception, tried_at = None, -math.inf
        if perception is None and now - tried_at >= STANDBY_RETRY_S:
            tried_at = now
            try:
                perception, camera = start_perception(
                    args.model,
                    camera_file,
                    args.allow_cpu,
                    black_box=REPO_ROOT / "data" / "follow",
                    truck=truck,
                    view=False,  # the game may well not be in front yet
                )
                perception.hold()
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                print(f"standby: perception not ready yet: {exc}", file=sys.stderr)
        try:
            block = reader.read().block
        except StaleTelemetryError:
            # Paused, or in its menus: the model kept, waited out. Given up
            # on, it was loaded again after each pause. Or the game closed and
            # started anew: its new block found (`reattached`), a "go" kept.
            if take_word(word_file, parent, only="stop") == "stop":
                return give_up(0)
            fresh = reattached(reader)
            if fresh is not None:
                reader.close()
                reader = fresh
                print("standby: the game started anew: its telemetry found again", flush=True)
                # And a new window: the capture held the old one and, held
                # (`PerceptionService.hold`), never grabbed to find it gone;
                # the beacon's drive then stopped for no view of the road.
                if perception is not None:
                    perception.stop()
                    perception, tried_at = None, -math.inf
                continue
            time.sleep(STANDBY_RETRY_S / 10.0)
            continue
        except (PalError, TelemetryError) as exc:
            print(f"standby: lost the game: {exc}", file=sys.stderr)
            return give_up(1)
        current = truck_id(block)
        if current != truck.truck and not current.startswith("unknown"):
            # Read once at the start, the truck was the one the game had then:
            # none yet, or another. A DAF XD drove on the Actros's camera,
            # saw the lorry ahead 7 to 12 m wide -- no vehicle ahead is -- and
            # ran into its back at 80 km/h (15 November). Perception is
            # started again with this truck's camera, view and dashboard.
            truck = load_profile(current)
            camera_file = truck.camera_file() if args.camera == DEFAULT_CAMERA else args.camera
            said = uncalibrated(truck) or "calibrated"
            print(f"standby: truck {truck.truck}: {said}", flush=True)
            if perception is not None:
                perception.stop()
                perception, tried_at = None, -math.inf
            continue
        if args.tap is not None:
            args.tap.pose(now, block)
            args.tap.status(now, "standby: switch the beacon on to drive")
        job = convert.job_target(block) if int(block["job"]["job_active"]) else None
        if job != weighed_for and not reader.read().is_paused:
            weighed_for = job
            weighing = time.perf_counter()
            plan_now(args.cache, thrown_away, to_gps=job is None, quiet=True)
            warm_depot_goals(args.cache)
            print(
                f"standby: ways weighed in {time.perf_counter() - weighing:.1f} s"
                + ("" if perception is not None else "; perception not ready yet"),
                flush=True,
            )
            if perception is not None:
                print("standby: ready", flush=True)
        said = take_word(word_file, parent)
        if said is None:
            time.sleep(STANDBY_LOOK_S)
            continue
        if said == "stop":
            return give_up(0)
        parts = said.split(maxsplit=2)
        if len(parts) != 3 or parts[0] != "go":
            continue
        went_at = time.perf_counter()
        to_gps, folder = parts[1] == "gps", Path(parts[2])
        outcome = folder / "outcome.json"
        if perception is None:
            write_outcome(outcome, "unsafe", "perception is not running: is the game in front?")
            return give_up(1)
        perception.resume()
        route = folder.with_suffix(".npz")
        route.parent.mkdir(parents=True, exist_ok=True)
        planning = time.perf_counter()
        code, wrong = plan_now(args.cache, route, to_gps)
        if code == GPS_ELSEWHERE and not to_gps:
            print("  the game's GPS goes elsewhere than the job: driving where it goes")
            to_gps = True
            code, wrong = plan_now(args.cache, route, True)
        elif code != 0 and not to_gps and float(block["navigation"]["distance"]) > 0.0:
            # A job cancelled still shows in the telemetry: at a border
            # north of Basel its depot could not be reached from the customs
            # road, and the drive stopped as if in a yard, the place marked
            # on the map never tried.
            retried = plan_now(args.cache, route, True)
            if retried[0] == 0:
                print("  no way to the job's depot from here: driving where the GPS goes")
                to_gps = True
                code, wrong = retried
        if to_gps and code == 0 and marked_place(route):
            # To a place marked on the map, which no telemetry tells: its
            # red line followed, not a point guessed from its distance. Near
            # Geneva the one point 1.7 km off by the map was 90 m ahead, and
            # the drive ended there, the place 1.5 km on. A depot the GPS
            # goes to -- a trailer to hitch -- is planned to as a job's is.
            args.gps_only, args.game_route = True, False
        if code != 0:
            lines = wrong.strip().splitlines()
            detail = YARD if "cannot be reached from" in wrong else (lines[-1] if lines else "")
            write_outcome(outcome, "no_route", detail or "no route")
            return give_up(1)
        looking = time.perf_counter()
        if not sees_the_road(perception, since=went_at, within=3.0):
            write_outcome(outcome, "unsafe", no_view(perception))
            return give_up(1)
        args.route, args.output, args.outcome = route, folder, outcome
        took = f"planning {looking - planning:.1f} s, a view {time.perf_counter() - looking:.1f} s"
        return Standing(reader, perception, camera, went_at, took, truck.truck)


def doublet(elapsed: float, amplitudes: list[float]) -> float:
    """The excitation to add to the steering command at `elapsed` seconds."""
    cycle, phase = divmod(elapsed, DOUBLET_PERIOD_S)
    amplitude = amplitudes[int(cycle) % len(amplitudes)]
    if DOUBLET_AT_S <= phase < DOUBLET_AT_S + DOUBLET_HALF_S:
        return amplitude
    if DOUBLET_AT_S + DOUBLET_HALF_S <= phase < DOUBLET_AT_S + 2 * DOUBLET_HALF_S:
        return -amplitude
    return 0.0


#: Hard ceiling whatever is asked for: the game's own truck speed limiter.
#: The road's limit decides below it (`--speed` is only a ceiling); steering
#: has been driven to 85 km/h.
ABSOLUTE_SPEED_LIMIT_KMH = 90.0

#: Where the game gives no limit -- yards, some junctions -- the truck keeps
#: to a town's.
NO_LIMIT_KMH = 50.0
# With --above-road-limit only: to catch traffic doing the 80 km/h limit, and
# so test following, the truck must go faster than it. Steering has been
# driven to 60 km/h; this is the first step past it, taken on purpose.
TEST_SPEED_LIMIT_KMH = 85.0

#: How far behind the tractor to look for the trailer before its own length is
#: known. Longer than any trailer measured, so the seed is never ahead of it.
TRAILER_LOOKBACK = 12.0


KEEP_FROM_MPS = 1.0

ROS_VIEW_EVERY_S = 0.2
"""How often a view of the road goes to the ROS 2 bridge (`--ros`): JPEG at
the frame's own size, encoded in the perception child."""


def ros_tap():
    """The ROS 2 bridge's mailbox (adapters/ros2/feed.py): it never waits,
    and with no bridge listening drops what it is given."""
    from adapters.ros2.feed import Feed, Tap

    feed = Feed()
    print(f"ROS 2: the drive's view goes to {feed.path} once a bridge listens", flush=True)
    return Tap(feed)


def tell_ros(tap, now: float, block, path, driven, report, camera, following) -> None:
    """One turn of the drive, told to the ROS 2 bridge."""
    tap.pose(now, block)
    position = block["truck"]["placement"]["position"]
    x, y, z = float(position["x"]), float(position["y"]), float(position["z"])
    if path is not None:
        tap.route(now, path.points, height=y)
    tap.lanes(now, getattr(driven, "network", None), x, z)
    if report is not None and camera is not None:
        lead = getattr(following, "lead_id", None) if following is not None else None
        tap.objects(now, report, camera, lead_id=lead)
    reason = following.decision.reason if following is not None else ""
    tap.status(now, "driving" if block["truck"]["light_beacon"] else "handed over", reason)


"""Views of the road are kept (`--keep-frames`) only moving at least this fast."""

JUNCTION_STOP_SNAPSHOT_S = 2.0
"""At rest in a junction this long, the black box is written out: in
Bremen the truck stood in a left turn for a vehicle 11 m ahead that was not
there, and nothing showed what it had seen."""

LIGHT_WAIT_SNAPSHOT_S = 15.0
"""Stopped for a light this long, the black box is written out: in Milan the
truck sat through two greens, and at the next light by a steady amber."""

FUEL_TELL_BELOW_KM = 600.0
FUEL_TELL_EVERY_KM = 100.0
"""The driver fills up: with the fuel range, as the dashboard shows it, under
`FUEL_TELL_BELOW_KM` they are told in the launcher's window, and again each
`FUEL_TELL_EVERY_KM` less: nothing pops up on the desktop (the driver's
word, 14 November)."""

SNAPSHOT_URGENT_MPS2 = 3.0
"""A deceleration asked for at least this hard has the black box written out."""


class LoopStalled(UnsafeCondition):
    """The loop stood still as long as the plugin waits before it stops the
    truck itself (`SAFETY_STOP_S`): the truck is being stopped, and the drive
    is over. Driven on, the loop would steer a truck it no longer has."""


SLOW_TURN_S = 0.25
"""A turn of the loop this long is told of, with where it went (`TurnClock`)."""


class TurnClock:
    """Where the time of a slow turn of the loop went.

    On Windows the loop stood still for 0.2-1.2 s every few seconds, with the
    game going on at 60 frames a second, and once long enough for the plugin
    to stop the truck at 80 km/h. Python's collector takes 5 ms here. So each
    turn is timed by its parts, with the loop's thread's own CPU time and the
    process's page faults: a part that took the time, the thread running or
    not -- waiting on the GIL, or not scheduled -- and memory read back from
    the disk tell the causes apart.
    """

    def __init__(self) -> None:
        try:
            import psutil

            self._process = psutil.Process()
        except ImportError:  # the timing is worth having without it
            self._process = None
        self._marks: list[tuple[str, float]] = []
        self._cpu = time.thread_time()
        self._faults = self._page_faults()

    def _page_faults(self) -> int | None:
        if self._process is None:
            return None
        return getattr(self._process.memory_info(), "num_page_faults", None)

    def mark(self, part: str) -> None:
        """The part of the turn that has just ended."""
        self._marks.append((part, time.perf_counter()))

    def turn(self, began: float, dt: float) -> str | None:
        """A new turn begins at `began`, `dt` after the last: what the last
        one's time went on, if it was slow; and the clock starts again."""
        cpu, faults = time.thread_time(), self._page_faults()
        said = None
        if dt >= SLOW_TURN_S:
            parts, start = [], began - dt
            for part, at in self._marks:
                parts.append((at - start, part))
                start = at
            parts.append((began - start, "sleep"))
            took, part = max(parts)
            said = f"a slow turn of the loop, {dt:.2f} s: {took:.2f} s of it {part}"
            said += f"; its thread on the CPU {cpu - self._cpu:.2f} s"
            if faults is not None and self._faults is not None:
                said += f", {faults - self._faults} page faults"
        self._marks.clear()
        self._cpu, self._faults = cpu, faults
        return said


def fuel_to_tell(fuel_range: float, told_at: float | None) -> bool:
    """Whether to tell the driver to fill up, `fuel_range` km left, having told
    them last at `told_at` km (`FUEL_TELL_BELOW_KM`)."""
    if fuel_range >= FUEL_TELL_BELOW_KM:
        return False
    return told_at is None or told_at - fuel_range >= FUEL_TELL_EVERY_KM


def check_loop(dt: float) -> None:
    """Raise `LoopStalled` if the loop's last turn took `SAFETY_STOP_S`."""
    if dt >= SAFETY_STOP_S:
        raise LoopStalled(f"the loop stood still {dt:.2f} s: the plugin is stopping the truck")


class GpsWorkerGone(UnsafeCondition):
    """The GPS worker's process has died: no route will change any more. By
    Uppsala it fell over on an 8 m route to the depot, and the truck drove on
    the old route, the game's new one never followed, into a kerb."""


class LostThePath(UnsafeCondition):
    """Too far from the reference to be following it.

    A lateral controller with a wrong idea of where the road is steers
    confidently in the wrong direction, so distance from the path is the one
    thing that says "stop trusting me".
    """


def is_planned(route: Path) -> bool:
    """A route planned on the map, rather than a recording of one being driven."""
    return route.suffix == ".npz"


def load_path(route: Path, spacing: float) -> ReferencePath:
    """Build a reference path from a planned route or a recorded drive."""
    if is_planned(route):
        # A planned route is the lane's middle; the tractor swings wide of
        # it in tight turns so the trailer keeps to the lane.
        planned = saved_route(route)
        if planned is None:
            points = load_points(route)
            return ReferencePath(widen_for_trailer(points[:, [0, 2]]), spacing=spacing)
        return route_path(planned, spacing)
    recording = open_recording(route)
    position = recording.telemetry["block"]["truck"]["placement"]["position"]
    points = np.column_stack([position["x"].astype(float), position["z"].astype(float)])
    return ReferencePath(points, spacing=spacing)


#: A new route that is already off to one side where the truck is -- a lane
#: change back out of an exit lane, eased across from the lane's start
#: behind the truck -- is joined from the old line over this far instead.
EASE_ONTO_M = 80.0
EASE_BEHIND_M = 30.0


def eased_onto(
    old: ReferencePath, new: ReferencePath, progress: float, spacing: float
) -> tuple[ReferencePath, float | None]:
    """`new`, joined from `old` at `progress` where they are apart there;
    and where on it the joining begins, or None where they are not apart."""
    here = old.sample_at(progress)
    near = where_on(new, here.x, here.z, progress)
    if math.hypot(near.x - here.x, near.z - here.z) < 0.5:
        return new, None
    points = new.points
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    begin = near.s
    joined = points.copy()
    span = (arc >= begin - EASE_BEHIND_M) & (arc < begin + EASE_ONTO_M)
    for k in np.flatnonzero(span):
        was = old.sample_at(max(0.0, progress + arc[k] - begin))
        t = min(max((arc[k] - begin) / EASE_ONTO_M, 0.0), 1.0)
        weight = t * t * (3.0 - 2.0 * t)
        joined[k] = (1.0 - weight) * np.array([was.x, was.z]) + weight * points[k]
    return ReferencePath(joined, spacing=spacing), begin


OFF_PUMP_STRAIGHT_M = 12.0
AT_A_PUMP_M = 15.0
"""Starting within `AT_A_PUMP_M` of a pump, the truck goes straight ahead for
`OFF_PUMP_STRAIGHT_M` before easing onto the route: its trailer stands
beside the pump's island. Steered at once onto a lane turning right, out of
a station in Sweden, the trailer cut into the canopy's pillar."""

JOIN_FROM_M = 3.0
JOIN_MAX_M = 10.0
JOIN_PER_M = 6.0
JOIN_LENGTHS_M = (20.0, 60.0)
"""A truck starting more than `JOIN_FROM_M` off its route, but no more than
`JOIN_MAX_M`, drives onto it from where it stands, over `JOIN_PER_M` metres
per metre off, within `JOIN_LENGTHS_M`. Off the pump at a service area out
of Frederikshavn it stood 6.3 m beside the forecourt's lane, and the drive
stopped at once, off its path. Nearer than that the steering closes the gap
itself: eased over 80 m from straight ahead, a truck 2.1 m off its route in
a Milan depot drove on into a railing where the route turned at once."""


def joined_from(
    path: ReferencePath,
    x: float,
    z: float,
    course: float,
    spacing: float,
    straight: float = 0.0,
) -> ReferencePath | None:
    """`path` from the truck at `(x, z)`, heading `course`: itself if the
    truck is on it or near enough, eased onto from straight ahead if it is a
    little further off, None if it is further off than `JOIN_MAX_M`. With
    `straight`, the first that many metres are straight ahead however near
    it is: off a pump (`OFF_PUMP_STRAIGHT_M`)."""
    # Anywhere along it: a route starts where its first lane does, which in
    # Milan was 115 m behind the truck.
    near = path.nearest(x, z)
    off = math.hypot(near.x - x, near.z - z)
    if off <= JOIN_FROM_M and straight <= 0.0:
        return path
    if off > JOIN_MAX_M:
        return None
    length = min(max(JOIN_PER_M * off, JOIN_LENGTHS_M[0]), JOIN_LENGTHS_M[1])
    length = max(length, straight + JOIN_LENGTHS_M[0]) if straight > 0.0 else length
    blended = []
    for u in np.arange(0.0, length, spacing):
        on = path.sample_at(min(near.s + u, path.length))
        t = max(0.0, u - straight) / (length - straight)
        weight = t * t * (3.0 - 2.0 * t)
        ahead = (x + math.sin(course) * u, z + math.cos(course) * u)
        blended.append(
            (
                (1.0 - weight) * ahead[0] + weight * on.x,
                (1.0 - weight) * ahead[1] + weight * on.z,
            )
        )
    points = path.points
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    rest = points[arc >= near.s + length]
    return ReferencePath(np.vstack([np.asarray(blended), rest]), spacing=spacing)


def where_on(path: ReferencePath, x: float, z: float, progress: float):
    """The point of `path` nearest `(x, z)`: looked for about `progress` first,
    as a route that keeps what it had shares its arc lengths with the old one;
    over the whole path if that finds nothing near. A route the game had
    planned anew started at the truck, 1,700 m before where the old one was,
    and the point found about the old progress was 9 m off: the drive stopped."""
    near = path.nearest(x, z, near=progress, window=200.0)
    if math.hypot(near.x - x, near.z - z) > NEW_ROUTE_OFF_M:
        near = path.nearest(x, z)
    return near


def keep_route(output: Path, name: str, route: Route, perception) -> None:
    """Keep a route the drive took or refused beside the recording, as
    `routes/<name>.npz` -- its lanes and steps, the line to be had from them
    and the map -- with a black-box snapshot of the view and the minimap if
    `perception` is given: twice the truck ended on a service road, and the
    routes that took it there were never seen again. Lanes only: the line of
    a route 1243 km long, made on a thread, held the drive's loop 0.86 s."""
    folder = output / "routes"
    with contextlib.suppress(OSError):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{name}.npz",
            lanes=np.asarray(route.lanes, dtype=np.int64),
            steps=np.asarray(route.steps, dtype=str),
        )
    if perception is not None:
        perception.snapshot(name)


def parts_ahead(
    old: ReferencePath,
    new: ReferencePath,
    x: float,
    z: float,
    progress: float,
    apart: float | None = None,
) -> float | None:
    """How far ahead of the truck at `(x, z)` the `new` path leaves the `old`
    one it is driving, by more than `apart` (`PART_M`), within `PART_LOOK_M`;
    None if it does not. Asked for no `apart`, None too if they are apart
    already where the truck is (a lane over, eased onto: `eased_onto`)."""
    over = a_lane_over(old, new, x, z, progress)
    if apart is None and over:
        return None
    apart = PART_M if apart is None else apart
    on_new = where_on(new, x, z, progress)
    on_old = old.nearest(x, z, near=progress)
    ahead = 0.0
    while ahead <= PART_LOOK_M:
        if on_old.s + ahead > old.length or on_new.s + ahead > new.length:
            return None
        point = new.sample_at(on_new.s + ahead)
        there = old.nearest(point.x, point.z, near=on_old.s + ahead, window=30.0)
        if math.hypot(point.x - there.x, point.z - there.z) > apart:
            return ahead
        ahead += PART_STEP_M
    return None


def a_lane_over(
    old: ReferencePath, new: ReferencePath, x: float, z: float, progress: float
) -> bool:
    """Whether the `new` path is already apart from the `old` one where the
    truck at `(x, z)` is, by no more than a fork (`PART_M`, `FORK_M`)."""
    on_new = where_on(new, x, z, progress)
    on_old = old.nearest(x, z, near=progress)
    return PART_M < math.hypot(on_new.x - on_old.x, on_new.z - on_old.z) <= FORK_M


def refused_as_parting(
    old: ReferencePath, new: ReferencePath, marks, x: float, z: float, progress: float, speed: float
) -> float | None:
    """How far on a `new` route forks from the `old` one, if that is too soon
    to take it (`FORK_M`, `FORK_EASE_M`); None if it is to be taken."""
    parts = parts_ahead(old, new, x, z, progress, apart=FORK_M)
    if parts is None:
        return None
    soon = max(PART_SOON_M, speed * PART_SOON_S)
    if a_lane_over(old, new, x, z, progress):
        return parts if parts < max(soon, FORK_EASE_M) else None
    here = where_on(new, x, z, progress).s
    changing = any(
        change.start <= here + soon and change.start + change.span >= here
        for change in marks.changes
    )
    return parts if parts < soon and changing else None


PART_M = 1.75
PART_LOOK_M = 200.0
PART_STEP_M = 5.0
PART_SOON_M = 60.0
PART_SOON_S = 3.5
FORK_M = 6.0
FORK_EASE_M = EASE_ONTO_M + 20.0
"""A new route a lane over already, its lane forking from the one driven in
less than this, the easing onto it (`EASE_ONTO_M`) not done by then, is not
taken either: drawn between the two, the line ran straight on between the
lanes as they parted. At 59 km/h, out of an exit lane 2.2 s from its fork, the
truck went into the barrier there.

A new route that forks from the one driven -- more than `FORK_M` apart,
past a lane over -- sooner than `PART_SOON_S` at the truck's speed (and no
less than `PART_SOON_M`), with a lane change to make before it, is not
taken: there is no driving onto it. Turned to a pump at 77 km/h with the
exit 60 m on, the truck went over the hatched gore between the motorway
and the exit, and hit its sign. The drive goes on its way. One turning off
from the lane the truck is in is taken, however soon: refused 35 m from a
junction at 49 km/h, the red line's way
was lost, and the old route took the truck through a service station. And
the fork is what counts, not where the lane change begins: an exit 100 m
on at 78 km/h, its lane change begun 20 m on, was refused, and an exit a
driver takes was missed."""

REPLAN_LEFT_M = 600.0
REPLAN_SHORT_M = 2500.0
"""With under `REPLAN_LEFT_M` of the path left, the depot more than
`REPLAN_SHORT_M` beyond its end and no port that near its end, the route is
running out on the road: the GPS worker is asked to plan afresh from where
the truck is. Near Birmingham, a lane change into a queue given up and the
red line's new ways refused as too soon, the truck drove its lane to the
end of the path and stopped there, in the middle of the road."""


def to_nearest_point(points, at) -> float:
    """Metres from `at`, world (x, z), to the nearest of `points`; inf if none."""
    return min((math.dist(p, (float(at[0]), float(at[1]))) for p in points), default=math.inf)


NEW_ROUTE_OFF_M = 5.0
"""A new route whose line passes further than this from the truck is not
taken: driving onto it would be a swerve -- 9.4 m over, onto another lane
of a junction, after the game planned anew out of Malmö. One lane over is
taken, eased onto (`eased_onto`): half way across into a lane, the red line
turning back the way it came from, the new route runs down the lane the
truck is leaving, 3.4-4.5 m off. Refused at 3 m, out of Calais and on the
way to Milan, the red line's way was lost."""


#: A view of the road up to this old is late, not lost: the game or the
#: detector hiccuped. Three drives in a row ended on views 0.50 to 0.51 s old
#: (the stop is at 0.5), on open road. Later than this, or a covered view,
#: still stops the truck.
LATE_VIEW_S = 1.0
#: While the view is late the target is this much under the truck's speed:
#: slowing, gently, rather than driving on blind at speed.
LATE_VIEW_SLOWING_MPS = 2.0

OUT_OF_CAB_GIVE_UP_S = 8.0
"""The game's camera out of the cab this long, the drive stops; till then the
truck slows as for a view late (`LATE_VIEW_SLOWING_MPS`)."""


def late_but_not_lost(report, now: float) -> bool:
    """Whether a view too old to drive on is only a moment late."""
    return report is not None and not report.covered and now - report.grabbed_at <= LATE_VIEW_S


WITHIN_LANE_M = 0.6
"""How far out of its lane's middle a tractor swings in a turn, where the
map has no lane beside it on the outside. Swung 1.2 m wide into a Milan
T-junction, it hit a guard house's island on the outside; swung 0.6 m on an
English slip road turning on 15 m, its trailer cut 2.2 m inside and touched
the barrier there, and it was let swing 0.9 m. At that, turning left on
11 m by night near Munich, it swung 0.87 m, ran 0.25 m wider still, as it
does in slow tight turns, and its cab hit a lamp post at the road's edge
(1.2 %); out of a yard near Milan a guard booth's roof. What stands on the
outside is hit by the cab; on the inside the trailer mostly rides over a
kerb: back to 0.6."""

OUTSIDE_LANE_M = 4.5
OUTSIDE_NEAR_M = 2.0
SAME_WAY_RAD = math.radians(60.0)
"""A lane passing within `OUTSIDE_NEAR_M` of the point a lane's width out
from the line, going the same way within `SAME_WAY_RAD`, is road to swing
wide onto. One coming the other way is not: turning right into the depot
road at Duisburg, the oncoming lane beyond a splitter island was taken
for room, the tractor swung 1.2 m wide, and its cab hit the island's sign."""

RING_EXIT_SWING_M = 0.0
RING_EXIT_NEAR_M = 2.0
"""Out of a roundabout the tractor does not swing wide: beside the way out,
on the outside of its turn, is the island between it and the way in, with
its signs. In Italy, at night, swung 0.4 m wide and drifting 0.3 m more,
its cab hit the direction sign there twice -- at 19 km/h, and at 11."""

ONCOMING_NEAR_M = 7.0
ONCOMING_SWING_M = 0.4
"""With a lane coming the other way this near on the outside of a turn, the
tractor swings no more than `ONCOMING_SWING_M` within its lane: what lies
between is often an island. Turning right on the way out of Hamburg, the
oncoming lane 6 m off beyond a median, the tractor swung 0.6 m within its
lane, drifted 0.3 m more, and its cab hit the island's sign. Not swung at
all on an English slip road, the oncoming lanes 6.1-6.8 m off, the trailer
cut 2.2 m inside and touched the barrier there, twice: where traffic keeps
left, the outside of the tight turns is the oncoming side."""


def room_outside(
    network,
    points: np.ndarray,
    outward: np.ndarray,
    wanted: np.ndarray,
    ways_out: np.ndarray | None = None,
) -> np.ndarray:
    """How far each point of a line may swing to the outside of its turn,
    `wanted` that far: all the way where the map has another lane there,
    `WITHIN_LANE_M` where it has none, and not at all where a lane coming
    the other way is beside it on the outside (`ONCOMING_NEAR_M`), nor on
    a roundabout's way out: within `RING_EXIT_NEAR_M` of `ways_out`, points
    along them (`ways_out_of_rings`)."""
    room = np.full(len(points), np.inf)
    leaving = np.zeros(len(points), dtype=bool)
    if ways_out is not None and len(ways_out):
        for i in np.flatnonzero(wanted > 0.0):
            gap = np.min(np.hypot(*(ways_out - points[i]).T))
            leaving[i] = gap <= RING_EXIT_NEAR_M
    for i in np.flatnonzero(wanted > 0.0):
        point, out = points[i], outward[i]
        if not out.any():
            continue
        way = None
        if len(points) > 1:
            ahead = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
            way = math.atan2(float(ahead[0]), float(ahead[1]))
        if leaving[i]:
            room[i] = RING_EXIT_SWING_M
            continue
        if way is not None and _oncoming_outside(network, point, out, way):
            room[i] = ONCOMING_SWING_M
            continue
        if wanted[i] <= WITHIN_LANE_M:
            continue
        probe = point + OUTSIDE_LANE_M * out
        near = network.lanes_near(float(probe[0]), float(probe[1]), OUTSIDE_NEAR_M)
        beside = False
        for lane in near:
            seen = network.project(int(lane), float(probe[0]), float(probe[1]))
            if seen.distance > OUTSIDE_NEAR_M:
                continue
            if way is not None and abs(math.remainder(seen.heading - way, 2 * math.pi)) > (
                SAME_WAY_RAD
            ):
                continue  # coming the other way
            beside = True
            break
        if not beside:
            room[i] = WITHIN_LANE_M
    return room


def ways_out_of_rings(route: Route, spacing: float = 0.5) -> np.ndarray:
    """`(n, 2)` points, `spacing` apart, along the lanes of `route` that lead
    out of a roundabout (`out_of_a_ring`)."""
    network = route.network
    found = []
    for lane in dict.fromkeys(route.lanes):
        if not out_of_a_ring(network, int(lane)):
            continue
        line = np.asarray(network.lane_points(int(lane)))[:, [0, 2]]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
        at = np.arange(0.0, s[-1] + spacing, spacing)
        found.append(np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])]))
    return np.vstack(found) if found else np.zeros((0, 2))


def _oncoming_outside(network, point: np.ndarray, out: np.ndarray, way: float) -> bool:
    """Whether a lane coming the other way runs within `ONCOMING_NEAR_M` of
    `point` on the side `out`."""
    for lane in network.lanes_near(float(point[0]), float(point[1]), ONCOMING_NEAR_M):
        seen = network.project(int(lane), float(point[0]), float(point[1]))
        if seen.distance > ONCOMING_NEAR_M or seen.distance < 1.0:
            continue
        if abs(math.remainder(seen.heading - way, 2 * math.pi)) < math.pi - SAME_WAY_RAD:
            continue
        # The lane's line is `lateral` to the right of the point, as the
        # lane goes: left of a heading h is (cos h, -sin h).
        beside = -seen.lateral * np.array([math.cos(seen.heading), -math.sin(seen.heading)])
        if float(beside @ out) > 0.5 * seen.distance:
            return True
    return False


def route_path(
    route: Route,
    spacing: float,
    shapes: dict[tuple[int, ...], np.ndarray | None] | None = None,
) -> ReferencePath:
    """The reference path along a planned route: the lanes' middle, the
    tractor swinging wide of it in tight turns so the trailer keeps to the
    lane, wider still through customs, and a taught crossing as driven.
    A junction turn too tight for that is driven along the line `shapes`
    has for it, if it has one (core.planning.tight_turns, `TurnShaping`)."""
    lanes = route.points()[:, [0, 2]]
    exact = customs_ends(route, taught=True)
    ways_out = ways_out_of_rings(route)
    points = widen_for_trailer(
        lanes,
        wider=customs_ends(route),
        exact=exact,
        room=lambda points, outward, wanted: room_outside(
            route.network, points, outward, wanted, ways_out
        ),
    )
    if shapes:
        for stretch in stretches(route):
            line = shapes.get(stretch.key)
            if line is not None:
                points = splice(points, line)
    return ReferencePath(points, spacing=spacing)


def rehearse(
    directory: Path, path: ReferencePath, profile: SpeedProfile, target_speed: float
) -> int:
    """Run the controllers over the recording, without touching the game.

    Reports what the lateral controller would have commanded at each recorded
    pose. It is not a closed loop -- the truck's actual positions came from a
    human, not from these commands -- so it cannot show whether the controller
    would have stayed on the path. What it does show is whether the commands
    are sane, continuous and pointed the right way, which is most of what goes
    wrong first.
    """
    recording = open_recording(directory)
    truck = recording.telemetry["block"]["truck"]
    position = truck["placement"]["position"]
    lateral = PurePursuit()

    progress = None
    errors, steering, headings, planned, taught = [], [], [], [], []
    for index in range(0, len(recording.telemetry), 3):
        block = recording.telemetry["block"][index]
        command = lateral.step(
            x=float(position["x"][index]),
            z=float(position["z"][index]),
            heading=convert.course_rad(block),
            speed=max(float(truck["speed"][index]), 1.0),
            path=path,
            progress=progress,
        )
        progress = command.progress
        if command.at_end:
            # Past the end the aim point is pinned to the last path point and
            # the geometry flips as the vehicle passes it. A real run stops
            # here, so including it would report a discontinuity that never
            # gets commanded.
            break
        errors.append(command.lateral_error)
        steering.append(command.steering)
        headings.append(command.heading_error)
        planned.append(min(target_speed, profile.speed_at(command.progress)))
        taught.append(float(truck["speed"][index]))

    errors_array = np.abs(np.array(errors))
    steering_array = np.array(steering)
    print(f"rehearsal over {directory.name}: {len(errors)} samples")
    print(f"  path              {path.length:.0f} m, {len(path)} points")
    print(
        f"  lateral error     max {errors_array.max():.3f} m, "
        f"RMS {np.sqrt((errors_array**2).mean()):.3f} m"
    )
    print(f"  heading error     max {np.degrees(np.abs(headings)).max():.2f} deg")
    print(f"  steering command  {steering_array.min():+.3f} .. {steering_array.max():+.3f}")
    print(f"  largest step      {np.abs(np.diff(steering_array)).max():.4f} per sample")

    planned_array = np.array(planned) * MS_TO_KMH
    taught_array = np.array(taught) * MS_TO_KMH
    moving = taught_array > 1.0
    print(
        f"  ceiling           {target_speed * MS_TO_KMH:.0f} km/h, "
        f"lookahead {lateral.lookahead_for(target_speed):.0f} m"
    )
    print(
        f"  planned speed     {planned_array.min():.1f} .. {planned_array.max():.1f} km/h, "
        f"mean {planned_array.mean():.1f}"
    )
    print(
        f"  taught speed      {taught_array[moving].min():.1f} .. "
        f"{taught_array.max():.1f} km/h, mean {taught_array[moving].mean():.1f}"
    )
    slower = float((planned_array < taught_array - 2.0).mean())
    print(f"  slower than taught over {slower * 100:.0f}% of the route")
    print(f"  tightest corner   {profile.slowest() * MS_TO_KMH:.1f} km/h")
    if errors_array.max() > 1.0:
        print("\n  the controller does not agree with the recorded line; check the")
        print("  path before driving it")
        return 1
    return 0


def grade_profile(route: Path, path: ReferencePath) -> np.ndarray:
    """Gravity's along-track contribution at each path point.

    Included because it is the largest longitudinal disturbance at these
    speeds -- a 5 degree grade is 0.86 m/s^2 against 0.24 of drag -- so a
    simulation on imagined flat ground would flatter the speed controller
    exactly where it is hardest pressed.

    From the drive's own pitch for a recording; for a planned route, from the
    heights of the map's lanes, which is the same road and a better source.
    """
    if is_planned(route):
        return _map_grade(load_points(route), path)
    recording = open_recording(route)
    truck = recording.telemetry["block"]["truck"]
    position = truck["placement"]["position"]
    pitch = truck["placement"]["orientation"]["pitch"].astype(float) * 2 * np.pi
    gravity = -convert.GRAVITY * np.sin(pitch)

    progress = None
    samples: list[tuple[float, float]] = []
    for index in range(0, len(recording.telemetry), 5):
        sample = path.nearest(
            float(position["x"][index]), float(position["z"][index]), near=progress
        )
        progress = sample.s
        samples.append((sample.s, float(gravity[index])))

    samples.sort()
    arc = np.array([s for s, _ in samples])
    values = np.array([g for _, g in samples])
    return np.interp(np.linspace(0.0, path.length, len(path)), arc, values)


def _map_grade(points: np.ndarray, path: ReferencePath) -> np.ndarray:
    """Along-track gravity from the lane heights, smoothed over 20 m.

    The lanes are sampled every 2 m and their heights come from splines, so a
    raw difference is noisy; a truck averages the slope over its own length.
    """
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 2]))
    s = np.concatenate([[0.0], np.cumsum(step)])
    grid = np.arange(0.0, s[-1], 1.0)
    height = np.interp(grid, s, points[:, 1])
    window = 21
    padded = np.pad(height, window // 2, mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    slope = np.gradient(smooth, grid)
    gravity = -convert.GRAVITY * np.sin(np.arctan(slope))
    return np.interp(np.linspace(0.0, path.length, len(path)), grid, gravity)


def simulate(
    directory: Path,
    path: ReferencePath,
    profile: SpeedProfile,
    target_speed: float,
    *,
    start: float,
    seconds: float,
    excite: list[float] | None = None,
    rate_gain: float | None = None,
    steering_schedule: bool = True,
    lookahead_time: float = 1.2,
    delay: float = 0.0,
) -> int:
    """Run the whole stack against a simulated vehicle on the real route.

    Unlike `--replay` this is a closed loop: the vehicle goes where the
    controllers steer it, so it answers the question that matters -- can they
    hold this route, including the corner that caps out at 18 km/h.

    The plant is a bicycle model behind the measured steering actuator and the
    measured throttle and brake maps, with the recorded gradient applied along
    the way. What it cannot model is what was never measured: tyre slip, the
    trailer, and how much of the steering map still holds at the lock these
    corners demand.
    """
    grades = grade_profile(directory, path)
    grade_at = lambda s: float(  # noqa: E731
        np.interp(s, np.linspace(0.0, path.length, len(path)), grades)
    )

    lateral = PurePursuit(lookahead_time=lookahead_time)
    steering = SteeringPositionController()
    speed_controller = SpeedController()
    actuators = ActuatorMap()

    sample = path.sample_at(start)
    x, z, course = sample.x, sample.z, sample.heading
    speed, applied = 0.0, 0.0
    progress = start
    dt = 1 / 60

    errors, speeds, planned_speeds, positions = [], [], [], []
    gain = rate_gain
    in_flight: deque[float] = deque()
    for tick in range(int(seconds / dt)):
        command = lateral.step(
            x=x, z=z, heading=course, speed=max(speed, 1.0), path=path, progress=progress
        )
        progress = command.progress
        errors.append(command.lateral_error)
        positions.append(progress)
        if command.at_end or abs(command.lateral_error) > MAX_LATERAL_ERROR_M:
            break

        planned = min(target_speed, profile.speed_at(progress))
        planned_speeds.append(planned)
        longitudinal = speed_controller.step(
            target_speed=planned,
            speed=speed,
            grade=grade_at(progress),
            dt=dt,
            coast_above=profile.coasting_at(progress, speed),
        )

        rate = steering.command(
            target=command.steering, measured=applied, speed=speed if steering_schedule else None
        )
        if excite:
            rate += doublet(tick * dt, excite)
        # A command reaches the game `delay` late: the input path measured
        # 15-28 ms, telemetry and the frame add more.
        in_flight.append(rate)
        rate = in_flight.popleft() if len(in_flight) > round(delay / dt) else 0.0
        # The plant is the actuator as measured while driving, unless told otherwise.
        plant = gain if gain is not None else rate_gain_at(speed)
        # And the game's pull back to centre, fitted to the drives: without it
        # the simulation under-read the error in bends two to three times.
        applied += (STEERING_CALIBRATION.sign * plant * rate - centring_at(speed) * applied) * dt
        applied = max(-1.0, min(1.0, applied))

        speed = max(
            0.0,
            speed
            + (
                actuators.acceleration_for(longitudinal.throttle, longitudinal.brake)
                + grade_at(progress)
            )
            * dt,
        )
        speeds.append(speed)
        course += (
            (speed / ACTROS_GEOMETRY.wheelbase)
            * math.tan(ACTROS_GEOMETRY.angle_for_command(applied))
            * dt
        )
        x += speed * math.sin(course) * dt
        z += speed * math.cos(course) * dt

    error_array = np.abs(np.array(errors))
    travelled = positions[-1] - start
    worst = int(np.argmax(error_array))

    print(f"simulation over {directory.name}, starting at s={start:.0f} m")
    print(f"  travelled         {travelled:.0f} m of {path.length:.0f} in {len(errors) * dt:.0f} s")
    print(
        f"  lateral error     max {error_array.max():.2f} m at s={positions[worst]:.0f}, "
        f"RMS {np.sqrt((error_array**2).mean()):.2f} m"
    )
    print(f"  over 0.5 m        {100 * (error_array > 0.5).mean():.0f}% of the route")
    print(f"  in bends          {bend_offset(np.array(errors), path, positions)}")
    if speeds:
        speed_array = np.array(speeds) * MS_TO_KMH
        planned_array = np.array(planned_speeds) * MS_TO_KMH
        print(f"  speed             {speed_array.min():.1f} .. {speed_array.max():.1f} km/h")
        print(
            f"  speed error       RMS "
            f"{np.sqrt(((speed_array - planned_array) ** 2).mean()):.1f} km/h"
        )
        lateral_g = (np.array(speeds) ** 2) * np.abs(
            [path.sample_at(s).curvature for s in positions[: len(speeds)]]
        )
        print(f"  lateral accel     max {lateral_g.max():.2f} m/s^2")

    if error_array.max() > MAX_LATERAL_ERROR_M:
        print("\n  LOST THE PATH -- do not drive this")
        return 1
    if error_array.max() > 1.5:
        print("\n  tracking is loose; worth understanding before driving it")
        return 1
    return 0


def bend_offset(errors: np.ndarray, path: ReferencePath, positions: list[float]) -> str:
    """Which way the truck drifts in bends, from signed lateral errors.

    A signed error with the curvature's sign is towards the inside: the
    uncompensated steering ran wide by up to 1.5 m, the first compensated
    drive ran inside by half a metre.
    """
    curvature = np.array([path.sample_at(s).curvature for s in positions[: len(errors)]])
    bends = np.abs(curvature) > 0.002
    if bends.sum() < 10:
        return "no bends"
    inward = errors[bends] * np.sign(curvature[bends])
    side = "inside" if np.mean(inward) > 0 else "outside"
    worst = np.abs(inward).max()
    return f"{abs(np.mean(inward)):.2f} m to the {side} on average (worst {worst:.2f} m)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "route",
        type=Path,
        nargs="?",
        help="recording directory, or a route from tools/plan_route.py; none with --standby",
    )
    parser.add_argument(
        "--standby",
        type=Path,
        default=None,
        metavar="WORD_FILE",
        help=(
            "make ready before the beacon is on and wait for word in WORD_FILE to plan "
            "from where the truck is and drive (tools/mission.py; standby_until_go)"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "data" / "map" / "installed",
        help="the map to plan on, with --standby",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=ABSOLUTE_SPEED_LIMIT_KMH,
        help="ceiling, km/h; below it the road's own limit decides",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="metres of clear road ahead *now* (required to drive)",
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--spacing", type=float, default=2.0, help="path resampling, m")
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "arc length to start from, metres. Needed on a closed loop, where "
            "the start and the end are the same place and an unseeded search "
            "finds whichever it likes"
        ),
    )
    parser.add_argument(
        "--lateral",
        type=float,
        default=2.5,
        help="cornering limit, m/s^2 (the taught drive reached 5.6)",
    )
    parser.add_argument("--replay", action="store_true", help="rehearse offline and exit")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="closed-loop simulation on this route, without the game",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--outcome",
        type=Path,
        default=None,
        help="write why the drive ended here, as JSON (tools/mission.py reads it)",
    )
    parser.add_argument(
        "--beacon",
        action="store_true",
        help=(
            "engaged by the truck's beacon (tools/mission.py): switched off, the "
            "controls are let go at once, the driver's"
        ),
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--sim-rate-gain",
        type=float,
        default=None,
        help="steering actuator gain for --simulate, /s (default: the calibration)",
    )
    parser.add_argument(
        "--sim-delay",
        type=float,
        default=0.0,
        help="for --simulate: seconds a steering command takes to reach the truck",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=1.2,
        help=(
            "pure pursuit aim, seconds of travel ahead. 1.2 since 2026-09-24: at 85 km/h "
            "it leaned 0.18 m into bends where 1.8 leaned 0.52"
        ),
    )
    parser.add_argument(
        "--old-steering",
        action="store_true",
        help=(
            "steer the way it did before 2026-09-24: the stationary calibration's gain "
            "and nothing for the game's pull back to centre. It ran 1.5 m wide at "
            "85 km/h; kept for comparison"
        ),
    )
    parser.add_argument(
        "--excite",
        type=float,
        nargs="+",
        metavar="AMPLITUDE",
        help="add steering doublets of these sizes, in turn, to identify the actuator",
    )
    parser.add_argument(
        "--acc",
        action="store_true",
        help="follow the vehicle ahead, seen by the camera (needs the perception extra)",
    )
    parser.add_argument("--time-gap", type=float, default=1.5, help="ACC time gap, s")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="drive with --acc even if the detector could only run on the CPU (slower)",
    )
    parser.add_argument(
        "--keep-frames",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="with --acc, record a view of the road this often while moving, for training",
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        help="send the drive's view to the ROS 2 bridge (adapters/ros2): pose, route, lanes, "
        "what perception sees, and a view of the road five times a second",
    )
    parser.add_argument(
        "--above-road-limit",
        action="store_true",
        help=(
            f"ignore the road's speed limit and allow up to {TEST_SPEED_LIMIT_KMH:.0f} km/h: "
            "a test of following traffic that keeps to the limit"
        ),
    )
    parser.add_argument(
        "--gps-only",
        action="store_true",
        help=(
            "with --gps: go where the minimap's red line goes and nowhere else -- no "
            "route of our own to the depot; start from plan_route.py --live --ahead"
        ),
    )
    parser.add_argument(
        "--gps",
        action="store_true",
        help=(
            "turn where the game's GPS turns, read off the minimap, re-planning to the "
            "job's depot (a route from plan_route.py --to-job, with --acc)"
        ),
    )
    parser.add_argument(
        "--game-route",
        action="store_true",
        help=(
            "drive the route the game's GPS takes, planned from the map and kept to the "
            "game's navigation distance and the minimap (core.planning.game_route); "
            "start from plan_route.py --live --to-job --game, with --acc"
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    args = parser.parse_args()

    standing = None
    args.tap = ros_tap() if args.ros else None
    if args.standby is not None:
        standing = standby_until_go(args)
        if isinstance(standing, int):
            return standing
    elif args.route is None:
        parser.error("give a route, or --standby")
    args.gps = args.gps or args.gps_only or args.game_route
    # Following the GPS, the loop begins on the route's first part and the
    # worker sends the whole (GpsWorker `whole`): the path of a whole 30 km
    # route took 6.3 s to make, and the truck stood waiting for it.
    driven = saved_route(args.route) if is_planned(args.route) else None
    if driven is not None and args.gps:
        driven = driven.ahead(0, FIRST_PART_M)
    game_scale = 1.0
    """The game's metres to the path's: what is said is said in the game's."""
    try:
        if driven is not None:
            path = route_path(driven, args.spacing)
        else:
            path = load_path(args.route, args.spacing)
    except (OSError, ValueError) as exc:
        print(f"cannot build a path from {args.route}: {exc}", file=sys.stderr)
        return 1

    target = args.speed / MS_TO_KMH
    profile = SpeedProfile(path, max_lateral_acceleration=args.lateral, max_speed=target)
    if driven is not None:
        game_scale = game_km(driven) * 1000.0 / max(path.length, 1.0)
    if args.replay:
        if is_planned(args.route):
            print("--replay needs a recording to replay; use --simulate", file=sys.stderr)
            return 2
        return rehearse(args.route, path, profile, target)
    if args.simulate:
        return simulate(
            args.route,
            path,
            profile,
            target,
            start=args.start or 0.0,
            seconds=args.duration,
            excite=args.excite,
            rate_gain=args.sim_rate_gain,
            steering_schedule=not args.old_steering,
            lookahead_time=args.lookahead,
            delay=args.sim_delay,
        )

    ceiling_kmh = TEST_SPEED_LIMIT_KMH if args.above_road_limit else ABSOLUTE_SPEED_LIMIT_KMH
    if args.speed > ceiling_kmh:
        print(f"target above the {ceiling_kmh:.0f} km/h ceiling", file=sys.stderr)
        return 2
    if args.distance is None:
        print(
            "--distance is required to drive. Give the metres of clear road\n"
            "actually ahead of the truck *now*, not what was there before.",
            file=sys.stderr,
        )
        return 2

    if args.gps and not (args.acc and is_planned(args.route)):
        print("--gps needs --acc and a route from plan_route.py --to-job", file=sys.stderr)
        return 2
    lights = TrafficLights(
        lines_on_path(load_stop_lines(args.route), path) if is_planned(args.route) else []
    )
    light_filter = LightFilter()
    barriers = marks_of(driven, path) if driven is not None else []
    taught_spans = taught_spans_on(driven, path) if driven is not None else []
    if lights.lines:
        print(f"{len(lights.lines)} traffic lights on the route", end="")
        print("" if args.acc else ": without --acc none is seen, and it stops at the first")

    if not args.yes:
        if args.acc:
            print("It keeps a gap to the vehicle ahead as the camera sees it, stops for")
            print("red lights, and stops if it loses sight of the road. It does not")
            print("give way at junctions or see pedestrians.")
        print(f"This drives the truck along {args.route.name} at {args.speed:.0f} km/h,")
        print(f"steering as well as accelerating, for up to {args.duration:.0f} s")
        print(f"or {args.distance:.0f} m. Ctrl-C disengages.")
        try:
            if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                return 1
        except EOFError:
            return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    output = args.output or REPO_ROOT / "data" / "follow" / f"{stamp}_follow"

    try:
        if standing is not None:
            reader = standing.reader
        else:
            reader = TelemetryReader()
            reader.open()
            reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot reach the game: {exc}", file=sys.stderr)
        return 1
    if reader.read().is_paused:
        print("the game is paused -- close the map and try again", file=sys.stderr)
        reader.close()
        return 1
    first = reader.read().block
    truck = load_profile(truck_id(first))
    truck_profile = truck  # `truck` is the telemetry's in the loop
    print(f"truck {truck.truck}: {uncalibrated(truck) or 'calibrated'}")
    camera_file = truck.camera_file() if args.camera == DEFAULT_CAMERA else args.camera
    if is_planned(args.route) and not args.start:
        block = reader.read().block
        position = block["truck"]["placement"]["position"]
        at_a_pump = False
        with contextlib.suppress(Exception):
            at_a_pump = any(
                math.hypot(
                    pump.position[0] - float(position["x"]), pump.position[1] - float(position["z"])
                )
                <= AT_A_PUMP_M
                for pump in load_pumps(route_cache(args.route))
            )
        joined = joined_from(
            path,
            float(position["x"]),
            float(position["z"]),
            convert.course_rad(block),
            args.spacing,
            straight=OFF_PUMP_STRAIGHT_M if at_a_pump else 0.0,
        )
        if at_a_pump:
            print(f"off a pump: {OFF_PUMP_STRAIGHT_M:.0f} m straight on first")
        if joined is None:
            print(
                f"the truck is more than {JOIN_MAX_M:.0f} m off the road: drive onto it first",
                file=sys.stderr,
            )
            if args.outcome is not None:
                write_outcome(args.outcome, "unsafe", "the truck is off the road")
            reader.close()
            return 1
        if joined is not path:
            path = joined
            profile = SpeedProfile(path, max_lateral_acceleration=args.lateral, max_speed=target)

    messages_file = None
    recorder = Recorder(
        output,
        notes=f"following {args.route.name} at {args.speed:.0f} km/h",
        metadata={
            "route": str(args.route),
            "target_speed_kmh": args.speed,
            "excitation": None
            if not args.excite
            else {
                "amplitudes": args.excite,
                "period_s": DOUBLET_PERIOD_S,
                "at_s": DOUBLET_AT_S,
                "half_s": DOUBLET_HALF_S,
            },
        },
    )
    perception = cruise = gate = camera = None
    indicators = None
    signalled: Route | None = None  # the route the indicators are for
    # Where a missed lane change left the route, and the speeds along the
    # lane the truck went on in from there.
    held_profile: tuple[float, SpeedProfile] | None = None
    if args.acc:
        try:
            if standing is not None and standing.truck != truck.truck:
                # Made ready for another truck, or for none: its camera would
                # put every distance wrong.
                print(
                    f"perception was ready for {standing.truck or 'no truck'}: "
                    f"started again for {truck.truck}",
                    file=sys.stderr,
                )
                standing.perception.stop()
                standing = replace(standing, perception=None)
            if standing is not None and standing.perception is not None:
                perception, camera = standing.perception, standing.camera
                perception.write_black_box_to(output / "events")
            else:
                perception, camera = start_perception(
                    args.model,
                    camera_file,
                    args.allow_cpu,
                    black_box=output / "events",
                    truck=truck,
                )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(f"cannot start perception: {exc}", file=sys.stderr)
            reader.close()
            return 1
        cruise = AdaptiveCruise(camera, GapPolicy(time_gap_s=args.time_gap))
        planned = driven
        if planned is not None:
            gate = LaneChangeGate(lane_changes(planned, path), camera, path)
            signalled = planned
            indicators = Indicators(route_signals(planned, path, gate.signalled()))
            cruise.junctions = junction_spans(planned, path)
            cruise.turns_across = turn_across_spans(planned, path)
            cruise.rings = ring_entry_spans(planned, path)
            cruise.ring_ways = ring_way_spans(planned, path)
            cruise.crossings = crossing_spans(planned, path)
            cruise.tight_turns = tight_turn_spans(planned, path)
            cruise.prefabs = prefab_spans(planned, path)
        print("perception running")
    acc_log: list[tuple] = []  # time, speed, lead id, distance, closing, target, urgent, age
    # (time, reason) each time following's reason changes: in the customs
    # queue out of Basel the truck drove on into the tanker it had stopped
    # behind, and acc.npy could not say why.
    acc_reasons: list[tuple[float, str]] = []
    # time, painted lane's left and right lines and the path's middle, metres
    # right of the eye `LANE_CHECK_M` ahead (NaN where unseen)
    lane_log: list[tuple] = []
    lane_sequence = -1
    kept_every = 0.0
    recorded_at = -math.inf
    following = None
    service_stop: float | None = None  # before a depot's own road (`PathMarks`)
    # How far before the next light's line to stop to keep it in view, and
    # that line (core.control.traffic_light.stop_short_to_see).
    light_stop, light_stop_at = STOP_SHORT_M, None
    light_view = view_bounds(truck_profile)
    # One lamp of the light ahead as it came nearer: (metres to its line,
    # x, y), for where it goes out of view (stop_short_from_sightings).
    lamp_sightings: list[tuple[float, float, float]] = []
    # The painted lane bounding the line where the map's leaves it
    # (core.control.lane_guard); its black box written once per keeping.
    lane_guard = LaneGuard(truck=-cruise.camera.seat_right_m) if cruise is not None else LaneGuard()
    guard_snapped = False
    impacts = Impacts()
    damage = DamageWatch()
    standstill = Standstill()
    was_urgent = False
    # time, distance to the stop line, filtered state, frame reading, lamp area, target, urgent
    light_log: list[tuple] = []
    light_sequence = -1

    commander = Commander(recorder=recorder)
    try:
        recorder.open()
        messages_file, logs = keep_messages(output, {"stdout": sys.stdout, "stderr": sys.stderr})
        sys.stdout, sys.stderr = logs["stdout"], logs["stderr"]
        commander.open()
    except (OSError, PalError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        stop_keeping(messages_file)
        recorder.close()
        reader.close()
        return 1

    gps = None
    replan_asked = None
    """The path a fresh plan was last asked for on (`REPLAN_LEFT_M`)."""
    boarding = []
    if driven is not None:
        boarding = [
            tuple(driven.network.lane_points(board)[-1][[0, 2]])
            for board in driven.network.ferry_links
        ]
    place: PlaceArrival | None = None
    """Following the red line to a place marked on the map, with no end of
    its own: the game's distance says when it is reached."""
    if args.gps:
        try:
            gps = GpsWorker(
                gps_router(
                    args.route, reader.read().block, only=args.gps_only, game=args.game_route
                ),
                spacing=args.spacing,
                lateral=args.lateral,
                max_speed=target,
                whole=driven is not None,
            )
        except (OSError, ValueError, RouteError, MapCacheError) as exc:
            print(f"cannot follow the GPS: {exc}", file=sys.stderr)
            commander.close()
            recorder.close()
            reader.close()
            if perception is not None:
                perception.stop()
            return 1
        depot = gps.router.depot
        if isinstance(gps.router, GpsFollower) and depot is None:
            place = PlaceArrival()
        where = f"{depot.company} in {depot.city}" if depot is not None else "wherever it goes"
        how = (
            "the game's route, planned from the map"
            if args.game_route
            else "the minimap's red line alone"
            if args.gps_only
            else "the GPS"
        )
        print(f"following {how} to {where}")

    # The truck's own: `geometry` further on is the trailer's, and a turn's
    # cap read that one and stopped six drives in their first seconds.
    truck_geometry = truck.geometry(first)
    lateral = PurePursuit(truck_geometry, lookahead_time=args.lookahead)
    steering = SteeringPositionController()

    def keep_lane(block) -> float:
        """The steering that keeps to the path now: for stopping in lane."""
        nonlocal progress
        position = block["truck"]["placement"]["position"]
        command = lateral.step(
            x=float(position["x"]),
            z=float(position["z"]),
            heading=convert.course_rad(block),
            speed=max(float(block["truck"]["speed"]), 1.0),
            path=path,
            progress=progress,
        )
        progress = command.progress
        return steering.command(
            target=command.steering,
            measured=float(block["truck"]["effective_steering"]),
            speed=float(block["truck"]["speed"]),
        )

    speed_controller = SpeedController(truck.longitudinal())
    parking = ParkingBrake()
    # The retarder, for slowing down more than coasting does and less than
    # the brake's floor (core.control.retarder).
    retarder_estimate = RetarderEstimate()
    # The brake's own strength, learned (core.control.longitudinal).
    brake_estimate = BrakeEstimate()
    retarder_lever = RetarderLever()
    # Round a tight turn wide while its mouth is clear (core.control.turn_line).
    turn_line = WideTurnChoice()
    turn_line_said: str | None = None
    out_of_cab_said = False
    wide_turns: list = []
    stop_latch = StopLatch()
    moving_off_since: float | None = None
    guard = SafetyMonitor(
        speed_limit_ms=(args.speed + 15.0) / MS_TO_KMH,
        distance_budget_m=args.distance,
    )

    print(f"\nfollowing {args.route.name} ({path.length:.0f} m) at {args.speed:.0f} km/h")
    print(f"recording to {output}\n")

    outcome = 0
    ending = ("end_of_route", "")
    """Why the drive ended, for --outcome: (reason, detail)."""
    light_wait_since: float | None = None
    junction_stop_since: float | None = None
    """When the truck stopped for the light ahead; inf once its wait is shot."""
    fuel_told_at: float | None = None
    """The fuel range, km, when the driver was last told to fill up."""
    reader_block = None
    progress = args.start
    paused_reported = False
    last_write_count = -1
    sent_brake: float | None = None
    """The brake last sent; at the start, the game's own (the driver's foot)."""
    # The pedals sent lately: what the game may still show as the stack's
    # (core.control.override.RecentPedals).
    recent_pedals = RecentPedals()
    """(throttle, brake) last sent, to tell the driver's feet by."""
    feet = OverrideDetector(pedal_tolerance=PEDAL_TAKEOVER, dwell=PEDAL_TAKEOVER_S)
    health = HealthSampler().start()
    previous = started = time.perf_counter()
    for log in (sys.stdout, sys.stderr):
        if isinstance(log, MessageLog):
            log.origin = started  # stamped as the events and logs are
    last_print = 0.0
    worst_error = 0.0
    last_error = 0.0
    worst_trailer = 0.0
    tracker: TrailerTracker | None = None
    hazards_off = HazardsOff()
    clock = TurnClock()
    engaged_said = False

    try:
        while True:
            now = time.perf_counter()
            dt, previous = now - previous, now
            slow = clock.turn(now, dt)
            if slow is not None:
                print(f"\n[{slow}]", file=sys.stderr)
            check_loop(dt)
            if now - started > args.duration:
                print("\n\nduration reached")
                ending = ("duration", "")
                break

            snapshot = reader.read()
            block = snapshot.block
            reader_block = block
            truck = block["truck"]

            if snapshot.is_paused:
                commander.send(ControlCommand(), engaged=False)
                recent_pedals.clear()
                speed_controller.reset()
                previous = time.perf_counter()
                if not paused_reported:
                    print("\n[paused -- controls released]", file=sys.stderr)
                    paused_reported = True
                time.sleep(0.05)
                continue
            if paused_reported:
                print("[resumed]", file=sys.stderr)
                paused_reported = False
            if args.beacon and not int(block["truck"]["light_beacon"]):
                print("\n\nthe beacon is off: the driver has the truck", file=sys.stderr)
                if not engaged_said and standing is not None:
                    # Off before the truck was ever driven: where the wait
                    # went. Near Geneva a drive stood 5.9 s, the driver
                    # switched the beacon off, and nothing said why.
                    print(
                        f"[never engaged, {now - standing.went_at:.1f} s after the word "
                        f"to go: {standing.took}, getting ready "
                        f"{started - standing.went_at:.1f} s, driving {now - started:.1f} s]",
                        file=sys.stderr,
                    )
                ending = ("disengaged", "")
                break
            # The driver's foot on a pedal is the driver taking the truck, as
            # the beacon going off is: at an automatic gate out of
            # Luxembourg the driver braked, and the truck pressed full
            # throttle against the brake until it gave up. Not judged across
            # a stalled turn, while the plugin held the throttle off.
            lately = recent_pedals.most()
            if lately is not None and dt < PEDAL_TAKEOVER_MAX_DT:
                foot = feet.update_pedals(
                    commanded_throttle=lately[0],
                    commanded_brake=lately[1],
                    input_throttle=float(truck["input_throttle"]),
                    input_brake=float(truck["input_brake"]),
                    dt=dt,
                )
                if foot is not None:
                    print(
                        f"\n\nthe driver's foot on the {foot.control}: the driver has the truck",
                        file=sys.stderr,
                    )
                    ending = ("disengaged", str(foot))
                    break
            else:
                feet.reset()

            speed = float(truck["speed"])
            position = truck["placement"]["position"]
            x, z = float(position["x"]), float(position["z"])
            job = block["job"]
            speed_controller.actuators.mass_ratio = mass_ratio(
                trailer=bool(int(block["trailers"][0]["connected"])),
                cargo_kg=float(job["cargo_mass"])
                if int(job["job_active"]) and int(job["cargo_loaded"])
                else 0.0,
            )
            retarder_level = int(truck["retarder_level"])
            speed_controller.actuators.retarder_steps = (
                int(truck["retarder_step_count"]) if retarder_lever.works else 0
            )
            retarder_estimate.observe(
                level=retarder_level,
                throttle=float(truck["effective_throttle"]),
                brake=float(truck["effective_brake"]),
                acceleration=convert.forward_acceleration(block),
                grade=convert.grade_acceleration(block),
                speed=speed,
                coast_deceleration=speed_controller.actuators.calibration.coast_deceleration,
                mass_ratio=speed_controller.actuators.mass_ratio,
                dt=max(dt, 0.0),
            )
            speed_controller.actuators.retarder_per_step = retarder_estimate.per_step
            brake_estimate.observe(
                brake=float(truck["effective_brake"]),
                throttle=float(truck["effective_throttle"]),
                acceleration=convert.forward_acceleration(block),
                grade=convert.grade_acceleration(block),
                speed=speed,
                calibration=speed_controller.actuators.calibration,
                dt=max(dt, 0.0),
            )
            if brake_estimate.seen_s >= BRAKE_TRUSTED_AFTER_S:
                speed_controller.actuators.brake_ratio = brake_estimate.ratio

            clock.mark("reading the telemetry")
            if gps is not None:
                # The minimap's route goes to the worker; a route it has
                # re-planned comes back, and is driven from here on.
                latest = perception.latest()
                if latest is not None:
                    gps.offer(
                        (x, z, convert.course_rad(block)),
                        latest.route_mask,
                        latest.road_mask,
                        float(block["navigation"]["distance"]),
                    )
                fuel_range = float(truck["fuel_range"])
                if fuel_to_tell(fuel_range, fuel_told_at):
                    fuel_told_at = fuel_range
                    print(f"\n[fuel for {fuel_range:.0f} km: fill up]")
                taken = gps.take()
                if not getattr(gps, "alive", True):
                    raise GpsWorkerGone("the GPS worker stopped: the route is no longer followed")
                for note in gps.notes():
                    print(f"\n[GPS: {note}]")
                    if "red line leaves" in note and perception is not None:
                        perception.snapshot(f"red-line-{now - started:07.1f}")
                if taken is not None:
                    rerouted, new_path, new_profile, marks = taken
                    at = where_on(new_path, x, z, progress)
                    if math.hypot(at.x - x, at.z - z) > NEW_ROUTE_OFF_M:
                        print(
                            f"\n[GPS: a new route {math.hypot(at.x - x, at.z - z):.1f} m from "
                            "the truck: not taken]",
                            file=sys.stderr,
                        )
                        taken = None
                        gps.refuse()
                if taken is not None:
                    parts = refused_as_parting(path, taken[1], taken[3], x, z, progress, speed)
                    if parts is not None:
                        keep_route(output, f"refused-{now - started:07.1f}", taken[0], perception)
                        print(
                            f"\n[GPS: a new route leaving this one {parts:.0f} m on, too soon "
                            f"at {speed * MS_TO_KMH:.0f} km/h: not taken]"
                        )
                        taken = None
                        gps.refuse()
                if taken is not None:
                    gps.drive()
                    profile = new_profile
                    held_profile = None
                    path, _ = eased_onto(path, new_path, progress, args.spacing)
                    previous_lights = lights
                    # Marked along the new path in the worker: eased onto
                    # from the old one over a few metres, its lengths are the
                    # new path's within centimetres.
                    lights = TrafficLights(marks.lights)
                    service_stop = marks.service_stop
                    lights.carry_over(previous_lights)
                    barriers = marks.barriers
                    taught_spans = marks.taught_spans
                    wide_turns = marks.wide_turns
                    if gate is not None:
                        gate = LaneChangeGate(marks.changes, gate.camera, path)
                        # A change begun behind the truck, still in its lane:
                        # held there, to begin from here, whole, once clear
                        # (core.control.lane_change) -- not squeezed in.
                        held = gate.adopt(where_on(path, x, z, progress).s, x, z)
                        if held is not None:
                            path = held
                            print("\n[a lane change begun behind the truck: held in its lane]")
                        signalled = rerouted
                        indicators = Indicators(route_signals(rerouted, path, gate.signalled()))
                        if cruise is not None:
                            cruise.junctions = marks.junctions
                            cruise.turns_across = marks.turns_across
                            cruise.rings = marks.rings
                            cruise.ring_ways = marks.ring_ways
                            cruise.crossings = marks.crossings
                            cruise.tight_turns = marks.tight_turns
                            cruise.prefabs = marks.prefabs
                    before = progress
                    progress = where_on(path, x, z, progress).s
                    if cruise is not None:
                        cruise.rebase(progress - before)
                    tracker = None
                    game_scale = game_km(rerouted) * 1000.0 / max(path.length, 1.0)
                    keep_route(output, f"taken-{now - started:07.1f}", rerouted, None)
                    print(f"\n[GPS: another way, {game_km(rerouted):.1f} km of it planned]")
                if gps.error:
                    print(f"\n[GPS: {gps.error}]", file=sys.stderr)
                    gps.error = None

            clock.mark("the GPS")
            wide_shift = turn_line.shift(
                spans=wide_turns,
                progress=progress,
                mouth_clear=cruise.mouth_clear if cruise is not None else None,
                dt=dt,
            )
            if turn_line.chosen is not None and turn_line.chosen != turn_line_said:
                turn_line_said = turn_line.chosen
                print(
                    "\n[a tight turn: round wide, its mouth clear]"
                    if turn_line.chosen == "wide"
                    else "\n[a tight turn: kept in, its mouth not seen clear]"
                )
            elif turn_line.chosen is None:
                turn_line_said = None
            lateral_command = lateral.step(
                x=x,
                z=z,
                heading=convert.course_rad(block),
                speed=max(speed, 1.0),
                path=path,
                progress=progress,
                aim_shift=lane_guard.shift,
                aim_shift_at=wide_shift,
            )
            if (
                abs(lateral_command.lateral_error) > PLACE_LOST_M
                and abs(last_error) <= MAX_LATERAL_ERROR_M
            ):
                lost_at = lateral_command.lateral_error
                found = path.nearest(x, z)
                with contextlib.suppress(OSError):
                    np.save(output / f"path-lost-{now - started:07.1f}.npy", path.points)
                print(
                    f"\n[the place on the path was lost, {lost_at:+.0f} m off: "
                    f"found again {math.hypot(found.x - x, found.z - z):.1f} m off]"
                )
                lateral_command = lateral.step(
                    x=x,
                    z=z,
                    heading=convert.course_rad(block),
                    speed=max(speed, 1.0),
                    path=path,
                    progress=found.s,
                    aim_shift=lane_guard.shift,
                    aim_shift_at=wide_shift,
                )
            last_error = lateral_command.lateral_error
            progress = lateral_command.progress
            worst_error = max(worst_error, abs(lateral_command.lateral_error))
            if abs(lateral_command.lateral_error) > MAX_LATERAL_ERROR_M:
                if perception is not None:
                    perception.snapshot(f"off-path-{now - started:07.1f}")
                raise LostThePath(f"{lateral_command.lateral_error:+.1f} m from the path")
            standstill.update(speed, dt)
            try:
                impacts.check(now, speed)
                damage.check(block)
            except Collided:
                if perception is not None:
                    perception.snapshot(f"hit-{now - started:07.1f}")
                # The frame that showed the hit, kept: raised before the end
                # of the loop, it was never written, and tools/report.py
                # counted no hit and no damage for a drive stopped by one.
                if snapshot.write_count != last_write_count:
                    last_write_count = snapshot.write_count
                    recorder.write_telemetry(snapshot)
                raise
            # The truck itself at the end, come to rest there
            # (core.control.end_stop): a depot's gate, a port.
            remaining = path.length - lateral_command.progress
            short_of_gate = (
                None if service_stop is None else service_stop - lateral_command.progress
            )
            if short_of_gate is not None:
                remaining = min(remaining, short_of_gate)
            if (
                gps is not None
                and short_of_gate is None
                and replan_asked is not path
                and remaining < REPLAN_LEFT_M
                and float(block["navigation"]["distance"]) / max(game_scale, 1.0)
                > remaining + REPLAN_SHORT_M
                and to_nearest_point(boarding, path.points[-1]) > REPLAN_SHORT_M
            ):
                # Running out short of the depot and of any port -- a lane
                # change given up, the way on not taken: planned afresh from
                # here, not stopped on the road.
                gps.replan()
                replan_asked = path
            if place is not None:
                to_go = place.to_go(float(block["navigation"]["distance"]), now, game_scale)
                if to_go is not None:
                    remaining = min(remaining, to_go)
            if reached(remaining, speed):
                if short_of_gate is not None and short_of_gate <= remaining + 1.0:
                    print("\n\nat the turn onto the depot's own road: the rest is the driver's")
                    ending = ("service_road", "")
                else:
                    print("\n\nreached the end of the route")
                break

            # The ceiling is a wish; the corner in front decides.
            # The ceiling asked for, the road's own limit, and the corner ahead:
            # whichever is lowest. The road's limit only ever lowers it.
            road_limit = None if args.above_road_limit else convert.road_speed_limit(block)
            if road_limit is None and not args.above_road_limit:
                road_limit = NO_LIMIT_KMH / MS_TO_KMH
            ceiling = target if road_limit is None else min(target, road_limit)
            planned_speed = min(ceiling, profile.speed_at(lateral_command.progress))
            urgent = None
            clock.mark("the path")
            report = perception.latest() if perception is not None else None
            if cruise is not None:
                # Following is one more cap on the speed -- or, when the gap
                # closes too fast for comfort, a deceleration commanded directly.
                try:
                    following = cruise.step(
                        report,
                        now,
                        path=path,
                        progress=lateral_command.progress,
                        x=x,
                        z=z,
                        course=convert.course_rad(block),
                        speed=speed,
                    )
                    out_of_cab_said = False
                except OutOfCab:
                    # A look about, or the camera nudged: slowed for, gently,
                    # the lane kept by the map, and gone on from once it is
                    # back. Stopped at once, the truck was braked at 6 m/s^2
                    # from 59 km/h on a motorway's slip road for a moment's
                    # look aside (18 November). Out for good, the drive ends.
                    out_for = getattr(cruise, "out_of_cab_s", 0.0)
                    if not out_of_cab_said:
                        out_of_cab_said = True
                        # What the cab check saw, kept: near Rostock it
                        # stopped a drive with the camera in the cab, the
                        # telemetry's head unmoved, and no frame of it kept.
                        perception.snapshot(f"out-of-cab-{now - started:07.1f}")
                        print(
                            "\n[the game's camera is not in the cab: slowing; back to the cab"
                            f" view within {OUT_OF_CAB_GIVE_UP_S:.0f} s, or the drive stops]"
                        )
                    if out_for >= OUT_OF_CAB_GIVE_UP_S:
                        raise
                    following = None
                except PerceptionLost:
                    if not late_but_not_lost(report, now):
                        raise
                    following = None
                if following is None:
                    # A view a moment late is not the road gone: slow a little
                    # and carry on when it comes back.
                    planned_speed = min(planned_speed, max(0.0, speed - LATE_VIEW_SLOWING_MPS))
                elif following.decision.target_speed is not None:
                    planned_speed = min(planned_speed, following.decision.target_speed)
                if following is not None:
                    urgent = following.decision.urgent_deceleration
                    lead = following.lead
                    acc_log.append(
                        (
                            now - started,
                            speed,
                            following.lead_id or 0,
                            lead.distance_m if lead else np.nan,
                            lead.closing_mps if lead else np.nan,
                            planned_speed,
                            urgent or 0.0,
                            following.report_age_s,
                        )
                    )
                    said = following.decision.reason
                    if not acc_reasons or acc_reasons[-1][1] != said:
                        acc_reasons.append((round(now - started, 3), said))
            # A red light is one more cap, and a light seen late one more
            # urgent deceleration; the larger of the two urgent ones wins.
            clock.mark("following")
            if report is not None and report.sequence != light_sequence:
                light_sequence = report.sequence
                light_filter.update(report.lamps, report.grabbed_at)
                if cruise is not None:
                    if light_filter.state is not LightState.GREEN:
                        cruise.green_since = None
                    elif cruise.green_since is None:
                        cruise.green_since = report.grabbed_at
                next_light = lights.next_line(lateral_command.progress)
                if (
                    next_light is not None
                    and camera is not None
                    and light_filter.state in (LightState.RED, LightState.AMBER)
                    and next_light.s - lateral_command.progress <= LIGHT_STOP_FROM_M
                ):
                    points = [
                        (lamp.x, lamp.y)
                        for lamp in report.lamps
                        if lamp.colour is light_filter.state
                    ]
                    if points:
                        to_line = next_light.s - lateral_command.progress
                        before = STOP_SHORT_M
                        if light_stop_at != next_light.s:
                            lamp_sightings = []
                        else:
                            before = light_stop
                        lamp_sightings = follow_lamp(lamp_sightings, to_line, points)
                        light_stop_at = next_light.s
                        wanted = stop_short_to_see(
                            to_line, points, camera, eye_ahead_m(block), light_view
                        )
                        measured = stop_short_from_sightings(
                            lamp_sightings, (camera.centre_col, camera.horizon_row), light_view
                        )
                        if measured is not None:
                            wanted = max(wanted, measured)
                        light_stop = moved_stop(before, wanted, to_line, speed)
            next_light = lights.next_line(lateral_command.progress)
            light = lights.step(
                progress=lateral_command.progress,
                speed=speed,
                state=light_filter.state,
                lamp=report.light if report is not None else None,
                minute_of_day=float(block["common"]["game_time"]) % 1440,
                now=now,
                traffic_going=cruise is not None and cruise.traffic_going,
                stop_short=(
                    light_stop
                    if next_light is not None and next_light.s == light_stop_at
                    else STOP_SHORT_M
                ),
            )
            waiting_at_light = light.target_speed == 0.0 and abs(speed) < 0.3
            if not waiting_at_light:
                light_wait_since = None
            elif light_wait_since is None:
                light_wait_since = now
            elif now - light_wait_since >= LIGHT_WAIT_SNAPSHOT_S and perception is not None:
                # Long enough for a red to have gone green: what is it seeing?
                perception.snapshot(f"light-wait-{now - started:07.1f}")
                light_wait_since = math.inf
            standing_in_a_junction = (
                cruise is not None
                and abs(speed) < 0.3
                and cruise.in_a_junction(lateral_command.progress)
            )
            if not standing_in_a_junction:
                junction_stop_since = None
            elif junction_stop_since is None:
                junction_stop_since = now
            elif now - junction_stop_since >= JUNCTION_STOP_SNAPSHOT_S and perception is not None:
                perception.snapshot(f"junction-stop-{now - started:07.1f}")
                junction_stop_since = math.inf
            if perception is not None:
                # The pole beside the line can stand a little before it.
                perception.set_light_distance(
                    None if light.distance_m is None else light.distance_m - LIGHT_POLE_BEFORE_M
                )
                perception.set_daylight(is_daylight(float(block["common"]["game_time"])))
            if light.target_speed is not None:
                planned_speed = min(planned_speed, light.target_speed)
            planned_speed = min(
                planned_speed,
                barrier_speed(barriers, lateral_command.progress),
                speed_to_stop_in(remaining),
            )
            lane_reason = ""
            if gate is not None and report is not None:
                # Into another lane only when it is clear (core.control.lane_change).
                if perception is not None:
                    perception.watch_mirror(gate.watching(lateral_command.progress))
                crossing_over = gate.step(
                    progress=lateral_command.progress,
                    speed=speed,
                    pose=(x, z, convert.course_rad(block)),
                    tracks=report.tracks,
                    unseen=report.unseen,
                    now=now,
                    mirror=getattr(report, "mirror", None),
                    mirror_far=getattr(report, "mirror_far", None),
                    night=not is_daylight(float(block["common"]["game_time"])),
                )
                if crossing_over.target_speed is not None:
                    planned_speed = min(planned_speed, crossing_over.target_speed)
                if crossing_over.reason:
                    lane_reason = f"{crossing_over.reason}  "
                if crossing_over.path is not None:
                    path = crossing_over.path
                    indicators = Indicators(route_signals(signalled, path, gate.signalled()))
                if crossing_over.missed:
                    # On in the lane the truck is in, off the route: its
                    # bends, not the route's, set the speed until the game
                    # plans anew.
                    print(f"\n[{crossing_over.reason}]")
                    if perception is not None:
                        perception.snapshot(f"missed-change-{now - started:07.1f}")
                    if cruise is not None:
                        cruise.forget_lead()
                    indicators = Indicators(route_signals(signalled, path, gate.signalled()))
                    held_from = max(0.0, lateral_command.progress - 50.0)
                    held_profile = (
                        held_from,
                        SpeedProfile(
                            ReferencePath(
                                path.points[np.arange(len(path)) * path.spacing >= held_from],
                                spacing=args.spacing,
                            ),
                            max_lateral_acceleration=args.lateral,
                            max_speed=target,
                        ),
                    )
            if held_profile is not None:
                planned_speed = min(
                    planned_speed,
                    held_profile[1].speed_at(lateral_command.progress - held_profile[0]),
                )
            turning = turning_cap(truck_geometry, float(truck["effective_steering"]))
            if turning is not None:
                # Not faster while the wheel is turned; never braked for.
                planned_speed = min(planned_speed, max(turning, speed))
            if light.urgent_deceleration is not None:
                urgent = max(urgent or 0.0, light.urgent_deceleration)
            if perception is not None:
                hard = (urgent or 0.0) >= SNAPSHOT_URGENT_MPS2
                if hard and not was_urgent:
                    perception.snapshot(f"brake-{now - started:07.1f}")
                was_urgent = hard
                if report is not None and report.sequence != lane_sequence:
                    lane_sequence = report.sequence
                    ahead = lane_ahead(path, progress, x, z, convert.course_rad(block), camera)
                    lane_log.append(
                        (
                            now - started,
                            report.lane_left_m,
                            report.lane_right_m,
                            ahead.right_at(LANE_CHECK_M) if ahead is not None else np.nan,
                        )
                    )
                    on_taught = any(
                        s0 - TAUGHT_GUARD_OFF_M <= lateral_command.progress <= s1
                        for s0, s1 in taught_spans
                    )
                    if on_taught:
                        # A driver's line is the way, whatever the paint: in
                        # the customs north of Basel the guard kept the truck
                        # 1.5 m off it, and it ran into something.
                        lane_guard.step(now, progress=lateral_command.progress)
                    elif ahead is not None:
                        lane_guard.step(
                            now,
                            report.lane_left_m,
                            report.lane_right_m,
                            ahead.right_at(LANE_CHECK_M),
                            ahead.right_at(GOING_AT_M),
                            progress=lateral_command.progress,
                            speed=speed,
                        )
                    else:
                        lane_guard.step(now, progress=lateral_command.progress)
                else:
                    lane_guard.step(now, progress=lateral_command.progress)
                if abs(lane_guard.shift) >= 0.3:
                    lane_reason += f"kept in the painted lane, {lane_guard.shift:+.1f} m  "
                    if not guard_snapped:
                        perception.snapshot(f"lane-guard-{now - started:07.1f}")
                        guard_snapped = True
                elif lane_guard.shift == 0.0:
                    guard_snapped = False
            if light.distance_m is not None:
                light_log.append(
                    (
                        now - started,
                        light.distance_m,
                        list(LightState).index(light_filter.state),
                        list(LightState).index(report.light) if report is not None else 3,
                        report.lamp_area if report is not None else 0,
                        np.nan if light.target_speed is None else light.target_speed,
                        light.urgent_deceleration or 0.0,
                    )
                )
            # A stop, once made, lasts until there is a real reason to go.
            stopping = stop_latch.stopping(
                target=planned_speed, speed=speed, stop_target=STOP_TARGET, hold_below=HOLD_BELOW
            )
            if stopping:
                planned_speed = 0.0
            if urgent is not None:
                longitudinal = speed_controller.urgent(urgent, max(dt, 1e-4))
            else:
                longitudinal = speed_controller.step(
                    target_speed=planned_speed,
                    speed=speed,
                    grade=convert.grade_acceleration(block),
                    dt=max(dt, 1e-4),
                    coast_above=profile.coasting_at(lateral_command.progress, speed),
                )
            # A stop that lasts is held on the parking brake, and it is let go
            # before moving off; the service brake alone did not hold a
            # stopped truck on a slope (core.control.parking).
            parked = bool(int(truck["parking_brake"]))
            want_parked = parking.should_hold(
                speed=speed, stopping=stopping, now=now, grade=convert.grade_acceleration(block)
            ) or (parked and stopping)
            if parked and not stopping:
                # Moving off: pull first, and let go of the parking brake
                # only once the throttle has been on for a moment.
                moving_off_since = moving_off_since if moving_off_since is not None else now
                want_parked = now - moving_off_since < MOVE_OFF_S
            else:
                moving_off_since = None
            press = parking.press(want_set=want_parked, is_set=parked, now=now)
            if parked and stopping:
                # Held by the parking brake: no pedals.
                longitudinal = LongitudinalCommand(0.0, 0.0, "parking brake on")
            elif not parked:
                # Never the brake pressed at a stop with it up: that is
                # reverse, for the game's gearbox.
                if sent_brake is None:
                    sent_brake = float(truck["input_brake"])
                longitudinal = held_still(longitudinal, speed, brake_down=sent_brake > 0.0)
            steer = steering.command(
                target=lateral_command.steering,
                measured=float(truck["effective_steering"]),
                speed=None if args.old_steering else speed,
            )
            if args.excite:
                steer = float(np.clip(steer + doublet(now - started, args.excite), -1.0, 1.0))
            clock.mark("lights, lanes and stops")
            commander.send(
                ControlCommand(
                    steer=steer,
                    throttle=longitudinal.throttle,
                    brake=longitudinal.brake,
                    buttons=(frozenset({Button.HANDBRAKE}) if press else frozenset())
                    | hazards_off.buttons(
                        on=bool(int(truck["hazard_warning"])), since_start=now - started
                    )
                    | (
                        indicators.buttons(
                            progress=lateral_command.progress,
                            left_on=bool(int(truck["indicator_left"])),
                            right_on=bool(int(truck["indicator_right"])),
                            now=now,
                        )
                        if indicators is not None
                        else frozenset()
                    )
                    | retarder_lever.buttons(
                        wanted=longitudinal.retarder, level=retarder_level, now=now
                    ),
                )
            )
            if not engaged_said:
                engaged_said = True
                if standing is not None:
                    print(
                        f"\n[engaged {now - standing.went_at:.1f} s after the word to go: "
                        f"{standing.took}]"
                    )
            sent_brake = longitudinal.brake
            recent_pedals.add(now, longitudinal.throttle, longitudinal.brake)

            # The tractor's error is not the vehicle's error: the trailer
            # cuts inside every corner, and on this route it reached 2.2 m
            # where the tractor held 0.9.
            trailer = block["trailers"][0]
            if int(trailer["connected"]):
                if tracker is None:
                    geometry = TrailerGeometry.from_block(block)
                    tracker = TrailerTracker(
                        path, geometry.length if geometry is not None else TRAILER_LOOKBACK
                    )
                trailer_position = trailer["placement"]["position"]
                trailer_error = tracker.update(
                    float(trailer_position["x"]),
                    float(trailer_position["z"]),
                    lateral_command.progress,
                )
                if trailer_error is not None:
                    worst_trailer = max(worst_trailer, abs(trailer_error))

            clock.mark("the command")
            if snapshot.write_count != last_write_count:
                last_write_count = snapshot.write_count
                recorder.write_telemetry(snapshot)
            if perception is not None and (args.keep_frames > 0.0 or args.tap is not None):
                # Views of the road for training on (the lane model): while
                # moving only, on the drive's own timeline. For the ROS view,
                # standing too, and only to it.
                keeping = args.keep_frames if abs(speed) >= KEEP_FROM_MPS else 0.0
                if args.tap is not None:
                    keeping = min(keeping or ROS_VIEW_EVERY_S, ROS_VIEW_EVERY_S)
                if keeping != kept_every:
                    perception.keep_frames(keeping)
                    kept_every = keeping
                for view in perception.kept():
                    if (
                        args.keep_frames > 0.0
                        and abs(speed) >= KEEP_FROM_MPS
                        and view.timestamp - recorded_at >= args.keep_frames * 0.99
                    ):
                        recorded_at = view.timestamp
                        recorder.write_encoded_frame(
                            view.jpeg, view.timestamp, view.sequence, view.width, view.height
                        )
                    if args.tap is not None:
                        args.tap.image(now, view.jpeg, camera, view.width, view.height)
            if args.tap is not None:
                tell_ros(args.tap, now, block, path, driven, report, camera, following)
            if perception is not None:
                learned = perception.cab_learned()
                if learned is not None:
                    truck_profile = truck_profile.with_cab(learned)
                    save_profile(truck_profile)
                    print(f"\n[the cab's dashboard of {truck_profile.truck}: learned]")
                top = perception.view_learned()
                if top is not None:
                    truck_profile = truck_profile.with_view_top(top)
                    save_profile(truck_profile)
                    light_view = view_bounds(truck_profile)
                    print(
                        f"\n[the windscreen of {truck_profile.truck}: from row "
                        f"{light_view.top:.0f}, learned]"
                    )
                for slow in perception.slow_turns():
                    print(f"\n[a slow turn of perception, {slow}]", file=sys.stderr)

            guard.check(
                speed=speed,
                throttle=longitudinal.throttle,
                brake=longitudinal.brake,
                position=(x, z),
                expect_forward=True,
                applied=pedals_taken(truck),
                frame=snapshot.write_count,
            )

            if now - last_print > 0.4:
                last_print = now
                print(
                    f"\r{convert.speed_kmh(block):5.1f}/{planned_speed * MS_TO_KMH:4.1f} km/h  "
                    f"lat {lateral_command.lateral_error:+6.2f} m  "
                    f"hdg {np.degrees(lateral_command.heading_error):+6.2f} deg  "
                    f"steer {lateral_command.steering:+.3f}  "
                    f"{lateral_command.progress * game_scale / 1000.0:5.1f}"
                    f"/{path.length * game_scale / 1000.0:.1f} km  "
                    + (
                        f"lead {acc_log[-1][3]:5.1f} m  "
                        if cruise is not None and acc_log and acc_log[-1][2]
                        else ""
                    )
                    + (
                        f"light {light.distance_m:4.0f} m {light_filter.state.value}  "
                        if light.distance_m is not None
                        else ""
                    )
                    + lane_reason
                    + f"{longitudinal.reason}",
                    end="",
                    flush=True,
                )
            clock.mark("recording and checks")
            time.sleep(LOOP_INTERVAL)
    except UnsafeCondition as exc:
        print(f"\n\nSTOP: {exc}", file=sys.stderr)
        outcome = 1
        ending = ("unsafe", str(exc))
        if isinstance(exc, LostThePath):
            keep_lane = None  # the path is what was lost
    except KeyboardInterrupt:
        print("\n\ninterrupted", file=sys.stderr)
        ending = ("interrupted", "")
    except (PalError, TelemetryError) as exc:
        print(f"\n\nlost the game: {exc}", file=sys.stderr)
        outcome = 1
        ending = ("lost_game", str(exc))
    finally:
        if ending[0] != "disengaged":
            try:
                _brake_to_rest(commander, reader, guard, keep_lane=keep_lane)
                _hand_over(commander)
            except Exception:
                # Never in the way of letting go of the truck below; but said.
                print("\nstopping failed:", file=sys.stderr)
                traceback.print_exc()
        with contextlib.suppress(Exception):
            # The driver is not handed a truck with the retarder on.
            _retarder_off(commander, reader)
        if brake_estimate.ratio is not None:
            print(
                f"brake: {brake_estimate.ratio:.2f} of the calibrated truck's, "
                f"against {speed_controller.actuators.mass_ratio:.2f} for the load, "
                f"seen {brake_estimate.seen_s:.0f} s"
            )
        if retarder_estimate.seen_s > 0.0:
            print(
                f"retarder: {retarder_estimate.per_step:.2f} m/s2 a step, "
                f"seen at work {retarder_estimate.seen_s:.0f} s, "
                f"{retarder_lever.presses} presses of the lever"
            )
        with contextlib.suppress(Exception):
            commander.disengage()
        commander.close()
        recorder.close()
        if args.tap is not None:
            args.tap.status(time.perf_counter(), f"stopped: {ending[0]}", ending[1])
            time.sleep(0.1)
            args.tap.feed.close()
        reader.close()
        if gps is not None:
            gps.stop()
        if perception is not None:
            perception.stop()
            if acc_log:
                np.save(output / "acc.npy", np.array(acc_log, dtype=float))
            if acc_reasons:
                (output / "acc_reasons.json").write_text(
                    json.dumps(acc_reasons, indent=0), encoding="utf-8"
                )
        if light_log:
            np.save(output / "lights.npy", np.array(light_log, dtype=float))
        if lane_log:
            np.save(output / "lanes.npy", np.array(lane_log, dtype=float))
        np.save(output / "health.npy", health.stop())

    print(
        f"\n{recorder.telemetry_count} frames, {recorder.duration:.1f}s, "
        f"{guard.distance_travelled * game_scale / 1000.0:.1f} km"
    )
    print(f"worst lateral error  tractor {worst_error:.2f} m", end="")
    if worst_trailer > 0.0:
        geometry = TrailerGeometry.from_block(reader_block) if reader_block is not None else None
        suffix = f" (length {geometry.length:.1f} m)" if geometry else ""
        print(f", trailer {worst_trailer:.2f} m{suffix}")
    else:
        print()
    print(f"recording: {output}")
    if args.outcome is not None:
        write_outcome(args.outcome, *ending)
    stop_keeping(messages_file)
    return outcome


def stop_keeping(messages_file) -> None:
    """The console's own streams back, and the drive's messages closed."""
    for name in ("stdout", "stderr"):
        log = getattr(sys, name)
        if isinstance(log, MessageLog):
            setattr(sys, name, log.stream)
    if messages_file is not None:
        messages_file.close()


def write_outcome(path: Path, reason: str, detail: str = "") -> None:
    """Why a drive ended, for tools/mission.py: `end_of_route`, `service_road`
    (stopped before the turn onto a depot's own road), `disengaged`,
    `unsafe`, `interrupted`, `lost_game`, `duration`; or `not_started`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reason": reason, "detail": detail}), encoding="utf-8")


def route_cache(route_file: Path) -> Path:
    """The map cache a route from plan_route.py was planned on."""
    with np.load(route_file, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    return Path(metadata.get("cache") or REPO_ROOT / "data" / "map" / "installed")


def saved_route(route_file: Path) -> Route | None:
    """The lanes of a route from plan_route.py, on the map it was planned on."""
    if not is_planned(route_file):
        return None
    with np.load(route_file, allow_pickle=False) as data:
        if "lanes" not in data:
            return None
        metadata = json.loads(str(data["metadata"]))
        lanes = tuple(int(lane) for lane in data["lanes"])
        steps = tuple(str(step) for step in data["steps"])
    cache = Path(metadata.get("cache") or REPO_ROOT / "data" / "map" / "installed")
    try:
        return Route(RoadNetwork.open(cache), lanes, steps)
    except MapCacheError:
        return None


def customs_ends(route: Route, *, taught: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Where each of a route's customs stretches begins and ends, world (x, z):
    those without a taught crossing; or, with `taught`, where each taught
    line is -- through customs or any other place -- driven as it was."""
    ends = []
    if taught:
        for _, _, points in route.taught_crossings():
            ends.append((points[0, [0, 2]].astype(float), points[-1, [0, 2]].astype(float)))
        return ends
    shown = {(first, last) for first, last, _ in route.taught_crossings()}
    for first, last in route.quiet_stretches():
        if (first, last) in shown:
            continue
        start = route.network.lane_points(route.lanes[first])[0]
        end = route.network.lane_points(route.lanes[last])[-1]
        ends.append((np.array([start[0], start[2]]), np.array([end[0], end[2]])))
    return ends


def game_km(route: Route) -> float:
    """How long `route` is in the game's kilometres, as its navigation and
    fuel range count them: the driver reads those, not the world's metres,
    which are some 19 times fewer out of town."""
    lanes, steps = route.lanes, route.steps
    return (
        sum(
            nav_length(route.network, lanes[k])
            for k in range(len(lanes))
            if k == 0 or steps[k - 1] != CHANGE
        )
        / 1000.0
    )


def marks_of(route: Route, path: ReferencePath) -> list[float]:
    """The toll gates of `route`, as distances along `path`."""
    return [line.s for line in lines_on_path(route.barriers(), path)]


@dataclass(frozen=True)
class PathMarks:
    """Where along a route's path the loop has something to do: its traffic
    lights, toll gates, lane changes, and the junctions and
    roundabouts given way in (core.control.acc).

    Made with the path in the GPS worker. Made in the loop for a new route of
    30 km they took 1.3 s, the junctions' spans alone a second: in a
    roundabout in Torino the loop stood still, the plugin stopped the truck
    for a stack gone (plugin/src/safety_stop.h), and the turn was not taken.
    """

    lights: list[LightLine]
    barriers: list[float]
    changes: list[LaneChange]
    junctions: list[tuple[float, float]]
    turns_across: list[tuple[float, float]]
    rings: list[tuple[float, float, int]]
    taught_spans: list[tuple[float, float]] = dataclass_field(default_factory=list)
    """Where the path is a driver's taught line (`taught_spans_on`)."""

    ring_ways: list[tuple[float, float]] = dataclass_field(default_factory=list)
    """Where the path goes through roundabouts (`ring_way_spans`)."""

    prefabs: list[tuple[float, float]] = dataclass_field(default_factory=list)
    """Where the path goes through junctions (`prefab_spans`)."""

    crossings: list[tuple[float, float]] = dataclass_field(default_factory=list)
    """Where the path goes through junctions crossed on the level (`crossing_spans`)."""

    tight_turns: list = dataclass_field(default_factory=list)
    """The tight turns along the path (`tight_turn_spans`)."""

    wide_turns: list = dataclass_field(default_factory=list)
    """The tight turns along the path to be gone round wide while their
    mouth is clear (core.planning.tight_turns.wide_spans)."""

    service_stop: float | None = None
    """Where along the path the truck stops short of its depot's gate, before
    the turn onto the depot's own road (core.planning.route.service_road_stop);
    None for a route that ends at the gate, or not at a depot."""


def path_marks(route: Route, path: ReferencePath, depot=None, wides=None) -> PathMarks:
    """What `route` has along `path`, as distances along it; `depot`, the
    job's, if the route may end at its gate; `wides`, the wide lines of its
    tight turns by their stretches' keys (`TurnShaping.wide`)."""
    barriers = marks_of(route, path)
    return PathMarks(
        lights=lines_on_path(route.stop_lines(), path),
        barriers=barriers,
        changes=lane_changes(route, path),
        junctions=junction_spans(route, path),
        turns_across=turn_across_spans(route, path),
        rings=ring_entry_spans(route, path),
        taught_spans=taught_spans_on(route, path),
        ring_ways=ring_way_spans(route, path),
        prefabs=prefab_spans(route, path),
        crossings=crossing_spans(route, path),
        tight_turns=tight_turn_spans(route, path),
        wide_turns=wide_spans(route, path, wides) if wides else [],
        service_stop=service_stop_on(route, path, depot),
    )


def service_stop_on(route: Route, path: ReferencePath, depot) -> float | None:
    """Where along `path` to stop before the turn onto the depot's own road,
    short of its gate; only for a route that ends at one of `depot`'s gates,
    not one that ends a few kilometres on, wherever that is."""
    if depot is None or not route.lanes:
        return None
    try:
        if int(route.lanes[-1]) not in depot_goals(route.network, depot):
            return None
    except RouteError:
        return None
    at = service_road_stop(route.network, route.lanes)
    return None if at is None else float(path.nearest(at[0], at[1]).s)


#: The lane guard leaves the path alone from this far before a taught line.
TAUGHT_GUARD_OFF_M = 20.0


def taught_spans_on(route: Route, path: ReferencePath) -> list[tuple[float, float]]:
    """Where along `path` each of `route`'s taught crossings runs, (start,
    end) distances: the driver's line, which the lane guard keeps off."""
    spans = []
    for _, _, points in route.taught_crossings():
        length = float(np.hypot(*np.diff(points[:, [0, 2]], axis=0).T).sum())
        s0 = path.nearest(float(points[0][0]), float(points[0][2])).s
        s1 = path.nearest(
            float(points[-1][0]), float(points[-1][2]), near=s0 + length, window=length + 50.0
        ).s
        spans.append((s0, max(s1, s0)))
    return spans


def barrier_speed(barriers: list[float], progress: float, deceleration: float = 1.0) -> float:
    """The speed cap for the toll gates around `progress`, m/s; inf if none."""
    cap = math.inf
    slow = BARRIER_SPEED_KMH / MS_TO_KMH
    for s in barriers:
        ahead = s - progress
        if -BARRIER_AFTER_M <= ahead <= 0.0:
            cap = min(cap, slow)
        elif 0.0 < ahead <= BARRIER_APPROACH_M:
            cap = min(cap, math.sqrt(slow * slow + 2.0 * deceleration * ahead))
    return cap


class TurnShaping:
    """The tight turns of the routes the GPS worker sends, shaped
    (core.planning.tight_turns) in a thread of its own: about 4 s a turn,
    which the worker cannot stop for while the red line is to be watched.

    Each route's turns are shaped in order along it, from the truck on, and
    kept by their lanes for the next route through them; `route_path` drives
    those shaped so far.
    """

    SOON_M = 2000.0
    """A turn shaped this near the truck is driven at once; the others wait
    for the rest of the route's, the path not changed under the truck for
    each."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network
        self.lines: dict[tuple[int, ...], np.ndarray | None] = {}
        self.wide: dict[tuple[int, ...], WideTurn] = {}
        """Lines over the mouth of the road turned into, for turns whose
        shaped line keeps off it, to be driven while it is clear."""
        self._jobs: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._route: Route | None = None
        self._queued = 0
        self._waiting = False
        threading.Thread(target=self._run, name="turns", daemon=True).start()

    def shape(self, route: Route, points: np.ndarray, at: int) -> None:
        """Shape the turns of `route` not yet shaped, beyond lane `at` where
        the truck is, along `points`, its line driven; those of an earlier
        route are dropped."""
        self._route = route
        self._queued = 0
        self._waiting = False
        for stretch in stretches(route):
            if stretch.begins > at and stretch.key not in self.lines:
                self._jobs.put((route, stretch, points))
                self._queued += 1

    def newly_shaped(self, at: int) -> bool:
        """Whether a turn of the route has been shaped since last asked that
        is to be driven now: one soon after lane `at` where the truck is, or
        any once the route's last is done."""
        route, drive = self._route, False
        while True:
            try:
                done_for, stretch, shaped = self._done.get_nowait()
            except queue.Empty:
                break
            if done_for is not route:
                continue
            self._queued -= 1
            self._waiting = self._waiting or shaped
            if shaped and self._ahead(route, at, stretch) <= self.SOON_M:
                drive = True
        if self._waiting and (drive or self._queued <= 0):
            self._waiting = False
            return True
        return False

    def _ahead(self, route: Route, at: int, stretch: Stretch) -> float:
        lengths = self.network.lane_length
        return float(sum(lengths[lane] for lane in route.lanes[max(at, 0) : stretch.begins]))

    def _run(self) -> None:
        while True:
            route, stretch, points = self._jobs.get()
            if route is not self._route:
                continue
            line = self.lines.get(stretch.key)
            if stretch.key not in self.lines:
                first, last = between(points, stretch.approach[0], stretch.exit_[-1])
                try:
                    line = shaped_turn(
                        self.network,
                        stretch.approach,
                        stretch.exit_,
                        points[first : last + 1],
                        stretch.junction,
                    )
                except Exception:  # a turn not shaped is driven as it was
                    traceback.print_exc()
                    line = None
                if line is not None:
                    try:
                        wide = wide_turn(
                            self.network,
                            stretch.approach,
                            stretch.exit_,
                            points[first : last + 1],
                            stretch.junction,
                        )
                    except Exception:  # kept to the line kept in, as before
                        traceback.print_exc()
                        wide = None
                    if wide is not None:
                        self.wide[stretch.key] = wide
                self.lines[stretch.key] = line
            self._done.put((route, stretch, line is not None))


FIRST_PART_M = 3_000.0
"""A new route goes to the loop twice: first this far of it, then the whole.
The whole of a 30 km route's path took 6 s to make on Windows; out of Bern
the red line turned the route 125 m before its junction, and by the time the
new route came the junction was passed. 3 km is under half a second, and two
minutes of driving at 90 km/h for the rest to come in."""

MASK_BYTES = -(-(GPS_REGION[2] - GPS_REGION[0]) * (GPS_REGION[3] - GPS_REGION[1]) // 8)
"""A packed minimap mask (core.planning.gps.unpack_mask)."""


class NewestView:
    """The newest minimap offered to the GPS worker, in shared memory: each
    offer replaces the last. Queued instead, out of Bern the worker was on
    minimaps seconds old when the red line left the route 150 m ahead --
    offers waited behind a slow look, and on Windows the pipe holds less than
    one -- and a look from where the truck was is the only one of use."""

    def __init__(self) -> None:
        self._lock = mp.Lock()
        self._number = mp.Value("Q", 0, lock=False)
        # x, z, course, nav distance (NaN: none), and the masks' lengths
        self._numbers = mp.Array("d", 6, lock=False)
        self._masks = mp.Array("B", 2 * MASK_BYTES, lock=False)
        self.fresh = mp.Event()
        """Set at each offer, and by whatever else wants the worker woken."""

    def put(
        self, pose: tuple[float, float, float], route: bytes, roads: bytes, nav: float | None
    ) -> None:
        route, roads = route[:MASK_BYTES], roads[:MASK_BYTES]
        with self._lock:
            self._numbers[:] = [*pose, math.nan if nav is None else nav, len(route), len(roads)]
            self._masks[: len(route)] = route
            self._masks[MASK_BYTES : MASK_BYTES + len(roads)] = roads
            self._number.value += 1
        self.fresh.set()

    def newer_than(self, number: int):
        """(number, pose, route, roads, nav distance) if offered since `number`; else None."""
        with self._lock:
            if self._number.value == number:
                return None
            x, z, course, nav, route_length, roads_length = self._numbers[:]
            masks = bytes(self._masks)
            number = self._number.value
        route = masks[: int(route_length)]
        roads = masks[MASK_BYTES : MASK_BYTES + int(roads_length)]
        return number, (x, z, course), route, roads, None if math.isnan(nav) else nav


class GpsWorker:
    """Checks the minimap's route in a process of its own (core.planning.gps).

    A new route to a depot 80 km off takes the planner about 4 s, and its
    speed profile another second. In a thread of the control loop's process
    they held the loop up (the GIL, and the profile built in the loop
    itself): the newest view of the road was 1.3 s old when the loop came
    back, and the truck stopped. So both run in a child process, on the
    newest minimap offered, at most twice a second; a new route, with its
    path and profile ready to drive, waits in `take` for the loop to adopt.
    """

    INTERVAL_S = 0.5

    def __init__(
        self,
        router: GpsRouter,
        *,
        spacing: float,
        lateral: float,
        max_speed: float,
        whole: bool = False,
    ) -> None:
        """`whole`: the loop began on the route's first part only
        (`FIRST_PART_M`); the worker makes the whole of it and sends it at once."""
        self.router = router
        self.error: str | None = None
        self._notes: list[str] = []
        self._offers: mp.Queue = mp.Queue()
        """What the worker is told besides the minimap: small, in order."""
        self._view = NewestView()
        self._results: mp.Queue = mp.Queue()
        self._received: queue.Queue = queue.Queue()
        """What the worker sent, read off its pipe by `_receive`."""
        self._receiving = threading.Event()
        self._offered_at: float | None = None
        # The worker numbers the routes it sends; 0 is the one it began on.
        self._taken = 0
        self._driven = 0
        self._refused = 0
        self._process = mp.Process(
            target=_gps_main,
            args=(
                router,
                self._offers,
                self._view,
                self._results,
                spacing,
                lateral,
                max_speed,
                whole,
            ),
            name="gps",
            daemon=True,
        )
        self._process.start()
        # The pipe is read on a thread of its own, never in the loop: a whole
        # route of 42 km, 1.5 MB, comes through the Windows pipe a little at a
        # time while the worker plans on, and the loop, reading it, stood
        # still 0.76 s waiting for the rest.
        self._receiving.set()
        self._receiver = threading.Thread(target=self._receive, name="gps-reader", daemon=True)
        self._receiver.start()

    def _receive(self) -> None:
        while self._receiving.is_set():
            try:
                self._received.put(self._results.get(timeout=0.1))
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return

    def offer(
        self,
        pose: tuple[float, float, float],
        packed_route: bytes,
        packed_roads: bytes,
        nav_distance: float | None = None,
    ) -> None:
        now = time.perf_counter()
        if self._offered_at is not None and now - self._offered_at < self.INTERVAL_S:
            return
        self._offered_at = now
        self._view.put(pose, packed_route, packed_roads, nav_distance)

    def notes(self) -> list[str]:
        """What the router has had to say since last asked."""
        said, self._notes = self._notes, []
        return said

    def take(self) -> tuple[Route, ReferencePath, SpeedProfile, PathMarks] | None:
        """The newest route the GPS has turned onto, with its path, profile
        and marks: all of it made in the worker, none in the loop."""
        taken = None
        while True:
            try:
                result = self._received.get_nowait()
            except queue.Empty:
                break
            if result[0] == "error":
                self.error = result[1]
                continue
            if result[0] == "note":
                self._notes.append(result[1])
                continue
            _, refused, number, lanes, steps, path, profile, marks = result
            if refused < self._refused:
                continue  # planned on from a route refused, before the worker heard
            taken = (Route(self.router.network, lanes, steps), path, profile, marks)
            self._taken = number
        if taken is not None:
            self.router.route = taken[0]
        return taken

    @property
    def alive(self) -> bool:
        """Whether the worker's process is still running."""
        return self._process.is_alive()

    def replan(self) -> None:
        """Have the worker plan afresh from where the truck is
        (core.planning.game_route.GameRouter.replan_here)."""
        self._tell(("replan",))

    def refuse(self) -> None:
        """The route last taken is not driven: the worker's router goes back
        to the one that is. Out of Calais a route 4.1 m over was refused, and
        the router, planning on from it, and the drive went two ways."""
        self._tell(("driving", self._driven))
        self._taken = self._driven
        self._refused += 1

    def drive(self) -> None:
        """The route last taken is driven."""
        self._driven = self._taken

    def _tell(self, message) -> None:
        self._offers.put(message)
        self._view.fresh.set()

    def stop(self) -> None:
        self._tell(None)
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
        self._receiving.clear()
        self._receiver.join(timeout=1.0)


def _gps_main(
    router: GpsRouter,
    offers: mp.Queue,
    view: NewestView,
    results: mp.Queue,
    spacing: float,
    lateral: float,
    max_speed: float,
    whole: bool = False,
) -> None:
    """The GPS worker's process: each newest minimap in, each new route out;
    and, when the loop does not drive a route sent, back to the one it does.
    A route whose tight turns have been shaped since is sent again."""
    sent = {0: router.route}
    seen = 0  # the number of the newest minimap looked at
    number = 0
    refused = 0
    shaping = TurnShaping(router.network)
    # The route last sent: as driven, and as the router has it.
    current: tuple[Route, Route] | None = None

    def put(route: Route) -> ReferencePath:
        nonlocal number
        path = route_path(route, spacing, shapes=shaping.lines)
        profile = SpeedProfile(path, max_lateral_acceleration=lateral, max_speed=max_speed)
        number += 1
        sent[number] = router.route
        marks = path_marks(route, path, getattr(router, "depot", None), dict(shaping.wide))
        results.put(("route", refused, number, route.lanes, route.steps, path, profile, marks))
        return path

    def send(route: Route, *, first: bool = True) -> None:
        nonlocal current
        new = current is None or current[0] is not route
        if new and first:
            first = route.ahead(getattr(router, "at", 0), FIRST_PART_M)
            if first is not route:
                put(first)  # to drive on with while the rest is made
        path = put(route)
        if new:
            shaping.shape(route, path.points, getattr(router, "at", 0))
        current = (route, router.route)

    parent = mp.parent_process()
    if whole:
        send(router.route, first=False)  # the loop has its first part already
    while True:
        if (
            current is not None
            and router.route is current[1]
            and shaping.newly_shaped(getattr(router, "at", 0))
        ):
            results.put(("note", "a tight turn ahead: into the junction deeper, then round"))
            send(current[0])
        if parent is not None and not parent.is_alive():
            return  # killed, the loop leaves nobody to stop this one
        view.fresh.wait(timeout=GpsWorker.INTERVAL_S)
        view.fresh.clear()
        messages = []
        with contextlib.suppress(queue.Empty):
            while True:
                messages.append(offers.get_nowait())
        if None in messages:
            return
        for message in messages:
            if message[0] == "replan" and hasattr(router, "replan_here"):
                router.replan_here()
            if message[0] == "driving":
                driven = message[1]
                router.route, router.at, router._pending = sent[driven], 0, None
                sent = {k: v for k, v in sent.items() if k >= driven}
                refused += 1
        offered = view.newer_than(seen)
        if offered is None:
            continue
        seen, *offered = offered
        pose, packed_route, packed_roads, nav_distance = offered
        mask = unpack_mask(packed_route)
        if mask is None:
            continue
        extra = {"nav_distance": nav_distance} if isinstance(router, GameRouter) else {}
        try:
            route = router.update(
                pose, mask, unpack_mask(packed_roads), time.perf_counter(), **extra
            )
        except RouteError as exc:
            results.put(("error", str(exc)))
            continue
        finally:
            for note in getattr(router, "notes", ())[:]:
                results.put(("note", note))
            if isinstance(router, GameRouter):
                router.notes.clear()
        if route is not None:
            send(route.to_ferry())  # driven to the ship; the crossing is the game's


def gps_router(
    route_file: Path, block, *, only: bool = False, game: bool = False
) -> GpsRouter | GpsFollower | GameRouter:
    """The router for a planned route, to the depot of the job in `block`; or,
    `only`, the follower of the minimap's red line alone
    (core.planning.gps_follow), starting from the route; or, `game`, the
    game's route planned from the map (core.planning.game_route)."""
    with np.load(route_file, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        lanes = tuple(int(lane) for lane in data["lanes"])
        steps = tuple(str(step) for step in data["steps"])
    cache = Path(metadata.get("cache") or REPO_ROOT / "data" / "map" / "installed")
    network = RoadNetwork.open(cache)
    job = block["job"]
    places = Places.open(cache)
    alternatives = tuple(place_from(a, places) for a in metadata.get("alternatives") or ())
    if metadata.get("destination"):
        # Found from the game's distance (plan_route.py --to-gps): no job says.
        depot = place_from(metadata["destination"], places)
        company, city = depot.company, depot.city
        if game:
            return GameRouter.along(
                network, depot, Route(network, lanes, steps), alternatives=alternatives
            )
    elif not int(job["job_active"]):
        raise ValueError("no job: the GPS has nowhere to go")
    else:
        company, city = convert.job_target(block)
        depot = places.company(company, city)
    if only:
        if depot is not None and depot.company == GPS_COMPANY:
            # A point guessed from the game's distance, not the place
            # marked: the red line goes to that, and the game's distance
            # says when it is reached (`PLACE_ARRIVED_NAV_M`).
            depot = None
        return GpsFollower.along(network, depot, Route(network, lanes, steps))
    if depot is None:
        raise ValueError(f"{company} in {city} is not on this map")
    if game:
        hitched = bool(int(block["trailers"][0]["connected"]))
        source = (
            Places.open(cache).company(
                bytes(job["source_company"]).split(b"\0", 1)[0].decode(),
                bytes(job["source_city"]).split(b"\0", 1)[0].decode(),
            )
            if hitched
            else None
        )
        return GameRouter.along(
            network,
            depot,
            Route(network, lanes, steps),
            origin=source.position if source is not None else None,
        )
    return GpsRouter(network, depot, Route(network, lanes, steps))


def _brake_to_rest(
    commander: Commander,
    reader: TelemetryReader,
    guard: SafetyMonitor,
    timeout: float = 25.0,
    keep_lane=None,
) -> None:
    """Stop, with the guards suspended so they cannot block their own stop.

    With `keep_lane` -- the steering that keeps to the path, given the
    telemetry -- the truck stays in its lane as it stops. Without it the
    steering is released: the wheel holds its angle, and a controller that
    has just been told it lost the path should not keep steering on that idea.
    """
    deadline = time.perf_counter() + timeout
    with guard.suspend():
        while time.perf_counter() < deadline:
            block, waited = _read_braking(reader, commander)
            deadline += waited
            if abs(float(block["truck"]["speed"])) < 0.15:
                break
            steer = 0.0
            if keep_lane is not None:
                with contextlib.suppress(Exception):
                    steer = keep_lane(block)
            commander.send(ControlCommand(steer=steer, brake=0.45))
            time.sleep(LOOP_INTERVAL)
        # Stopped: set the parking brake before letting go. On a slope a
        # truck left to the service brake rolled back within a quarter of a
        # second (core.control.parking).
        parking = ParkingBrake()
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            block, waited = _read_braking(reader, commander)
            deadline += waited
            is_set = bool(int(block["truck"]["parking_brake"]))
            press = parking.press(want_set=True, is_set=is_set, now=time.perf_counter())
            if is_set and not press:
                break
            buttons = frozenset({Button.HANDBRAKE}) if press else frozenset()
            commander.send(ControlCommand(brake=0.45, buttons=buttons))
            time.sleep(LOOP_INTERVAL)


GAME_BACK_WAIT_S = 600.0
"""How long a truck the game has stopped showing is braked for, blind,
waiting for it to come back (`_read_braking`)."""


def _read_braking(reader: TelemetryReader, commander: Commander):
    """The newest telemetry block, and the seconds waited for it: while the
    game shows none -- its window put away, it stops; and goes on where it
    was when it comes back -- the brake is kept on, so that the plugin has a
    stack braking the truck the moment it does. Near Venlo (18 November) the
    game stood still for a minute at 31 km/h short of a junction; the stack
    gave up on it after a second, and the truck came back to nobody, and
    crashed. Raises the reader's error once `GAME_BACK_WAIT_S` is up."""
    began = time.perf_counter()
    said = False
    while True:
        try:
            return reader.read().block, time.perf_counter() - began
        except StaleTelemetryError:
            if time.perf_counter() - began > GAME_BACK_WAIT_S:
                raise
            if not said:
                said = True
                print("\n[the game shows nothing: braking until it is back]", file=sys.stderr)
            commander.send(ControlCommand(brake=0.45))
            time.sleep(LOOP_INTERVAL)


def _retarder_off(commander: Commander, reader: TelemetryReader, timeout: float = 3.0) -> None:
    """The retarder's lever back to off, a press at a time, pedals left alone."""
    lever = RetarderLever()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        level = int(reader.read().block["truck"]["retarder_level"])
        now = time.perf_counter()
        buttons = lever.buttons(wanted=0, level=level, now=now)
        if level <= 0 and not buttons:
            return
        commander.send(ControlCommand(buttons=buttons))
        time.sleep(LOOP_INTERVAL)


def _hand_over(commander: Commander) -> None:
    """At rest: hazard lights on, and the horn twice -- the driver's cue to
    take the controls (core.control.handover)."""
    began = time.perf_counter()
    while (since := time.perf_counter() - began) < HANDOVER_S:
        commander.send(ControlCommand(buttons=handover_buttons(since)))
        time.sleep(LOOP_INTERVAL)
    for _ in range(60):
        commander.send(ControlCommand())
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
