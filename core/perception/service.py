"""Perception in a process of its own, beside the control loop.

The control loop runs at 60 Hz and must not wait: a frame takes 4 ms to grab
and 20-odd to detect, and a steering command 25 ms late is a steering
command the truck feels. So capture, detection and tracking run in a child
process, and the loop takes the newest report whenever it wants one,
without blocking.

A report carries every track with its estimate, stamped with the time its
frame was grabbed. The lead vehicle is chosen by the caller, because the
lane it must be on comes from the route the caller is driving.

Timestamps are `time.perf_counter()`, which on both Linux and Windows reads
a clock shared by every process on the machine, so the caller can tell how
old a report is.

The capture and detector are made inside the child by factories the caller
passes, so tests run the real process with a fake window and a fake model.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from core.perception.blackbox import BlackBox
from core.perception.cab import (
    CAB_REFERENCE_FILE,
    DashboardLearner,
    RoofLearner,
    in_cab,
    load_reference,
)
from core.perception.camera import GroundCamera
from core.perception.detection import Kind
from core.perception.lanes import lane_lines, lateral_at
from core.perception.lights import Lamp, LightState, bright_view, find_lamps, reading
from core.perception.tracking import CUT_MARGIN_PX, Tracker, plausible
from core.planning.gps import road_mask, route_mask

__all__ = ["PerceptionReport", "PerceptionService", "TrackReport", "clipped_at"]

CLIP_MARGIN_PX = 4.0
"""A box within this of a side of the view is cut off by it."""


def clipped_at(box, view, width: int, height: int) -> int:
    """-1 when `box` reaches the left side of what the detector is shown,
    1 the right, 0 neither -- or no view."""
    if view is None:
        return 0
    left, _, right, _ = view.windscreen.pixels(width, height)
    if box[0] <= left + CLIP_MARGIN_PX:
        return -1
    if box[2] >= right - CLIP_MARGIN_PX:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class TrackReport:
    """One track as the child saw it. Duck-types `Track` for `lead_vehicle`."""

    id: int
    kind: str
    confirmed: bool
    plausible: bool
    distance: float
    closing: float
    lateral: float
    width: float
    measurement_age: float = 0.0
    clipped: int = 0
    """-1 when its box is cut off by the left side of the view, 1 by the
    right, 0 neither (core.perception.tracking.CLIPPED_REACH_M)."""

    def estimate(self) -> tuple[float, float, float, float]:
        return self.distance, self.closing, self.lateral, self.width


@dataclass(frozen=True, slots=True)
class KeptFrame:
    """A view of the road kept for the drive's recording, JPEG-encoded in
    the child: for training on (`PerceptionService.keep_frames`)."""

    timestamp: float
    """`time.perf_counter()` at capture, on the caller's clock."""

    sequence: int
    width: int
    height: int
    jpeg: bytes


KEEP_QUALITY: Final = 90
"""JPEG quality of a kept view, as core.recording.writer.JPEG_QUALITY."""


@dataclass(frozen=True, slots=True)
class PerceptionReport:
    grabbed_at: float
    """perf_counter() when the frame was grabbed."""

    sequence: int
    tracks: tuple[TrackReport, ...]
    detect_ms: float
    covered: bool = False
    """True when the game was not in front and nothing could be seen."""

    light: LightState = LightState.UNKNOWN
    """The colour of the lit traffic-light lamps in this frame, by area
    (core.perception.lights.reading); unfiltered."""

    lamp_area: int = 0
    """Total area of the lamps behind `light`, pixels."""

    lamps: tuple[Lamp, ...] = ()
    """Every lit traffic-light lamp found, for core.perception.lights.LightFilter."""

    mirror: tuple[int, float] | None = None
    """(side, how near) of the nearest vehicle in the mirror on the side
    asked for (`PerceptionService.watch_mirror`, Detector.nearest_in_mirror),
    in the frames a mirror was looked at so; None in the others."""

    mirror_far: tuple[int, float] | None = None
    """The same, found in the mirror as it looks (Detector.far_in_mirror):
    far vehicles too, for how fast one comes up; in the frames a mirror was
    looked at so, None in the others."""

    route_mask: bytes = b""
    """The minimap's GPS route (core.planning.gps.route_mask), np.packbits'd;
    empty for a frame of another size."""

    road_mask: bytes = b""
    """The minimap's roads (core.planning.gps.road_mask), packed the same way,
    for measuring its zoom."""

    lane_left_m: float = float("nan")
    lane_right_m: float = float("nan")
    """The painted lines of the truck's lane (core.perception.lanes), metres
    right of the driver's eye `LANE_CHECK_M` ahead; NaN where not found. For
    measuring where the map's lane and the painted one part."""

    unseen: tuple[tuple[int, float], ...] = ()
    """(id, box middle's column, pixels) of each confirmed
    track with no distance -- its foot hidden by the dashboard, so close --
    for core.control.lane_change."""

    in_cab: bool = True
    """Whether the game's camera is the cab's (core.perception.cab), which
    everything else here assumes."""

    covered_why: str = ""
    """Why nothing could be seen, where `covered`: the capture's word."""


LANE_EVERY = 3
"""The lane's lines are looked for in every third frame: 9 ms a frame, beside
20-odd for the detector, for a measurement that changes over metres."""


STOP_WITHIN_S: Final = 5.0
"""How long the perception process is given to write its black box and stop."""


READY_TIMEOUT_S: Final = 180.0
"""How long the child may take to load its model. The first start of a new
install on Windows loads CUDA's libraries from a fresh folder, which the
virus scanner reads first: 60 s was not enough (20 November)."""


class PerceptionService:
    """Starts the child, and hands the caller its newest report."""

    def __init__(
        self,
        make_capture: Callable[[], Any],
        make_detector: Callable[[], Any],
        camera: GroundCamera,
        rate_hz: float = 30.0,
        black_box: Path | None = None,
        cab_reference: Path = CAB_REFERENCE_FILE,
        learn_cab: Path | None = None,
        learn_roof: bool = False,
    ) -> None:
        """`black_box`: where `snapshot` writes the frames about a moment;
        `cab_reference`: the truck's dashboard (core.perception.cab);
        `learn_cab`: for a truck with none of its own, where to save the one
        learned from the first frames (`cab_learned`), not checking the cab
        until then; `learn_roof`: for a truck with no view of its own, find
        where its windscreen begins and look from there (`view_learned`)."""
        self._cab_reference = cab_reference
        self._learn_cab = learn_cab
        self._cab_learned: Path | None = None
        self._learn_roof = learn_roof
        self._view_learned: float | None = None
        self._make_capture = make_capture
        self._black_box = black_box
        self._snapshots: mp.Queue = mp.Queue()
        self._make_detector = make_detector
        self._camera = camera
        self._interval = 1.0 / rate_hz
        self._process: mp.Process | None = None
        self._receiver = None
        self._stop = None
        self._lock = threading.Lock()
        self._drainer: threading.Thread | None = None
        self._draining = threading.Event()
        self._light_distance = mp.Value("d", float("nan"), lock=False)
        self._keep_every = mp.Value("d", 0.0, lock=False)
        self._kept: list[KeptFrame] = []
        self._slow: list[str] = []
        self._daylight = mp.Value("b", 0, lock=False)
        self._mirror_side = mp.Value("b", 0, lock=False)
        self._held = mp.Event()
        self._latest: PerceptionReport | None = None
        self.error: str | None = None
        self.provider: str = ""
        """Where the child's model runs, as its detector reported it; empty
        for a detector that does not say."""

    def start(self, ready_timeout: float = READY_TIMEOUT_S) -> None:
        """Start the child and wait until it has loaded its model."""
        receiver, sender = mp.Pipe(duplex=False)
        self._stop = mp.Event()
        self._process = mp.Process(
            target=_run,
            args=(
                sender,
                self._stop,
                self._make_capture,
                self._make_detector,
                self._camera,
                self._interval,
                self._light_distance,
                self._black_box,
                self._snapshots,
                self._daylight,
                self._cab_reference,
                self._keep_every,
                self._held,
                self._mirror_side,
                self._learn_cab,
                self._learn_roof,
            ),
            name="perception",
            daemon=True,
        )
        self._process.start()
        sender.close()
        self._receiver = receiver
        if not receiver.poll(ready_timeout):
            self.stop()
            raise TimeoutError("perception did not start in time")
        message = receiver.recv()
        if not (isinstance(message, tuple) and message[0] == "ready"):
            self.stop()
            raise RuntimeError(f"perception failed to start: {message}")
        self.provider = message[1]
        # The pipe is read all the time, not when the caller asks: unread, it
        # fills and the child waits on it, grabbing nothing. On Windows it
        # holds less than one report, so the 1.25 s follow.py spent setting
        # up after this made its first report 1.25 s old, and the drive stopped.
        self._draining.set()
        self._drainer = threading.Thread(
            target=self._drain, args=(receiver,), name="perception-reader", daemon=True
        )
        self._drainer.start()

    def _drain(self, receiver) -> None:
        with contextlib.suppress(EOFError, OSError):
            while self._draining.is_set():
                if not receiver.poll(0.05):
                    continue
                try:
                    message = receiver.recv()
                except (EOFError, OSError):
                    raise
                except Exception as exc:
                    with self._lock:
                        self.error = f"a report could not be read: {_said(exc)}"
                    continue
                with self._lock:
                    if isinstance(message, PerceptionReport):
                        self._latest = message
                    elif isinstance(message, KeptFrame):
                        self._kept.append(message)
                    elif isinstance(message, tuple) and message[0] == "slow":
                        self._slow.append(message[1])
                    elif isinstance(message, tuple) and message[0] == "cab":
                        self._cab_learned = Path(message[1])
                    elif isinstance(message, tuple) and message[0] == "view":
                        self._view_learned = float(message[1])
                    elif isinstance(message, str):
                        self.error = message

    def cab_learned(self) -> Path | None:
        """Where the truck's dashboard was saved, once, when learned
        (`learn_cab`); None before, and after it has been asked."""
        with self._lock:
            learned, self._cab_learned = self._cab_learned, None
        return learned

    def view_learned(self) -> float | None:
        """Where the windscreen was found to begin, as a share of the frame's
        height, once (`learn_roof`); None before, and after it has been asked."""
        with self._lock:
            learned, self._view_learned = self._view_learned, None
        return learned

    def latest(self) -> PerceptionReport | None:
        """The newest report so far; never blocks. None before the first."""
        with self._lock:
            return self._latest

    def set_light_distance(self, metres: float | None) -> None:
        """How far the nearest traffic light can be, from the map; None for
        no light near. Lamps are looked for no lower than one 1.6 m up at
        that distance (core.perception.lights.lowest_row)."""
        self._light_distance.value = float("nan") if metres is None else float(metres)

    def keep_frames(self, every_s: float) -> None:
        """Keep a view of the road every `every_s` seconds for the recording;
        0 for none."""
        self._keep_every.value = max(float(every_s), 0.0)

    def kept(self) -> list[KeptFrame]:
        """The views kept since last asked."""
        with self._lock:
            kept, self._kept = self._kept, []
        return kept

    def slow_turns(self) -> list[str]:
        """The perception loop's slow turns since last asked, each told part
        by part (`SLOW_PERCEPTION_S`)."""
        with self._lock:
            slow, self._slow = self._slow, []
        return slow

    def watch_mirror(self, side: int) -> None:
        """Look in the mirror on `side` -- +1 right, -1 left -- or, with 0, in
        none (`PerceptionReport.mirror`): for a lane change on that side."""
        self._mirror_side.value = int(side)

    def set_daylight(self, daylight: bool) -> None:
        """Whether it is day in the game (core.perception.lights.is_daylight)."""
        self._daylight.value = int(daylight)

    def snapshot(self, label: str) -> None:
        """Have the frames about now written out, as `label` (core.perception.blackbox)."""
        if self._black_box is not None:
            self._snapshots.put((str(Path(self._black_box) / label), time.perf_counter()))

    def write_black_box_to(self, directory: Path) -> None:
        """Where `snapshot` writes from now on: a drive that waited for the
        beacon (tools/follow.py --standby) learns its folder only then."""
        self._black_box = directory

    def hold(self) -> None:
        """Stop looking at the road until `resume`, the model kept loaded:
        waiting for the beacon (tools/follow.py --standby), the detector
        would take the GPU from the game the driver is driving."""
        self._held.set()

    def resume(self) -> None:
        self._held.clear()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._process is not None:
            # The reader goes on reading while the child winds down: blocked
            # on a full pipe it never got to write the black box, and was
            # killed. After two hits the frames that would have shown them
            # were lost.
            self._process.join(timeout=STOP_WITHIN_S)
            if self._process.is_alive():
                self._process.terminate()
        self._draining.clear()
        if self._drainer is not None:
            self._drainer.join(timeout=1.0)
        if self._receiver is not None:
            self._receiver.close()
        self._process = self._receiver = self._drainer = None

    def __enter__(self) -> PerceptionService:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _run(
    sender,
    stop,
    make_capture,
    make_detector,
    camera,
    interval,
    light_distance,
    black_box_dir=None,
    snapshots=None,
    daylight=None,
    cab_reference_file: Path = CAB_REFERENCE_FILE,
    keep_every=None,
    held=None,
    mirror_side=None,
    learn_cab: Path | None = None,
    learn_roof: bool = False,
) -> None:
    """The child: grab, detect, track, report, until told to stop."""
    black_box = BlackBox() if black_box_dir is not None else None
    try:
        capture = make_capture()
        capture.open()
        detector = make_detector()
    except Exception as exc:  # the parent needs to hear why, whatever it was
        sender.send(f"{type(exc).__name__}: {exc}")
        return
    tracker = Tracker(camera)
    cab_reference = load_reference(cab_reference_file)
    learner = DashboardLearner() if learn_cab is not None else None
    roof = RoofLearner() if learn_roof and getattr(detector, "view", None) is not None else None
    if learner is not None:
        cab_reference = None  # another truck's: not checked against until learned
    sender.send(("ready", str(getattr(detector, "provider", ""))))
    kept_at = -math.inf
    try:
        from pal.base import WindowNotInFrontError

        parent = mp.parent_process()
        while not stop.is_set():
            if parent is not None and not parent.is_alive():
                return  # killed, the parent leaves nobody to stop this one
            if held is not None and held.is_set():
                time.sleep(interval)
                continue
            began = time.perf_counter()
            try:
                frame = capture.grab()
            except WindowNotInFrontError as exc:
                sender.send(
                    PerceptionReport(began, -1, (), 0.0, covered=True, covered_why=str(exc))
                )
                time.sleep(interval)
                continue
            except Exception as exc:
                # Any other failure to grab ended the child, said only to a
                # pipe nobody read out: on Windows every beacon then stopped
                # for "no view of the road: is the game in front?" with the
                # game in front (20 November). It is a view not had, and why.
                why = f"the game window could not be grabbed: {_said(exc)}"
                sender.send(PerceptionReport(began, -1, (), 0.0, covered=True, covered_why=why))
                time.sleep(max(interval, GRAB_RETRY_S))
                continue
            marks = [("grab", time.perf_counter())]
            if black_box is not None:
                black_box.add(frame.image, frame.timestamp)
                _snapshots_due(black_box, snapshots, black_box_dir, frame.timestamp)
            marks.append(("black box", time.perf_counter()))
            every = keep_every.value if keep_every is not None else 0.0
            if every > 0.0 and frame.timestamp - kept_at >= every:
                kept_at = frame.timestamp
                _keep(sender, frame)
            marks.append(("keep", time.perf_counter()))
            detected_at = time.perf_counter()
            found = detector.detect(frame.image)
            detect_ms = (time.perf_counter() - detected_at) * 1000.0
            marks.append(("detect", time.perf_counter()))
            view = getattr(detector, "view", None)
            cut_row = None
            if view is not None:
                height, width = frame.image.shape[:2]
                cut_row = view.windscreen.pixels(width, height)[3]
            tracks = tracker.update(found, frame.timestamp, cut_row)
            marks.append(("track", time.perf_counter()))
            light, lamp_area, lamps = LightState.UNKNOWN, 0, []
            spec = getattr(detector, "spec", None)
            heads = (
                [d.box for d in found if d.kind is Kind.TRAFFIC_LIGHT]
                if spec is not None and spec.signal_heads
                else None
            )
            if view is not None:
                lamps = find_lamps(
                    frame.image,
                    view,
                    camera,
                    [d.box for d in found if d.kind is not Kind.TRAFFIC_LIGHT],
                    nearest_m=light_distance.value,
                    daylight=(bool(daylight.value) if daylight is not None else False)
                    or bright_view(frame.image, view),
                    heads=heads,
                )
                light = reading(lamps)
                lamp_area = sum(lamp.area for lamp in lamps if lamp.colour is light)
            marks.append(("lights", time.perf_counter()))
            route = roads = b""
            lane_left = lane_right = float("nan")
            if frame.image.shape[:2] == (1080, 1920):
                route = np.packbits(route_mask(frame.image)).tobytes()
                roads = np.packbits(road_mask(frame.image)).tobytes()
                if frame.sequence % LANE_EVERY == 0:
                    left_line, right_line = lane_lines(frame.image)
                    lane_left = lateral_at(left_line, camera)
                    lane_right = lateral_at(right_line, camera)
            marks.append(("minimap and lanes", time.perf_counter()))
            reports = []
            unseen = []
            for track in tracks:
                estimate = track.estimate()
                if estimate is None:
                    # Its foot under the dashboard, so close -- not above the
                    # horizon, far up a rise: up onto the Öresund bridge a
                    # car far ahead, taken for one 8 m off, stopped the truck
                    # again and again.
                    hidden = cut_row is not None and track.box[3] >= cut_row - CUT_MARGIN_PX
                    if track.confirmed and hidden:
                        middle = (track.box[0] + track.box[2]) / 2.0
                        unseen.append((track.id, middle))
                    continue
                reports.append(
                    TrackReport(
                        track.id,
                        str(track.kind),
                        track.confirmed,
                        plausible(track),
                        *estimate,
                        track.measurement_age,
                        clipped_at(track.box, view, frame.image.shape[1], frame.image.shape[0]),
                    )
                )
            if roof is not None and frame.image.shape[:2] == (1080, 1920):
                row = roof.offer(frame.image, time.perf_counter())
                if row is not None:
                    roof = None
                    widened = _widened(detector.view, row, frame.image.shape[0])
                    if widened is not None:
                        detector.view = widened
                        sender.send(("view", widened.windscreen.top))
            if learner is not None and frame.image.shape[:2] == (1080, 1920):
                learned = learner.offer(frame.image, time.perf_counter())
                if learned is not None:
                    learner = None
                    cab_reference = learned
                    try:
                        learn_cab.parent.mkdir(parents=True, exist_ok=True)
                        np.save(learn_cab, learned)
                        sender.send(("cab", str(learn_cab)))
                    except OSError as exc:
                        sender.send(("slow", f"the dashboard learned was not saved: {exc}"))
            cab = (
                cab_reference is None
                or frame.image.shape[:2] != (1080, 1920)
                or in_cab(frame.image, cab_reference)
            )
            marks.append(("cab", time.perf_counter()))
            sender.send(
                PerceptionReport(
                    frame.timestamp,
                    frame.sequence,
                    tuple(reports),
                    detect_ms,
                    light=light,
                    lamp_area=lamp_area,
                    lamps=tuple(lamps),
                    route_mask=route,
                    road_mask=roads,
                    lane_left_m=lane_left,
                    lane_right_m=lane_right,
                    unseen=tuple(unseen),
                    in_cab=cab,
                    mirror=_in_mirror(detector, frame, mirror_side),
                    mirror_far=_far_in_mirror(detector, frame, mirror_side),
                )
            )
            marks.append(("send", time.perf_counter()))
            if marks[-1][1] - began > SLOW_PERCEPTION_S:
                sender.send(("slow", _slow_turn(began, marks)))
            spare = interval - (time.perf_counter() - began)
            if spare > 0:
                time.sleep(spare)
    except Exception as exc:
        with contextlib.suppress(Exception):
            sender.send(_said(exc))
    finally:
        if black_box is not None:
            with contextlib.suppress(Exception):
                _snapshots_due(black_box, snapshots, black_box_dir, time.perf_counter(), True)
        with contextlib.suppress(Exception):
            capture.close()
        sender.close()


GRAB_RETRY_S: Final = 0.5
"""After a grab that failed for a reason other than another window in front,
how long before the next try."""


def _said(exc: BaseException) -> str:
    """An exception as one line: its type, its word, and where in the code it
    was raised -- the child's traceback does not reach the parent."""
    import traceback

    where = traceback.extract_tb(exc.__traceback__)[-1:] if exc.__traceback__ else []
    at = f" ({Path(where[0].filename).name}:{where[0].lineno})" if where else ""
    return f"{type(exc).__name__}: {exc}{at}"


SLOW_PERCEPTION_S: Final = 0.4
"""A turn of the perception loop this long is told of, part by part: on a
motorway at 80 km/h no view came for over a second, and the drive stopped,
nothing said of where the time went."""


def _slow_turn(began: float, marks: list[tuple[str, float]]) -> str:
    """How a slow turn of the loop went, as `marks` -- (part, when it ended)
    -- have it: "0.92 s: grab 0.85 s, detect 0.04 s, ..."."""
    parts, last = [], began
    for name, at in marks:
        parts.append(f"{name} {at - last:.2f} s")
        last = at
    return f"{last - began:.2f} s: " + ", ".join(parts)


def _widened(view, row: int, height: int):
    """`view` with its windscreen begun at `row`, if that is higher up than
    it begins now: a view is only widened by what is learned, never cut.
    Learned from a minute at a depot, an Actros's and a Volvo's came out at
    286 and 313, under the 250 their windscreens do begin at."""
    from dataclasses import replace

    from core.perception.detection import Rect

    top = row / height
    screen = view.windscreen
    if top >= screen.top - 10 / height or row < ROOF_LOWEST_ROW:
        return None
    return replace(view, windscreen=Rect(screen.left, top, screen.right, screen.bottom))


ROOF_LOWEST_ROW = 80
"""Above this the frame is the game's own: the mirror insets begin at 80."""


def _keep(sender, frame) -> None:
    """Send the parent a view to keep, encoded here: a raw 1080p frame is
    6 MB down the pipe, a JPEG a few hundred kB."""
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, KEEP_QUALITY])
    if ok:
        height, width = frame.image.shape[:2]
        sender.send(KeptFrame(frame.timestamp, frame.sequence, width, height, encoded.tobytes()))


def _snapshots_due(black_box, snapshots, directory, now, stopping=False) -> None:
    """Take the snapshots asked for, and write out the ones whose time has come."""
    with contextlib.suppress(queue.Empty):
        while True:
            label, asked = snapshots.get_nowait()
            black_box.request(label, asked)
    for label, frames in black_box.due(now, all_of_them=stopping):
        # The label carries its folder (`PerceptionService.snapshot`).
        where = (Path(directory) / Path(label).parent, Path(label).name, frames)
        if stopping:
            BlackBox.write(*where)
        else:
            # Some 24 files, on a laptop's hard disk: written by the loop, it
            # saw nothing while they went, 0.4 s after a lane-guard snapshot.
            threading.Thread(target=BlackBox.write, args=where, daemon=True).start()


MIRROR_EVERY: Final = 2
"""The mirror is looked at in every second frame, and only on the side of a
lane change at hand: the detector takes about 20 ms a frame. The looks take
turns: `Detector.nearest_in_mirror` in one, `Detector.far_in_mirror` in the
next, so watching the mirror costs no more than it did with one."""


def _mirror_side(detector, frame, mirror_side, turn: int, method: str) -> int:
    """The side to look in with `method` in this frame; 0 for none."""
    side = int(mirror_side.value) if mirror_side is not None else 0
    if side == 0 or not hasattr(detector, method) or frame.image.shape[:2] != (1080, 1920):
        return 0
    if frame.sequence % (2 * MIRROR_EVERY) != turn * MIRROR_EVERY:
        return 0
    return side


def _in_mirror(detector, frame, mirror_side) -> tuple[int, float] | None:
    """(side, how near) of the nearest vehicle in the mirror on the side
    asked for, in its turn of the frames; None in the others, or none asked."""
    side = _mirror_side(detector, frame, mirror_side, 0, "nearest_in_mirror")
    return (side, float(detector.nearest_in_mirror(frame.image, side))) if side else None


def _far_in_mirror(detector, frame, mirror_side) -> tuple[int, float] | None:
    """(side, how near) by `Detector.far_in_mirror`, in its turn; or None."""
    side = _mirror_side(detector, frame, mirror_side, 1, "far_in_mirror")
    return (side, float(detector.far_in_mirror(frame.image, side))) if side else None
