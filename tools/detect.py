#!/usr/bin/env python3
"""Run the vehicle detector on the game window, or on a recording's images.

    python tools/detect.py --live --seconds 60 --save data/detect/live
    python tools/detect.py --recording data/runs/traffic --save data/detect/traffic

Prints what it sees and how long seeing it took. With `--save`, frames with
something in them are written with the boxes drawn, and with the windscreen
the model is shown (green) and the mirrors and HUD it ignores (grey) -- so a
wrong cab layout is as visible as a wrong box.

The model comes from tools/export_detector.py. Nothing here drives.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.perception.detection import Detection, Detector  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "dfine-nano"
SAVE_INTERVAL_S = 0.5
GPU_FIRST = ("CUDAExecutionProvider", "CPUExecutionProvider")


def annotate(image: np.ndarray, found: list[Detection], detector: Detector) -> np.ndarray:
    import cv2

    out = image.copy()
    height, width = out.shape[:2]
    for rect in detector.view.covered:
        left, top, right, bottom = rect.pixels(width, height)
        cv2.rectangle(out, (left, top), (right - 1, bottom - 1), (128, 128, 128), 1)
    left, top, right, bottom = detector.view.windscreen.pixels(width, height)
    cv2.rectangle(out, (left, top), (right - 1, bottom - 1), (0, 200, 0), 2)
    for item in found:
        x0, y0, x1, y1 = (round(v) for v in item.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        label = f"{item.kind} {item.score:.2f}"
        cv2.putText(out, label, (x0, max(y0 - 5, 12)), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


class Tally:
    def __init__(self) -> None:
        self.latency: list[float] = []
        self.kinds: collections.Counter = collections.Counter()
        self.frames = 0
        self.with_something = 0

    def add(self, found: list[Detection], seconds: float) -> None:
        self.frames += 1
        self.latency.append(seconds)
        self.with_something += bool(found)
        self.kinds.update(str(item.kind) for item in found)

    def report(self) -> str:
        if not self.frames:
            return "no frames"
        ms = np.array(self.latency) * 1000
        kinds = ", ".join(f"{n} {k}" for k, n in self.kinds.most_common()) or "nothing"
        return (
            f"{self.frames} frames, {self.with_something} with something in them: {kinds}\n"
            f"detection takes median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms, "
            f"max {ms.max():.1f} ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="the game window, now")
    source.add_argument("--recording", type=Path, help="a recording made with --images")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to run --live")
    parser.add_argument("--save", type=Path, help="write annotated frames here")
    parser.add_argument("--cpu", action="store_true", help="run the model on the CPU")
    args = parser.parse_args()

    providers = ("CPUExecutionProvider",) if args.cpu else GPU_FIRST
    detector = Detector.open(args.model, threshold=args.threshold, providers=providers)
    print(f"{detector.spec.source} on {detector.provider}")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    tally = Tally()
    last_saved = -np.inf

    def handle(image: np.ndarray, name: str, stamp: float) -> None:
        nonlocal last_saved
        before = time.perf_counter()
        found = detector.detect(image)
        tally.add(found, time.perf_counter() - before)
        if found:
            print(
                f"{name}: "
                + ", ".join(
                    f"{d.kind} {d.score:.2f} @({d.ground_point[0]:.0f}, {d.ground_point[1]:.0f})"
                    for d in found
                )
            )
        # The first frame is saved whatever is in it, to check the cab layout by.
        first = tally.frames == 1
        if args.save and (found or first) and stamp - last_saved >= SAVE_INTERVAL_S:
            import cv2

            cv2.imwrite(str(args.save / f"{name}.jpg"), annotate(image, found, detector))
            last_saved = stamp

    if args.recording:
        from core.recording import open_recording

        recording = open_recording(args.recording)
        if not len(recording.frames):
            print(f"{args.recording} has no images: record with --images", file=sys.stderr)
            return 1
        for index in range(len(recording.frames)):
            handle(recording.image(index), f"{index:06d}", float(recording.frames["t"][index]))
    else:
        from pal import GAME_WINDOW_TITLE, WindowNotInFrontError, get_backend

        covered = 0
        with get_backend().screen_capture(GAME_WINDOW_TITLE) as capture:
            end = time.perf_counter() + args.seconds
            while time.perf_counter() < end:
                try:
                    frame = capture.grab()
                except WindowNotInFrontError:
                    covered += 1
                    time.sleep(0.2)
                    continue
                handle(frame.image, f"{frame.sequence:06d}", frame.timestamp)
        if covered:
            print(f"({covered} grabs skipped with another window in front of the game)")

    print(tally.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
