#!/usr/bin/env python3
"""Record a driving session to disk.

    python tools/record.py data/runs/a2-east --notes "highway, dry, daytime"
    python tools/record.py data/runs/probe --duration 30
    python tools/record.py data/runs/traffic --images 10

Drive manually or let something else drive; this only watches. Ctrl-C stops and
finalises the recording.

`--images HZ` also records the game window that many times a second, on the
telemetry's timeline -- the data perception is measured and trained on. The
images are encoded on a thread of their own so telemetry is never late for
them. While another window is in front of the game nothing is grabbed: the
screen would show that window, not the road.

Recordings go under `data/`, which is git-ignored: they are large, they are
reproducible only by driving again, and they are nobody else's business.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bridge.reader import StaleTelemetryError, TelemetryError, TelemetryReader  # noqa: E402
from core.recording import Recorder  # noqa: E402
from pal import PalError  # noqa: E402

#: Poll a little faster than the game publishes, so no frame is missed to
#: aliasing. Duplicate reads are dropped by the write-count check below.
POLL_INTERVAL = 0.004


def default_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    return REPO_ROOT / "data" / "runs" / stamp


class ImageRecorder:
    """Grabs the game window at a fixed rate and records it off-thread."""

    def __init__(self, recorder: Recorder, rate: float) -> None:
        from pal import GAME_WINDOW_TITLE, get_backend

        self._recorder = recorder
        self._interval = 1.0 / rate
        self._capture = get_backend().screen_capture(GAME_WINDOW_TITLE)
        self._pending: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._write, name="images", daemon=True)
        self._next = 0.0
        self.covered = 0
        self.dropped = 0

    def open(self) -> None:
        self._capture.open()
        self._thread.start()

    def poll(self) -> None:
        """Grab if one is due. Called from the telemetry loop; takes ~3 ms when it grabs."""
        from pal import WindowNotInFrontError

        now = time.perf_counter()
        if now < self._next:
            return
        self._next = max(self._next + self._interval, now)
        try:
            frame = self._capture.grab()
        except WindowNotInFrontError:
            self.covered += 1
            return
        try:
            self._pending.put_nowait(frame)
        except queue.Full:
            # The disk is not keeping up. Dropping an image is better than
            # stalling telemetry; the count says it happened.
            self.dropped += 1

    def close(self) -> None:
        self._pending.put(None)
        self._thread.join()
        self._capture.close()

    def report(self) -> str:
        notes = []
        if self.covered:
            notes.append(f"{self.covered} skipped with another window in front")
        if self.dropped:
            notes.append(f"{self.dropped} dropped, the disk not keeping up")
        return f"{self._recorder.frame_count} images" + (f" ({', '.join(notes)})" if notes else "")

    def _write(self) -> None:
        while (frame := self._pending.get()) is not None:
            self._recorder.write_frame(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="recording directory (default: data/runs/<timestamp>)",
    )
    parser.add_argument("--notes", default="", help="free text stored in the manifest")
    parser.add_argument(
        "--images",
        type=float,
        default=0.0,
        metavar="HZ",
        help="also record the game window this many times a second (default: none)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds (default: until Ctrl-C)",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="stop once this file exists: for a recording started in a window of its "
        "own, where nobody can press Ctrl-C (a border driven to teach it)",
    )
    args = parser.parse_args()

    path = args.path or default_path()

    try:
        reader = TelemetryReader()
        reader.open()
        reader.wait_for_frame()
    except (PalError, TelemetryError) as exc:
        print(f"cannot read telemetry: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "game_version": int(reader.read().block["header"]["game_version"]),
        "game_id": int(reader.read().block["header"]["game_id"]),
    }

    recorder = Recorder(path, notes=args.notes, metadata=metadata)
    try:
        recorder.open()
    except (OSError, FileExistsError) as exc:
        print(f"cannot start recording: {exc}", file=sys.stderr)
        reader.close()
        return 1

    images = ImageRecorder(recorder, args.images) if args.images > 0 else None
    if images is not None:
        try:
            images.open()
        except PalError as exc:
            print(f"cannot capture the game window: {exc}", file=sys.stderr)
            recorder.close()
            reader.close()
            return 1

    print(f"recording to {path}")
    print("Ctrl-C to stop.\n")

    last_write_count = -1
    duplicates = 0
    stale_since: float | None = None
    started = time.perf_counter()

    try:
        while True:
            if args.duration and time.perf_counter() - started >= args.duration:
                break
            if args.stop_file is not None and args.stop_file.exists():
                break
            try:
                snapshot = reader.read()
            except StaleTelemetryError:
                # A pause is part of the session, not the end of it. Keep the
                # recording open so the run survives a menu visit.
                if stale_since is None:
                    stale_since = time.perf_counter()
                    print("\n[paused or game closed -- still recording]", file=sys.stderr)
                time.sleep(0.1)
                continue

            if stale_since is not None:
                print(f"[resumed after {time.perf_counter() - stale_since:.1f}s]", file=sys.stderr)
                stale_since = None

            # The reader polls faster than the game publishes, so the same
            # frame comes back more than once. Recording duplicates would
            # inflate the file and put fictitious frames into any replay.
            if snapshot.write_count == last_write_count:
                duplicates += 1
                time.sleep(POLL_INTERVAL)
                continue
            last_write_count = snapshot.write_count

            recorder.write_telemetry(snapshot)
            if images is not None:
                images.poll()
            if recorder.telemetry_count % 600 == 0:
                print(
                    f"\r{recorder.telemetry_count:7d} frames  "
                    f"{recorder.duration:6.1f}s  "
                    f"{float(snapshot.block['truck']['speed']) * 3.6:5.1f} km/h",
                    end="",
                    flush=True,
                )
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        pass
    except (PalError, TelemetryError) as exc:
        print(f"\nrecording stopped: {exc}", file=sys.stderr)
    finally:
        if images is not None:
            images.close()
        recorder.close()
        reader.close()

    print(f"\n\n{recorder.telemetry_count} frames, {recorder.duration:.1f}s -> {path}")
    if images is not None:
        print(images.report())
    if duplicates:
        print(f"({duplicates} duplicate reads dropped)")
    return 0 if recorder.telemetry_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
