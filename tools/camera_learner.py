"""Learn a truck's camera while its driver drives it, before it drives itself.

A truck with no camera of its own sees through another truck's: distances,
places across the road and widths all come out wrong, and a DAF XD on the
Actros's camera ran into the lorry ahead at 80 km/h (15 November). So such a
truck is not driven by the stack (tools/mission.py) until its camera is
measured, and it is measured from the driver's own driving:

- the drive is recorded in chunks (`CHUNK_S`) with views of the road
  (tools/record.py --images), under data/calibration/<truck>/;
- after each chunk the camera is worked out from all of them
  (tools/calibrate_camera.calibrate), on a thread of its own;
- once that is enough (`calibrate_camera.enough`), it is written beside the
  truck's profile, and the truck can be driven.
"""

from __future__ import annotations

import contextlib
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR: Final = REPO_ROOT / "data" / "calibration"

CHUNK_S: Final = 300.0
"""Recorded at a time before the camera is worked out from what there is."""

RETRY_S: Final = 30.0
"""A chunk that ended in under half its time -- the game not running, its
window not in front -- is tried again this long after, not at once."""

IMAGES_HZ: Final = 2.0
"""Views of the road a second: a DAF XD's camera came from 1092 over 13 min."""


class CameraLearner:
    """Records the driver's driving of `truck` and measures its camera."""

    def __init__(
        self,
        truck: str,
        cache: Path,
        *,
        root: Path = CALIBRATION_DIR,
        chunk_s: float = CHUNK_S,
        launch=subprocess.Popen,
        calibrate=None,
        save=None,
        clock=time.monotonic,
    ) -> None:
        self.truck = truck
        self.cache = cache
        self.folder = root / truck
        self.chunk_s = chunk_s
        self._launch = launch
        self._calibrate = calibrate or _calibrate
        self._save = save or _save
        self._recorder = None
        self._working: threading.Thread | None = None
        self._chunks = 0
        self._clock = clock
        self._began = 0.0
        self._not_before = -float("inf")
        self.learned = False
        self.why = "nothing recorded yet"
        """What the recordings so far lack, in words for the driver."""

    def step(self) -> None:
        """Called on each turn of the mission's loop: keeps a chunk being
        recorded, and works the camera out after each one."""
        if self.learned or (self._working is not None and self._working.is_alive()):
            return
        if self._recorder is not None and self._recorder.poll() is None:
            return
        now = self._clock()
        if self._recorder is not None:
            self._recorder = None
            if now - self._began < self.chunk_s / 2.0:
                self._not_before = now + RETRY_S
            self._working = threading.Thread(target=self._work, daemon=True)
            self._working.start()
            return
        if now < self._not_before:
            return
        self._began = now
        self._chunks += 1
        self._recorder = self._launch(
            [
                sys.executable,
                "tools/record.py",
                str(self.folder / f"{datetime.now():%Y%m%d-%H%M%S}-{self._chunks:03d}"),
                "--images",
                str(IMAGES_HZ),
                "--duration",
                str(self.chunk_s),
                "--notes",
                f"learning the camera of {self.truck}",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _work(self) -> None:
        chunks = sorted(
            path
            for path in self.folder.glob("*")
            if (path / "manifest.json").is_file() and any((path / "images").glob("*.jpg"))
        )
        if not chunks:
            self.why = "no views of the road recorded: is the game in front?"
            return
        try:
            document, why = self._calibrate(self.cache, chunks)
        except Exception as exc:  # a chunk cut short, a bad frame: try again with more
            document, why = None, f"could not be worked out yet ({type(exc).__name__}: {exc})"
        if document is None:
            self.why = why
            return
        self._save(self.truck, document, len(chunks))
        self.learned = True
        self.why = ""

    def stop(self) -> None:
        """Ends a chunk being recorded, keeping what it has (tools/record.py
        finalises on an interrupt)."""
        if self._recorder is not None and self._recorder.poll() is None:
            self._recorder.send_signal(signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._recorder.wait(timeout=10.0)
        self._recorder = None


def _calibrate(cache: Path, chunks: list[Path]):
    from core.map import RoadNetwork
    from tools.calibrate_camera import calibrate

    return calibrate(RoadNetwork.open(cache), chunks, say=lambda *_a, **_k: None)


def _save(truck: str, document: dict, chunks: int) -> None:
    """The camera beside the truck's profile, and the profile told of it."""
    from core.calibration import PROFILES_DIR, load_profile, save_profile

    name = f"{truck}.camera.json"
    (PROFILES_DIR / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    profile = load_profile(truck).with_camera(
        name,
        f"learned from the driver's driving, {chunks} recordings under "
        f"data/calibration/{truck}: focal {document['focal_px']:.0f} px, horizon row "
        f"{document['horizon_row']:.1f}, eye {document['height_m']:.2f} m up",
    )
    save_profile(profile)
