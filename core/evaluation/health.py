"""The drive's own health over a long run: memory and processor, sampled.

A leak shows on a day's drive, not on a test's: `HealthSampler` samples the
process and its children (the perception child) every `EVERY_S` on a thread
of its own, and `memory_growth` turns the samples into megabytes an hour --
over the run's later part, once models and caches are loaded.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Final

import numpy as np

__all__ = ["EVERY_S", "HealthSampler", "memory_growth"]

EVERY_S: Final = 10.0
SETTLED_SHARE: Final = 0.25
"""The first quarter of a run is loading, not leaking: left out of the slope."""


class HealthSampler:
    """Samples (seconds since start, resident MB of this process and its
    children, CPU % of them) until stopped."""

    def __init__(self, every_s: float = EVERY_S, process=None) -> None:
        import psutil

        self._process = process or psutil.Process()
        self._every = every_s
        self._samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._began = time.perf_counter()

    def start(self) -> HealthSampler:
        self._thread = threading.Thread(target=self._run, name="health", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> np.ndarray:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.samples()

    def samples(self) -> np.ndarray:
        return np.array(self._samples, float).reshape(-1, 3)

    def sample(self) -> None:
        import psutil

        processes = [self._process]
        with contextlib.suppress(psutil.Error):
            processes += self._process.children(recursive=True)
        resident = cpu = 0.0
        for process in processes:
            try:
                resident += process.memory_info().rss / 1e6
                cpu += process.cpu_percent(interval=None)
            except psutil.Error:
                continue
        self._samples.append((time.perf_counter() - self._began, resident, cpu))

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self.sample()


def memory_growth(samples: np.ndarray) -> float:
    """Resident megabytes gained an hour over the run's settled part; nan
    for too few samples."""
    samples = np.asarray(samples, float).reshape(-1, 3)
    if len(samples) < 4:
        return float("nan")
    settled = samples[int(len(samples) * SETTLED_SHARE) :]
    if len(settled) < 3 or settled[-1, 0] - settled[0, 0] <= 0:
        return float("nan")
    slope = np.polyfit(settled[:, 0], settled[:, 1], 1)[0]
    return float(slope * 3600.0)
