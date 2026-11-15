"""Fake platform backend: deterministic, in-process, no operating system.

This is what makes the rest of the stack testable. Perception, control and
planning can be exercised against recorded or synthetic data with no game
running, no GPU and no X server -- which is also what lets CI run them.

The fakes are deliberately *observable*: `FakeSharedMemoryWriter` keeps what
was written to it, `FakeHotkeyListener` lets a test inject a key press, and
`FakeSharedMemoryReader` can be made to fail on demand so that reconnection
logic gets exercised rather than assumed.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pal.base import (
    Backend,
    Frame,
    GamePaths,
    HotkeyAction,
    HotkeyListener,
    OverlayWindow,
    PalNotConnectedError,
    ScreenCapture,
    SharedMemoryReader,
    SharedMemoryWriter,
    WindowGeometry,
)

__all__ = [
    "FakeBackend",
    "FakeGamePaths",
    "FakeHotkeyListener",
    "FakeOverlayWindow",
    "FakeScreenCapture",
    "FakeSharedMemoryReader",
    "FakeSharedMemoryWriter",
]


class FakeSharedMemoryReader(SharedMemoryReader):
    """Serves pre-supplied byte blobs, cycling once the source is exhausted.

    Set `fail_next` to make the following `read()` raise, simulating the game
    exiting mid-run.
    """

    def __init__(self, size: int, blobs: Sequence[bytes] | None = None) -> None:
        self._size = size
        self._blobs = list(blobs) if blobs else [bytes(size)]
        if any(len(b) != size for b in self._blobs):
            raise ValueError(f"every blob must be exactly {size} bytes")
        self._cursor = 0
        self._open = False
        self.fail_next = False
        self.read_count = 0

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> int:
        return self._size

    def read(self) -> bytes:
        if not self._open:
            raise PalNotConnectedError("shared memory reader is not open")
        if self.fail_next:
            self.fail_next = False
            self._open = False
            raise PalNotConnectedError("simulated shared memory disconnect")
        blob = self._blobs[self._cursor % len(self._blobs)]
        self._cursor += 1
        self.read_count += 1
        return blob


class FakeSharedMemoryWriter(SharedMemoryWriter):
    """Writes into a buffer a test can read back.

    Backed by the same `bytearray` the fake reader for that segment name uses,
    so a round trip through the PAL behaves like the real thing: write a
    command block, read it back, and check what the other side would see.
    """

    def __init__(self, size: int, buffer: bytearray | None = None) -> None:
        self._size = size
        self.buffer = buffer if buffer is not None else bytearray(size)
        if len(self.buffer) != size:
            raise ValueError(f"buffer must be exactly {size} bytes")
        self._open = False
        self.write_count = 0

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> int:
        return self._size

    def write(self, data: bytes, offset: int = 0) -> None:
        if not self._open:
            raise PalNotConnectedError("shared memory writer is not open")
        if offset < 0 or offset + len(data) > self._size:
            raise ValueError(
                f"write of {len(data)} bytes at offset {offset} does not fit "
                f"in a {self._size}-byte segment"
            )
        self.buffer[offset : offset + len(data)] = data
        self.write_count += 1


class FakeScreenCapture(ScreenCapture):
    """Serves frames from a finite sequence, or generates blank ones.

    The generated frames are not blank in a useless way: each carries its
    sequence number in the top-left pixel, so a test can tell frames apart and
    detect reordering.
    """

    def __init__(
        self,
        geometry: WindowGeometry | None = None,
        frames: Iterable[NDArray[np.uint8]] | None = None,
    ) -> None:
        self._geometry = geometry or WindowGeometry(0, 0, 64, 48)
        self._frames: list[NDArray[np.uint8]] | None = list(frames) if frames is not None else None
        self._counter = itertools.count()
        self._open = False
        self.grab_count = 0

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def geometry(self) -> WindowGeometry:
        if not self._open:
            raise PalNotConnectedError("screen capture is not open")
        return self._geometry

    def set_geometry(self, geometry: WindowGeometry) -> None:
        """Simulate the user moving or resizing the game window."""
        self._geometry = geometry

    def grab(self) -> Frame:
        if not self._open:
            raise PalNotConnectedError("screen capture is not open")
        sequence = next(self._counter)
        if self._frames is not None:
            if sequence >= len(self._frames):
                raise PalNotConnectedError("recorded frames exhausted")
            image = self._frames[sequence]
        else:
            image = np.zeros((self._geometry.height, self._geometry.width, 3), dtype=np.uint8)
            image[0, 0] = sequence % 256
        self.grab_count += 1
        return Frame(image=image, timestamp=time.perf_counter(), sequence=sequence)


class FakeHotkeyListener(HotkeyListener):
    """Delivers hotkeys only when a test injects them."""

    def __init__(self) -> None:
        self._open = False
        self._callbacks: list[Callable[[HotkeyAction], None]] = []

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def subscribe(self, callback: Callable[[HotkeyAction], None]) -> None:
        self._callbacks.append(callback)

    def inject(self, action: HotkeyAction) -> None:
        """Simulate the user pressing the key bound to `action`."""
        if not self._open:
            raise PalNotConnectedError("hotkey listener is not open")
        for callback in self._callbacks:
            callback(action)


@dataclass(slots=True)
class FakeGamePaths(GamePaths):
    """Points at a throwaway directory laid out like a real installation.

    `create()` builds that layout under a tmp_path so that tools which expect
    real files -- archive scanning, config backup -- can be tested for real.
    """

    root: Path
    archive_names: tuple[str, ...] = field(default=("base.scs", "def.scs", "dlc_east.scs"))

    @property
    def game_root(self) -> Path:
        return self.root / "game"

    @property
    def binary_dir(self) -> Path:
        return self.game_root / "bin" / "fake_x64"

    @property
    def plugins_dir(self) -> Path:
        return self.binary_dir / "plugins"

    @property
    def user_dir(self) -> Path:
        return self.root / "user"

    def archives(self) -> Sequence[Path]:
        return tuple(self.game_root / name for name in self.archive_names)

    def create(self) -> FakeGamePaths:
        """Materialise the layout on disk and return self."""
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        for archive in self.archives():
            archive.touch()
        (self.user_dir / "config.cfg").write_text('uset g_fps_limit "60"\n', encoding="utf-8")
        return self


class FakeOverlayWindow(OverlayWindow):
    """Keeps the last image it was handed; draws nothing."""

    def __init__(self) -> None:
        self._open = False
        self.last_image: NDArray[np.uint8] | None = None
        self.draw_count = 0

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def draw(self, image: NDArray[np.uint8]) -> None:
        if not self._open:
            raise PalNotConnectedError("overlay window is not open")
        self.last_image = image
        self.draw_count += 1


class FakeBackend(Backend):
    """In-process backend with no OS dependencies.

    Each factory method returns the *same* object every time it is called, so a
    test can configure a fake, hand the backend to production code, and then
    inspect what that code did.
    """

    name = "fake"

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path.cwd() / ".fake-pal"
        self._shared_memory: dict[str, FakeSharedMemoryReader] = {}
        self._writers: dict[str, FakeSharedMemoryWriter] = {}
        self._captures: dict[str, FakeScreenCapture] = {}
        self._hotkeys = FakeHotkeyListener()
        self._paths = FakeGamePaths(self._root)
        self._overlay = FakeOverlayWindow()

    def shared_memory(self, name: str, size: int) -> FakeSharedMemoryReader:
        if name not in self._shared_memory:
            self._shared_memory[name] = FakeSharedMemoryReader(size)
        return self._shared_memory[name]

    def shared_memory_writer(self, name: str, size: int) -> FakeSharedMemoryWriter:
        if name not in self._writers:
            self._writers[name] = FakeSharedMemoryWriter(size)
        return self._writers[name]

    def set_shared_memory(self, name: str, reader: FakeSharedMemoryReader) -> None:
        """Serve `reader` for this segment name.

        Lets a test script the exact byte sequence production code will see --
        including the ones that only occur when a reader races a writer, which
        is precisely what cannot be reproduced on demand against the game.
        """
        self._shared_memory[name] = reader

    def screen_capture(self, window_title: str) -> FakeScreenCapture:
        if window_title not in self._captures:
            self._captures[window_title] = FakeScreenCapture()
        return self._captures[window_title]

    def hotkeys(self) -> FakeHotkeyListener:
        return self._hotkeys

    def game_paths(self) -> FakeGamePaths:
        return self._paths

    def overlay(self) -> FakeOverlayWindow:
        return self._overlay


def blank_frames(count: int, width: int = 64, height: int = 48) -> Iterator[NDArray[np.uint8]]:
    """Generate `count` distinguishable frames for tests."""
    for index in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[0, 0] = index % 256
        yield image
