"""Platform abstraction layer: interfaces.

Everything the stack needs from the operating system is declared here and
nowhere else. `core/` talks to these types; it never learns which platform it
is running on. Each backend module (`pal.linux`, `pal.windows`, `pal.fake`)
supplies one concrete implementation per interface, and all of them are held to
the same contract tests in `tests/contract/`.

Note what is *not* here: there is no input device. ADR-0002 established that
the telemetry plugin can register one with the game itself, so feeding controls
in needs no OS-specific code on either platform. `core.common.controls` holds
the control value types, and `core.bridge.command` does the writing.

Two rules shape these interfaces:

* **Raw in, parsed elsewhere.** `SharedMemoryReader` hands back bytes, not
  telemetry fields. The SCS struct layout is identical on both platforms, so
  parsing it is platform-independent work and belongs in `core/bridge/`.
* **Close never fails.** `PalResource.close()` is idempotent and does not
  raise, because it runs in `finally` blocks and on the panic path, where an
  exception would mask the failure that got us there.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Backend",
    "Frame",
    "GamePaths",
    "HotkeyAction",
    "HotkeyListener",
    "OverlayWindow",
    "PalError",
    "PalNotConnectedError",
    "PalResource",
    "PalUnsupportedError",
    "ScreenCapture",
    "SharedMemoryReader",
    "SharedMemoryWriter",
    "WindowGeometry",
    "WindowNotFoundError",
]


GAME_WINDOW_TITLE: Final = "Euro Truck Simulator 2"
"""The game window's title, on every platform, whatever the language."""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PalError(Exception):
    """Base class for every failure raised by a platform backend."""


class PalUnsupportedError(PalError):
    """The backend cannot provide this capability on this platform.

    Raised for capabilities that are genuinely absent -- an overlay window
    under Wayland, for instance -- as opposed to ones that are merely not
    written yet, which raise `NotImplementedError`.
    """


class PalNotConnectedError(PalError):
    """The resource is not open, or the thing behind it went away.

    Typically: the game was closed, so the shared memory segment or the window
    no longer exists. Callers are expected to recover by reopening.
    """


class WindowNotFoundError(PalNotConnectedError):
    """No window matched the requested title."""


class WindowNotInFrontError(PalError):
    """The game window exists but another window is in front of it.

    A capture that reads the screen would return whatever covers the game --
    a browser, a terminal -- and hand someone's private window to perception
    as if it were the road. Transient: bring the game to the front and grab
    again.
    """


class WindowOffScreenError(WindowNotInFrontError):
    """The game window reaches past the edge of the screen.

    A capture that reads the screen cannot read what is not on it: on X11 the
    grab fails outright. On a 1920x1080 screen a 1920x1080 game window pushed
    down by the desktop's top bar ran 27 px off the bottom, and every drive
    stopped at once for "no view of the road". Make the game full screen.
    """


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """Position and size of a window in screen coordinates, in pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured frame.

    `timestamp` comes from `time.perf_counter()` and is taken as close to the
    actual capture as the backend allows. It shares a time axis with telemetry
    reads and control commands, which is what makes latency measurable.
    """

    image: NDArray[np.uint8]
    """Pixel data, shape (height, width, 3), BGR, uint8."""

    timestamp: float
    """`time.perf_counter()` value at capture."""

    sequence: int
    """Monotonically increasing frame counter; gaps mean dropped frames."""

    @property
    def shape(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return height, width


class HotkeyAction(enum.StrEnum):
    """What the user asked for, independent of which key they pressed."""

    ENGAGE = "engage"
    DISENGAGE = "disengage"
    PANIC = "panic"


# --------------------------------------------------------------------------- #
# Resource base
# --------------------------------------------------------------------------- #


class PalResource(abc.ABC):
    """A platform resource with an explicit lifetime.

    Subclasses implement `open()` and `close()`; the context manager protocol
    comes for free. `close()` must be idempotent and must not raise, so that it
    is safe to call from an error path.
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the underlying OS resource.

        Raises:
            PalNotConnectedError: the resource does not exist right now.
            PalUnsupportedError: this platform cannot provide it at all.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the resource. Idempotent; never raises."""

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        """Whether the resource is currently usable."""

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #


class SharedMemoryReader(PalResource):
    """Read-only view of the telemetry plugin's shared memory segment.

    Returns raw bytes. Interpreting them is `core/bridge/`'s job -- the SCS
    struct layout is the same on Linux and Windows, so parsing is not
    platform-specific work and must not be duplicated per backend.
    """

    @abc.abstractmethod
    def read(self) -> bytes:
        """Return a consistent snapshot of the segment.

        The returned buffer is a copy: the plugin keeps writing, and callers
        hold on to snapshots across frames.

        Raises:
            PalNotConnectedError: the segment vanished (game closed).
        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Size of the mapped segment in bytes."""


class SharedMemoryWriter(PalResource):
    """Write access to a shared memory segment created by someone else.

    The counterpart to `SharedMemoryReader`, and the direction that carries
    control commands: the plugin creates the segment inside the game process
    and the stack writes setpoints into it.

    Writing, not creating. If the segment does not exist, the game is not
    running and there is nothing to command -- so `open()` fails rather than
    conjuring a block nobody reads.
    """

    @abc.abstractmethod
    def write(self, data: bytes, offset: int = 0) -> None:
        """Copy `data` into the segment at `offset`.

        Raises:
            PalNotConnectedError: the segment went away.
            ValueError: the write would run past the end of the segment.
        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Size of the mapped segment in bytes."""


class ScreenCapture(PalResource):
    """Grabs frames from the game window."""

    @abc.abstractmethod
    def grab(self) -> Frame:
        """Capture the current contents of the window.

        Raises:
            PalNotConnectedError: the window was closed or is not mapped.
            WindowNotInFrontError: another window covers the game, so what
                would be captured is not the game.
        """

    @property
    @abc.abstractmethod
    def geometry(self) -> WindowGeometry:
        """Current window geometry.

        Re-read after every reopen: the user may move or resize the window, and
        the capture region has to follow.
        """


class HotkeyListener(PalResource):
    """Listens for the engage / disengage / panic keys globally.

    Global, because the game window has focus while the stack is driving.
    Callbacks run on the listener's own thread; keep them short and hand work
    to the main loop.
    """

    @abc.abstractmethod
    def subscribe(self, callback: Callable[[HotkeyAction], None]) -> None:
        """Register a callback invoked once per hotkey press."""


class GamePaths(abc.ABC):
    """Locates the game's installation and user files.

    Everything here is derived, never configured: absolute paths must not
    appear in config files, because they do not survive moving between
    machines or platforms.
    """

    @property
    @abc.abstractmethod
    def game_root(self) -> Path:
        """Installation root, the directory containing `bin/` and `base.scs`.

        Raises:
            PalNotConnectedError: the game could not be located.
        """

    @property
    @abc.abstractmethod
    def binary_dir(self) -> Path:
        """Platform binary directory (`bin/linux_x64`, `bin/win_x64`)."""

    @property
    @abc.abstractmethod
    def plugins_dir(self) -> Path:
        """Where the telemetry plugin goes. May not exist yet."""

    @property
    @abc.abstractmethod
    def user_dir(self) -> Path:
        """User data directory holding `config.cfg`, profiles and saves."""

    @abc.abstractmethod
    def archives(self) -> Sequence[Path]:
        """Every `.scs` archive of the base game and installed DLC.

        Ordering is significant: later archives override earlier ones, the same
        way the game layers them.
        """


class OverlayWindow(PalResource):
    """A click-through window drawn on top of the game.

    Optional. Straightforward on Windows, awkward on X11 and impossible under
    Wayland, so backends are allowed to raise `PalUnsupportedError` from
    `open()`. The separate dashboard window is the supported path; this is a
    nicety.
    """

    @abc.abstractmethod
    def draw(self, image: NDArray[np.uint8]) -> None:
        """Replace the overlay contents with an RGBA image."""


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


class Backend(abc.ABC):
    """Factory for one platform's implementations.

    Resources are created, not opened: the caller decides lifetime, usually via
    `with`.
    """

    name: str
    """Short identifier, e.g. "linux". Used in logs and test parametrisation."""

    @abc.abstractmethod
    def shared_memory(self, name: str, size: int) -> SharedMemoryReader:
        """Reader for the named segment, expected to be `size` bytes."""

    @abc.abstractmethod
    def shared_memory_writer(self, name: str, size: int) -> SharedMemoryWriter:
        """Writer for the named segment, which must already exist."""

    @abc.abstractmethod
    def screen_capture(self, window_title: str) -> ScreenCapture:
        """Capture bound to the first window whose title contains the pattern."""

    @abc.abstractmethod
    def hotkeys(self) -> HotkeyListener:
        """Global hotkey listener."""

    @abc.abstractmethod
    def game_paths(self) -> GamePaths:
        """Locator for the game's installation and user files."""

    @abc.abstractmethod
    def overlay(self) -> OverlayWindow:
        """On-top overlay window.

        Raises:
            PalUnsupportedError: this platform or session cannot host one.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
