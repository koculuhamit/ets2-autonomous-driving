"""Linux platform backend.

The primary development target: Ubuntu 22.04 on an **X11** session. Wayland is
out of scope by decision, not by accident -- screen capture and global hotkeys
both require compositor-specific portals there, and the resulting latency is
not something this project wants to fight. Backends raise a clear error rather
than degrading silently.

`LinuxGamePaths` is implemented; the rest are skeletons filled in as Phase 0
and Phase 1 proceed, each documenting the approach it will take.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from pal._image import bgra_to_bgr
from pal._steam import ETS2_APP_ID, app_install_dir
from pal.base import (
    Backend,
    Frame,
    GamePaths,
    HotkeyAction,
    HotkeyListener,
    OverlayWindow,
    PalNotConnectedError,
    PalUnsupportedError,
    ScreenCapture,
    SharedMemoryReader,
    SharedMemoryWriter,
    WindowGeometry,
    WindowNotFoundError,
    WindowNotInFrontError,
    WindowOffScreenError,
)

__all__ = [
    "LinuxBackend",
    "LinuxGamePaths",
    "LinuxHotkeyListener",
    "LinuxOverlayWindow",
    "LinuxScreenCapture",
    "LinuxSharedMemoryReader",
    "LinuxSharedMemoryWriter",
    "bgra_to_bgr",
    "is_x11_session",
]

_TODO = "not implemented on Linux"

#: Steam roots to probe, in order. The Debian/Ubuntu package and the Flatpak
#: build each use their own, and `~/.steam/steam` is a symlink whose target
#: varies, so all of them are checked rather than guessed.
_STEAM_ROOT_CANDIDATES = (
    Path.home() / ".steam" / "debian-installation",
    Path.home() / ".steam" / "steam",
    Path.home() / ".local" / "share" / "Steam",
    Path.home() / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam",
)


def is_x11_session() -> bool:
    """Whether the current session is X11.

    Checked explicitly so failures say "this needs an X11 session" instead of
    surfacing as an obscure Xlib error three layers down.
    """
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"


def _require_x11(capability: str) -> None:
    if not is_x11_session():
        session = os.environ.get("XDG_SESSION_TYPE") or "unknown"
        raise PalUnsupportedError(
            f"{capability} requires an X11 session, but XDG_SESSION_TYPE={session!r}. "
            "Set WaylandEnable=false in /etc/gdm3/custom.conf and log in again."
        )


class LinuxGamePaths(GamePaths):
    """Locates the game through Steam's own metadata.

    No configured path: absolute paths in config files do not survive moving
    between machines, let alone between platforms.
    """

    def __init__(self, steam_root: Path | None = None, app_id: int = ETS2_APP_ID) -> None:
        self._explicit_root = steam_root
        self._app_id = app_id
        self._cached_root: Path | None = None

    @property
    def steam_root(self) -> Path:
        """The Steam installation root.

        Raises:
            PalNotConnectedError: no Steam installation was found.
        """
        if self._cached_root is not None:
            return self._cached_root
        candidates = (
            (self._explicit_root,) if self._explicit_root is not None else _STEAM_ROOT_CANDIDATES
        )
        for candidate in candidates:
            if (candidate / "steamapps").is_dir():
                self._cached_root = candidate.resolve()
                return self._cached_root
        searched = ", ".join(str(c) for c in candidates)
        raise PalNotConnectedError(f"no Steam installation found; looked in: {searched}")

    @property
    def game_root(self) -> Path:
        install_dir = app_install_dir(self.steam_root, self._app_id)
        if install_dir is None:
            raise PalNotConnectedError(
                f"Steam app {self._app_id} is not installed under {self.steam_root}"
            )
        return install_dir

    @property
    def binary_dir(self) -> Path:
        return self.game_root / "bin" / "linux_x64"

    @property
    def plugins_dir(self) -> Path:
        return self.binary_dir / "plugins"

    @property
    def user_dir(self) -> Path:
        """`~/.local/share/Euro Truck Simulator 2`.

        Fixed by the game, independent of where Steam put the installation.
        """
        return Path.home() / ".local" / "share" / "Euro Truck Simulator 2"

    def archives(self) -> Sequence[Path]:
        """Base-game and DLC `.scs` archives, in the game's own load order.

        The game reads `base.scs` and `def.scs` first, then DLC archives; later
        archives override earlier ones. Sorting the DLC names keeps the result
        reproducible across machines, which matters because the map cache is
        keyed on it.
        """
        root = self.game_root
        if not root.is_dir():
            raise PalNotConnectedError(f"game root does not exist: {root}")
        priority = ("base.scs", "base_share.scs", "base_vehicle.scs", "core.scs", "def.scs")
        ordered = [root / name for name in priority if (root / name).is_file()]
        rest = sorted(p for p in root.glob("*.scs") if p not in ordered)
        return tuple(ordered + rest)

    def is_game_installed(self) -> bool:
        """Whether a *complete* installation is present.

        A download in progress creates the directory and the manifest long
        before the binary exists, so the binary is what gets checked.
        """
        try:
            return (self.binary_dir / "eurotrucks2").is_file()
        except PalNotConnectedError:
            return False


class LinuxSharedMemoryReader(SharedMemoryReader):
    """Maps the segment the telemetry plugin publishes.

    On Linux a POSIX shared memory object is an ordinary file under
    `/dev/shm`, so this is a read-only `mmap` of that path -- no `shm_open`
    binding needed, and the same code works for any producer.

    `read()` returns a copy. The plugin keeps writing while we work and callers
    hold snapshots across frames, so handing out a view of live memory would
    mean every stored "snapshot" silently changed underneath them.
    """

    def __init__(self, name: str, size: int) -> None:
        self._name = name
        self._size = size
        self._file: BinaryIO | None = None
        self._map: mmap.mmap | None = None

    @property
    def path(self) -> Path:
        """Where the segment appears in the filesystem."""
        return Path("/dev/shm") / self._name.lstrip("/")

    def open(self) -> None:
        self.close()
        path = self.path
        if not path.exists():
            # "No segment" is equally consistent with the game not running, the
            # plugin not being installed, and the plugin failing during init.
            # The game does not log the SDK's own messages on Linux, so the
            # plugin keeps its own diagnostics file; point at it.
            diagnostics = plugin_log()
            hint = (
                f"The plugin wrote {diagnostics}; check it for the reason."
                if diagnostics.is_file()
                else "The plugin left no diagnostics file: if the game is running, it "
                "never loaded the plugin. Close the game and run the launcher "
                "(ets2ad.sh) again, which installs it; from a checkout, "
                "python scripts/install_plugin.py"
            )
            raise PalNotConnectedError(f"{path} does not exist. Is the game running? {hint}")

        actual = path.stat().st_size
        if actual < self._size:
            raise PalNotConnectedError(
                f"{path} is {actual} bytes, expected at least {self._size}. "
                f"This usually means the installed plugin is an older build."
            )

        try:
            self._file = path.open("rb")
            self._map = mmap.mmap(self._file.fileno(), self._size, prot=mmap.PROT_READ)
        except OSError as exc:
            self.close()
            raise PalNotConnectedError(f"could not map {path}: {exc}") from exc

    def close(self) -> None:
        if self._map is not None:
            with contextlib.suppress(BufferError, ValueError, OSError):
                self._map.close()
            self._map = None
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    @property
    def is_open(self) -> bool:
        return self._map is not None

    @property
    def size(self) -> int:
        return self._size

    def read(self) -> bytes:
        if self._map is None:
            raise PalNotConnectedError("shared memory reader is not open")
        try:
            return self._map[: self._size]
        except (ValueError, OSError) as exc:
            # The mapping outlives shm_unlink, so this means something more
            # unusual than the game exiting -- but the recovery is the same.
            self.close()
            raise PalNotConnectedError(f"shared memory read failed: {exc}") from exc


class LinuxSharedMemoryWriter(SharedMemoryWriter):
    """Maps the plugin's command segment for writing.

    Opened `r+b` and mapped `MAP_SHARED`: the segment already exists, created
    by the plugin inside the game process. Creating it here would produce a
    block with nobody on the other end, which is worse than failing.
    """

    def __init__(self, name: str, size: int) -> None:
        self._name = name
        self._size = size
        self._file: BinaryIO | None = None
        self._map: mmap.mmap | None = None

    @property
    def path(self) -> Path:
        return Path("/dev/shm") / self._name.lstrip("/")

    def open(self) -> None:
        self.close()
        path = self.path
        if not path.exists():
            raise PalNotConnectedError(
                f"{path} does not exist; the game is not running with the plugin loaded"
            )
        actual = path.stat().st_size
        if actual < self._size:
            raise PalNotConnectedError(
                f"{path} is {actual} bytes, expected at least {self._size}; "
                f"the installed plugin is probably an older build"
            )
        try:
            self._file = path.open("r+b")
            self._map = mmap.mmap(self._file.fileno(), self._size, flags=mmap.MAP_SHARED)
        except OSError as exc:
            self.close()
            raise PalNotConnectedError(f"could not map {path} for writing: {exc}") from exc

    def close(self) -> None:
        if self._map is not None:
            with contextlib.suppress(BufferError, ValueError, OSError):
                self._map.close()
            self._map = None
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    @property
    def is_open(self) -> bool:
        return self._map is not None

    @property
    def size(self) -> int:
        return self._size

    def write(self, data: bytes, offset: int = 0) -> None:
        if self._map is None:
            raise PalNotConnectedError("shared memory writer is not open")
        if offset < 0 or offset + len(data) > self._size:
            raise ValueError(
                f"write of {len(data)} bytes at offset {offset} does not fit "
                f"in a {self._size}-byte segment"
            )
        try:
            self._map[offset : offset + len(data)] = data
        except (ValueError, OSError) as exc:
            self.close()
            raise PalNotConnectedError(f"shared memory write failed: {exc}") from exc


class LinuxScreenCapture(ScreenCapture):
    """Captures the game window: found by title with Xlib, grabbed with mss.

    mss reads the screen through MIT-SHM, which avoids copying every frame
    through the X protocol socket. Measured on this machine it takes 3.3 ms in
    the median for the whole 1920x1080 window with the game running -- about
    300 frames a second, a long way past what anything downstream can use.
    See docs/results/capture_bench.md.

    It captures the screen where the window is, not the window's own drawable:
    a GL or Vulkan window read directly under a compositor can come back
    black. So whatever covers the game would be captured instead of it --
    someone's browser, handed to perception as the road. A grab therefore
    refuses unless the game is the active window, which it must be while
    driving anyway: inputs only reach the focused window. A window manager
    that does not say which window is active is trusted.

    The window's position and size are re-read on every grab -- one X round
    trip -- because the user may move or resize it mid-run.
    """

    def __init__(self, window_title: str) -> None:
        self._window_title = window_title
        self._display = None
        self._window = None
        self._grabber = None
        self._sequence = 0

    def open(self) -> None:
        _require_x11("screen capture")
        if self.is_open:
            return
        import mss
        from Xlib import display as xdisplay

        self._display = xdisplay.Display()
        # Windows come and go while the tree is walked, and asking one that has
        # just gone for its name is an X error Xlib prints to stderr even when
        # the exception is handled. Quiet only for the walk.
        self._display.set_error_handler(lambda *_: None)
        window = _find_window(self._display.screen().root, self._window_title)
        self._display.set_error_handler(None)
        if window is None:
            self._display.close()
            self._display = None
            raise WindowNotFoundError(f"no window titled {self._window_title!r}")
        self._window = window
        self._grabber = mss.MSS()
        self._sequence = 0

    def close(self) -> None:
        if self._grabber is not None:
            with contextlib.suppress(Exception):
                self._grabber.close()
            self._grabber = None
        if self._display is not None:
            with contextlib.suppress(Exception):
                self._display.close()
            self._display = None
        self._window = None

    @property
    def is_open(self) -> bool:
        return self._grabber is not None

    @property
    def geometry(self) -> WindowGeometry:
        if self._display is None or self._window is None:
            raise PalNotConnectedError("screen capture is not open")
        from Xlib import error as xerror

        try:
            size = self._window.get_geometry()
            origin = self._window.translate_coords(self._display.screen().root, 0, 0)
        except (xerror.BadWindow, xerror.BadDrawable) as exc:
            self.close()
            raise PalNotConnectedError("the game window has gone") from exc
        # translate_coords gives the root's origin in window coordinates, so
        # the window's origin on the screen is its negation.
        return WindowGeometry(x=-origin.x, y=-origin.y, width=size.width, height=size.height)

    @property
    def in_front(self) -> bool:
        """Whether the game is the active window, as the window manager says.

        One X round trip. True when the window manager does not publish an
        active window at all.
        """
        if self._display is None or self._window is None:
            raise PalNotConnectedError("screen capture is not open")
        return is_active(self._display, self._window)

    def grab(self) -> Frame:
        if self._grabber is None:
            raise PalNotConnectedError("screen capture is not open")
        where = self.geometry
        screen = self._display.screen()
        if off_screen(where, screen.width_in_pixels, screen.height_in_pixels):
            raise WindowOffScreenError(
                f"the game window ({where.width}x{where.height} at {where.x},{where.y}) "
                f"reaches past the screen ({screen.width_in_pixels}x"
                f"{screen.height_in_pixels}): make the game full screen"
            )
        if not self.in_front:
            raise WindowNotInFrontError(f"{self._window_title!r} is not the active window")
        region = {"left": where.x, "top": where.y, "width": where.width, "height": where.height}
        before = time.perf_counter()
        shot = self._grabber.grab(region)
        after = time.perf_counter()
        self._sequence += 1
        return Frame(
            image=bgra_to_bgr(shot.raw, where.height, where.width),
            timestamp=(before + after) / 2.0,
            sequence=self._sequence,
        )


def plugin_log(environment: Mapping[str, str] = os.environ) -> Path:
    """Where the plugin writes its diagnostics (plugin/src/diagnostics.cpp):
    the user's runtime directory, else `$TMPDIR`, else /tmp."""
    base = environment.get("XDG_RUNTIME_DIR") or environment.get("TMPDIR") or "/tmp"
    return Path(base) / "ets2ad_plugin.log"


def off_screen(where: WindowGeometry, width: int, height: int) -> bool:
    """Whether the window at `where` reaches past a `width` x `height` screen."""
    return (
        where.x < 0
        or where.y < 0
        or where.x + where.width > width
        or where.y + where.height > height
    )


def is_active(display, window) -> bool:
    """Whether `window` is the one `_NET_ACTIVE_WINDOW` names.

    The active window can be the game's own window or a frame the window
    manager wrapped around it, so either counts. No such property -- a
    window manager without EWMH -- counts as active: there is nothing to go
    on, and refusing every grab would make capture unusable there.
    """
    root = display.screen().root
    prop = root.get_full_property(display.intern_atom("_NET_ACTIVE_WINDOW"), 0)
    if prop is None or len(prop.value) == 0:
        return True
    active = int(prop.value[0])
    return active != 0 and (active == window.id or _has_ancestor(window, active))


def _has_ancestor(window, candidate: int) -> bool:
    """Whether a window above `window` in the tree, short of the root, is `candidate`."""
    with contextlib.suppress(Exception):
        tree = window.query_tree()
        while tree.parent is not None and tree.parent.id != tree.root.id:
            if tree.parent.id == candidate:
                return True
            tree = tree.parent.query_tree()
    return False


def _find_window(root, title: str):
    """The first window in the tree whose name is exactly `title`, or None."""
    pending = [root]
    while pending:
        window = pending.pop()
        with contextlib.suppress(Exception):
            if window.get_wm_name() == title:
                return window
        with contextlib.suppress(Exception):
            pending.extend(window.query_tree().children)
    return None


class LinuxHotkeyListener(HotkeyListener):
    """Planned: evdev, reading keyboard devices directly.

    Global by necessity -- the game window has focus while the stack drives, so
    an X11 grab on the application window would never see the keys. Requires
    membership of the `input` group.
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[HotkeyAction], None]] = []

    def open(self) -> None:
        raise NotImplementedError(_TODO)

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return False

    def subscribe(self, callback: Callable[[HotkeyAction], None]) -> None:
        self._callbacks.append(callback)


class LinuxOverlayWindow(OverlayWindow):
    """Low priority: a click-through, always-on-top X11 window.

    Needs a running compositor and the shape extension, and it competes with
    the game for the GPU. The separate dashboard window is the supported path;
    this exists so the interface is honest about being optional.
    """

    def open(self) -> None:
        _require_x11("overlay window")
        raise NotImplementedError(_TODO)

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return False

    def draw(self, image: NDArray[np.uint8]) -> None:
        raise NotImplementedError(_TODO)


class LinuxBackend(Backend):
    """Assembles the Linux implementations."""

    name = "linux"

    def __init__(self, steam_root: Path | None = None) -> None:
        self._steam_root = steam_root

    def shared_memory(self, name: str, size: int) -> SharedMemoryReader:
        return LinuxSharedMemoryReader(name, size)

    def shared_memory_writer(self, name: str, size: int) -> SharedMemoryWriter:
        return LinuxSharedMemoryWriter(name, size)

    def screen_capture(self, window_title: str) -> ScreenCapture:
        return LinuxScreenCapture(window_title)

    def hotkeys(self) -> HotkeyListener:
        return LinuxHotkeyListener()

    def game_paths(self) -> LinuxGamePaths:
        return LinuxGamePaths(self._steam_root)

    def overlay(self) -> OverlayWindow:
        return LinuxOverlayWindow()
