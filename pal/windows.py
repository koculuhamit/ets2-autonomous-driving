"""Windows platform backend.

What the stack needs from Windows, in the shape the Linux backend gives it:

* **shared memory** -- the plugin (plugin/src/shared_memory.cpp) creates
  `Local\\<name>` mappings in the game; they are opened by name and mapped
  (`OpenFileMappingW`, `MapViewOfFile`), never created here: a block with no
  game on the other end is worse than failing.
* **screen capture** -- the window found by its exact title (`FindWindowW`),
  its client area on the screen (`GetClientRect`, `ClientToScreen`), grabbed
  with mss. As on Linux the screen is read where the window is, so a grab
  refuses unless the game is the foreground window. The process is made
  per-monitor DPI aware first, or a scaled desktop reports the window in
  scaled pixels and the grab is the wrong part of the screen. Exclusive
  full screen can come back black: run the game windowed or borderless.
* **game paths** -- Steam's root from the registry
  (`HKCU\\Software\\Valve\\Steam\\SteamPath`), the library walk shared with
  Linux (pal._steam); the profiles under Documents.
* **notifications** -- a toast through PowerShell and a system sound, each
  best effort.

The Win32 calls go through `Win32Api`, one small class, so the logic around
them runs in tests on any platform with a stand-in. **None of this has run on
Windows yet**: written to the documented APIs, checked on Linux against
stand-ins only.

Hotkeys and the overlay are not built: the stack engages with the truck's
beacon (tools/mission.py) and draws nothing over the game.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

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
)

__all__ = [
    "Win32Api",
    "WindowsBackend",
    "WindowsGamePaths",
    "WindowsHotkeyListener",
    "WindowsOverlayWindow",
    "WindowsScreenCapture",
    "WindowsSharedMemoryReader",
    "WindowsSharedMemoryWriter",
]

_NOT_BUILT = "not built for Windows: the stack engages with the truck's beacon"

FILE_MAP_WRITE = 0x0002
FILE_MAP_READ = 0x0004
ERROR_FILE_NOT_FOUND = 2

_STEAM_FALLBACK = Path("C:/Program Files (x86)/Steam")


class Win32Api:
    """The Win32 calls the backend makes, and nothing else."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise PalUnsupportedError("the Win32 API exists only on Windows")
        from ctypes import wintypes

        self._wintypes = wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._open_mapping = kernel32.OpenFileMappingW
        self._open_mapping.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
        self._open_mapping.restype = wintypes.HANDLE
        self._map_view = kernel32.MapViewOfFile
        self._map_view.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_size_t,
        )
        self._map_view.restype = ctypes.c_void_p
        self._unmap_view = kernel32.UnmapViewOfFile
        self._unmap_view.argtypes = (ctypes.c_void_p,)
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._find_window = user32.FindWindowW
        self._find_window.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        self._find_window.restype = wintypes.HWND
        self._is_window = user32.IsWindow
        self._is_window.argtypes = (wintypes.HWND,)
        self._client_rect = user32.GetClientRect
        self._client_rect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        self._client_to_screen = user32.ClientToScreen
        self._client_to_screen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
        self._foreground = user32.GetForegroundWindow
        self._foreground.restype = wintypes.HWND
        self._root_owner = user32.GetAncestor
        self._root_owner.argtypes = (wintypes.HWND, wintypes.UINT)
        self._root_owner.restype = wintypes.HWND

    def open_mapping(self, name: str, *, write: bool) -> int:
        """A handle to the named mapping; raises PalNotConnectedError if none."""
        access = FILE_MAP_READ | (FILE_MAP_WRITE if write else 0)
        handle = self._open_mapping(access, False, name)
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_FILE_NOT_FOUND:
                raise PalNotConnectedError(f"no mapping named {name!r}")
            raise PalNotConnectedError(f"OpenFileMapping({name!r}) failed, error {error}")
        return int(handle)

    def map_view(self, handle: int, size: int, *, write: bool) -> int:
        access = FILE_MAP_READ | (FILE_MAP_WRITE if write else 0)
        address = self._map_view(handle, access, 0, 0, size)
        if not address:
            raise PalNotConnectedError(f"MapViewOfFile failed, error {ctypes.get_last_error()}")
        return int(address)

    def unmap_view(self, address: int) -> None:
        self._unmap_view(address)

    def close_handle(self, handle: int) -> None:
        self._close_handle(handle)

    def find_window(self, title: str) -> int | None:
        hwnd = self._find_window(None, title)
        return int(hwnd) if hwnd else None

    def client_area(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """(x, y, width, height) of the window's client area on the screen;
        None once the window has gone."""
        if not self._is_window(hwnd):
            return None
        rect = self._wintypes.RECT()
        if not self._client_rect(hwnd, ctypes.byref(rect)):
            return None
        origin = self._wintypes.POINT(0, 0)
        if not self._client_to_screen(hwnd, ctypes.byref(origin)):
            return None
        return origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top

    def is_foreground(self, hwnd: int) -> bool:
        active = self._foreground()
        if not active:
            return False
        ga_root = 2
        return int(active) == hwnd or int(self._root_owner(active, ga_root) or 0) == hwnd

    def make_dpi_aware(self) -> None:
        """Window and screen coordinates in physical pixels, as mss grabs them."""
        with contextlib.suppress(AttributeError, OSError):
            per_monitor_aware = 2
            ctypes.WinDLL("shcore").SetProcessDpiAwareness(per_monitor_aware)
            return
        with contextlib.suppress(AttributeError, OSError):
            ctypes.WinDLL("user32").SetProcessDPIAware()


def _api(api: Win32Api | None) -> Win32Api:
    return api if api is not None else Win32Api()


class _Mapping:
    """A named mapping, opened and mapped; shared by the reader and writer."""

    def __init__(self, name: str, size: int, write: bool, api: Win32Api | None) -> None:
        self._name = name
        self._size = size
        self._write = write
        self._given_api = api
        self._api: Win32Api | None = None
        self._handle: int | None = None
        self.address: int | None = None

    @property
    def object_name(self) -> str:
        """As the plugin creates it: in the session's namespace."""
        return "Local\\" + self._name.lstrip("/")

    def open(self) -> None:
        self.close()
        self._api = _api(self._given_api)
        try:
            self._handle = self._api.open_mapping(self.object_name, write=self._write)
        except PalNotConnectedError as exc:
            raise PalNotConnectedError(
                f"{exc}: is the game running, with the plugin installed? With the "
                "game closed the launcher (ets2ad.bat) installs it; from a checkout, "
                "python scripts/install_plugin.py"
            ) from exc
        try:
            self.address = self._api.map_view(self._handle, self._size, write=self._write)
        except PalNotConnectedError:
            self.close()
            raise

    def close(self) -> None:
        if self._api is not None:
            if self.address is not None:
                with contextlib.suppress(Exception):
                    self._api.unmap_view(self.address)
            if self._handle is not None:
                with contextlib.suppress(Exception):
                    self._api.close_handle(self._handle)
        self.address = None
        self._handle = None


class WindowsSharedMemoryReader(SharedMemoryReader):
    """Maps the segment the telemetry plugin publishes, read-only.

    `read()` returns a copy, for the reason the Linux reader does: callers
    keep snapshots, and a view of live memory would change underneath them.
    """

    def __init__(self, name: str, size: int, api: Win32Api | None = None) -> None:
        self._mapping = _Mapping(name, size, write=False, api=api)
        self._size = size

    def open(self) -> None:
        self._mapping.open()

    def close(self) -> None:
        self._mapping.close()

    @property
    def is_open(self) -> bool:
        return self._mapping.address is not None

    @property
    def size(self) -> int:
        return self._size

    def read(self) -> bytes:
        if self._mapping.address is None:
            raise PalNotConnectedError("shared memory reader is not open")
        return ctypes.string_at(self._mapping.address, self._size)


class WindowsSharedMemoryWriter(SharedMemoryWriter):
    """Maps the plugin's command segment for writing; never creates it."""

    def __init__(self, name: str, size: int, api: Win32Api | None = None) -> None:
        self._mapping = _Mapping(name, size, write=True, api=api)
        self._size = size

    def open(self) -> None:
        self._mapping.open()

    def close(self) -> None:
        self._mapping.close()

    @property
    def is_open(self) -> bool:
        return self._mapping.address is not None

    @property
    def size(self) -> int:
        return self._size

    def write(self, data: bytes, offset: int = 0) -> None:
        if self._mapping.address is None:
            raise PalNotConnectedError("shared memory writer is not open")
        if offset < 0 or offset + len(data) > self._size:
            raise ValueError(
                f"write of {len(data)} bytes at offset {offset} does not fit "
                f"in a {self._size}-byte segment"
            )
        ctypes.memmove(self._mapping.address + offset, data, len(data))


class WindowsScreenCapture(ScreenCapture):
    """Captures the game window: found by title, grabbed with mss.

    The window's place is read again on every grab: the user may move it.
    """

    def __init__(self, window_title: str, api: Win32Api | None = None, grabber=None) -> None:
        self._window_title = window_title
        self._given_api = api
        self._given_grabber = grabber
        self._api: Win32Api | None = None
        self._hwnd: int | None = None
        self._grabber = None
        self._sequence = 0

    def open(self) -> None:
        if self.is_open:
            return
        api = _api(self._given_api)
        api.make_dpi_aware()
        hwnd = api.find_window(self._window_title)
        if hwnd is None:
            raise WindowNotFoundError(f"no window titled {self._window_title!r}")
        if self._given_grabber is not None:
            grabber = self._given_grabber
        else:
            import mss

            grabber = mss.MSS()
        self._api, self._hwnd, self._grabber = api, hwnd, grabber
        self._sequence = 0

    def close(self) -> None:
        if self._grabber is not None and self._grabber is not self._given_grabber:
            with contextlib.suppress(Exception):
                self._grabber.close()
        self._grabber = None
        self._hwnd = None
        self._api = None

    @property
    def is_open(self) -> bool:
        return self._grabber is not None

    @property
    def geometry(self) -> WindowGeometry:
        if self._api is None or self._hwnd is None:
            raise PalNotConnectedError("screen capture is not open")
        area = self._api.client_area(self._hwnd)
        if area is None:
            self.close()
            raise PalNotConnectedError("the game window has gone")
        x, y, width, height = area
        return WindowGeometry(x=x, y=y, width=width, height=height)

    @property
    def in_front(self) -> bool:
        """Whether the game is the foreground window."""
        if self._api is None or self._hwnd is None:
            raise PalNotConnectedError("screen capture is not open")
        return self._api.is_foreground(self._hwnd)

    def grab(self) -> Frame:
        if self._grabber is None:
            raise PalNotConnectedError("screen capture is not open")
        where = self.geometry
        if not self.in_front:
            raise WindowNotInFrontError(f"{self._window_title!r} is not the foreground window")
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


class WindowsHotkeyListener(HotkeyListener):
    """Not built: the stack engages with the truck's beacon, read from the
    telemetry, so no key needs hooking outside the game."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[HotkeyAction], None]] = []

    def open(self) -> None:
        raise PalUnsupportedError(_NOT_BUILT)

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return False

    def subscribe(self, callback: Callable[[HotkeyAction], None]) -> None:
        self._callbacks.append(callback)


def _registry_steam_root() -> Path | None:
    """Steam's root as Steam wrote it into the registry; None off Windows."""
    if sys.platform != "win32":
        return None
    import winreg

    with (
        contextlib.suppress(OSError),
        winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key,
    ):
        value, _ = winreg.QueryValueEx(key, "SteamPath")
        return Path(value)
    return None


def _documents() -> Path:
    """The user's Documents folder, where it has been moved to."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class _Guid(ctypes.Structure):
            _fields_ = (
                ("data1", wintypes.DWORD),
                ("data2", wintypes.WORD),
                ("data3", wintypes.WORD),
                ("data4", ctypes.c_ubyte * 8),
            )

        # FOLDERID_Documents {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
        folder = _Guid(
            0xFDD39AD0,
            0x238F,
            0x46AF,
            (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
        )
        found = ctypes.c_wchar_p()
        with contextlib.suppress(AttributeError, OSError):
            shell32 = ctypes.WinDLL("shell32")
            if (
                shell32.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(found))
                == 0
            ):
                path = Path(found.value)
                ctypes.WinDLL("ole32").CoTaskMemFree(found)
                return path
    return Path.home() / "Documents"


class WindowsGamePaths(GamePaths):
    """Locates the game through Steam's own metadata, as on Linux."""

    def __init__(self, steam_root: Path | None = None, app_id: int = ETS2_APP_ID) -> None:
        self._explicit_root = steam_root
        self._app_id = app_id

    @property
    def steam_root(self) -> Path:
        """The Steam installation root.

        Raises:
            PalNotConnectedError: no Steam installation was found.
        """
        candidates = (
            (self._explicit_root,)
            if self._explicit_root is not None
            else tuple(p for p in (_registry_steam_root(), _STEAM_FALLBACK) if p is not None)
        )
        for candidate in candidates:
            if (candidate / "steamapps").is_dir():
                return candidate
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
        return self.game_root / "bin" / "win_x64"

    @property
    def plugins_dir(self) -> Path:
        return self.binary_dir / "plugins"

    @property
    def user_dir(self) -> Path:
        """`Documents\\Euro Truck Simulator 2`: profiles, logs, mods."""
        return _documents() / "Euro Truck Simulator 2"

    def archives(self) -> Sequence[Path]:
        """Base-game and DLC `.scs` archives, in the game's own load order
        (as `LinuxGamePaths.archives`: the same files on either platform)."""
        root = self.game_root
        if not root.is_dir():
            raise PalNotConnectedError(f"game root does not exist: {root}")
        priority = ("base.scs", "base_share.scs", "base_vehicle.scs", "core.scs", "def.scs")
        ordered = [root / name for name in priority if (root / name).is_file()]
        rest = sorted(p for p in root.glob("*.scs") if p not in ordered)
        return tuple(ordered + rest)


class WindowsOverlayWindow(OverlayWindow):
    """Not built: nothing is drawn over the game."""

    def open(self) -> None:
        raise PalUnsupportedError("no overlay: nothing is drawn over the game")

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return False

    def draw(self, image: NDArray[np.uint8]) -> None:
        raise PalUnsupportedError("no overlay: nothing is drawn over the game")


class WindowsBackend(Backend):
    """Assembles the Windows implementations."""

    name = "windows"

    def shared_memory(self, name: str, size: int) -> SharedMemoryReader:
        return WindowsSharedMemoryReader(name, size)

    def shared_memory_writer(self, name: str, size: int) -> SharedMemoryWriter:
        return WindowsSharedMemoryWriter(name, size)

    def screen_capture(self, window_title: str) -> ScreenCapture:
        return WindowsScreenCapture(window_title)

    def hotkeys(self) -> HotkeyListener:
        return WindowsHotkeyListener()

    def game_paths(self) -> GamePaths:
        return WindowsGamePaths()

    def overlay(self) -> OverlayWindow:
        return WindowsOverlayWindow()
