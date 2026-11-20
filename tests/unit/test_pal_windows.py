"""The Windows backend's logic, around stand-ins for the Win32 calls.

None of this proves the Win32 calls themselves: that needs Windows. It
proves what is done with them -- the names, the copies, the bounds, the
refusals -- on any platform.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from pal.base import (
    PalNotConnectedError,
    PalUnsupportedError,
    WindowNotFoundError,
    WindowNotInFrontError,
)
from pal.windows import (
    Win32Api,
    WindowsGamePaths,
    WindowsScreenCapture,
    WindowsSharedMemoryReader,
    WindowsSharedMemoryWriter,
)
from tests.unit.test_steam import _make_library


class FakeApi:
    """Mappings as ctypes buffers, one window, a foreground window."""

    def __init__(self) -> None:
        self.mappings: dict[str, ctypes.Array] = {}
        self.opened: list[tuple[str, bool]] = []
        self.unmapped: list[int] = []
        self.closed: list[int] = []
        self.window: int | None = 7
        self.area: tuple[int, int, int, int] | None = (100, 50, 4, 3)
        self.foreground = 7

    def open_mapping(self, name: str, *, write: bool) -> int:
        self.opened.append((name, write))
        if name not in self.mappings:
            raise PalNotConnectedError(f"no mapping named {name!r}")
        return 1000 + list(self.mappings).index(name)

    def map_view(self, handle: int, size: int, *, write: bool) -> int:
        return ctypes.addressof(list(self.mappings.values())[handle - 1000])

    def unmap_view(self, address: int) -> None:
        self.unmapped.append(address)

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)

    def find_window(self, title: str) -> int | None:
        return self.window if title == "Euro Truck Simulator 2" else None

    def client_area(self, hwnd: int):
        return self.area

    def is_foreground(self, hwnd: int) -> bool:
        return self.foreground == hwnd

    def make_dpi_aware(self) -> None:
        return None


def test_the_win32_api_is_refused_off_windows() -> None:
    if __import__("sys").platform == "win32":
        pytest.skip("on Windows the API is real")
    with pytest.raises(PalUnsupportedError):
        Win32Api()


def test_the_plugin_s_mapping_is_read_by_its_local_name_as_a_copy() -> None:
    api = FakeApi()
    block = ctypes.create_string_buffer(b"telemetry!", 16)
    api.mappings["Local\\ets2ad_telemetry"] = block
    with WindowsSharedMemoryReader("ets2ad_telemetry", 10, api=api) as reader:
        assert reader.is_open
        first = reader.read()
        block[0] = b"T"
        assert first == b"telemetry!", "a copy, not a view"
        assert reader.read() == b"Telemetry!"
    assert api.opened == [("Local\\ets2ad_telemetry", False)]
    assert api.unmapped and api.closed == [1000]
    assert not reader.is_open


def test_no_game_is_not_connected_with_how_to_install() -> None:
    reader = WindowsSharedMemoryReader("ets2ad_telemetry", 10, api=FakeApi())
    with pytest.raises(PalNotConnectedError, match="install_plugin"):
        reader.open()
    assert not reader.is_open
    with pytest.raises(PalNotConnectedError):
        reader.read()


def test_commands_are_written_into_the_plugin_s_mapping_within_its_size() -> None:
    api = FakeApi()
    block = ctypes.create_string_buffer(8)
    api.mappings["Local\\ets2ad_command"] = block
    with WindowsSharedMemoryWriter("ets2ad_command", 8, api=api) as writer:
        writer.write(b"\x01\x02", offset=3)
        assert block.raw == b"\x00\x00\x00\x01\x02\x00\x00\x00"
        with pytest.raises(ValueError):
            writer.write(b"123456789")
    assert api.opened == [("Local\\ets2ad_command", True)]


def test_the_window_is_grabbed_where_it_is_and_only_in_front() -> None:
    api = FakeApi()

    class Shot:
        raw = bytes(np.arange(4 * 3 * 4, dtype=np.uint8))

    class Grabber:
        def __init__(self) -> None:
            self.regions: list[dict] = []

        def grab(self, region):
            self.regions.append(region)
            return Shot()

    grabber = Grabber()
    with WindowsScreenCapture("Euro Truck Simulator 2", api=api, grabber=grabber) as capture:
        frame = capture.grab()
        assert frame.image.shape == (3, 4, 3)
        assert grabber.regions == [{"left": 100, "top": 50, "width": 4, "height": 3}]
        api.foreground = 99
        with pytest.raises(WindowNotInFrontError):
            capture.grab()
        api.area = None
        with pytest.raises(PalNotConnectedError):
            capture.grab()
    assert not capture.is_open


def test_the_picture_is_made_at_the_size_it_was_grabbed() -> None:
    """With the display scaled, the grab and the window's size can differ."""
    api = FakeApi()

    class Shot:
        width, height = 8, 6
        raw = bytes(8 * 6 * 4)

    class Grabber:
        def grab(self, region):
            return Shot()

    with WindowsScreenCapture("Euro Truck Simulator 2", api=api, grabber=Grabber()) as capture:
        assert capture.grab().image.shape == (6, 8, 3)


def test_a_minimised_window_is_not_in_front() -> None:
    api = FakeApi()
    api.area = (100, 50, 0, 0)

    class Grabber:
        def grab(self, region):
            raise AssertionError("not grabbed")

    with (
        WindowsScreenCapture("Euro Truck Simulator 2", api=api, grabber=Grabber()) as capture,
        pytest.raises(WindowNotInFrontError, match="minimised"),
    ):
        capture.grab()


def test_no_game_window_is_not_found() -> None:
    capture = WindowsScreenCapture("Something Else", api=FakeApi(), grabber=object())
    with pytest.raises(WindowNotFoundError):
        capture.open()


def test_the_game_is_found_through_steam_s_libraries(tmp_path: Path) -> None:
    _make_library(tmp_path, 227300, "Euro Truck Simulator 2")
    paths = WindowsGamePaths(steam_root=tmp_path)
    assert paths.game_root == tmp_path / "steamapps" / "common" / "Euro Truck Simulator 2"
    assert paths.plugins_dir == paths.game_root / "bin" / "win_x64" / "plugins"
    (paths.game_root / "def.scs").touch()
    (paths.game_root / "base.scs").touch()
    (paths.game_root / "dlc_east.scs").touch()
    assert [p.name for p in paths.archives()] == ["base.scs", "def.scs", "dlc_east.scs"]


def test_no_steam_is_not_connected(tmp_path: Path) -> None:
    with pytest.raises(PalNotConnectedError):
        WindowsGamePaths(steam_root=tmp_path).game_root  # noqa: B018
