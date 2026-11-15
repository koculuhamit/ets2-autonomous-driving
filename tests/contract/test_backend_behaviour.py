"""Behavioural contract, shared by every backend that can actually run.

Each suite below is written once against the interface and inherited by one
class per backend. The fake runs everywhere; the Linux classes are marked
`hardware` and skip themselves when the capability is not available, so the
same assertions cover the real backend the moment it is implemented.

That inheritance is the point. A behavioural difference between the fake and
the real backend is exactly the kind of bug that makes offline replay lie, and
the only defence is running one test set against both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pal.base import (
    Backend,
    GamePaths,
    HotkeyAction,
    PalNotConnectedError,
    ScreenCapture,
    SharedMemoryReader,
)
from pal.fake import FakeBackend, FakeGamePaths, FakeHotkeyListener
from pal.linux import LinuxBackend

pytestmark = pytest.mark.filterwarnings("error")


# --------------------------------------------------------------------------- #
# Shared suites
# --------------------------------------------------------------------------- #


class SharedMemoryContract:
    """Behaviour every `SharedMemoryReader` must exhibit."""

    @pytest.fixture
    def reader(self, backend: Backend) -> SharedMemoryReader:
        raise NotImplementedError

    def test_read_before_open_raises(self, reader: SharedMemoryReader) -> None:
        with pytest.raises(PalNotConnectedError):
            reader.read()

    def test_read_returns_exactly_size_bytes(self, reader: SharedMemoryReader) -> None:
        with reader:
            blob = reader.read()
        assert isinstance(blob, bytes)
        assert len(blob) == reader.size

    def test_read_after_close_raises(self, reader: SharedMemoryReader) -> None:
        with reader:
            reader.read()
        with pytest.raises(PalNotConnectedError):
            reader.read()


class ScreenCaptureContract:
    """Behaviour every `ScreenCapture` must exhibit."""

    @pytest.fixture
    def capture(self, backend: Backend) -> ScreenCapture:
        raise NotImplementedError

    def test_grab_before_open_raises(self, capture: ScreenCapture) -> None:
        with pytest.raises(PalNotConnectedError):
            capture.grab()

    def test_frame_matches_geometry_and_is_bgr_uint8(self, capture: ScreenCapture) -> None:
        with capture:
            geometry = capture.geometry
            frame = capture.grab()
        assert frame.image.dtype == "uint8"
        assert frame.image.ndim == 3 and frame.image.shape[2] == 3
        assert frame.shape == (geometry.height, geometry.width)

    def test_sequence_increases_and_timestamps_advance(self, capture: ScreenCapture) -> None:
        with capture:
            first = capture.grab()
            second = capture.grab()
        assert second.sequence > first.sequence
        assert second.timestamp >= first.timestamp


class GamePathsContract:
    """Behaviour every `GamePaths` must exhibit."""

    @pytest.fixture
    def paths(self, backend: Backend) -> GamePaths:
        raise NotImplementedError

    def test_paths_are_absolute(self, paths: GamePaths) -> None:
        for path in (paths.game_root, paths.binary_dir, paths.plugins_dir, paths.user_dir):
            assert path.is_absolute(), f"{path} is not absolute"

    def test_plugins_dir_is_under_binary_dir(self, paths: GamePaths) -> None:
        assert paths.plugins_dir.parent == paths.binary_dir

    def test_binary_dir_is_under_game_root(self, paths: GamePaths) -> None:
        assert paths.game_root in paths.binary_dir.parents

    def test_archives_are_scs_files_under_the_game_root(self, paths: GamePaths) -> None:
        archives = paths.archives()
        assert all(a.suffix == ".scs" for a in archives)
        assert all(paths.game_root in a.parents for a in archives)

    def test_archive_order_is_stable(self, paths: GamePaths) -> None:
        """Later archives override earlier ones, and the map cache is keyed on
        the result -- an unstable order would silently invalidate it."""
        assert list(paths.archives()) == list(paths.archives())


# --------------------------------------------------------------------------- #
# Fake backend
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_backend(tmp_path: Path) -> FakeBackend:
    backend = FakeBackend(root=tmp_path)
    backend.game_paths().create()
    return backend


class TestFakeSharedMemory(SharedMemoryContract):
    @pytest.fixture
    def reader(self, fake_backend: FakeBackend) -> SharedMemoryReader:
        return fake_backend.shared_memory("scs_telemetry", 64)

    def test_simulated_disconnect_is_recoverable(self, reader: SharedMemoryReader) -> None:
        """Losing the game mid-run must look like a disconnect, not a crash."""
        reader.open()
        reader.read()
        reader.fail_next = True
        with pytest.raises(PalNotConnectedError):
            reader.read()
        assert reader.is_open is False
        reader.open()
        assert len(reader.read()) == reader.size


class TestFakeScreenCapture(ScreenCaptureContract):
    @pytest.fixture
    def capture(self, fake_backend: FakeBackend) -> ScreenCapture:
        return fake_backend.screen_capture("Euro Truck Simulator 2")

    def test_geometry_change_is_reflected_in_frames(self, capture: ScreenCapture) -> None:
        """The user can move or resize the window mid-run; capture must follow."""
        from pal.base import WindowGeometry

        with capture:
            capture.set_geometry(WindowGeometry(10, 20, 32, 24))
            frame = capture.grab()
        assert frame.shape == (24, 32)


class TestFakeGamePaths(GamePathsContract):
    @pytest.fixture
    def paths(self, fake_backend: FakeBackend) -> GamePaths:
        return fake_backend.game_paths()

    def test_created_layout_exists_on_disk(self, paths: FakeGamePaths) -> None:
        assert paths.plugins_dir.is_dir()
        assert (paths.user_dir / "config.cfg").is_file()
        assert all(a.is_file() for a in paths.archives())


def test_fake_hotkeys_deliver_only_when_injected() -> None:
    received: list[HotkeyAction] = []
    listener = FakeHotkeyListener()
    listener.subscribe(received.append)
    with listener:
        listener.inject(HotkeyAction.ENGAGE)
        listener.inject(HotkeyAction.PANIC)
    assert received == [HotkeyAction.ENGAGE, HotkeyAction.PANIC]


def test_fake_backend_returns_stable_instances() -> None:
    """Tests configure a fake, hand the backend to production code, then
    inspect what that code did -- which only works if it is the same object."""
    backend = FakeBackend()
    assert backend.hotkeys() is backend.hotkeys()
    assert backend.screen_capture("x") is backend.screen_capture("x")
    assert backend.screen_capture("x") is not backend.screen_capture("y")


# --------------------------------------------------------------------------- #
# Linux backend
# --------------------------------------------------------------------------- #

requires_linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux backend")


@requires_linux
@pytest.mark.hardware
class TestLinuxGamePaths(GamePathsContract):
    @pytest.fixture
    def paths(self) -> GamePaths:
        game_paths = LinuxBackend().game_paths()
        try:
            _ = game_paths.game_root
        except PalNotConnectedError as exc:
            pytest.skip(f"Euro Truck Simulator 2 is not installed: {exc}")
        return game_paths


@requires_linux
@pytest.mark.hardware
def test_the_game_window_can_be_captured() -> None:
    """Needs an X11 session and the game running; skipped otherwise."""
    from pal.base import PalUnsupportedError, WindowNotFoundError
    from pal.linux import LinuxScreenCapture

    capture = LinuxScreenCapture("Euro Truck Simulator 2")
    try:
        capture.open()
    except (PalUnsupportedError, WindowNotFoundError) as exc:
        pytest.skip(str(exc))
    if not capture.in_front:
        capture.close()
        pytest.skip("the game is not the active window")
    try:
        where = capture.geometry
        first, second = capture.grab(), capture.grab()
    finally:
        capture.close()

    assert first.image.shape == (where.height, where.width, 3)
    assert first.image.dtype == np.uint8
    assert second.sequence == first.sequence + 1
    assert second.timestamp > first.timestamp
    assert first.image.mean() > 5, "an all-black frame: the window is covered or not drawn"
