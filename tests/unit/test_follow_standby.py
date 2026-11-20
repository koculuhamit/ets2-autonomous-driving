"""Waiting for the beacon: word from the mission, and a game started anew."""

from __future__ import annotations

from pathlib import Path

from core.bridge.layout import BLOCK_SIZE
from core.bridge.reader import TelemetryReader
from pal import PalNotConnectedError
from pal.fake import FakeBackend, FakeSharedMemoryReader
from tests.unit.test_telemetry_reader import make_block
from tools.follow import reattached, take_word


def reader_on(count: int) -> TelemetryReader:
    """A reader open on a block whose publish counter stands at `count`."""
    backend = FakeBackend()
    block = make_block(write_count=count)
    backend.set_shared_memory("ets2ad_telemetry", FakeSharedMemoryReader(BLOCK_SIZE, [block]))
    reader = TelemetryReader(backend)
    reader.open()
    reader.read()
    return reader


class _Gone:
    def is_running(self) -> bool:
        return False


class _Here:
    def is_running(self) -> bool:
        return True


def test_a_game_started_anew_is_found_on_its_new_block() -> None:
    """The game closed and opened again: the standby's block was the old one,
    unlinked, and the beacon's word never got the truck moving."""
    old = reader_on(187858)
    fresh = reattached(old, opener=lambda: reader_on(12))
    assert fresh is not None and fresh.is_open
    assert fresh.last_write_count == 12


def test_a_paused_game_keeps_its_reader() -> None:
    old = reader_on(500)
    assert reattached(old, opener=lambda: reader_on(500)) is None


def test_no_game_at_all_keeps_the_reader_waiting() -> None:
    class Closed(TelemetryReader):
        def open(self) -> None:
            raise PalNotConnectedError("the game is not running")

    old = reader_on(500)
    assert reattached(old, opener=lambda: Closed(FakeBackend())) is None


def test_waiting_out_a_still_game_leaves_a_go_for_later(tmp_path: Path) -> None:
    word_file = tmp_path / "word"
    word_file.write_text("go job /tmp/drive-01", encoding="utf-8")
    assert take_word(word_file, _Here(), only="stop") is None
    assert word_file.exists(), "the go is kept"
    assert take_word(word_file, _Here()) == "go job /tmp/drive-01"
    assert not word_file.exists()


def test_stop_is_taken_even_while_waiting(tmp_path: Path) -> None:
    word_file = tmp_path / "word"
    word_file.write_text("stop\n", encoding="utf-8")
    assert take_word(word_file, None, only="stop") == "stop"
    assert not word_file.exists()


def test_the_mission_gone_is_a_stop_and_no_word_is_none(tmp_path: Path) -> None:
    assert take_word(tmp_path / "none", _Gone()) == "stop"
    assert take_word(tmp_path / "none", _Here()) is None


class _Perception:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.stopped = False

    def hold(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


class _StillReader:
    """A reader whose game stands still: every read is stale."""

    last_write_count = 7

    def open(self) -> None:
        pass

    def wait_for_frame(self) -> None:
        pass

    def read(self):
        from core.bridge.reader import StaleTelemetryError

        raise StaleTelemetryError("the game is paused")

    def close(self) -> None:
        pass


def _standby(monkeypatch, tmp_path: Path, perceptions: list[_Perception], fresh, words: list):
    """Run `standby_until_go` against fakes: the perceptions it starts in
    order, `fresh` for `reattached`, the words `take_word` returns."""
    import types

    import tools.follow as follow

    started: list[_Perception] = []

    def start_perception(*_args, **_kwargs):
        started.append(perceptions[len(started)])
        return started[-1], object()

    first = _StillReader()
    monkeypatch.setattr(follow, "TelemetryReader", lambda: first)
    monkeypatch.setattr(follow, "truck_id", lambda _block: "truck")
    monkeypatch.setattr(follow, "load_profile", lambda truck: follow.TruckProfile(truck))
    monkeypatch.setattr(follow, "start_perception", start_perception)
    monkeypatch.setattr(follow, "reattached", fresh)
    monkeypatch.setattr(follow, "take_word", lambda *_a, **_k: words.pop(0) if words else None)
    monkeypatch.setattr(follow.time, "sleep", lambda _s: None)
    reads = iter([types.SimpleNamespace(block=None)])  # the truck's, at the start

    def first_read():
        snapshot = next(reads, None)
        if snapshot is None:
            raise follow.StaleTelemetryError("still")
        return snapshot

    first.read = first_read
    args = types.SimpleNamespace(
        standby=tmp_path / "word", camera="camera.json", model=None, allow_cpu=True, cache=None
    )
    return follow.standby_until_go(args), started


def test_a_game_started_anew_starts_perception_anew(monkeypatch, tmp_path: Path) -> None:
    """The capture held the old game's window: switched on, the drive
    stopped at once, "no view of the road"."""
    old, new = _Perception(), _Perception()
    fresh = iter([_StillReader()])
    code, started = _standby(
        monkeypatch, tmp_path, [old, new], lambda _r: next(fresh, None), [None, "stop"]
    )
    assert code == 0
    assert started == [old, new], "started again on the new game"
    assert old.stopped


def test_a_perception_child_gone_is_started_again(monkeypatch, tmp_path: Path) -> None:
    dead, again = _Perception(alive=False), _Perception()
    code, started = _standby(
        monkeypatch, tmp_path, [dead, again], lambda _r: None, [None, None, "stop"]
    )
    assert code == 0
    assert started == [dead, again]
    assert dead.stopped


def test_no_view_says_what_the_capture_said() -> None:
    """A game window run off the screen was asked "is the game in front?"."""
    from types import SimpleNamespace

    from tools.follow import no_view

    def service(report):
        return SimpleNamespace(latest=lambda: report)

    off = SimpleNamespace(covered_why="the game window reaches past the screen")
    assert no_view(service(off)) == "no view of the road: the game window reaches past the screen"
    assert no_view(service(None)).endswith("is the game in front?")
    assert no_view(service(SimpleNamespace(covered_why=""))).endswith("is the game in front?")


def test_a_truck_loaded_after_the_standby_began_gets_perception_for_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Read once at the start, with no truck in the game yet, the profile
    stayed the default's: a DAF XD drove on the Actros's camera and ran into
    the lorry ahead. The truck that comes is seen, and perception is
    started again with its camera."""
    import types

    import tools.follow as follow

    blank, daf = _Perception(), _Perception()
    started, loaded = [], []

    def start_perception(*_args, truck=None, **_kwargs):
        started.append(([blank, daf][len(started)], truck.truck))
        return started[-1][0], object()

    def load_profile(truck):
        loaded.append(truck)
        return follow.TruckProfile(truck)

    reader = _StillReader()
    reads = iter(
        [types.SimpleNamespace(block="unknown.unknown"), types.SimpleNamespace(block="daf.xd")]
    )

    def read():
        snapshot = next(reads, None)
        if snapshot is None:
            raise follow.StaleTelemetryError("still")
        return snapshot

    reader.read = read
    words = [None, None, "stop"]
    monkeypatch.setattr(follow, "TelemetryReader", lambda: reader)
    monkeypatch.setattr(follow, "truck_id", lambda block: block)
    monkeypatch.setattr(follow, "load_profile", load_profile)
    monkeypatch.setattr(follow, "start_perception", start_perception)
    monkeypatch.setattr(follow, "reattached", lambda _r: None)
    monkeypatch.setattr(follow, "take_word", lambda *_a, **_k: words.pop(0) if words else "stop")
    monkeypatch.setattr(follow.time, "sleep", lambda _s: None)
    args = types.SimpleNamespace(
        standby=tmp_path / "word", camera="camera.json", model=None, allow_cpu=True, cache=None
    )
    assert follow.standby_until_go(args) == 0
    assert loaded == ["unknown.unknown", "daf.xd"]
    assert [truck for _, truck in started] == ["unknown.unknown", "daf.xd"]
    assert blank.stopped, "the default's perception let go"
