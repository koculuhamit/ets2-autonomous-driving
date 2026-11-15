"""Turning a screen grab into the image the rest of the stack sees."""

from __future__ import annotations

import numpy as np
import pytest

from pal.linux import bgra_to_bgr


def test_bgra_becomes_bgr_with_the_alpha_dropped() -> None:
    bgra = np.zeros((2, 3, 4), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
    bgr = bgra_to_bgr(bgra.tobytes(), 2, 3)
    assert bgr.shape == (2, 3, 3)
    assert bgr.dtype == np.uint8
    assert (bgr[..., 0] == 10).all() and (bgr[..., 1] == 20).all() and (bgr[..., 2] == 30).all()


def test_the_image_is_contiguous_so_a_model_can_take_it_as_it_is() -> None:
    bgr = bgra_to_bgr(np.arange(4 * 5 * 4, dtype=np.uint8).tobytes(), 4, 5)
    assert bgr.flags["C_CONTIGUOUS"]


def test_pixels_keep_their_place() -> None:
    bgra = np.random.default_rng(0).integers(0, 256, (7, 11, 4), dtype=np.uint8)
    assert (bgra_to_bgr(bgra.tobytes(), 7, 11) == bgra[..., :3]).all()


def test_a_grab_of_the_wrong_size_is_refused_not_reshaped_into_nonsense() -> None:
    """A window resized between reading its size and grabbing it gives this."""
    with pytest.raises(ValueError, match="not a 5x4 BGRA image"):
        bgra_to_bgr(bytes(4 * 4 * 4), 4, 5)


class _Prop:
    def __init__(self, value) -> None:
        self.value = value


class _Tree:
    def __init__(self, parent, root) -> None:
        self.parent, self.root = parent, root


class _Window:
    def __init__(self, wid: int, parent: _Window | None = None, active=None) -> None:
        self.id, self.parent, self._active = wid, parent, active

    def query_tree(self) -> _Tree:
        root = self
        while root.parent is not None:
            root = root.parent
        return _Tree(self.parent, root)

    def get_full_property(self, atom, kind):
        return self._active


class _Display:
    def __init__(self, root: _Window) -> None:
        self._root = root

    def screen(self):
        return type("Screen", (), {"root": self._root})()

    def intern_atom(self, name: str) -> str:
        return name


def _tree(active):
    """root(1) > frame(2) > game(3), with `active` as _NET_ACTIVE_WINDOW."""
    root = _Window(1, active=active)
    frame = _Window(2, parent=root)
    return _Display(root), _Window(3, parent=frame)


def test_the_game_is_in_front_when_it_is_the_active_window() -> None:
    from pal.linux import is_active

    display, game = _tree(_Prop([3]))
    assert is_active(display, game)


def test_the_game_is_in_front_when_the_frame_around_it_is_active() -> None:
    from pal.linux import is_active

    display, game = _tree(_Prop([2]))
    assert is_active(display, game)


def test_another_window_in_front_is_seen() -> None:
    """What happened: a grab with a browser over the game returned the browser."""
    from pal.linux import is_active

    display, game = _tree(_Prop([99]))
    assert not is_active(display, game)


def test_the_root_is_not_taken_for_a_frame_of_the_game() -> None:
    from pal.linux import is_active

    display, game = _tree(_Prop([1]))
    assert not is_active(display, game)


def test_no_active_window_at_all_is_not_the_game() -> None:
    from pal.linux import is_active

    display, game = _tree(_Prop([0]))
    assert not is_active(display, game)


def test_a_window_manager_that_does_not_say_is_trusted() -> None:
    from pal.linux import is_active

    display, game = _tree(None)
    assert is_active(display, game)


def test_a_window_past_the_screen_edge_is_off_screen() -> None:
    """A 1920x1080 window pushed down 27 px by the top bar on a 1920x1080
    screen: the grab failed, and every drive stopped for "no view"."""
    from pal.base import WindowGeometry
    from pal.linux import off_screen

    assert off_screen(WindowGeometry(x=1920, y=27, width=1920, height=1080), 3840, 1080)
    assert off_screen(WindowGeometry(x=-5, y=0, width=800, height=600), 1920, 1080)
    assert not off_screen(WindowGeometry(x=1920, y=0, width=1920, height=1080), 3840, 1080)
    assert not off_screen(WindowGeometry(x=0, y=0, width=1920, height=1080), 1920, 1080)


def test_the_plugin_log_is_looked_for_where_the_plugin_writes_it() -> None:
    """In the user's runtime directory first, not under a name another
    account could take in the shared /tmp (plugin/src/diagnostics.cpp)."""
    from pathlib import Path

    from pal.linux import plugin_log

    name = "ets2ad_plugin.log"
    assert plugin_log({"XDG_RUNTIME_DIR": "/run/user/1000", "TMPDIR": "/x"}) == Path(
        "/run/user/1000", name
    )
    assert plugin_log({"TMPDIR": "/x"}) == Path("/x", name)
    assert plugin_log({"XDG_RUNTIME_DIR": "", "TMPDIR": ""}) == Path("/tmp", name)
