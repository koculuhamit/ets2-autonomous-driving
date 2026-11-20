"""Whether the game's camera is in the cab."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.perception.cab import DashboardLearner, dashboard_edges, in_cab, load_reference


def dashboard() -> np.ndarray:
    """A frame whose dashboard has bright gauges on dark plastic."""
    image = np.full((1080, 1920, 3), 25, dtype=np.uint8)
    for x in range(650, 1450, 160):
        image[800:860, x : x + 60] = 200
    return image


def test_the_cab_s_own_dashboard_is_the_cab() -> None:
    frame = dashboard()
    assert in_cab(frame, dashboard_edges(frame))


def test_road_where_the_dashboard_should_be_is_not() -> None:
    """From the chase camera, in Bern: the truck's own trailer and the road."""
    road = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    road[:, 900:1000] = 230  # a lane line
    assert not in_cab(road, dashboard_edges(dashboard()))


def test_a_dark_dashboard_at_night_is_the_cab_whatever_its_edges() -> None:
    night = np.full((1080, 1920, 3), 10, dtype=np.uint8)
    assert in_cab(night, dashboard_edges(dashboard()))


def test_the_recorded_dashboard_is_shipped() -> None:
    reference = load_reference()
    assert reference is not None and reference.shape == (48, 120)


def _view(seed: int, noise_seed: int) -> np.ndarray:
    """A lit frame: with seed 1 a dashboard of gauges in its lower middle,
    with 2 the road and a trailer's stripes from the chase camera."""
    pattern = np.random.default_rng(seed)
    image = np.full((1080, 1920, 3), 90, np.uint8)
    if seed == 1:
        for _ in range(40):
            x, y = int(pattern.integers(560, 1400)), int(pattern.integers(680, 1050))
            cv2.circle(image, (x, y), 30, (220, 220, 220), 3)
    else:
        for y in range(680, 1060, 45):
            cv2.line(image, (500, y), (1500, y + 120), (220, 220, 220), 9)
    noise = np.random.default_rng(noise_seed).integers(0, 12, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_a_truck_s_own_dashboard_is_learned_from_its_drive() -> None:
    learner = DashboardLearner(frames=12, every_s=0.3)
    learned = None
    for k in range(12):
        learned = learner.offer(_view(1, k), k * 0.3)
    assert learned is not None
    assert in_cab(_view(1, 99), learned)
    assert not in_cab(_view(2, 99), learned), "another view is not the cab's"


def test_frames_closer_than_the_spacing_are_not_counted() -> None:
    learner = DashboardLearner(frames=3, every_s=0.3)
    assert all(learner.offer(_view(1, k), k * 0.01) is None for k in range(10))


def test_a_view_that_changed_half_way_is_not_learned() -> None:
    """The camera switched while learning: no dashboard from the two."""
    learner = DashboardLearner(frames=12, every_s=0.3)
    results = [learner.offer(_view(1 if k < 6 else 2, k), k * 0.3) for k in range(12)]
    assert all(result is None for result in results)
    # Then the second view goes on: it is what is learned.
    for k in range(12, 24):
        learned = learner.offer(_view(2, k), k * 0.3)
    assert learned is not None and in_cab(_view(2, 99), learned)


def _drive_top(roof_row_px: int, frames: int = 30, seed: int = 0) -> np.ndarray:
    """(frames, 400, cols) grey crops of the frame's top: a still dark roof
    down to `roof_row_px`, a moving view below it."""
    from core.perception.cab import ROOF_BAND

    rng = np.random.default_rng(seed)
    cols = int((ROOF_BAND[3] - ROOF_BAND[2]) * 1920)
    out = np.empty((frames, ROOF_BAND[1] - ROOF_BAND[0], cols), np.uint8)
    for k in range(frames):
        view = rng.integers(40, 230, (out.shape[1], cols))
        view[:roof_row_px] = 20 + (k % 2)
        out[k] = view
    return out


def test_the_roof_s_edge_is_where_the_view_begins_to_move() -> None:
    from core.perception.cab import roof_row

    assert roof_row(_drive_top(162)) == 162
    assert roof_row(_drive_top(250)) == 250


def test_a_still_view_shows_no_roof() -> None:
    from core.perception.cab import roof_row

    still = np.repeat(_drive_top(162, frames=1), 30, axis=0)
    assert roof_row(still) is None


def test_a_view_is_only_ever_widened_by_what_is_learned() -> None:
    """From a minute at a depot an Actros's came out at 286: its view, from
    250, stays; a DAF XD's 170 widens it."""
    from core.perception.detection import DEFAULT_CAB_VIEW
    from core.perception.service import _widened

    assert _widened(DEFAULT_CAB_VIEW, 286, 1080) is None
    assert _widened(DEFAULT_CAB_VIEW, 245, 1080) is None, "no closer than 10 px: nothing gained"
    assert _widened(DEFAULT_CAB_VIEW, 40, 1080) is None, "the mirror insets' rows"
    wider = _widened(DEFAULT_CAB_VIEW, 170, 1080)
    assert wider.windscreen.top == pytest.approx(170 / 1080)
    assert wider.windscreen.bottom == DEFAULT_CAB_VIEW.windscreen.bottom
    assert wider.covered == DEFAULT_CAB_VIEW.covered
