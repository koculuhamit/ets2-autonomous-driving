"""Whether the game's camera is in the cab."""

from __future__ import annotations

import numpy as np

from core.perception.cab import dashboard_edges, in_cab, load_reference


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
