"""Stopping the truck while the game shows nothing: braked blind until it is back."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE
from core.bridge.reader import StaleTelemetryError


class _Gone:
    """A reader whose game shows nothing for `gone` reads, then a truck at
    `speed`, stopping by `slowing` a read, its parking brake set at rest."""

    def __init__(self, gone: int, speed: float = 8.6, slowing: float = 2.0) -> None:
        self.gone, self.speed, self.slowing, self.reads = gone, speed, slowing, 0

    def read(self):
        self.reads += 1
        if self.reads <= self.gone:
            raise StaleTelemetryError("no new frame for 1.0s; the game is not running")
        block = np.zeros(1, dtype=BLOCK_DTYPE)[0]
        block["truck"]["speed"] = self.speed
        block["truck"]["parking_brake"] = 1 if self.speed == 0.0 else 0
        self.speed = max(self.speed - self.slowing, 0.0)
        return type("Snapshot", (), {"block": block})()


class _Commander:
    def __init__(self) -> None:
        self.sent = []

    def send(self, command, *, engaged: bool = True) -> None:
        self.sent.append(command)


class _Guard:
    def suspend(self):
        return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    import tools.follow as follow

    monkeypatch.setattr(follow, "LOOP_INTERVAL", 0.0)


def test_while_the_game_shows_nothing_the_truck_is_braked_and_stopped_when_it_is_back() -> None:
    """Near Venlo (18 November) the game stood still for a minute at 31 km/h
    short of a junction; the stack gave up after a second, and the truck came
    back to nobody, and crashed."""
    import tools.follow as follow

    reader, commander = _Gone(gone=50), _Commander()
    follow._brake_to_rest(commander, reader, _Guard())
    blind = commander.sent[:50]
    assert len(blind) == 50 and all(c.brake > 0.0 for c in blind), "braked while it was gone"
    assert reader.speed == 0.0, "and stopped once it was back"


def test_a_game_gone_for_good_is_given_up_on(monkeypatch) -> None:
    import tools.follow as follow

    monkeypatch.setattr(follow, "GAME_BACK_WAIT_S", 0.0)
    with pytest.raises(StaleTelemetryError):
        follow._brake_to_rest(_Commander(), _Gone(gone=10**6), _Guard())
