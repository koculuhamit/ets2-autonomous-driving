"""Handing the truck back: hazard lights once, then the horn twice."""

from __future__ import annotations

from core.common.controls import Button
from core.control.handover import HANDOVER_S, handover_buttons


def test_the_hazards_are_switched_on_once_then_the_horn_sounds_twice() -> None:
    held = [handover_buttons(k * 0.05) for k in range(int(HANDOVER_S / 0.05) + 2)]
    presses = []
    for before, now in zip([frozenset(), *held], held, strict=False):
        presses.extend(sorted(now - before))
    assert presses == [Button.HAZARD, Button.HORN, Button.HORN]


def test_after_the_signal_nothing_is_held() -> None:
    assert handover_buttons(HANDOVER_S + 0.1) == frozenset()


def test_a_drive_switches_off_the_hazards_its_last_hand_over_left_on() -> None:
    from core.control.handover import HazardsOff

    off = HazardsOff()
    pressed = [bool(off.buttons(on=True, since_start=k * 0.05)) for k in range(10)]
    assert pressed[:4] == [True] * 4 and not any(pressed[4:]), "pressed once, not toggled back"
    assert not off.buttons(on=False, since_start=0.6)


def test_hazards_the_driver_switches_on_later_are_left_alone() -> None:
    from core.control.handover import HazardsOff

    assert not HazardsOff().buttons(on=True, since_start=60.0)
