"""Coming to rest at a gas pump: the cap, and closed round the truck of
test_traffic_light (the measured actuators, a 0.3 s lag)."""

from __future__ import annotations

import pytest

from core.control.end_stop import PAST_M, reached, speed_to_stop_in
from core.control.longitudinal import SpeedController

KMH = 1 / 3.6


def test_the_cap_falls_to_nothing_at_the_end() -> None:
    assert speed_to_stop_in(0.0) == 0.0
    assert speed_to_stop_in(-3.0) == 0.0
    assert speed_to_stop_in(50.0) == pytest.approx(10.0)


def test_the_end_is_reached_by_the_truck_not_its_aim() -> None:
    """Out of Linz the drive ended 10.7 m short of the pump."""
    assert not reached(10.7, 7.4)
    assert not reached(1.0, 2.0)
    assert not reached(0.2, 1.0), "still going: let it come to rest"
    assert reached(1.0, 0.0)
    assert reached(-PAST_M, 1.0)


@pytest.mark.parametrize("road", [30 * KMH, 50 * KMH, 80 * KMH])
def test_the_truck_comes_to_rest_at_the_pump_gently(road: float) -> None:
    pump, dt, lag = 300.0, 1 / 60, 0.3
    controller = SpeedController()
    actuators = controller.actuators
    position, speed, acceleration, hardest = 0.0, road, 0.0, 0.0
    for _ in range(int(120 / dt)):
        remaining = pump - position
        if reached(remaining, speed):
            break
        target = min(road, speed_to_stop_in(remaining))
        command = controller.step(target_speed=target, speed=speed, grade=0.0, dt=dt)
        delivered = actuators.acceleration_for(command.throttle, command.brake)
        acceleration += (delivered - acceleration) * dt / lag
        speed = max(0.0, speed + acceleration * dt)
        position += speed * dt
        if speed > 3.0:
            hardest = max(hardest, -acceleration)
    else:
        pytest.fail("never reached the pump")
    assert abs(pump - position) < 0.3
    assert speed < 0.3, "come to rest there, not braked from speed"
    assert hardest < 2.6, "a comfortable stop"
