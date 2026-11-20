"""Per-truck calibration: identified, fitted, kept, and read back."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bridge.layout import BLOCK_DTYPE
from core.calibration import (
    TruckProfile,
    fit_brake,
    fit_steering_map,
    load_profile,
    save_profile,
    truck_id,
)
from core.control.lateral import ACTROS_GEOMETRY
from core.control.longitudinal import ACTROS_EMPTY
from core.perception.detection import DEFAULT_CAB_VIEW


def block_of(brand: bytes = b"scania", model: bytes = b"s_2016"):
    block = np.zeros(1, dtype=BLOCK_DTYPE)[0]
    block["truck"]["brand_id"] = brand
    block["truck"]["model_id"] = model
    wheels = block["truck"]["wheels"]
    wheels["count"] = 6
    # An 8x4's like: two steered axles ahead, two driven behind (+z is back).
    for k, (z, steered, powered) in enumerate(
        [(-2.0, 1, 0), (-2.0, 1, 0), (-0.6, 1, 0), (-0.6, 1, 0), (2.2, 0, 1), (2.2, 0, 1)]
    ):
        wheels["position"][k]["z"] = z
        wheels["steerable"][k] = steered
        wheels["powered"][k] = powered
    return block


def test_the_truck_is_known_by_its_brand_and_model() -> None:
    assert truck_id(block_of()) == "scania.s_2016"
    assert truck_id(block_of(b"", b"")) == "unknown.unknown"


def test_a_truck_with_no_profile_runs_on_the_actros_numbers_and_its_own_wheelbase() -> None:
    profile = TruckProfile("scania.s_2016")
    geometry = profile.geometry(block_of())
    assert geometry.wheelbase == pytest.approx(2.2 - (-2.0 - 2.0 - 0.6 - 0.6) / 4)
    assert geometry.steering_map == ACTROS_GEOMETRY.steering_map
    assert profile.longitudinal() is ACTROS_EMPTY
    assert profile.cab_view() is DEFAULT_CAB_VIEW
    assert profile.camera_file().name == "camera.json"
    assert profile.missing() == ["steering", "brake", "camera", "cab", "view"]


def test_steering_steps_both_ways_make_one_rising_map() -> None:
    samples = [
        (0.1, math.radians(4.0)),
        (-0.1, math.radians(-4.2)),
        (0.5, math.radians(21.0)),
        (-0.5, math.radians(-21.0)),
        (0.95, math.radians(39.0)),
        (0.3, math.radians(12.0)),
        (0.0, 0.0),
    ]
    fitted = fit_steering_map(samples)
    assert fitted[0] == (0.0, 0.0)
    assert [c for c, _ in fitted] == [0.0, 0.1, 0.3, 0.5, 0.95]
    assert math.degrees(fitted[1][1]) == pytest.approx(4.1, abs=0.01)
    # It is a valid map for the controllers.
    geometry = TruckProfile("t", steering_map=fitted).geometry(block_of())
    assert geometry.command_for_angle(math.radians(-21.0)) == pytest.approx(-0.5, abs=1e-3)


def test_too_few_steering_steps_are_no_map() -> None:
    with pytest.raises(ValueError):
        fit_steering_map([(0.5, 0.3)])


def test_brake_runs_give_the_dead_zone_and_a_rising_curve() -> None:
    samples = [(0.04, 0.01), (0.05, 0.02), (0.06, 1.8), (0.1, 2.8), (0.12, 2.7), (0.4, 6.8)]
    dead_zone, curve = fit_brake(samples)
    assert dead_zone == pytest.approx(0.055)
    assert curve == ((0.06, 1.8), (0.1, 2.8), (0.4, 6.8))
    profile = TruckProfile("t", brake_dead_zone=dead_zone, brake_curve=curve)
    calibration = profile.longitudinal()
    assert calibration.brake_curve == curve
    assert calibration.throttle_curve == ACTROS_EMPTY.throttle_curve


def test_a_profile_is_saved_and_read_back(tmp_path) -> None:
    profile = TruckProfile(
        "scania.s_2016",
        steering_map=((0.0, 0.0), (0.5, 0.35), (0.9, 0.6)),
        brake_dead_zone=0.05,
        brake_curve=((0.06, 1.8), (0.4, 6.8)),
        camera="scania.s_2016.camera.json",
        cab_reference="scania.s_2016.cab.npy",
        view={
            "windscreen": [0.1, 0.2, 1.0, 0.7],
            "covered": [[0.0, 0.1, 0.1, 0.4]],
        },
        measured={"steering": "2026-10-05"},
    )
    save_profile(profile, tmp_path)
    back = load_profile("scania.s_2016", tmp_path)
    assert back == profile
    assert back.camera_file(tmp_path) == tmp_path / "scania.s_2016.camera.json"
    view = back.cab_view()
    assert view.covered[0].right == pytest.approx(0.1)
    assert view.windscreen.left == pytest.approx(0.1)
    assert back.missing() == []


def test_an_unknown_truck_has_an_empty_profile(tmp_path) -> None:
    assert load_profile("daf.xf", tmp_path) == TruckProfile("daf.xf")


class _Segment:
    def __init__(self, phase: str, command: float, t, block) -> None:
        self.phase, self.command, self.t, self.block = phase, command, t, block


def test_a_held_steering_step_reads_as_its_command_and_its_wheels_angle() -> None:
    from core.calibration.samples import steering_samples

    blocks = np.zeros(40, dtype=BLOCK_DTYPE)
    truck = blocks["truck"]
    truck["effective_steering"] = -0.5
    truck["wheels"]["steerable"][:, :4] = 1
    # Turns: 20 degrees on the inner wheels, 22 on the outer, either axle.
    truck["wheels"]["steering"][:, [0, 2]] = -20.0 / 360.0
    truck["wheels"]["steering"][:, [1, 3]] = -22.0 / 360.0
    t = np.linspace(0.0, 2.0, 40)
    ((command, angle),) = steering_samples([_Segment("steer", -0.5, t, blocks)])
    assert command == pytest.approx(-0.5)
    assert math.degrees(angle) == pytest.approx(-21.0)
    assert steering_samples([_Segment("centre", 0.0, t, blocks)]) == []


def test_a_braking_run_reads_as_its_deceleration() -> None:
    from core.calibration.samples import brake_samples

    t = np.linspace(0.0, 3.0, 90)
    blocks = np.zeros(len(t), dtype=BLOCK_DTYPE)
    blocks["truck"]["effective_brake"] = 0.25
    blocks["truck"]["speed"] = 11.0 - 5.27 * t
    ((command, deceleration),) = brake_samples([_Segment("brake", 0.25, t, blocks)])
    assert command == 0.25
    assert deceleration == pytest.approx(5.27, abs=1e-3)


def test_a_view_is_set_from_boxes_in_fractions_and_round_trips() -> None:
    from tools.calibrate import parse_box, view_as_dict, view_from

    drawn = view_from("0.17,0.23,1,0.65", ["0,0.07,0.14,0.37", "0.86,0.07,1,0.37"])
    assert drawn["windscreen"] == [0.17, 0.23, 1.0, 0.65]
    assert drawn["covered"][0] == [0.0, 0.07, 0.14, 0.37]
    assert len(drawn["covered"]) == 4, "both mirrors and the HUD's two"
    view = TruckProfile("t", view=drawn).cab_view()
    assert view_as_dict(view) == drawn
    assert view_as_dict(DEFAULT_CAB_VIEW)["windscreen"][1] == pytest.approx(250 / 1080)
    with pytest.raises(ValueError):
        parse_box("0.5,0.5,0.4,0.6")
    with pytest.raises(ValueError):
        view_from("0,0,1,1", ["0,0,2,1"])


def test_a_truck_without_its_own_dashboard_learns_one(tmp_path) -> None:
    from core.calibration import DEFAULTS_TRUCK

    scania = TruckProfile("scania.vehicle.scania.r_2016")
    assert scania.learns_cab(tmp_path) == tmp_path / "scania.vehicle.scania.r_2016.cab.npy"
    learned = scania.with_cab(tmp_path / "scania.vehicle.scania.r_2016.cab.npy")
    assert learned.cab_reference == "scania.vehicle.scania.r_2016.cab.npy"
    assert "learned" in learned.measured["cab"] and "cab" not in learned.missing()
    assert learned.learns_cab(tmp_path) is None, "learned once"
    assert TruckProfile(DEFAULTS_TRUCK).learns_cab(tmp_path) is None, "the default is its own"
    assert TruckProfile(DEFAULTS_TRUCK).missing() == []


def test_the_driver_is_told_what_a_truck_is_not_calibrated_for() -> None:
    from core.calibration import DEFAULTS_TRUCK
    from tools.follow import uncalibrated

    said = uncalibrated(TruckProfile("scania.vehicle.scania.r_2016"))
    assert "not calibrated" in said and "distances" in said and "cab view (1)" in said
    assert uncalibrated(TruckProfile(DEFAULTS_TRUCK)) == ""


def test_a_truck_without_a_view_of_its_own_learns_its_windscreen() -> None:
    from core.calibration import DEFAULTS_TRUCK

    xd = TruckProfile("daf.vehicle.daf.xd")
    assert xd.learns_roof()
    learned = xd.with_view_top(170 / 1080)
    assert learned.cab_view().windscreen.top == pytest.approx(170 / 1080, abs=1e-4)
    assert learned.cab_view().covered == xd.cab_view().covered
    assert not learned.learns_roof(), "learned once"
    assert "view" not in learned.missing()
    assert not TruckProfile(DEFAULTS_TRUCK).learns_roof()


def test_a_truck_without_its_own_camera_is_to_have_it_learned() -> None:
    from core.calibration import DEFAULTS_TRUCK

    xf = TruckProfile("daf.vehicle.daf.xf_electric")
    assert xf.learns_camera()
    learned = xf.with_camera("daf.vehicle.daf.xf_electric.camera.json", "learned")
    assert not learned.learns_camera() and "learned" in learned.measured["camera"]
    assert not TruckProfile(DEFAULTS_TRUCK).learns_camera()
