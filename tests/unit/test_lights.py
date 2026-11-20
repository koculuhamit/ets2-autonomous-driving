"""Reading traffic lights from the picture: lamps, a frame's verdict, and the filter."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.perception.camera import GroundCamera
from core.perception.detection import DEFAULT_CAB_VIEW
from core.perception.lights import (
    CONFIRM_FRAMES,
    HOLD_S,
    RED_MEMORY_S,
    RED_STEADY_S,
    Lamp,
    LightFilter,
    LightState,
    find_lamps,
    in_same_head,
    reading,
)

CAMERA = GroundCamera(focal_px=819.0, horizon_row=438.0, centre_col=960.0, height_m=2.88)
HORIZON = CAMERA.horizon_row
RED, AMBER, GREEN, UNKNOWN = LightState.RED, LightState.AMBER, LightState.GREEN, LightState.UNKNOWN
BGR = {
    "red": (40, 40, 255),
    "amber": (0, 190, 255),
    "green": (170, 255, 60),
    "unlit green": (90, 110, 40),
}


def night(*lamps: tuple[str, int, int, int]) -> np.ndarray:
    """A dark 1920x1080 frame with lit discs: (colour, x, y, radius)."""
    image = np.full((1080, 1920, 3), 12, dtype=np.uint8)
    for colour, x, y, radius in lamps:
        cv2.circle(image, (x, y), radius, BGR[colour], -1)
    return image


def colours(image: np.ndarray, vehicles=(), nearest_m=None) -> list[LightState]:
    return [
        lamp.colour for lamp in find_lamps(image, DEFAULT_CAB_VIEW, CAMERA, vehicles, nearest_m)
    ]


@pytest.mark.parametrize("colour", ["red", "amber", "green"])
def test_a_lit_lamp_above_the_road_is_found_in_its_colour(colour: str) -> None:
    assert colours(night((colour, 1100, 300, 6))) == [LightState(colour)]


def test_an_unlit_lens_is_not_a_lamp() -> None:
    """The dark green lens under a red lamp is still teal, only dim."""
    assert colours(night(("red", 1500, 400, 8), ("unlit green", 1500, 440, 8))) == [LightState.RED]


def test_a_tail_light_below_the_horizon_is_not_a_traffic_light() -> None:
    assert colours(night(("red", 1000, int(HORIZON) + 30, 5))) == []


def test_near_the_line_the_pole_s_lower_lamps_are_seen() -> None:
    """Stopped 7 m before the line, the green lamp beside it is 100 px below
    the horizon: out of sight of a fixed horizon gate, not of the map's."""
    low_green = night(("green", 1535, int(HORIZON) + 100, 8))
    assert colours(low_green) == []
    assert colours(low_green, nearest_m=6.0) == [LightState.GREEN]
    assert colours(low_green, nearest_m=60.0) == [], "that low, 60 m off, it is not up a pole"


def test_a_red_blob_too_big_for_a_lamp_that_far_off_is_not_one() -> None:
    """Out of Calais at night: 400-700 px of red, 81 m from the line, was
    taken for the light's red; a lamp that far is a few pixels across."""
    blob = night(("red", 1300, 380, 12))  # 450 px
    assert colours(blob, nearest_m=81.0) == []
    assert colours(blob, nearest_m=12.0) == [LightState.RED]


def red_ring_sign() -> np.ndarray:
    """A speed-limit sign: a red rim round a white face."""
    image = night()
    cv2.circle(image, (1300, 380), 9, BGR["red"], -1)
    cv2.circle(image, (1300, 380), 5, (235, 235, 235), -1)
    return image


def test_by_day_a_red_ringed_sign_is_not_a_lamp() -> None:
    """Out of Calais two speed-limit signs held the truck at a light the map
    has and the road does not."""
    lamps = find_lamps(red_ring_sign(), DEFAULT_CAB_VIEW, CAMERA, (), 20.0, daylight=True)
    assert lamps == []


def swiss_flag() -> np.ndarray:
    """A Swiss flag hung corner down: a red diamond, a white cross in it."""
    image = night()
    diamond = np.array([(1300, 365), (1315, 380), (1300, 395), (1285, 380)], np.int32)
    cv2.fillPoly(image, [diamond], BGR["red"])
    cv2.rectangle(image, (1294, 378), (1306, 382), (235, 235, 235), -1)
    cv2.rectangle(image, (1298, 374), (1302, 386), (235, 235, 235), -1)
    return image


def test_by_day_a_flag_with_a_white_cross_is_not_a_lamp() -> None:
    """On the bridge in Geneva the Swiss flags along its rail were read as a
    red lamp, a flag at a time, while the truck waited at the line."""
    lamps = find_lamps(swiss_flag(), DEFAULT_CAB_VIEW, CAMERA, (), 20.0, daylight=True)
    assert lamps == []


def test_a_lit_red_lamp_is_a_filled_disc_and_is_found_by_day() -> None:
    lamp_image = night(("red", 1300, 380, 9))
    lamps = find_lamps(lamp_image, DEFAULT_CAB_VIEW, CAMERA, (), 20.0, daylight=True)
    assert [lamp.colour for lamp in lamps] == [LightState.RED]


def test_by_night_a_ring_may_be_a_lamp_whitened_by_its_own_glare() -> None:
    lamps = find_lamps(red_ring_sign(), DEFAULT_CAB_VIEW, CAMERA, (), 20.0, daylight=False)
    assert [lamp.colour for lamp in lamps] == [LightState.RED]


def red_in_a_red_picture() -> np.ndarray:
    """A bright red patch in a dark red figure on a poster's black."""
    image = night()
    image[330:430, 1260:1340] = (25, 25, 120)  # the figure: dark, saturated red
    cv2.circle(image, (1300, 380), 9, BGR["red"], -1)
    return image


def test_by_day_a_red_patch_in_a_red_picture_is_not_a_lamp() -> None:
    """In Bern a poster's pink, on an advertising column at a junction, held
    the truck at a light it could not see."""
    lamps = find_lamps(red_in_a_red_picture(), DEFAULT_CAB_VIEW, CAMERA, (), 20.0, daylight=True)
    assert lamps == []
    by_night = find_lamps(red_in_a_red_picture(), DEFAULT_CAB_VIEW, CAMERA, (), 20.0)
    assert [lamp.colour for lamp in by_night] == [LightState.RED], "its glow reddens the housing"


def test_day_is_seven_to_seven_in_the_game() -> None:
    from core.perception.lights import is_daylight

    assert is_daylight(10 * 60 + 25) and not is_daylight(23 * 60) and not is_daylight(1440 + 60)


def test_a_lamp_in_a_signal_head_is_the_light_s_whatever_stands_behind_it() -> None:
    """A tanker drew up behind the pole, the head inside its box: its red and
    green were taken for the tanker's, and the truck stood at the green."""
    tanker = (900.0, 380.0, 1400.0, 600.0)
    head = (980.0, 370.0, 1020.0, 450.0)

    def found(image, heads):
        return [
            lamp.colour
            for lamp in find_lamps(image, DEFAULT_CAB_VIEW, CAMERA, [tanker], None, heads=heads)
        ]

    assert found(night(("green", 1000, 410, 5)), [head]) == [GREEN]
    assert found(night(("red", 1000, 410, 5)), [head]) == [RED]
    assert found(night(("red", 1200, 410, 5)), [head]) == [], "the tanker's own, off the head"


def test_a_lamp_on_a_detected_vehicle_is_the_vehicle_s() -> None:
    """A trailer's top marker lamps are red and four metres up."""
    image = night(("red", 1000, 400, 4))
    assert colours(image, vehicles=[(900.0, 380.0, 1100.0, 600.0)]) == []
    assert colours(image, vehicles=[(1200.0, 380.0, 1400.0, 600.0)]) == [LightState.RED]


def test_a_lit_surface_is_not_a_lamp() -> None:
    """Lit by a sodium lamp, a wall is amber -- and surrounded by more wall."""
    image = night()
    image[300:360, 900:1100] = (50, 140, 190)  # a lit wall, not quite lamp-bright
    image[325:333, 996:1004] = (0, 190, 255)  # a bright patch on it
    assert colours(image) == []


def test_a_red_wedge_twice_as_wide_as_tall_is_not_a_lamp() -> None:
    """A café's red and white parasol across a junction in Nynäshamn held the
    truck at a light that was not there for 38 s: one red panel, a wedge."""
    image = night()
    wedge = np.array([[1088, 306], [1112, 306], [1100, 294]], dtype=np.int32)
    cv2.fillPoly(image, [wedge], BGR["red"])
    assert colours(image) == []


def test_two_lenses_run_together_are_still_a_lamp() -> None:
    """Drawn out, but full and round-edged as discs are."""
    image = night(("red", 1094, 300, 7), ("red", 1106, 300, 7))
    assert colours(image) == [LightState.RED]


def test_a_lamp_in_a_mirror_is_not_ahead() -> None:
    assert colours(night(("red", 1800, 300, 6))) == []


def test_the_nearest_light_outweighs_the_next_junction_s() -> None:
    """Near lamps are large, far ones a few pixels: the verdict goes by area."""
    lamps = find_lamps(
        night(("green", 1300, 400, 9), ("red", 900, 430, 2), ("red", 950, 430, 2)),
        DEFAULT_CAB_VIEW,
        CAMERA,
    )
    assert reading(lamps) is LightState.GREEN


def test_nothing_lit_reads_as_unknown() -> None:
    assert reading(find_lamps(night(), DEFAULT_CAB_VIEW, CAMERA)) is LightState.UNKNOWN


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #


def lamp(colour: LightState, x: float = 1000.0, y: float = 300.0, area: int = 40) -> Lamp:
    return Lamp(colour, x, y, area)


def feed(light_filter: LightFilter, frames, start: float = 0.0, dt: float = 1 / 30):
    out = []
    for i, lamps in enumerate(frames):
        out.append(light_filter.update(lamps, start + i * dt))
    return out


RED_LAMP = [lamp(RED, y=276.0)]
AMBER_LAMP = [lamp(AMBER, y=288.0)]
GREEN_LAMP = [lamp(GREEN, y=300.0)]
NOTHING: list[Lamp] = []
STEADY = round(RED_STEADY_S * 30) + CONFIRM_FRAMES
"""Frames, at 30 a second, for a red to be believed."""


def test_a_colour_is_believed_after_a_few_frames_in_a_row() -> None:
    light_filter = LightFilter()
    out = feed(light_filter, [GREEN_LAMP] * CONFIRM_FRAMES)
    assert out[-2] is UNKNOWN
    assert out[-1] is GREEN


def test_a_red_is_believed_once_lit_for_a_while() -> None:
    out = feed(LightFilter(), [RED_LAMP] * STEADY)
    assert out[STEADY - CONFIRM_FRAMES - 2] is UNKNOWN
    assert out[-1] is RED


def test_a_flashing_red_warning_lamp_is_not_a_light() -> None:
    """Out of Calais at night: aircraft warning lamps on a pylon, on for
    0.2-0.4 s, out for a second, at 1574 and 1593 px, 110 px above the
    horizon, where a pole's head would be."""
    warning = [lamp(RED, x=1574.0, y=326.0, area=35), lamp(RED, x=1593.0, y=331.0, area=25)]
    flashing = ([warning] * 12 + [NOTHING] * 30) * 5
    assert set(feed(LightFilter(), flashing)) == {UNKNOWN}


def test_two_warning_lamps_flashing_out_of_step_are_not_a_light() -> None:
    """A pylon's and a chimney's, far apart: between them red is nearly
    always lit somewhere, but neither lamp for long."""
    pylon = [lamp(RED, x=1574.0, y=326.0, area=35)]
    chimney = [lamp(RED, x=700.0, y=250.0, area=30)]
    flashing = ([pylon] * 10 + [chimney] * 10) * 10
    assert set(feed(LightFilter(), flashing)) == {UNKNOWN}


def test_a_red_lamp_drifting_across_the_view_on_the_move_is_steady() -> None:
    frames = [[lamp(RED, x=1000.0 + 2.0 * k, y=276.0 - 0.5 * k)] for k in range(STEADY)]
    assert feed(LightFilter(), frames)[-1] is RED


def test_one_odd_frame_does_not_change_the_state() -> None:
    light_filter = LightFilter()
    frames = [RED_LAMP] * STEADY + [[lamp(GREEN, x=1500.0)]] + [RED_LAMP] * 5
    assert set(feed(light_filter, frames)[STEADY:]) == {RED}


def test_a_light_goes_green_amber_red_green_in_one_head() -> None:
    """The cycle of city-01, lamp by lamp: 12 px apart, green lowest."""
    frames = [GREEN_LAMP] * 5 + [AMBER_LAMP] * 20 + [RED_LAMP] * 20 + [AMBER_LAMP] * 10
    frames += [GREEN_LAMP] * 5
    out = feed(LightFilter(), frames)
    assert [out[4], out[24], out[44], out[54], out[-1]] == [GREEN, AMBER, RED, RED, GREEN]


def test_amber_out_of_nowhere_is_not_believed() -> None:
    assert feed(LightFilter(), [AMBER_LAMP] * 10)[-1] is UNKNOWN


def test_amber_away_from_the_green_lamp_is_a_lit_sign() -> None:
    """In city-01 a row of amber blobs 170 px from the green lamp stopped the
    truck at a green light; the green's own amber comes on just above it."""
    sign = [lamp(AMBER, x=1170.0, y=290.0, area=110)]
    out = feed(LightFilter(), [GREEN_LAMP] * 5 + [sign] * 10)
    assert out[-1] is GREEN


def test_green_at_a_red_light_must_come_on_in_its_head() -> None:
    """A green sign beside a red light is not the light turning green."""
    green_sign = [lamp(RED, y=276.0, area=30), lamp(GREEN, x=1300.0, y=300.0, area=200)]
    out = feed(LightFilter(), [RED_LAMP] * STEADY + [green_sign] * 20)
    assert out[-1] is RED


def test_green_after_the_red_has_gone_out_is_believed() -> None:
    """city-01: the red on the pole went out, and the green that came on was
    overhead, in another head. Between them, amber alone for 2 s."""
    light_filter = LightFilter()
    feed(light_filter, [RED_LAMP] * STEADY)
    later = RED_MEMORY_S + 1.0
    out = feed(light_filter, [[lamp(GREEN, x=1500.0)]] * 5, start=later)
    assert out[-1] is GREEN


def test_a_state_outlives_its_lamps_for_a_while() -> None:
    """Close to the line the lamps leave the view; the last colour stands."""
    light_filter = LightFilter()
    feed(light_filter, [GREEN_LAMP] * 5)
    assert light_filter.update(NOTHING, 0.2 + HOLD_S - 0.1) is GREEN
    assert light_filter.update(NOTHING, 0.2 + HOLD_S + 0.1) is UNKNOWN


def test_one_head_s_lamps_are_told_from_another_s() -> None:
    green = lamp(GREEN, x=956.0, y=332.0, area=38)
    assert in_same_head(lamp(AMBER, x=956.0, y=321.0, area=35), green, 1.0)
    assert not in_same_head(lamp(AMBER, x=1059.0, y=330.0), green, 1.0), "the next head"
    assert not in_same_head(lamp(AMBER, x=956.0, y=345.0), green, 1.0), "below it"


def test_by_day_a_lamp_in_its_housing_against_the_sky_is_found() -> None:
    """Overhead lamps by day are framed by sky as much as by their housing;
    judged by the median of their surroundings, every one was missed."""
    image = np.full((1080, 1920, 3), (235, 225, 215), dtype=np.uint8)  # pale sky
    image[280:334, 991:1009] = (25, 25, 25)  # a signal head, three lamps tall
    cv2.circle(image, (1000, 290), 6, BGR["red"], -1)
    assert colours(image) == [LightState.RED]


def test_a_big_yellow_sign_does_not_outvote_a_green_lamp() -> None:
    """In the game: a priority-road sign above the light, 1460 px of amber,
    against the green lamp's 343."""
    sign = lamp(AMBER, x=1317.0, y=314.0, area=1460)
    green = lamp(GREEN, x=1298.0, y=493.0, area=343)
    assert reading([sign, green]) is GREEN
    assert reading([sign]) is AMBER


def test_green_replacing_red_under_a_yellow_sign_is_believed() -> None:
    sign = lamp(AMBER, x=1317.0, y=314.0, area=1460)
    red = lamp(RED, x=1300.0, y=430.0, area=450)
    green = lamp(GREEN, x=1298.0, y=493.0, area=343)
    out = feed(LightFilter(), [[sign, red]] * STEADY + [[sign]] * 20 + [[sign, green]] * 5)
    assert out[STEADY - 1] is RED
    assert out[-1] is GREEN


def test_a_red_shape_on_a_lit_poster_is_not_a_lamp() -> None:
    """In the game, at night: a poster's red, on the poster's bright blue,
    held the truck at a roundabout with no light. A lamp's neighbours are
    unlit lenses in a black housing."""
    image = night()
    image[478:520, 1782:1798] = (170, 100, 30)  # the poster's blue, below the red
    cv2.circle(image, (1790, 470), 6, BGR["red"], -1)
    assert colours(image, nearest_m=11.0) == []


def test_a_lamp_in_its_head_is_found_whichever_it_is() -> None:
    image = night()
    image[300:360, 994:1006] = (20, 20, 20)  # a head
    cv2.circle(image, (1000, 308), 5, BGR["red"], -1)
    assert colours(image) == [LightState.RED]
    image = night()
    image[300:360, 994:1006] = (20, 20, 20)
    cv2.circle(image, (1000, 352), 5, BGR["green"], -1)
    assert colours(image) == [LightState.GREEN]


def test_a_red_lamp_under_a_yellow_sign_is_still_seen() -> None:
    """In the game: the red lamp's glow and the priority-road sign just above
    it were found as one tall blob, not a lamp; a green sign was then taken
    for the light turning green, and the truck went on red."""
    image = night()
    image[396:420, 1226:1250] = (60, 200, 255)  # the yellow sign, bright
    image[426:470, 1231:1245] = (20, 20, 20)  # the head
    cv2.circle(image, (1238, 432), 6, BGR["red"], -1)
    assert LightState.RED in colours(image, nearest_m=11.0)


def test_amber_a_second_on_and_drifted_up_is_the_green_s_head() -> None:
    """In the game: amber 25 px above and 5 px beside the green seen a second
    before, the truck coming on at 49 km/h."""
    green = lamp(GREEN, x=975.0, y=390.0, area=14)
    amber = lamp(AMBER, x=980.0, y=365.0, area=20)
    out = feed(LightFilter(), [[green]] * 5 + [[amber]] * 5, dt=0.1)
    assert out[-1] is AMBER


def test_a_red_beside_a_green_that_is_still_lit_does_not_end_it() -> None:
    """Moving off at a green in Rotterdam: a lorry's tail lamps came into view
    beside the pole, 80-110 px against the green's 37, 60 px to one side and
    50 px above, and held the truck at the green until it gave up."""
    light = LightFilter()
    green = lamp(GREEN, 1618.0, 499.0, 37)
    tail_lamps = (lamp(RED, 1555.0, 443.0, 82), lamp(RED, 1575.0, 448.0, 111))
    assert feed(light, [(green,)] * 5)[-1] is GREEN
    assert set(feed(light, [(green, *tail_lamps)] * 30, start=1.0)) == {GREEN}


def test_a_green_that_goes_out_for_a_red_elsewhere_is_red() -> None:
    light = LightFilter()
    feed(light, [GREEN_LAMP] * 5)
    assert feed(light, [[lamp(RED, x=1300.0)]] * STEADY, start=1.0)[-1] is RED


def test_green_is_believed_once_our_red_is_out_though_another_red_stays_lit() -> None:
    """Milan: the crossing road's red stayed in view while ours went green,
    and the truck sat through two greens."""
    ours = lamp(RED, x=1000.0, y=276.0)
    theirs = lamp(RED, x=1400.0, y=250.0)
    green = lamp(GREEN, x=1000.0, y=300.0)
    light = LightFilter()
    assert feed(light, [[ours]] * STEADY)[-1] is RED
    assert feed(light, [[ours, theirs]] * 30, start=1.0)[-1] is RED
    out = feed(light, [[green, theirs]] * 10, start=2.0)
    assert out[-1] is GREEN


def test_with_our_red_still_lit_a_green_elsewhere_is_not_believed() -> None:
    ours = lamp(RED, x=1000.0, y=276.0)
    elsewhere = lamp(GREEN, x=1500.0, y=300.0, area=200)
    light = LightFilter()
    feed(light, [[ours]] * STEADY)
    assert set(feed(light, [[ours, elsewhere]] * 60, start=1.0)) == {RED}


def test_a_steady_amber_after_a_red_does_not_hold_the_red() -> None:
    """Milan: a steady amber, not believed after a red, was taken for the red
    standing, and held the truck at the line for good."""
    light = LightFilter()
    feed(light, [RED_LAMP] * STEADY)
    out = feed(light, [AMBER_LAMP] * round((HOLD_S + 1.0) * 30), start=1.0)
    assert out[-1] is UNKNOWN


def test_a_red_seen_late_is_ours_and_its_green_believed() -> None:
    """Bern: the state turned red for a red give-way sign above the light;
    the light's own red, seen after, was never taken for ours, and its green,
    the sign still red above it, held the truck at the line."""
    sign = lamp(RED, x=1640.0, y=230.0, area=60)
    red = lamp(RED, x=1640.0, y=276.0)
    green = lamp(GREEN, x=1640.0, y=300.0)
    light = LightFilter()
    assert feed(light, [[sign]] * STEADY)[-1] is RED
    assert feed(light, [[sign, red]] * 30, start=1.0)[-1] is RED
    assert feed(light, [[sign, green]] * 10, start=2.0)[-1] is GREEN


def test_a_daylit_view_is_day_whatever_the_clock() -> None:
    """19:50 in Bern: the sun up, the game's clock past day."""
    from core.perception.detection import DEFAULT_CAB_VIEW
    from core.perception.lights import bright_view

    day = np.full((1080, 1920, 3), 245, np.uint8)
    night = np.full((1080, 1920, 3), 20, np.uint8)
    night[300:304, 900:904] = 255  # a street lamp
    assert bright_view(day, DEFAULT_CAB_VIEW)
    assert not bright_view(night, DEFAULT_CAB_VIEW)


def test_a_dull_red_patch_in_dark_leaves_is_not_a_lit_lamp() -> None:
    """Outside Bern a salmon house front seen through dark leaves passed
    every other test and held the truck at a stop line with no light in
    sight: it peaked at 220 of 255, a lit red lamp at 232 or more."""
    image = night()
    cv2.circle(image, (1000, 367), 3, (110, 115, 215), -1)  # V 215, S 124
    assert find_lamps(image, DEFAULT_CAB_VIEW, CAMERA, (), 2.4, daylight=True) == []
    lit = night(("red", 1000, 367, 3))
    assert [lamp.colour for lamp in find_lamps(lit, DEFAULT_CAB_VIEW, CAMERA, (), 2.4)] == [RED]


def lamps_with_heads(image: np.ndarray, heads, nearest_m=None) -> list[LightState]:
    return [
        lamp.colour
        for lamp in find_lamps(image, DEFAULT_CAB_VIEW, CAMERA, (), nearest_m, heads=heads)
    ]


def test_with_heads_a_lamp_counts_only_in_one() -> None:
    """Short of Linz a filling station's logo, red under yellow among trees,
    was taken for a light's red: the detector found no head there."""
    image = night()
    image[300:360, 994:1006] = (20, 20, 20)  # a head
    cv2.circle(image, (1000, 308), 5, BGR["red"], -1)
    head = (990.0, 298.0, 1010.0, 362.0)
    assert lamps_with_heads(image, [head]) == [LightState.RED]
    assert lamps_with_heads(image, []) == [], "no head: not a light's lamp"
    assert lamps_with_heads(image, [(1300.0, 300.0, 1320.0, 360.0)]) == [], "another head"
    assert colours(image) == [LightState.RED], "without heads, as before"


def test_a_head_far_beyond_a_near_line_is_the_next_light_s() -> None:
    """Short of Linz, 10 m from a line, the red of the next junction's light
    -- a head 14 px tall at the horizon -- held the truck where there was none."""
    from core.perception.lights import ACROSS_M, head_distance

    image = night()
    cv2.circle(image, (1000, 400), 3, BGR["red"], -1)
    far = (996.0, 393.0, 1004.0, 407.0)  # 14 px tall: about 88 m
    assert head_distance(far, CAMERA) > (10.0 + ACROSS_M)
    assert lamps_with_heads(image, [far], nearest_m=10.0) == []
    assert lamps_with_heads(image, [far], nearest_m=40.0) == [LightState.RED], "it may be ours"


def _head_image(colour_bgr, row_share: float, lamp_px: int = 3):
    """A dark frame with one standing signal head at (950..962, 350..380) and
    a small lit blob `lamp_px` across at `row_share` of its height."""
    image = np.full((1080, 1920, 3), 40, np.uint8)
    image[350:381, 950:963] = (25, 25, 25)
    y = int(350 + row_share * 30)
    image[y : y + lamp_px, 955 : 955 + lamp_px] = colour_bgr
    image[y + lamp_px + 1 : y + 2 * lamp_px + 1, 955 : 955 + lamp_px] = colour_bgr
    return image, (950.0, 350.0, 962.0, 380.0)


def test_a_far_head_too_small_for_its_lamp_is_read_whole() -> None:
    """Near Venlo at sunset a red, a few pixels and broken in two by its
    lens, was found lamp by lamp in one view of five, and confirmed 15 m
    short of the line at 48 km/h. Read whole, a detected head shows it."""
    from core.perception.detection import DEFAULT_CAB_VIEW
    from core.perception.lights import LightState, find_lamps, reading

    camera = GroundCamera(820.0, 438.0, 960.0, 2.9)
    red, head = _head_image((30, 30, 250), 0.2)
    assert reading(find_lamps(red, DEFAULT_CAB_VIEW, camera, heads=[head])) is LightState.RED
    assert reading(find_lamps(red, DEFAULT_CAB_VIEW, camera, heads=[])) is LightState.UNKNOWN
    green, _ = _head_image((120, 250, 30), 0.8)
    assert reading(find_lamps(green, DEFAULT_CAB_VIEW, camera, heads=[head])) is LightState.GREEN


def test_a_colour_where_its_lamp_cannot_be_is_not_read() -> None:
    """A red at the bottom of a standing head is no red lamp of it."""
    from core.perception.detection import DEFAULT_CAB_VIEW
    from core.perception.lights import LightState, find_lamps, reading

    camera = GroundCamera(820.0, 438.0, 960.0, 2.9)

    def scattered(row_share: float):
        # Single lit pixels, two apart: no lamp lamp by lamp, read whole only.
        image, head = _head_image((25, 25, 25), 0.0, lamp_px=1)
        y = int(350 + row_share * 30)
        for dx in (0, 2, 4, 6):
            for dy in (0, 2, 4):
                image[y + dy, 953 + dx] = (30, 30, 250)
        return image, head

    high, head = scattered(0.15)
    assert reading(find_lamps(high, DEFAULT_CAB_VIEW, camera, heads=[head])) is LightState.RED
    low, head = scattered(0.75)
    assert reading(find_lamps(low, DEFAULT_CAB_VIEW, camera, heads=[head])) is LightState.UNKNOWN
