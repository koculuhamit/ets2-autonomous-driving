"""Traffic lights: which colour is lit, from the picture alone.

The map knows where the lights are -- a lane a light governs starts at its
stop line (core.planning.route.Route.stop_lines) -- but not what they show,
and neither does the telemetry. So the colour is read from the image, and the
map only says when it matters.

A lit lamp is a small, bright, saturated blob in a dark housing, above the
road. That is what `find_lamps` looks for:

- **Bright and saturated**: V >= 200 and S >= 100 (OpenCV HSV). An unlit
  green lens is still teal, but dim.
- **Above the road**: at least `MIN_LAMP_HEIGHT_M` up, if it is no nearer
  than the stop line, which the map gives (`nearest_m`). Far from the line
  that is about the horizon, where tail lights, at 1 m, are well below; at
  the line the green lamp of the pole beside it is 100 px below the horizon,
  and has to be seen. Without a distance, the horizon, give or take
  `BELOW_HORIZON_PX`.
- **No bigger than a lamp**: no bigger than `LENS_RADIUS_MAX_M` makes one
  at `nearest_m`.
- **Not on a vehicle**: a trailer's top marker lamps are red and 4 m up, so
  blobs inside a detected vehicle's box are dropped.
- **Round, and dark around**: a lamp fills its bounding box and its
  surroundings are the housing or the night sky. Lit foliage, windows and
  road surface under sodium lamps are the same hue as an amber lamp, but they
  are surrounded by more of themselves.

Measured on city-01 (a night drive through Italian junctions, see
docs/results/traffic_lights.md): the lamps of the light ahead were seen from
110 m, and each red-green change the driver waited through was seen.

With the detector's signal heads (`heads`, core.perception.detection's
`TRAFFIC_LIGHT`), a lamp counts only inside one: a filling station's
logo, red under yellow among trees, stopped the truck short of Linz, and
every red thing elsewhere -- posters, signs, tail lamps -- needed a rule of
its own. A head's height gives how far off it is (`head_distance`): one far
beyond the light whose line is near is the next junction's, whose red held
the truck at that junction with no light of its own.

`LightFilter` turns the per-frame reading into a state that does not flicker:
a colour must be seen in `CONFIRM_FRAMES` frames in a row, and the changes a
lookalike could fake -- amber after green, green after red -- must happen in
one signal head.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core.perception.camera import GroundCamera
from core.perception.detection import CabView

__all__ = [
    "Lamp",
    "LightFilter",
    "LightState",
    "bright_view",
    "find_lamps",
    "head_distance",
    "in_same_head",
    "is_daylight",
    "reading",
]

V_MIN: Final = 200
S_MIN: Final = 100
CORE_V_MIN: Final = 200
RED_PEAK_V: Final = 228
"""A red lamp is lit: its brightest pixel is at least this. Over city-01
and traffic-01/02 every red lamp found peaked at 232 or more -- the small,
far ones too -- most at 255; a salmon house front seen through dark leaves
outside Bern, which passed every other test and held the truck at a stop
line with no light in sight, peaked at 220."""
"""A lamp is lit through: the middle of its blob is bright, coloured or burnt
white. The red ring of a no-parking sign is a ring, blue in the middle -- by
day it held the truck at a light that was not there. (Its red, at a median
brightness of 214, is dimmer than a near lamp's 252, but no dimmer than a
far lamp's at night, so brightness alone would not do.)"""

DISC_CHECK_PX: Final = 40
"""Blobs this big or bigger must be disc-like (`_disc_like`); smaller ones
are too few pixels to have a shape."""

ELONGATED: Final = 1.6
DISC_FILL: Final = 0.65
DISC_ROUND: Final = 0.7
"""A blob `ELONGATED` times as long as it is wide must fill its box or be
round as discs are: a disc fills 0.79 and has a roundness (4 pi area /
perimeter squared) of 1, and two lenses run together, as seen in city-01,
0.8 and 0.92. The parasol's red panel, 1.9 to 2.1 long: at most 0.63 and
0.63."""

MIN_AREA_PX: Final = 4
MAX_AREA_PX: Final = 3000
MIN_FILL: Final = 0.35
"""Share of its bounding box a lamp's blob covers."""

ASPECT: Final = (0.4, 2.5)
DARK_RING: Final = 0.6
"""The darker quarter of what is around a lamp is at most this share of the
lamp's brightness. Not the median: by day an overhead lamp is framed by sky
as much as by its black housing, and the median rejected every one of them.
Lit surfaces are next to more of themselves all round."""

DARK_RING_QUANTILE: Final = 25

HOUSING: Final = 0.55
"""Where a lamp's unlit neighbours are, the median brightness is at most this
share of the lamp's. Unlit lenses in the sun were 0.5 of a lit red lamp; a
poster's blue under its red, 0.74."""

BELOW_HORIZON_PX: Final = 8
"""How far below the horizon row a lamp's centre may be at least. A lamp 3 m
up is level with the driver's eye; this is the horizon's session-to-session
drift, plus a little."""

MIN_LAMP_HEIGHT_M: Final = 1.6
"""The lowest lamp of a signal head. Stopped 7 m before a line in city-01,
the green lens on the pole beside it was 1.9 m up."""

NEAREST_LAMP_M: Final = 3.0

LENS_RADIUS_MAX_M: Final = 0.7
"""No lamp is bigger, glow and all, than a disc this wide seen from the
nearest the light can be: the green lamps of every drive so far, summed
per frame, came to 0.59 m at most. Out of Calais at night a red blob of
400-700 px, 81 m from the line -- a lamp 1.1-1.5 m across -- was taken for
the light's red."""

TINTED_SHARE: Final = 0.6
"""By day a red lamp sits in a black housing. A red blob whose darker
surroundings are this share or more saturated red is part of a red picture,
not a lamp: in Bern the pink of a figure on a poster, on an advertising
column at a junction, held the truck at a green it could not see; its dark
surroundings were 0.91 red, the daylight reds of every other drive 0-0.15."""

SIGN_HOLE: Final = 0.12
DAYLIGHT_MINUTES: Final = (7 * 60, 19 * 60)
"""By day a lit lamp is a filled disc of its colour. A red-ringed sign is a
ring round a white face: out of Calais two speed-limit signs, far down the
road where a light the map has and the road does not would stand, held the
truck at a standstill. A Swiss flag is red round a white cross: on the
bridge in Geneva the flags along its rail were a red lamp a frame at a time,
the nearest 0.22 white within its outline. A red blob whose outline holds
this share or more of not-red is no lamp; the 44 red lamps seen by day with
their signal heads in the frames kept so far held 0.01 at most. By night a
lamp's own glare whitens its middle, so the test is only made by day,
between these minutes of the game's day."""

MIN_LAMP_AREA_PX: Final = 12
"""Least total lamp area in a frame for a colour to count."""

HEAD_HEIGHT_M: Final = 1.5
"""A signal head's long side as the detector boxes it, three lenses and the
board behind them: how far off a head is, from its box (`head_distance`).
Waiting 10 m from the line in Bern, Milan and Linz, the light's head was
112-140 px tall: 1.4-1.75 m."""

HEAD_MARGIN: Final = 0.3
"""A lamp's centre may lie this share of its head's box outside it: the box
is the detector's, and a lit lamp's glow spills past the housing."""

ACROSS_M: Final = 40.0
"""How far past its stop line a light's head may stand: on the far side of a
wide junction. Beyond that, a head is the next light's."""

HEAD_DISTANCE_SLACK: Final = 1.3
"""How far a head may seem beyond `ACROSS_M` and still be the light's: a
head's height is not every head's, nor its box exactly the head. Held 10 m
from a line short of Linz by the next junction's red, its head was 13-14 px
tall: 88 m, against the 65 m a light's could be."""

CONFIRM_FRAMES: Final = 3
GREEN_MEMORY_S: Final = 1.0
RED_MEMORY_S: Final = 1.0
"""A light turning green has put its red out: these lights show amber alone
for 2 s between the two. A red lamp seen this recently means a green blob is
something else -- unless it is in that red's own head."""

RED_STEADY_S: Final = 0.5
RED_GAP_S: Final = 0.15
"""A red is believed once a red lamp has been lit this long without going
out for more than `RED_GAP_S`: a light's red is lit for seconds. Out of Calais at night the
aircraft warning lamps on a pylon and a chimney flash red, 0.2-0.4 s on and
a second out, high on the right where a pole's head would be; each flash
was a red, and each held the state for `HOLD_S`, past the next -- the truck
stood at a junction without a light until the driver gave up."""

LAMP_SPACING_RADII: Final = 4.0
HEAD_SLACK_PX: Final = 4.0
DRIFT_PX: Final = 15.0

HOLD_S: Final = 2.0
"""A state is kept this long after its lamps are last seen. Close to the stop
line the lamps leave the view -- overhead ones go behind the cab roof -- and a
green that was there a moment ago is still the best guess."""


class LightState(enum.Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Lamp:
    """One lit lamp in the frame."""

    colour: LightState
    x: float
    y: float
    area: int


HUES: Final = {
    LightState.RED: (169, 7),
    LightState.AMBER: (18, 34),
    LightState.GREEN: (50, 100),
}
"""OpenCV hue ranges of each lamp colour; red wraps round 0."""


def colour_of(hue: float) -> LightState | None:
    """The lamp colour of an OpenCV hue (0-180), or None for none of them."""
    if hue < 8.0 or hue > 168.0:
        return LightState.RED
    if 18.0 <= hue <= 34.0:
        return LightState.AMBER
    if 50.0 <= hue <= 100.0:
        return LightState.GREEN
    return None


def lowest_row(camera: GroundCamera, nearest_m: float | None) -> float:
    """The lowest image row a lamp can be on, if it is at least `nearest_m` away."""
    below = BELOW_HORIZON_PX
    if nearest_m is not None and math.isfinite(nearest_m):
        drop = camera.height_m - MIN_LAMP_HEIGHT_M
        below = max(below, drop * camera.focal_px / max(nearest_m, NEAREST_LAMP_M))
    return camera.horizon_row + below


def head_distance(box: Sequence[float], camera: GroundCamera) -> float:
    """How far off a signal head is, metres, from its box: (left, top,
    right, bottom) in frame pixels, upright or on its side."""
    size = max(box[2] - box[0], box[3] - box[1], 1.0)
    return camera.focal_px * HEAD_HEIGHT_M / size


def _in_a_head(
    x: float,
    y: float,
    heads: Sequence[Sequence[float]],
    camera: GroundCamera,
    nearest_m: float | None,
) -> bool:
    """Whether a lamp at (x, y) is in one of `heads` near enough to be the
    light's whose line is `nearest_m` off (`ACROSS_M`)."""
    farthest = math.inf
    if nearest_m is not None and math.isfinite(nearest_m):
        farthest = (max(nearest_m, 0.0) + ACROSS_M) * HEAD_DISTANCE_SLACK
    for left, top, right, bottom in heads:
        dx, dy = (right - left) * HEAD_MARGIN, (bottom - top) * HEAD_MARGIN
        if not (left - dx <= x <= right + dx and top - dy <= y <= bottom + dy):
            continue
        if head_distance((left, top, right, bottom), camera) <= farthest:
            return True
    return False


def find_lamps(
    image: NDArray[np.uint8],
    view: CabView,
    camera: GroundCamera,
    vehicles: Iterable[Sequence[float]] = (),
    nearest_m: float | None = None,
    daylight: bool = False,
    heads: Sequence[Sequence[float]] | None = None,
) -> list[Lamp]:
    """Lit traffic-light lamps in one BGR frame of the whole game window.

    `vehicles` are detected vehicle boxes, (left, top, right, bottom) in
    frame pixels; a blob inside one is a vehicle's lamp. `nearest_m` is how
    far the nearest light can be: the distance to the next stop line, a
    little less. `heads` are detected signal heads' boxes: given, a blob
    counts only in one of them, near enough to be the light's (`_in_a_head`),
    and whatever vehicle stands behind it: at a light in Sweden a tanker
    drew up behind the pole, the head inside its box, the light's red and
    green were taken for the tanker's, and the truck stood at the green.
    """
    lowest = lowest_row(camera, nearest_m)
    largest = MAX_AREA_PX
    if nearest_m is not None and math.isfinite(nearest_m):
        widest = camera.focal_px * LENS_RADIUS_MAX_M / max(nearest_m, NEAREST_LAMP_M)
        largest = min(largest, math.pi * widest * widest)
    height, width = image.shape[:2]
    left, top, right, _ = view.windscreen.pixels(width, height)
    bottom = int(min(height, lowest + 1))
    if bottom <= top:
        return []
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (value >= V_MIN) & (saturation >= S_MIN)
    boxes = [tuple(box) for box in vehicles]
    lamps: list[Lamp] = []
    # One colour at a time: closed together, a red lamp's glow and the yellow
    # priority-road sign above it were one blob in the game, too long to be a
    # lamp, and the red that was lit was not seen.
    for colour, (low, high) in HUES.items():
        in_family = (hue >= low) & (hue <= high) if low <= high else (hue >= low) | (hue <= high)
        mask = (bright & in_family).astype(np.uint8)
        # Closed, so that a red ring broken by a fence in front is fewer
        # blobs, the largest centred on the sign's middle rather than on itself.
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(joined, connectivity=8)
        for k in range(1, count):
            x0, y0, w, h, joined_area = (int(v) for v in stats[k])
            blob = (labels[y0 : y0 + h, x0 : x0 + w] == k) & (mask[y0 : y0 + h, x0 : x0 + w] > 0)
            area = int(blob.sum())
            if not MIN_AREA_PX <= area <= largest:
                continue
            if joined_area < MIN_FILL * w * h or not ASPECT[0] <= w / h <= ASPECT[1]:
                continue
            x, y = float(centroids[k][0] + left), float(centroids[k][1] + top)
            if y > lowest or view.is_covered(x, y, width, height):
                continue
            if heads is not None:
                if not _in_a_head(x, y, heads, camera, nearest_m):
                    continue
            elif any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes):
                continue
            pad = max(w, h)
            ring = value[max(0, y0 - pad) : y0 + h + pad, max(0, x0 - pad) : x0 + w + pad]
            cx, cy = round(x) - left, round(y) - top
            core = value[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2]
            if core.size == 0 or float(np.median(core)) < CORE_V_MIN:
                continue
            if colour is LightState.RED and int(value[y0 : y0 + h, x0 : x0 + w].max()) < RED_PEAK_V:
                continue
            brightness = float(value[y0 : y0 + h, x0 : x0 + w][blob].mean())
            if float(np.percentile(ring, DARK_RING_QUANTILE)) > DARK_RING * brightness:
                continue
            if not _in_housing(image, colour, x0 + left, y0 + top, w, h, brightness):
                continue
            if area >= DISC_CHECK_PX and not _disc_like(blob, w, h):
                continue
            if daylight and colour is LightState.RED and _ring(blob):
                continue
            if daylight and colour is LightState.RED and _tinted(hsv, cx, cy, area):
                continue
            lamps.append(Lamp(colour, x, y, area))
    if heads is not None:
        lamps.extend(_read_heads(image, heads, lamps, view, camera, nearest_m, lowest))
    return lamps


HEAD_V_MIN: Final = 170
HEAD_MIN_PX: Final = 5
HEAD_OVER_NEXT: Final = 2.0
HEAD_ROWS: Final = {
    LightState.RED: (0.0, 0.45),
    LightState.AMBER: (0.3, 0.7),
    LightState.GREEN: (0.55, 1.0),
}
"""A detected signal head in which no lamp was found lamp by lamp is read
whole (`_read_heads`): its lit pixels -- `HEAD_V_MIN` bright, `S_MIN`
saturated -- of one colour, `HEAD_MIN_PX` or more and `HEAD_OVER_NEXT` times
any other's, where that colour's lamp sits in a standing head (`HEAD_ROWS`,
shares of its height from the top). From 40 to 60 m a lamp is a few pixels,
broken in two by its lens, and lamp by lamp it was found in one view of
five: at sunset near Venlo a red was confirmed 15 m short of its line, at
48 km/h, and gone through (18 November). Read whole, the head showed red in
every view from 3.8 s out."""


def _read_heads(image, heads, found, view, camera, nearest_m, lowest) -> list[Lamp]:
    """Lamps of the heads `heads` none of `found` is in, read head by head."""
    height, width = image.shape[:2]
    read: list[Lamp] = []
    for box in heads:
        left, top, right, bottom = (round(float(v)) for v in box)
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, width - 1), min(bottom, height - 1)
        tall = bottom - top
        if tall < 4 or right - left < 2 or tall < 1.5 * (right - left):
            continue  # not a standing head: where each colour sits is not known
        if any(left <= lamp.x <= right and top <= lamp.y <= bottom for lamp in found):
            continue
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        if cy > lowest or view.is_covered(cx, cy, width, height):
            continue
        if not _in_a_head(cx, cy, [box], camera, nearest_m):
            continue
        hsv = cv2.cvtColor(image[top : bottom + 1, left : right + 1], cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (value >= HEAD_V_MIN) & (saturation >= S_MIN)
        counts = {}
        for colour, (low, high) in HUES.items():
            family = (hue >= low) & (hue <= high) if low <= high else (hue >= low) | (hue <= high)
            counts[colour] = np.argwhere(lit & family)
        best = max(counts, key=lambda c: len(counts[c]))
        points = counts[best]
        others = max(len(counts[c]) for c in counts if c is not best)
        if len(points) < HEAD_MIN_PX or len(points) < HEAD_OVER_NEXT * others:
            continue
        share = float(points[:, 0].mean()) / tall
        low_share, high_share = HEAD_ROWS[best]
        if not low_share <= share <= high_share:
            continue
        read.append(
            Lamp(
                best,
                float(left + points[:, 1].mean()),
                float(top + points[:, 0].mean()),
                len(points),
            )
        )
    return read


def _disc_like(blob: NDArray[np.bool_], w: int, h: int) -> bool:
    """Whether a blob could be a lens: not drawn out and ragged at once.

    At a junction in Nynäshamn a café's red and white parasol, across the
    road, was a red lamp for 38 s: one red panel, a wedge twice as wide as
    tall, filling half its box. A lens is round; drawn out, it is two lenses
    run together, which fill their box as discs do (in city-01, 2.0 wide and
    0.8 full). Close under a red lamp its glare bites a white crescent out of
    the red, which is ragged but not drawn out.
    """
    if max(w, h) < ELONGATED * min(w, h):
        return True
    area = int(blob.sum())
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = max(cv2.arcLength(max(contours, key=cv2.contourArea), True), 1.0)
    fill = area / float(w * h)
    roundness = 4.0 * math.pi * area / perimeter**2
    return fill >= DISC_FILL or roundness >= DISC_ROUND


def _tinted(hsv: NDArray[np.uint8], cx: int, cy: int, area: int) -> bool:
    """Whether the darker half of what is round a blob, three of its radii
    out, is `TINTED_SHARE` or more saturated red: a picture, not a housing."""
    reach = max(4, round(3.0 * math.sqrt(area / math.pi)))
    window = hsv[max(0, cy - reach) : cy + reach, max(0, cx - reach) : cx + reach].reshape(-1, 3)
    hue, saturation, value = (window[:, k].astype(float) for k in range(3))
    dark = value < 0.5 * value.max()
    if not dark.any():
        return False
    low, high = HUES[LightState.RED]
    red = ((hue >= low) | (hue <= high)) & (saturation >= S_MIN)
    return float(red[dark].mean()) >= TINTED_SHARE


def _ring(blob: NDArray[np.bool_]) -> bool:
    """Whether a blob is a ring: its outline holds `SIGN_HOLE` or more of
    not-blob -- a sign's white face in its red rim."""
    mask = blob.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    inside = int(filled.sum())
    return inside > 0 and (inside - int(mask.sum())) / inside >= SIGN_HOLE


def is_daylight(minute_of_day: float) -> bool:
    """Whether it is day in the game, for `find_lamps`."""
    start, end = DAYLIGHT_MINUTES
    return start <= minute_of_day % 1440 < end


BRIGHT_VIEW_V: Final = 200
"""The top third of the windscreen this bright at its 90th percentile (HSV
value), the view is daylit, whatever the game's clock: at 19:50 in Bern the
sun was up, the clock said night, and a red give-way sign above a light was
taken for its lamp -- the rules that tell a sign from a lamp are daylight's.
Over the recorded black boxes night was 5-60, broad day 240-255."""


def bright_view(image: NDArray[np.uint8], view) -> bool:
    """Whether the road ahead is seen in daylight (`BRIGHT_VIEW_V`)."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = view.windscreen.pixels(width, height)
    band = image[y0 : y0 + max(1, (y1 - y0) // 3), x0:x1]
    if band.size == 0:
        return False
    value = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[..., 2]
    return float(np.percentile(value, 90)) >= BRIGHT_VIEW_V


def _in_housing(
    image: NDArray[np.uint8], colour: LightState, x0: int, y0: int, w: int, h: int, lit: float
) -> bool:
    """Whether a lamp has the dark of a signal head where its other lenses are:
    below a red lamp, above a green one, both sides of an amber.

    A lit poster's red, in the game, stood on the poster's bright blue and
    stopped the truck at a roundabout with no light; a lamp's neighbours are
    unlit lenses in a black housing, day and night.
    """
    gap = max(1, h // 3)
    span = max(3, 2 * h)
    sides = []
    if colour in (LightState.RED, LightState.AMBER):
        sides.append(image[y0 + h + gap : y0 + h + gap + span, x0 : x0 + w])
    if colour in (LightState.GREEN, LightState.AMBER):
        sides.append(image[max(0, y0 - gap - span) : max(0, y0 - gap), x0 : x0 + w])
    for side in sides:
        if side.size == 0:
            return False
        if float(np.median(side.max(axis=2))) > HOUSING * lit:
            return False
    return True


def reading(lamps: Iterable[Lamp]) -> LightState:
    """What one frame's lamps say: red or green, whichever has more lamp area;
    amber only when neither is lit.

    The light ahead is the nearest, so its lamps are the largest; the next
    junction's, visible beyond it, are a few pixels each. Amber does not take
    part in the vote: a yellow priority-road sign above a green lamp in the
    game was 1460 px of "amber" against the lamp's 343, and outvoted it for
    45 s. A light's own amber shows with its red and green lamps both out.
    """
    areas = dict.fromkeys((LightState.RED, LightState.AMBER, LightState.GREEN), 0)
    for lamp in lamps:
        areas[lamp.colour] += lamp.area
    red, green = areas[LightState.RED], areas[LightState.GREEN]
    if max(red, green) >= MIN_LAMP_AREA_PX:
        return LightState.RED if red >= green else LightState.GREEN
    return LightState.AMBER if areas[LightState.AMBER] >= MIN_LAMP_AREA_PX else LightState.UNKNOWN


def _radius(lamp: Lamp) -> float:
    return math.sqrt(lamp.area / math.pi)


def in_same_head(lamp: Lamp, under: Lamp, spacings: float) -> bool:
    """Whether `lamp` could be the lamp up to `spacings` above `under` in one
    signal head. In city-01 the amber came on 11 px above the green, of a
    glow 3.5 px in radius, and the red 12 px above the amber.

    `under` may have been seen up to a second before, and on the move a
    head's lamps drift across the view: in the game an amber came on 25 px
    above and 5 px beside the green seen a second earlier, was not believed,
    and the truck, still taking the light for green, went through its red.
    So `DRIFT_PX` more either way.
    """
    r = _radius(under)
    rise = under.y - lamp.y
    across = abs(lamp.x - under.x)
    return across <= max(HEAD_SLACK_PX, 1.5 * r) + DRIFT_PX and -DRIFT_PX / 3 < rise <= (
        spacings * (LAMP_SPACING_RADII * r + HEAD_SLACK_PX) + DRIFT_PX
    )


@dataclass
class LightFilter:
    """Each frame's lamps in, a steady state out.

    Two changes are only believed where a light's head says they can happen,
    because each has a lookalike that is common and costly:

    - **Green to amber** only if the amber lamp comes on just above a green
      one just seen. Streets lit by sodium lamps are amber everywhere; one
      false amber stopped the truck at a green light in city-01.
    - **Red to green** only once no red lamp has been seen for
      `RED_MEMORY_S`, or if the green comes on in the head of a red one --
      lit, or the one the state turned red for, gone out: in Milan the
      crossing road's red stayed lit while ours was green, through two
      greens. A
      green sign is a green blob, and a false green at a red light is the
      worst mistake here. The red need not be in the same head as the green:
      in city-01 the red that went out was on the pole, and the green that
      came on was overhead.
    - **Green to red**, while the green is still lit, only if the red comes on
      in its head. Moving off at a green in Rotterdam, a lorry's tail lamps
      came into view beside the pole, bigger than the green, and held the
      truck at a green light until it gave up. A green gone out lets any red
      through.
    """

    state: LightState = LightState.UNKNOWN
    _candidate: LightState = LightState.UNKNOWN
    _run: int = 0
    _seen_at: float | None = None
    _last: dict[LightState, tuple[float, tuple[Lamp, ...]]] = field(default_factory=dict)
    _reds: list[list[float]] = field(default_factory=list)
    """Red lamps being followed: x, y, lit since, last seen."""

    _ours: list[Lamp] = field(default_factory=list)
    """The red lamps the state turned red for, where last seen."""

    def update(self, lamps: Sequence[Lamp], now: float) -> LightState:
        steady = self._follow_reds(lamps, now)
        self._ours = [self._lit_as(red, lamps) or red for red in self._ours]
        if self.state is LightState.RED:
            # A red lit while the state is red is ours too, seen late or not:
            # in Bern the state turned red for a red give-way sign above the
            # light, the light's own red was never taken for ours, and its
            # green, the sign still red above it, was never believed.
            for lamp in lamps:
                if lamp.colour is LightState.RED and not any(
                    self._lit_as(red, (lamp,)) for red in self._ours
                ):
                    self._ours.append(lamp)
        seen = reading(lamps)
        if self.state is LightState.RED and self._green_in_our_head(lamps):
            seen = LightState.GREEN  # however big another head's red
        if seen is LightState.RED and self.state is not LightState.RED and not steady:
            seen = LightState.UNKNOWN  # not yet: it may be a flash
        if seen is not LightState.UNKNOWN and not self._possible(seen, lamps, now):
            # Not the light's. The state stands only if its own colour is lit:
            # taken for the state as it stood whatever was lit, a steady amber
            # in Milan -- not believed after a red -- held the red for good.
            lit = any(lamp.colour is self.state for lamp in lamps)
            seen = self.state if lit else LightState.UNKNOWN
        for colour in (LightState.RED, LightState.AMBER, LightState.GREEN):
            mine = tuple(lamp for lamp in lamps if lamp.colour is colour)
            if mine:
                self._last[colour] = (now, mine)
        if seen is LightState.UNKNOWN:
            if self._seen_at is None or now - self._seen_at > HOLD_S:
                self.state = LightState.UNKNOWN
            self._run = 0
            return self.state
        self._seen_at = now
        if seen is self._candidate:
            self._run += 1
        else:
            self._candidate, self._run = seen, 1
        if self._run >= CONFIRM_FRAMES:
            if seen is LightState.RED and self.state is not LightState.RED:
                self._ours = [lamp for lamp in lamps if lamp.colour is LightState.RED]
            self.state = seen
        return self.state

    def _follow_reds(self, lamps: Sequence[Lamp], now: float) -> bool:
        """Follow each red lamp from frame to frame; whether one has been lit
        for `RED_STEADY_S`. Lamp by lamp: a pylon's warning lamps and a
        chimney's, flashing out of step, made one red that hardly went out."""
        self._reds = [red for red in self._reds if now - red[3] <= RED_GAP_S]
        steady = False
        for lamp in lamps:
            if lamp.colour is not LightState.RED:
                continue
            reach = DRIFT_PX + 2.0 * _radius(lamp)
            near = [r for r in self._reds if math.hypot(r[0] - lamp.x, r[1] - lamp.y) <= reach]
            if near:
                red = min(near, key=lambda r: math.hypot(r[0] - lamp.x, r[1] - lamp.y))
                red[0], red[1], red[3] = lamp.x, lamp.y, now
            else:
                red = [lamp.x, lamp.y, now, now]
                self._reds.append(red)
            steady = steady or now - red[2] >= RED_STEADY_S
        return steady

    def _green_in_our_head(self, lamps: Sequence[Lamp]) -> bool:
        """Whether a red the state turned red for has gone out and a green
        come on in its head."""
        greens = [lamp for lamp in lamps if lamp.colour is LightState.GREEN]
        return any(
            self._lit_as(red, lamps) is None and in_same_head(red, green, 2.0)
            for red in self._ours
            for green in greens
        )

    @staticmethod
    def _lit_as(red: Lamp, lamps: Sequence[Lamp]) -> Lamp | None:
        """The red lamp of `lamps` that is `red` drifted a little, if lit."""
        reach = DRIFT_PX + 2.0 * _radius(red)
        near = [
            lamp
            for lamp in lamps
            if lamp.colour is LightState.RED and math.hypot(lamp.x - red.x, lamp.y - red.y) <= reach
        ]
        return min(near, key=lambda lamp: math.hypot(lamp.x - red.x, lamp.y - red.y), default=None)

    def _recent(self, colour: LightState, now: float, within: float) -> tuple[Lamp, ...]:
        when, lamps = self._last.get(colour, (-math.inf, ()))
        return lamps if now - when <= within else ()

    def _possible(self, seen: LightState, lamps: Sequence[Lamp], now: float) -> bool:
        mine = [lamp for lamp in lamps if lamp.colour is seen]
        if seen is LightState.AMBER:
            if self.state is LightState.AMBER:
                return True
            if self.state is not LightState.GREEN:
                # Amber out of nowhere is far more often a lit sign than a light.
                return False
            greens = self._recent(LightState.GREEN, now, GREEN_MEMORY_S)
            return any(in_same_head(a, g, 1.0) for a in mine for g in greens)
        if seen is LightState.GREEN and self.state is not LightState.GREEN:
            if self.state is LightState.RED and self._green_in_our_head(lamps):
                # Our red has gone out and green come on in its head; a red
                # still lit is another head's, as the crossing road's is.
                return True
            reds = self._recent(LightState.RED, now, RED_MEMORY_S)
            return not reds or any(in_same_head(r, g, 2.0) for g in mine for r in reds)
        if seen is LightState.RED and self.state is LightState.GREEN:
            greens = [lamp for lamp in lamps if lamp.colour is LightState.GREEN]
            return not greens or any(in_same_head(r, g, 2.0) for r in mine for g in greens)
        return True
