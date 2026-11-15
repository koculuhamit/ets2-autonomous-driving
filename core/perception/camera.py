"""Where a detection is on the road, from where it is in the image.

The game's interior camera is a pinhole looking along the road. With the road
flat, the row where a vehicle meets the road says how far away it is: rows
further below the horizon are nearer. The column then says how far to the
side it is, and the width of its box how wide it is in metres.

    distance = focal * height / (row - horizon)
    lateral  = (column - centre) * distance / focal

Four numbers describe the camera, and none of them come from the game's
settings: the FOV is not in the config files, and the telemetry gives the
head's offset from its default position, not the position itself. They are
measured from recordings instead (tools/calibrate_camera.py):

- **horizon**, where lane lines meet on a straight road.
- **height**, from how fast the lane widens below the horizon: the lane's
  width in pixels grows by `lane_width / height` per row, and the map knows
  the lane width.
- **focal x height**, the product every distance is proportional to, from
  driving straight: seen from above (`bird_eye_maps`), the road moves by what
  the camera's scale says it moved, and the telemetry knows how far it did.
  The focal length is this over the height.

Measuring the focal length from turns -- far scenery shifts by
`focal * tan(yaw)` -- gave 739 to 891 px depending on the drive. The game
turns the interior camera into bends on its own, so the camera yaws more than
the truck, and by an amount nobody reports. `focal_from_yaw` stays as the
first guess the road-motion measurement starts from.

The flat-road assumption breaks on crests and dips, and the horizon moves
when the cab pitches or the player looks around. Both make distances wrong
in proportion to how far the horizon is off; see
docs/results/distance_estimation.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CAMERA_FORMAT = "ets2ad-camera-1"

__all__ = [
    "GroundCamera",
    "Line",
    "bird_eye_maps",
    "focal_from_yaw",
    "fov_deg",
    "height_from_lane",
    "vanishing_point",
]


@dataclass(frozen=True, slots=True)
class GroundCamera:
    focal_px: float
    horizon_row: float
    """Image row of the horizon: where the road meets the sky, straight ahead."""

    centre_col: float
    """Image column straight ahead."""

    height_m: float
    """Of the eye above the road."""

    seat_right_m: float = 0.0
    """Of the eye from the truck's centreline, to the right; negative on the
    left, where a left-hand-drive truck's driver sits."""

    def distance(self, row: float) -> float | None:
        """How far ahead the road is at an image row; None at or above the horizon."""
        below = row - self.horizon_row
        if below <= 0:
            return None
        return self.focal_px * self.height_m / below

    def lateral(self, column: float, distance: float) -> float:
        """Metres to the right of straight ahead, of a point at `distance`."""
        return (column - self.centre_col) * distance / self.focal_px

    def size(self, pixels: float, distance: float) -> float:
        """Metres across, of something `pixels` wide at `distance`."""
        return pixels * distance / self.focal_px

    def locate(self, box: Sequence[float]) -> tuple[float, float, float] | None:
        """(distance, lateral, width) in metres of a box standing on the road.

        None when its bottom is not below the horizon -- nothing standing on
        a flat road can be there.
        """
        left, _, right, bottom = box
        distance = self.distance(bottom)
        if distance is None:
            return None
        lateral = self.lateral((left + right) / 2.0, distance)
        return distance, lateral, self.size(right - left, distance)

    @classmethod
    def from_json(cls, document: dict) -> GroundCamera:
        if document.get("format") != CAMERA_FORMAT:
            raise ValueError(f"not a camera ({document.get('format')!r})")
        return cls(
            focal_px=float(document["focal_px"]),
            horizon_row=float(document["horizon_row"]),
            centre_col=float(document["centre_col"]),
            height_m=float(document["height_m"]),
            seat_right_m=float(document.get("seat_right_m", 0.0)),
        )

    def to_json(self) -> dict:
        return {
            "format": CAMERA_FORMAT,
            "focal_px": round(self.focal_px, 1),
            "horizon_row": round(self.horizon_row, 1),
            "centre_col": round(self.centre_col, 1),
            "height_m": round(self.height_m, 3),
            "seat_right_m": round(self.seat_right_m, 2),
        }


def bird_eye_maps(
    camera: GroundCamera,
    ahead: tuple[float, float],
    across: tuple[float, float],
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.remap` maps showing the road from above.

    Row k of the result is `ahead[0] + k * step` metres ahead, column j is
    `across[0] + j * step` metres to the right. A camera whose scale is
    wrong by a factor stretches the view along the road by that factor.
    """
    distances = np.arange(ahead[0], ahead[1], step)
    offsets = np.arange(across[0], across[1], step)
    z, x = np.meshgrid(distances, offsets, indexing="ij")
    rows = camera.horizon_row + camera.focal_px * camera.height_m / z
    columns = camera.centre_col + camera.focal_px * x / z
    return columns.astype(np.float32), rows.astype(np.float32)


@dataclass(frozen=True, slots=True)
class Line:
    """An image line as column = a + b * row."""

    a: float
    b: float

    @classmethod
    def through(cls, x1: float, y1: float, x2: float, y2: float) -> Line:
        if y1 == y2:
            raise ValueError("a horizontal segment is not a lane line")
        b = (x2 - x1) / (y2 - y1)
        return cls(x1 - b * y1, b)

    def column(self, row: float) -> float:
        return self.a + self.b * row


def vanishing_point(lines: Sequence[Line]) -> tuple[float, float]:
    """(column, row) where lines meet, least squares over all of them.

    Each line says column - b * row = a; two or more with different slopes
    pin the point down.
    """
    if len(lines) < 2:
        raise ValueError("a vanishing point needs two lines")
    system = np.array([[1.0, -line.b] for line in lines])
    target = np.array([line.a for line in lines])
    if np.ptp(system[:, 1]) < 1e-6:
        raise ValueError("parallel lines in the image do not meet")
    (column, row), *_ = np.linalg.lstsq(system, target, rcond=None)
    return float(column), float(row)


def height_from_lane(left: Line, right: Line, lane_width_m: float) -> float:
    """The eye's height above the road, from the two lines bounding one lane.

    Their separation in pixels grows by `lane_width / height` per row below
    the horizon, whatever the focal length.
    """
    widening = right.b - left.b
    if widening <= 0:
        raise ValueError("the right line must open away from the left one below the horizon")
    return lane_width_m / widening


def focal_from_yaw(
    shifts_px: Sequence[float], yaws_rad: Sequence[float], tolerance_px: float = 3.0
) -> tuple[float, int]:
    """Focal length from how far distant scenery moved when the truck yawed.

    A least-squares slope of shift against tan(yaw) through the origin,
    refitted on the pairs within `tolerance_px` of the first fit: frames
    where something near moved through the view, or the player looked
    around, fit badly and are left out. Returns (focal, pairs used).
    """
    x = np.tan(np.asarray(yaws_rad, dtype=float))
    y = np.asarray(shifts_px, dtype=float)
    if x.size < 2 or not np.any(x):
        raise ValueError("no yaw to measure against")
    focal = float(np.dot(x, y) / np.dot(x, x))
    for _ in range(3):
        inliers = np.abs(y - focal * x) < tolerance_px
        if inliers.sum() < 2:
            break
        focal = float(np.dot(x[inliers], y[inliers]) / np.dot(x[inliers], x[inliers]))
    return abs(focal), int(np.sum(np.abs(y - focal * x) < tolerance_px))


def fov_deg(focal_px: float, width_px: float) -> float:
    """The horizontal field of view a focal length gives, for a frame width."""
    return math.degrees(2.0 * math.atan(width_px / 2.0 / focal_px))
