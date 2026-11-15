"""Reference paths: a route the vehicle should follow.

A path is a polyline in the world's ground plane, parameterised by arc length,
with the queries a lateral controller needs: where am I on it, how far off it
am I, which way is it pointing, and how sharply does it bend.

Where the points come from is deliberately not this module's business. Today
they come from a recorded drive -- teach-and-repeat, which needs no map at all
and is enough to build and tune a lateral controller against. Later they will
come from the road geometry in the game's own archives. The controller does not
change when the source does, which is why the source is kept out of here.

Positions are world `(x, z)`, the ground plane the game uses; `y` is height and
plays no part in lateral control. Headings are the direction of travel along
the path, in radians, using `atan2(dx, dz)` so that they compose with
`core.bridge.convert.forward_unit` without a conversion nobody remembers.

**The turn conventions here are measured, not assumed.** A lateral controller
built on an inverted one steers away from the path, and nothing about the
symptom says which sign is wrong, so the chain is written down:

* The game's control mix makes a negative `steering` value a right turn.
* In a recorded drive, steering value and yaw rate correlate at 0.948 with a
  positive slope -- so **turning right decreases heading**.
* Facing heading zero the vehicle travels towards `+z`; turning right therefore
  swings it towards `-x`, which makes **`+x` the left-hand side**.

From which everything else follows, and is asserted in the tests:

| | heading | curvature | direction |
|---|---|---|---|
| right turn | decreases | negative | towards `-x` |
| left turn | increases | positive | towards `+x` |

and `lateral_error` is positive when the vehicle is **left** of the path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["PathSample", "ReferencePath", "TrailerTracker", "widen_for_trailer"]

#: Points closer together than this add nothing but noise to the heading and
#: curvature estimates, which are derived by differencing.
MIN_SPACING_M: Final = 0.5


@dataclass(frozen=True, slots=True)
class PathSample:
    """The path, evaluated at one arc length."""

    s: float
    """Distance along the path from its start, metres."""

    x: float
    z: float
    heading: float
    """Direction of travel at this point, radians, `atan2(dx, dz)`."""

    curvature: float
    """Signed curvature, 1/m. Positive is a left turn; see the module docstring.

    Derived by differencing a polyline, so it is noisy at short spacings and
    meaningless at spacings below `MIN_SPACING_M`. A lateral controller should
    use it for speed limiting and look-ahead, not as a feedforward it trusts
    pointwise.
    """


class ReferencePath:
    """An arc-length parameterised polyline with nearest-point queries.

    Built from raw points -- typically a recorded drive -- which are resampled
    to a uniform spacing first. Resampling matters: a drive that stops at a
    junction leaves hundreds of points in one place, and every geometric
    quantity derived by differencing goes to nonsense there.
    """

    def __init__(self, points: np.ndarray, *, spacing: float = 2.0) -> None:
        """
        Args:
            points: `(n, 2)` array of world `(x, z)`.
            spacing: metres between resampled points.

        Raises:
            ValueError: fewer than two distinct points, or a spacing too small
                to derive a heading from.
        """
        if spacing < MIN_SPACING_M:
            raise ValueError(f"spacing must be at least {MIN_SPACING_M} m")
        raw = np.asarray(points, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("points must be an (n, 2) array of (x, z)")

        raw = _drop_duplicates(raw)
        if len(raw) < 2:
            raise ValueError("a path needs at least two distinct points")

        self._spacing = spacing
        self._points = _resample(raw, spacing)
        self._s = _arc_length(self._points)
        self._heading = _headings(self._points)
        self._curvature = _curvatures(self._points, self._s, self._heading)

    @property
    def points(self) -> np.ndarray:
        """Resampled `(n, 2)` path points."""
        return self._points

    @property
    def spacing(self) -> float:
        """Metres between the resampled points."""
        return self._spacing

    @property
    def length(self) -> float:
        """Total arc length, metres."""
        return float(self._s[-1])

    def __len__(self) -> int:
        return len(self._points)

    def sample_at(self, s: float) -> PathSample:
        """The path at arc length `s`, clamped to its ends."""
        s = min(max(float(s), 0.0), float(self._s[-1]))
        index = int(self._s.searchsorted(s, side="right")) - 1
        index = max(0, min(index, len(self._points) - 2))
        span = self._s[index + 1] - self._s[index]
        t = 0.0 if span <= 0.0 else (s - self._s[index]) / span

        x, z = self._points[index] + t * (self._points[index + 1] - self._points[index])
        return PathSample(
            s=s,
            x=float(x),
            z=float(z),
            heading=_interpolate_angle(self._heading[index], self._heading[index + 1], t),
            curvature=float(
                self._curvature[index] + t * (self._curvature[index + 1] - self._curvature[index])
            ),
        )

    def nearest(
        self, x: float, z: float, *, near: float | None = None, window: float = 60.0
    ) -> PathSample:
        """The point on the path closest to `(x, z)`.

        Args:
            x, z: query position in the world ground plane.
            near: arc length to search around, normally the previous result.
                Without it the whole path is searched, which is both slower and
                wrong where a route crosses itself -- the nearest point in
                space is not necessarily the one the vehicle is on.
            window: how far either side of `near` to look, metres.
        """
        if near is None:
            lo, hi = 0, len(self._points)
        else:
            lo = int(np.searchsorted(self._s, near - window))
            hi = int(np.searchsorted(self._s, near + window)) + 1
            lo, hi = max(0, lo), min(len(self._points), hi)
            if hi - lo < 2:
                lo, hi = 0, len(self._points)

        segment = self._points[lo:hi]
        distances = np.hypot(segment[:, 0] - x, segment[:, 1] - z)
        index = lo + int(np.argmin(distances))

        # Refine onto the adjacent segment rather than returning the vertex: at
        # a 2 m spacing a vertex is up to a metre from the true foot of the
        # perpendicular, which is the same order as the error being controlled.
        return self.sample_at(self._project(index, x, z))

    def lateral_error(self, x: float, z: float, sample: PathSample) -> float:
        """Signed distance from the path at `sample`, positive to the left.

        So a positive error means the vehicle must steer right to return. Which
        side is "left" was established by measurement rather than assumed; see
        the module docstring.
        """
        dx, dz = x - sample.x, z - sample.z
        # Left of a heading pointing along (sin h, cos h) is (cos h, -sin h).
        return dx * math.cos(sample.heading) - dz * math.sin(sample.heading)

    def _project(self, index: int, x: float, z: float) -> float:
        """Arc length of the foot of the perpendicular near vertex `index`."""
        best = self._s[index]
        best_distance = math.inf
        for i in (index - 1, index):
            if i < 0 or i + 1 >= len(self._points):
                continue
            (x0, z0), (x1, z1) = self._points[i], self._points[i + 1]
            dx, dz = x1 - x0, z1 - z0
            span = dx * dx + dz * dz
            if span <= 0.0:
                continue
            t = ((x - x0) * dx + (z - z0) * dz) / span
            t = min(max(t, 0.0), 1.0)
            px, pz = x0 + t * dx, z0 + t * dz
            distance = (px - x) ** 2 + (pz - z) ** 2
            if distance < best_distance:
                best_distance = distance
                best = self._s[i] + t * (self._s[i + 1] - self._s[i])
        return float(best)


def _drop_duplicates(points: np.ndarray) -> np.ndarray:
    """Remove consecutive points closer than a centimetre.

    A stationary vehicle produces thousands of these, and they make every
    derived quantity divide by roughly zero.
    """
    if len(points) < 2:
        return points
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    keep = np.concatenate(([True], step > 0.01))
    return points[keep]


class TrailerTracker:
    """Where the trailer is on the path, and how far off it.

    Separate from the tractor's own projection for one reason: the trailer is
    behind the tractor, and behind the start of a closed loop is the far end
    of it. An unseeded search there does not merely lose precision, it answers
    with the other side of the loop -- 227 m of lateral error in a report that
    was otherwise sound.

    So the search is seeded from the tractor, and nothing is reported until
    the trailer is actually on the route.
    """

    def __init__(self, path: ReferencePath, behind: float) -> None:
        """
        Args:
            path: the route being followed.
            behind: distance from the tractor's reference point back to the
                trailer's, metres. Only needs to be roughly right; it seeds a
                search, it does not answer one.
        """
        if behind <= 0.0:
            raise ValueError("behind must be positive")
        self._path = path
        self._behind = behind
        self._progress: float | None = None

    @property
    def progress(self) -> float | None:
        """Arc length of the trailer along the path, or None before it joins."""
        return self._progress

    def update(self, x: float, z: float, tractor_progress: float) -> float | None:
        """Project the trailer onto the path.

        Args:
            x, z: the trailer's world position in the ground plane.
            tractor_progress: where the tractor is, to seed the search.

        Returns:
            Signed lateral error, positive to the left, or None while the
            trailer is still behind the start of the route.
        """
        if tractor_progress <= self._behind:
            return None
        seed = self._progress if self._progress is not None else tractor_progress - self._behind
        sample = self._path.nearest(x, z, near=seed)
        self._progress = sample.s
        return self._path.lateral_error(x, z, sample)


TRAILER_LENGTH_M: Final = 8.0
"""Kingpin to trailer axles of the box trailer driven, near enough (8.1)."""

WIDEN_SHARE: Final = 0.5
WIDEN_MAX_M: Final = 1.2
WIDEN_LEAD_M: Final = 15.0

WIDER_SHARE: Final = 1.0
WIDER_MAX_M: Final = 2.5
WIDER_RAMP_M: Final = 20.0
"""Where turns are taken wider still (`widen_for_trailer`'s `wider`): a
border's customs lanes, where trucks wait on the inside of the bends.
Taking half the trailer's cut, the truck passed one waiting there close
enough for the supervisor to stop it."""


def widen_for_trailer(
    points: np.ndarray,
    *,
    trailer_length: float = TRAILER_LENGTH_M,
    share: float = WIDEN_SHARE,
    max_shift: float = WIDEN_MAX_M,
    lead: float = WIDEN_LEAD_M,
    spacing: float = 1.0,
    wider: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    exact: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    room=None,
) -> np.ndarray:
    """The line the tractor should drive for the trailer to follow the lane.

    A trailer tracks inside its tractor: in a steady turn of radius R its
    axles run R - sqrt(R^2 - L^2) further in, 2.7 m at a 12 m junction
    corner, where it met whatever stood on the inside in the game. So in
    turns the tractor goes `share` of that further out, at most `max_shift`,
    beginning `lead` metres before the turn and easing back after it, as a
    driver swings wide. Straights are left where they are.

    `room`, given `(points, outward, wanted)` -- each point, the unit vector to
    the outside of its turn, and how far out it would go -- says how far out
    each may go: in Milan a T-junction
    swung 1.2 m wide put the tractor onto a guard house's island on the outside.

    Args:
        points: `(n, 2)` world `(x, z)` in the direction of travel.

    Returns:
        `(m, 2)` points, resampled at `spacing`.
    """
    raw = _drop_duplicates(np.asarray(points, dtype=float))
    if len(raw) < 3:
        return raw
    line = _resample(raw, spacing)
    reach = max(1, round(lead / spacing))
    if len(line) < 5 or len(line) <= max(reach, round(WIDER_RAMP_M / spacing)):
        # Shorter than the windows the shift is smoothed over: no turn on it
        # to swing wide for. By Uppsala a route of 8 m to the depot took the
        # GPS worker down, and the truck drove on blind to the old one.
        return line
    s = _arc_length(line)
    heading = _headings(line)
    curvature = _curvatures(line, s, heading)
    window = max(1, round(5.0 / spacing))
    curvature = np.convolve(curvature, np.ones(window) / window, mode="same")
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-6)
    inside = np.where(
        radius > trailer_length,
        radius - np.sqrt(np.maximum(radius**2 - trailer_length**2, 0.0)),
        trailer_length,
    )
    # Positive curvature turns left; the outside of a left turn is the right.

    def between(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        weight = np.zeros(len(line))
        for start, end in pairs:
            i0 = int(np.argmin(np.hypot(*(line - np.asarray(start)[:2]).T)))
            i1 = i0 + int(np.argmin(np.hypot(*(line[i0:] - np.asarray(end)[:2]).T)))
            weight[i0 : i1 + 1] = 1.0
        ramp = max(1, round(WIDER_RAMP_M / spacing))
        return np.convolve(weight, np.ones(ramp) / ramp, mode="same")

    def spread(share_: float, max_: float) -> np.ndarray:
        shift = np.minimum(share_ * inside, max_) * np.sign(curvature)
        shift = _spread(shift, reach)
        return np.convolve(shift, np.ones(reach) / reach, mode="same")

    shift = spread(share, max_shift)
    if wider:
        # Between each (start, end) of `wider`, turns go wider still.
        weight = between(wider)
        shift = (1.0 - weight) * shift + weight * spread(WIDER_SHARE, WIDER_MAX_M)
    if exact:
        # A line a driver drove already swings wide where it needs to.
        shift = (1.0 - between(exact)) * shift
    left = np.column_stack([np.cos(heading), -np.sin(heading)])
    if room is not None:
        outward = -np.sign(shift)[:, None] * left
        cap = np.asarray(room(line, outward, np.abs(shift)), dtype=float)
        padded = np.pad(cap, reach, mode="edge")
        cap = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1).min(axis=1)
        shift = np.sign(shift) * np.minimum(np.abs(shift), cap)
        shift = np.convolve(shift, np.ones(reach) / reach, mode="same")
    return line - shift[:, None] * left


def _spread(values: np.ndarray, reach: int) -> np.ndarray:
    """Each value replaced by the largest, either way, within `reach` of it."""
    padded_pos = np.pad(np.clip(values, 0.0, None), reach)
    padded_neg = np.pad(np.clip(-values, 0.0, None), reach)
    windows = 2 * reach + 1
    pos = np.lib.stride_tricks.sliding_window_view(padded_pos, windows).max(axis=1)
    neg = np.lib.stride_tricks.sliding_window_view(padded_neg, windows).max(axis=1)
    return pos - neg


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample to a uniform arc-length spacing, the last step shorter: the
    path ends where its points do. Cut to the spacing, a path to a gas pump
    ended 1.8 m short of it, and the truck stopped where the game would not
    fill the tank."""
    s = _arc_length(points)
    if s[-1] < spacing:
        return points
    targets = np.arange(0.0, s[-1] + 1e-9, spacing)
    if s[-1] - targets[-1] > 0.01 * spacing:
        targets = np.append(targets, s[-1])
    return np.column_stack(
        [np.interp(targets, s, points[:, 0]), np.interp(targets, s, points[:, 1])]
    )


def _arc_length(points: np.ndarray) -> np.ndarray:
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate(([0.0], np.cumsum(step)))


def _headings(points: np.ndarray) -> np.ndarray:
    """Direction of travel at each point, radians.

    `atan2(dx, dz)` rather than the usual `atan2(dy, dx)`, so that a heading
    here means the same thing as one from `convert.forward_unit`.
    """
    dx = np.gradient(points[:, 0])
    dz = np.gradient(points[:, 1])
    return np.arctan2(dx, dz)


def _curvatures(points: np.ndarray, s: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Signed curvature from the rate of change of heading with arc length."""
    unwrapped = np.unwrap(heading)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.gradient(unwrapped, s)
    return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)


def _interpolate_angle(a: float, b: float, t: float) -> float:
    """Interpolate between two angles the short way round."""
    delta = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + delta * t
