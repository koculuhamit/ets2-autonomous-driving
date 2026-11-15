"""Dubins paths: the shortest way forward between two poses at a turning radius.

Poses are (x, z, heading), heading `atan2(dx, dz)` -- increasing to the
left. A path is three pieces, each a left turn (`L`), right turn (`R`) or
straight (`S`), of the six words LSL, RSR, LSR, RSL, RLR and LRL; the
shortest is taken. Written for the parking planner (removed 7 October
2026); the way through a junction turn too tight for the lane
(core.planning.tight_turns) is one.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = ["dubins_length", "dubins_path"]

_WORDS: Final = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def _to_standard(start, end, radius):
    """The end pose in the start's frame, scaled to unit radius, in the
    usual (x along heading, y to the left, angle counter-clockwise) form."""
    dx, dz = end[0] - start[0], end[1] - start[1]
    heading = start[2]
    forward = dx * math.sin(heading) + dz * math.cos(heading)
    left = dx * math.cos(heading) - dz * math.sin(heading)
    d = math.hypot(forward, left) / radius
    theta = math.atan2(left, forward)
    alpha = _mod(-theta)
    beta = _mod(end[2] - heading - theta)
    return d, alpha, beta


def _segments(d: float, a: float, b: float) -> dict[str, tuple[float, float, float]]:
    sa, ca, sb, cb = math.sin(a), math.cos(a), math.sin(b), math.cos(b)
    cab = math.cos(a - b)
    found = {}
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= -1e-9:
        t = math.atan2(cb - ca, d + sa - sb)
        found["LSL"] = (_mod(-a + t), math.sqrt(max(p2, 0.0)), _mod(b - t))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= -1e-9:
        t = math.atan2(ca - cb, d - sa + sb)
        found["RSR"] = (_mod(a - t), math.sqrt(max(p2, 0.0)), _mod(-b + t))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= -1e-9:
        p = math.sqrt(max(p2, 0.0))
        t = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        found["LSR"] = (_mod(-a + t), p, _mod(-_mod(b) + t))
    p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
    if p2 >= -1e-9:
        p = math.sqrt(max(p2, 0.0))
        t = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        found["RSL"] = (_mod(a - t), p, _mod(b - t))
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1:
        p = _mod(2 * math.pi - math.acos(tmp))
        t = _mod(a - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        found["RLR"] = (t, p, _mod(a - b - t + p))
    tmp = (6.0 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1:
        p = _mod(2 * math.pi - math.acos(tmp))
        t = _mod(-a - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        found["LRL"] = (t, p, _mod(_mod(b) - a - t + p))
    return found


def _best(start, end, radius):
    d, a, b = _to_standard(start, end, radius)
    options = _segments(d, a, b)
    if not options:
        return None
    word, pieces = min(options.items(), key=lambda kv: sum(kv[1]))
    return word, pieces


def dubins_length(start, end, radius: float) -> float:
    """The shortest forward path's length, metres; inf if none."""
    best = _best(start, end, radius)
    return math.inf if best is None else radius * sum(best[1])


def dubins_path(
    start, end, radius: float, step: float = 0.5
) -> list[tuple[float, float, float, float]]:
    """The shortest forward path as (x, z, heading, curvature) every `step`
    metres -- curvature positive turning left -- ending at `end`."""
    best = _best(start, end, radius)
    if best is None:
        return []
    word, pieces = best
    x, z, heading = start
    out = [(x, z, heading, 0.0)]
    for letter, piece in zip(word, pieces, strict=True):
        length = piece * radius
        if length < 1e-6:
            continue
        curvature = {"L": 1.0 / radius, "R": -1.0 / radius, "S": 0.0}[letter]
        count = max(1, math.ceil(length / step))
        ds = length / count
        for _ in range(count):
            mid = heading + 0.5 * curvature * ds
            x += ds * math.sin(mid)
            z += ds * math.cos(mid)
            heading += curvature * ds
            out.append((x, z, heading, curvature))
    return out


def _mod(angle: float) -> float:
    return angle % (2.0 * math.pi)
