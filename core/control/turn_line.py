"""Round a tight turn wide, over the mouth of the road turned into, when
nothing waits there; kept off it otherwise.

Kept off that mouth always (the driver's rule, 11 November, after lorries and
a coach waiting there were hit), the line round a 9.8 m turn in Uppsala was
pulled in, and the trailer cut 3.4 m inside and knocked a sign down
(16 November). The driver's choice: wide when the mouth is clear.

The path keeps the line kept off it (core.planning.tight_turns.kept_in); a
`WideSpan` says how far right of it the wide line runs, and while the mouth
is seen clear the steering aims that much aside, eased in and out at
`SHARE_RATE`. From where the junction begins the choice is held: not changed
under the truck halfway round.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from core.planning.tight_turns import WideSpan

__all__ = ["SHARE_RATE", "WideTurnChoice"]

SHARE_RATE: Final = 0.5
"""How quickly the aim moves between the two lines: the share of the way
from one to the other, a second."""

LOOK_BEFORE_M: Final = 5.0
"""A span is looked at from this far before its shifts begin."""


class WideTurnChoice:
    """Which line round the tight turn ahead: the steering's aim shift."""

    def __init__(self) -> None:
        self._span: WideSpan | None = None
        self._share = 0.0
        self._wanted = 0.0
        self.chosen: str | None = None
        """"wide" or "kept", once held for the span; None before."""

    def shift(
        self,
        *,
        spans: Sequence[WideSpan],
        progress: float,
        mouth_clear: Callable[[object], bool] | None,
        dt: float,
    ) -> Callable[[float], float] | None:
        """The aim shift (metres right of the path, by distance along it) for
        now, or None away from any wide span. `mouth_clear` tells whether a
        mouth is seen clear; without one, the line is always kept."""
        span = next((s for s in spans if s.begins - LOOK_BEFORE_M <= progress <= s.ends), None)
        if span is not self._span:
            self._span, self._share, self._wanted, self.chosen = span, 0.0, 0.0, None
        if span is None:
            return None
        if progress < span.turns_at:
            clear = mouth_clear is not None and bool(mouth_clear(span.mouth))
            self._wanted = 1.0 if clear else 0.0
        elif self.chosen is None:
            self.chosen = "wide" if self._wanted > 0.5 else "kept"
        step = SHARE_RATE * max(dt, 0.0)
        if self._share < self._wanted:
            self._share = min(self._share + step, self._wanted)
        else:
            self._share = max(self._share - step, self._wanted)
        share = self._share
        if share <= 0.0:
            return None
        return lambda s: share * span.shift_at(s)
