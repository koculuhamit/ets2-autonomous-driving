"""How well a detector finds what a reference says is there.

The reference here is usually another detector's labels (tools/pseudo_label.py)
rather than hand-drawn boxes, so the numbers say how close a model comes to
the teacher, not to the truth. They are still the right numbers to compare
two students by.

Vehicles are matched regardless of kind: whether a van is a car or a truck
matters much less for following it than whether it is seen at all. Recall is
also broken down by box height, which stands in for distance -- a car 20
pixels tall is far away, and far away is where the pretrained model failed.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = ["HEIGHT_BINS", "Score", "match"]

MATCH_IOU: Final = 0.5
HEIGHT_BINS: Final = (0.0, 16.0, 32.0, 64.0, 128.0, float("inf"))
"""Box heights in frame pixels, far to near, for recall by distance."""

Box = tuple[float, float, float, float]


def match(
    found: Sequence[Box], reference: Sequence[Box], overlap: float = MATCH_IOU
) -> tuple[list[bool], list[bool]]:
    """Which found boxes are right, and which reference boxes were found.

    Greedy on overlap, best pair first; each box takes part in at most one
    pair. `found` is expected most certain first, as detectors return it,
    but the pairing does not depend on the order.
    """
    hits_found = [False] * len(found)
    hits_reference = [False] * len(reference)
    pairs = []
    for i, a in enumerate(found):
        for j, b in enumerate(reference):
            iou = _iou(a, b)
            if iou >= overlap:
                pairs.append((iou, i, j))
    for _, i, j in sorted(pairs, reverse=True):
        if not hits_found[i] and not hits_reference[j]:
            hits_found[i] = hits_reference[j] = True
    return hits_found, hits_reference


@dataclass
class Score:
    """Precision and recall accumulated over frames."""

    true_positives: int = 0
    false_positives: int = 0
    missed: int = 0
    by_height: dict[tuple[float, float], list[int]] = field(default_factory=dict)
    """(low, high) -> [found, total] reference boxes of that height."""

    def add(self, found: Sequence[Box], reference: Sequence[Box]) -> None:
        hits_found, hits_reference = match(found, reference)
        self.true_positives += sum(hits_found)
        self.false_positives += len(hits_found) - sum(hits_found)
        self.missed += len(hits_reference) - sum(hits_reference)
        for box, hit in zip(reference, hits_reference, strict=True):
            height = box[3] - box[1]
            for low, high in itertools.pairwise(HEIGHT_BINS):
                if low <= height < high:
                    tally = self.by_height.setdefault((low, high), [0, 0])
                    tally[0] += hit
                    tally[1] += 1

    @property
    def precision(self) -> float:
        judged = self.true_positives + self.false_positives
        return self.true_positives / judged if judged else float("nan")

    @property
    def recall(self) -> float:
        present = self.true_positives + self.missed
        return self.true_positives / present if present else float("nan")


def _iou(a: Box, b: Box) -> float:
    across = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    down = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = across * down
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return float(overlap / union) if union > 0 else 0.0
