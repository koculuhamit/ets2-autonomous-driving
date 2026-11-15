"""Scoring a detector against reference labels."""

from __future__ import annotations

import pytest

from core.perception.evaluation import Score, match


def test_each_reference_box_is_found_at_most_once() -> None:
    """Two boxes on one car: one is right, the other is a false positive."""
    found = [(0, 0, 10, 10), (1, 0, 11, 10)]
    hits_found, hits_reference = match(found, [(0, 0, 10, 10)])
    assert hits_found == [True, False]
    assert hits_reference == [True]


def test_the_better_pairing_wins_whatever_the_order() -> None:
    found = [(0, 0, 10, 10), (20, 0, 30, 10)]
    reference = [(21, 0, 31, 10), (3, 0, 13, 10)]
    assert match(found, reference) == ([True, True], [True, True])


def test_a_box_in_the_wrong_place_is_not_a_hit() -> None:
    assert match([(0, 0, 10, 10)], [(8, 0, 18, 10)]) == ([False], [False])


def test_precision_recall_and_recall_by_height() -> None:
    score = Score()
    near, far = (0, 0, 100, 80), (500, 300, 510, 310)
    score.add(found=[near, (900, 0, 950, 40)], reference=[near, far])
    score.add(found=[], reference=[far])
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1 / 3)
    assert score.by_height[(64.0, 128.0)] == [1, 1]
    assert score.by_height[(0.0, 16.0)] == [0, 2], "the far car was missed both times"


def test_nothing_to_judge_is_not_a_perfect_score() -> None:
    score = Score()
    score.add([], [])
    assert score.precision != score.precision and score.recall != score.recall  # NaN
