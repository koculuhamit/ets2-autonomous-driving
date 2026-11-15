"""Whose map it is: DLC guards, ownership, and the roads that follow from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.map import Sector, dlc


def test_every_guard_the_map_uses_is_known() -> None:
    """Europe uses guards 0 to 25 except 8. An unknown guard would be refused."""
    for guard in range(26):
        assert guard in dlc.GUARDS


def test_the_base_game_needs_nothing() -> None:
    assert dlc.allowed(0, ())


def test_a_single_dlc_guard_needs_that_dlc() -> None:
    assert dlc.allowed(4, {"italy"})
    assert not dlc.allowed(4, {"france"})


def test_a_border_between_two_dlcs_needs_both() -> None:
    """Where two DLC regions meet, the road exists only for someone with both."""
    assert not dlc.allowed(5, {"italy"})
    assert not dlc.allowed(5, {"france"})
    assert dlc.allowed(5, {"italy", "france"})


def test_an_unknown_guard_is_refused() -> None:
    """A new DLC's roads in a cache built for someone without it is the worse mistake."""
    assert not dlc.allowed(99, dlc.ALL_RELEASED)


def test_the_shelved_dlc_is_in_nobody_s_map() -> None:
    """Heart of Russia was never released, and its roads are in the data anyway."""
    assert "russia" not in dlc.ALL_RELEASED
    assert not dlc.allowed(13, dlc.ALL_RELEASED)
    assert not dlc.allowed(14, dlc.ALL_RELEASED)


def test_every_releasable_dlc_has_an_archive() -> None:
    assert set(dlc.ARCHIVES) == dlc.ALL_RELEASED


def test_installed_dlcs_are_read_from_the_archives_present(tmp_path: Path) -> None:
    (tmp_path / "dlc_north.scs").write_bytes(b"")
    (tmp_path / "dlc_it.scs").write_bytes(b"")
    (tmp_path / "dlc_christmas.scs").write_bytes(b"")  # not a map DLC
    assert dlc.installed(tmp_path) == {"scandinavia", "italy"}


def test_a_profile_can_be_named(tmp_path: Path) -> None:
    assert dlc.parse_profile("all") == dlc.ALL_RELEASED
    assert dlc.parse_profile("italy, france") == {"italy", "france"}
    (tmp_path / "dlc_fr.scs").write_bytes(b"")
    assert dlc.parse_profile("installed", tmp_path) == {"france"}


def test_a_misspelt_dlc_is_an_error_not_an_empty_map() -> None:
    with pytest.raises(ValueError, match="unknown DLC itly"):
        dlc.parse_profile("itly")


def test_the_shelved_dlc_cannot_be_asked_for() -> None:
    with pytest.raises(ValueError, match="unknown DLC russia"):
        dlc.parse_profile("russia")


def test_installed_needs_a_game_directory() -> None:
    with pytest.raises(ValueError, match="game directory"):
        dlc.parse_profile("installed")


def test_a_map_restricted_to_some_dlcs_drops_the_rest(tmp_path: Path) -> None:
    def road(uid: str, guard: int) -> dict:
        return {
            "uid": uid,
            "road_look": "look",
            "length": 10.0,
            "start_node": f"{uid}a",
            "end_node": f"{uid}b",
            "ai_vehicles": True,
            "left_hand_traffic": False,
            "hidden": False,
            "dlc_guard": guard,
            "country": 0,
            "points": [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        }

    document = {
        "format": "ets2ad-map-roads",
        "format_version": 6,
        "sector_size": 4000,
        "sample_spacing": 2.0,
        "sectors": [[0, 0]],
        "node_count": 0,
        "item_count": 0,
        "countries": {"0": "none"},
        "road_looks": {},
        "roads": [road("base", 0), road("it", 4), road("itfr", 5), road("ru", 13)],
        "prefabs": [],
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    sector = Sector.load(path)

    assert {r.uid for r in sector.restricted_to(()).roads} == {"base"}
    assert {r.uid for r in sector.restricted_to({"italy"}).roads} == {"base", "it"}
    assert {r.uid for r in sector.restricted_to(dlc.ALL_RELEASED).roads} == {
        "base",
        "it",
        "itfr",
    }
