"""Which DLCs a piece of road belongs to, and which of them a player has.

`base_map.scs` carries the roads of every map DLC on every installation. What
a player can actually drive is decided per road and per prefab by a *DLC
guard*: 0 for the base game, otherwise a number standing for one DLC or for
two of them together -- the roads where two DLC regions meet exist only for
someone who owns both.

The game states the numbers nowhere readable. The table below is the one the
TruckLib author documents at github.com/sk-zk/map-docs/wiki/DLC-guard, and it
was checked against the map rather than trusted: for each guard, the countries
its roads are in and the prefab families it uses. All 24 guards present in the
map agree -- 4 is in Italy and uses `dlc_it` prefabs, 11 is in Spain and uses
the Iberian ones, 21 straddles Greece, Turkey and Bulgaria. See
`docs/results/map_extraction.md`.

One of them is a DLC nobody can own. Guard 13 is Heart of Russia, shelved
before release, and its roads are in the map regardless.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

__all__ = [
    "ALL_RELEASED",
    "ARCHIVES",
    "GUARDS",
    "UNRELEASED",
    "allowed",
    "installed",
    "parse_profile",
]

#: Guard value to the DLCs a player must own for the item to exist.
GUARDS: Final[dict[int, frozenset[str]]] = {
    0: frozenset(),
    1: frozenset({"east"}),
    2: frozenset({"scandinavia"}),
    3: frozenset({"france"}),
    4: frozenset({"italy"}),
    5: frozenset({"italy", "france"}),
    6: frozenset({"baltic"}),
    7: frozenset({"baltic", "east"}),
    8: frozenset({"baltic", "scandinavia"}),
    9: frozenset({"black_sea"}),
    10: frozenset({"black_sea", "east"}),
    11: frozenset({"iberia"}),
    12: frozenset({"iberia", "france"}),
    13: frozenset({"russia"}),
    14: frozenset({"russia", "baltic"}),
    15: frozenset({"krone"}),
    16: frozenset({"west_balkans"}),
    17: frozenset({"west_balkans", "east"}),
    18: frozenset({"west_balkans", "black_sea"}),
    19: frozenset({"feldbinder"}),
    20: frozenset({"greece"}),
    21: frozenset({"greece", "black_sea"}),
    22: frozenset({"greece", "west_balkans"}),
    23: frozenset({"nordic"}),
    24: frozenset({"nordic", "baltic"}),
    25: frozenset({"nordic", "scandinavia"}),
}

#: DLCs that exist in the map data and cannot be owned.
UNRELEASED: Final = frozenset({"russia"})

#: Every DLC a player could own.
ALL_RELEASED: Final = frozenset().union(*GUARDS.values()) - UNRELEASED

#: The archive a DLC installs, which is how an installation says it has it.
#:
#: Checked on this machine for scandinavia, krone and feldbinder, which are
#: installed. The map DLCs' names match the game's own achievement files in
#: `core.scs` (`achievements.dlc_it.sii` and so on). `east` has no achievement
#: file and its name is the community's, not verified here.
ARCHIVES: Final[dict[str, str]] = {
    "east": "dlc_east.scs",
    "scandinavia": "dlc_north.scs",
    "france": "dlc_fr.scs",
    "italy": "dlc_it.scs",
    "baltic": "dlc_balt.scs",
    "black_sea": "dlc_balkan_e.scs",
    "iberia": "dlc_iberia.scs",
    "krone": "dlc_krone.scs",
    "west_balkans": "dlc_balkan_w.scs",
    "feldbinder": "dlc_feldbinder.scs",
    "greece": "dlc_greece.scs",
    "nordic": "dlc_polar.scs",
}


def allowed(guard: int, owned: Iterable[str]) -> bool:
    """Whether an item with this guard exists for someone owning `owned`.

    An unknown guard is refused rather than let through: a new DLC's roads
    appearing in a cache built for somebody without it is the worse mistake.
    """
    required = GUARDS.get(guard)
    return required is not None and required <= frozenset(owned)


def installed(game_directory: str | Path) -> frozenset[str]:
    """The DLCs whose archives are present in a game installation."""
    directory = Path(game_directory)
    return frozenset(dlc for dlc, archive in ARCHIVES.items() if (directory / archive).is_file())


def parse_profile(profile: str, game_directory: str | Path | None = None) -> frozenset[str]:
    """Turn a `--dlc` argument into a set of owned DLCs.

    `all` is every released DLC, which is what a cache meant for someone else
    should cover. `installed` is what this game directory has. Anything else
    is a comma-separated list of names from `ALL_RELEASED`.

    Raises:
        ValueError: an unknown name, or `installed` without a directory.
    """
    if profile == "all":
        return ALL_RELEASED
    if profile == "installed":
        if game_directory is None:
            raise ValueError("--dlc installed needs the game directory")
        return installed(game_directory)
    names = frozenset(name.strip() for name in profile.split(",") if name.strip())
    unknown = names - ALL_RELEASED
    if unknown:
        raise ValueError(
            f"unknown DLC {', '.join(sorted(unknown))}; known: {', '.join(sorted(ALL_RELEASED))}"
        )
    return names
