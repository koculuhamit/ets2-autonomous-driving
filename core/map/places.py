"""Where things are: company depots and cities, by the ids a job uses.

The telemetry's job names its destination as a city id and a company id --
`berlin`, `tree_et` -- and nothing else: no position, no road. The map knows
where every depot is and which prefab is its yard, and a yard's control nodes
are its gates onto the road. So a job's destination becomes a place on the
network by looking it up here.

Kept beside the compiled network as `places.json`: a few hundred entries, not
worth arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from core.map.sector import Sector

__all__ = ["CityPlace", "CompanyPlace", "Places"]

PLACES_FILE: Final = "places.json"


@dataclass(frozen=True, slots=True)
class CompanyPlace:
    company: str
    city: str
    position: tuple[float, float]
    gates: tuple[tuple[float, float], ...]
    """Where the yard meets the road: the yard prefab's control nodes."""

    spots: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(frozen=True, slots=True)
class CityPlace:
    city: str
    position: tuple[float, float]
    shown: bool


@dataclass(frozen=True)
class Places:
    companies: tuple[CompanyPlace, ...]
    cities: tuple[CityPlace, ...]

    @classmethod
    def from_sector(cls, sector: Sector) -> Places:
        """Depots and cities of an extraction, each depot with its gates."""
        prefabs = {prefab.uid: prefab for prefab in sector.prefabs}
        companies = []
        for company in sector.companies:
            yard = prefabs.get(company.prefab) if company.prefab else None
            gates = (
                tuple((float(p[0]), float(p[2])) for p in yard.node_positions)
                if yard is not None
                else ()
            )
            companies.append(
                CompanyPlace(
                    company=company.company,
                    city=company.city,
                    position=_xz(company.position),
                    gates=gates,
                    spots=tuple((kind, _xz(where)) for kind, where in company.spots),
                )
            )
        cities = tuple(CityPlace(c.city, _xz(c.position), c.shown) for c in sector.cities)
        return cls(tuple(companies), cities)

    def save(self, directory: str | Path) -> None:
        document = {
            "companies": [
                {
                    "company": c.company,
                    "city": c.city,
                    "position": c.position,
                    "gates": c.gates,
                    "spots": [{"type": kind, "position": at} for kind, at in c.spots],
                }
                for c in self.companies
            ],
            "cities": [
                {"city": c.city, "position": c.position, "shown": c.shown} for c in self.cities
            ],
        }
        (Path(directory) / PLACES_FILE).write_text(json.dumps(document), encoding="utf-8")

    @classmethod
    def open(cls, directory: str | Path) -> Places:
        document = json.loads((Path(directory) / PLACES_FILE).read_text(encoding="utf-8"))
        companies = tuple(
            CompanyPlace(
                company=c["company"],
                city=c["city"],
                position=tuple(c["position"]),
                gates=tuple(tuple(g) for g in c["gates"]),
                spots=tuple((s["type"], tuple(s["position"])) for s in c["spots"]),
            )
            for c in document["companies"]
        )
        cities = tuple(
            CityPlace(c["city"], tuple(c["position"]), bool(c["shown"])) for c in document["cities"]
        )
        return cls(companies, cities)

    def company(self, company: str, city: str) -> CompanyPlace | None:
        """The depot a job names, or None if it is not on this map.

        None is a real answer: the destination may be in a DLC this map was
        not built with.
        """
        for place in self.companies:
            if place.company == company and place.city == city:
                return place
        return None

    def city(self, city: str) -> CityPlace | None:
        for place in self.cities:
            if place.city == city:
                return place
        return None


def _xz(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[2])
