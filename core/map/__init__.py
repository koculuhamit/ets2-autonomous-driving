"""The game's road network, as extracted from its archives.

`tools/map_extract` reads the archives and writes what it finds; everything
here reads that and turns it into something a planner can use. The split is
deliberate: parsing the game's formats is a .NET problem (ADR-0006) and lives
outside the driving loop entirely, while what the truck actually plans on is
Python like the rest of the stack.
"""

from core.map import dlc
from core.map.graph import Lane, LaneGraph
from core.map.matcher import LaneMatch, LaneMatcher
from core.map.network import MapCacheError, Projection, RoadNetwork
from core.map.places import CityPlace, CompanyPlace, Places
from core.map.sector import (
    LaneMeasurement,
    Prefab,
    PrefabLane,
    Road,
    RoadLook,
    Sector,
    Semaphore,
    SemaphoreProfile,
    Signal,
)

__all__ = [
    "CityPlace",
    "CompanyPlace",
    "Lane",
    "LaneGraph",
    "LaneMatch",
    "LaneMatcher",
    "LaneMeasurement",
    "MapCacheError",
    "Places",
    "Prefab",
    "PrefabLane",
    "Projection",
    "Road",
    "RoadLook",
    "RoadNetwork",
    "Sector",
    "Semaphore",
    "SemaphoreProfile",
    "Signal",
    "dlc",
]
