"""What is measured per truck, kept per truck, and read back for it.

Steering ratio, brakes, the camera's seat, the dashboard and the mirrors
differ from truck to truck: the stack was measured on a Mercedes Actros
(core.control.lateral.ACTROS_GEOMETRY, longitudinal.ACTROS_EMPTY,
configs/camera.json, configs/cab_dashboard.npy, detection.DEFAULT_CAB_VIEW),
and drove an 8x4 in Bern on those. `tools/calibrate.py` measures a truck and
writes its profile to `configs/trucks/<brand>.<model>.json` with its files
beside it; the tools read it for the truck the telemetry reports
(`truck_id`), and fall back to the Actros's numbers for what it lacks.

The wheelbase is not kept: it is read off the telemetry's wheels each time
(`geometry`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Final

from core.control.lateral import ACTROS_GEOMETRY, VehicleGeometry
from core.control.longitudinal import ACTROS_EMPTY, LongitudinalCalibration

__all__ = [
    "PROFILES_DIR",
    "TruckProfile",
    "fit_brake",
    "fit_steering_map",
    "load_profile",
    "save_profile",
    "truck_id",
]

PROFILES_DIR: Final = Path(__file__).resolve().parents[2] / "configs" / "trucks"
DEFAULT_CAMERA_FILE: Final = PROFILES_DIR.parent / "camera.json"
DEFAULT_CAB_FILE: Final = PROFILES_DIR.parent / "cab_dashboard.npy"

FORMAT: Final = "ets2ad-truck-1"

DEFAULTS_TRUCK: Final = "mercedes.vehicle.mercedes.actros2014"
"""The truck the default camera and dashboard were measured on: the
Actros. Another without its own drives on them (`TruckProfile.missing`)."""

NO_BRAKING_MPS2: Final = 0.2
"""Deceleration under which a brake command counts as doing nothing: 0.05
of command gave 0.02 m/s^2 on the Actros, coasting all but."""


@dataclass(frozen=True, slots=True)
class TruckProfile:
    truck: str
    """`brand.model`, as `truck_id` reads it."""

    steering_map: tuple[tuple[float, float], ...] | None = None
    """`effective_steering` to the steered wheels' mean angle, radians
    (lateral.VehicleGeometry.steering_map)."""

    brake_dead_zone: float | None = None
    brake_curve: tuple[tuple[float, float], ...] | None = None
    """Brake command to deceleration, m/s^2 (longitudinal.LongitudinalCalibration)."""

    camera: str | None = None
    """File name, beside the profile, of the camera's calibration
    (tools/calibrate_camera.py)."""

    cab_reference: str | None = None
    """File name, beside the profile, of the dashboard's edges
    (tools/calibration/cab_reference.py)."""

    view: dict | None = None
    """The cab view's regions as fractions of the frame:
    {"windscreen": [l, t, r, b], "covered": [[l, t, r, b], ...]}
    (detection.CabView): the mirrors and the HUD are what is covered."""

    measured: dict[str, str] = field(default_factory=dict)
    """What was measured, and when or from what."""

    def geometry(self, block) -> VehicleGeometry:
        """The bicycle model's: the wheelbase and the driven axle's place
        behind the truck's origin from the telemetry, the steering map from
        this profile or the Actros's."""
        wheels = block["truck"]["wheels"]
        count = int(wheels["count"])
        steered = [
            float(wheels["position"][k]["z"]) for k in range(count) if wheels["steerable"][k]
        ]
        driven = [float(wheels["position"][k]["z"]) for k in range(count) if wheels["powered"][k]]
        wheelbase = ACTROS_GEOMETRY.wheelbase
        axle_behind = 0.0
        if steered and driven:
            # The truck's +z points back, from its origin.
            axle_behind = sum(driven) / len(driven)
            wheelbase = max(axle_behind - sum(steered) / len(steered), 1.0)
        return VehicleGeometry(
            wheelbase=wheelbase,
            steering_map=self.steering_map or ACTROS_GEOMETRY.steering_map,
            axle_behind=axle_behind,
        )

    def longitudinal(self) -> LongitudinalCalibration:
        """The pedals': the brake from this profile, the rest the Actros's."""
        if self.brake_curve is None:
            return ACTROS_EMPTY
        return replace(
            ACTROS_EMPTY,
            brake_curve=self.brake_curve,
            brake_dead_zone=(
                self.brake_dead_zone
                if self.brake_dead_zone is not None
                else ACTROS_EMPTY.brake_dead_zone
            ),
        )

    def camera_file(self, directory: Path = PROFILES_DIR) -> Path:
        return directory / self.camera if self.camera else DEFAULT_CAMERA_FILE

    def cab_file(self, directory: Path = PROFILES_DIR) -> Path:
        return directory / self.cab_reference if self.cab_reference else DEFAULT_CAB_FILE

    def cab_view(self):
        """The detector's view of the cab (detection.CabView)."""
        from core.perception.detection import DEFAULT_CAB_VIEW, CabView, Rect

        if not self.view:
            return DEFAULT_CAB_VIEW
        return CabView(
            windscreen=Rect(*self.view["windscreen"]),
            covered=tuple(Rect(*r) for r in self.view.get("covered", ())),
        )

    def learns_cab(self, directory: Path = PROFILES_DIR) -> Path | None:
        """Where this truck's dashboard is to be saved once learned from a
        drive (core.perception.cab.DashboardLearner): for a truck with none
        of its own and not the one the default was recorded on; else None."""
        if self.cab_reference or self.truck in (DEFAULTS_TRUCK, "unknown.unknown"):
            return None
        return directory / f"{self.truck}.cab.npy"

    def with_cab(self, learned: Path, directory: Path = PROFILES_DIR) -> TruckProfile:
        """This profile with its dashboard learned and saved at `learned`."""
        from datetime import date

        return replace(
            self,
            cab_reference=learned.name,
            measured={
                **self.measured,
                "cab": f"{date.today().isoformat()}, learned from the first frames of a drive",
            },
        )

    def learns_camera(self) -> bool:
        """Whether this truck's camera is still to be measured, before it is
        driven: with another truck's, distances, places across and widths
        all come out wrong, and a DAF XD ran into the lorry ahead."""
        return not self.camera and self.truck not in (DEFAULTS_TRUCK, "unknown.unknown")

    def with_camera(self, name: str, note: str) -> TruckProfile:
        """This profile with its camera measured, in `name` beside it."""
        from datetime import date

        return replace(
            self,
            camera=name,
            measured={**self.measured, "camera": f"{date.today().isoformat()}, {note}"},
        )

    def learns_roof(self) -> bool:
        """Whether this truck's windscreen is to be found from a drive: one
        with no view of its own, not the one the default was measured on."""
        return not self.view and self.truck not in (DEFAULTS_TRUCK, "unknown.unknown")

    def with_view_top(self, top: float) -> TruckProfile:
        """This profile with the default view, its windscreen begun at `top`
        (a share of the frame's height), as learned from a drive."""
        from datetime import date

        view = self.cab_view()

        def box(rect) -> list[float]:
            return [rect.left, rect.top, rect.right, rect.bottom]

        windscreen = box(view.windscreen)
        windscreen[1] = round(float(top), 4)
        return replace(
            self,
            view={"windscreen": windscreen, "covered": [box(r) for r in view.covered]},
            measured={
                **self.measured,
                "view": f"{date.today().isoformat()}, windscreen's top learned from a drive",
            },
        )

    def missing(self) -> list[str]:
        """What this truck still runs on the Actros's numbers for: nothing,
        for the Actros."""
        if self.truck == DEFAULTS_TRUCK:
            return []
        names = {
            "steering": self.steering_map,
            "brake": self.brake_curve,
            "camera": self.camera,
            "cab": self.cab_reference,
            "view": self.view,
        }
        return [name for name, value in names.items() if not value]


def truck_id(block) -> str:
    """`brand.model` of the truck the telemetry reports, e.g. `mercedes.actros`."""
    truck = block["truck"]

    def text(name: str) -> str:
        return bytes(truck[name]).split(b"\0", 1)[0].decode(errors="replace")

    return f"{text('brand_id') or 'unknown'}.{text('model_id') or 'unknown'}"


def load_profile(truck: str, directory: Path = PROFILES_DIR) -> TruckProfile:
    """The truck's profile; an empty one, all the Actros's, where there is none."""
    path = directory / f"{truck}.json"
    if not path.is_file():
        return TruckProfile(truck)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} truck profile")

    def pairs(name: str):
        value = data.get(name)
        return tuple((float(a), float(b)) for a, b in value) if value else None

    return TruckProfile(
        truck=data["truck"],
        steering_map=pairs("steering_map"),
        brake_dead_zone=data.get("brake_dead_zone"),
        brake_curve=pairs("brake_curve"),
        camera=data.get("camera"),
        cab_reference=data.get("cab_reference"),
        view=data.get("view"),
        measured=dict(data.get("measured", {})),
    )


def save_profile(profile: TruckProfile, directory: Path = PROFILES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.truck}.json"
    data = {"format": FORMAT, **asdict(profile)}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def fit_steering_map(
    samples: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """A steering map from held steps: (applied `effective_steering`, the
    steered wheels' mean angle, radians), either side. Each side's steps to
    the same command are averaged by magnitude; the map starts at (0, 0)
    and rises in both columns, a step that does not left out."""
    by_command: dict[float, list[tuple[float, float]]] = {}
    for command, angle in samples:
        if abs(command) < 1e-3:
            continue
        by_command.setdefault(round(abs(command), 2), []).append((abs(command), abs(angle)))
    points = sorted(
        (
            sum(c for c, _ in group) / len(group),
            sum(a for _, a in group) / len(group),
        )
        for group in by_command.values()
    )
    fitted = [(0.0, 0.0)]
    for command, angle in points:
        if command > fitted[-1][0] and angle > fitted[-1][1]:
            fitted.append((round(command, 4), round(angle, 5)))
    if len(fitted) < 3:
        raise ValueError("too few steering steps to make a map of")
    return tuple(fitted)


def fit_brake(samples: list[tuple[float, float]]) -> tuple[float, tuple[tuple[float, float], ...]]:
    """(dead zone, curve) from braking runs: (command, deceleration m/s^2).
    The dead zone ends midway between the largest command that did not
    brake (`NO_BRAKING_MPS2`) and the first that did -- 0.055 on the Actros,
    as measured by hand; the curve is the rest, rising in both columns."""
    ordered = sorted(samples)
    live = [(c, d) for c, d in ordered if d >= NO_BRAKING_MPS2]
    if len(live) < 2:
        raise ValueError("too few braking runs that braked")
    dead = [c for c, d in ordered if d < NO_BRAKING_MPS2 and c < live[0][0]]
    dead_zone = (max(dead) + live[0][0]) / 2.0 if dead else 0.0
    curve = [live[0]]
    for command, deceleration in live[1:]:
        if command > curve[-1][0] and deceleration > curve[-1][1]:
            curve.append((command, deceleration))
    if len(curve) < 2:
        raise ValueError("the braking runs do not rise with the command")
    return round(dead_zone, 3), tuple((round(c, 3), round(d, 2)) for c, d in curve)


def degrees_map(steering_map: tuple[tuple[float, float], ...]) -> str:
    """A steering map for people: command -> degrees."""
    return ", ".join(f"{c:.2f}->{math.degrees(a):.1f}" for c, a in steering_map)
