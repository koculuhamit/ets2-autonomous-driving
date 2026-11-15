"""Whether the game's camera is in the cab: what perception is calibrated for.

Everything perception measures assumes the interior camera: distances from
where a vehicle meets the road, lights by where they stand in the frame, the
mirrors by their insets. Switched to the chase camera in Bern, the truck's own
trailer filled the view 9.5 m ahead at the truck's speed; it was braked for
at 5 m/s^2, remembered once out of view, and the truck never moved again.

From the cab the lower middle of the frame is the dashboard and the wheel:
the same edges whatever the light (`DASHBOARD`), or, at night, dark. From
outside it is road and trailer. A frame is the cab's if the dashboard's edges
match those of the cab frames recorded (`EDGE_MATCH_MIN`), or most of it is
dark (`DARK_SHARE_MIN`): every one of 911 recorded cab frames was, and no
frame of the chase camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["CAB_REFERENCE_FILE", "dashboard_edges", "in_cab", "load_reference"]

CAB_REFERENCE_FILE: Final = Path(__file__).resolve().parents[2] / "configs" / "cab_dashboard.npy"

DASHBOARD: Final = (0.62, 0.98, 0.30, 0.78)
"""(top, bottom, left, right) of the frame, as shares: the dashboard and wheel."""

DARK_BAND: Final = (0.78, 0.95, 0.30, 0.72)
SIZE: Final = (120, 48)
"""(width, height) the dashboard is compared at."""

EDGE_MATCH_MIN: Final = 0.06
"""Cab frames matched the recorded dashboard by 0.11 or more, but for two at
night; the chase camera's by 0.02 at most."""

DARK_V: Final = 60
DARK_SHARE_MIN: Final = 0.5
"""Of the dashboard below the wheel's hub, the share darker than `DARK_V`
that is the cab's at night: 0.79 and more where the edges were faint; 0.31
at most from outside."""


def dashboard_edges(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """The dashboard's edge strength, normalised, `SIZE` (as rows, columns)."""
    height, width = image.shape[:2]
    top, bottom, left, right = DASHBOARD
    band = image[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
    grey = cv2.resize(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA)
    grey = grey.astype(np.float32)
    magnitude = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    return (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)


def in_cab(image: NDArray[np.uint8], reference: NDArray[np.float32]) -> bool:
    """Whether a BGR frame of the game window is seen from the cab."""
    if float((dashboard_edges(image) * reference).mean()) >= EDGE_MATCH_MIN:
        return True
    height, width = image.shape[:2]
    top, bottom, left, right = DARK_BAND
    band = image[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
    value = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[..., 2]
    return float((value < DARK_V).mean()) >= DARK_SHARE_MIN


def load_reference(path: Path = CAB_REFERENCE_FILE) -> NDArray[np.float32] | None:
    """The recorded dashboard's edges (tools/calibration/cab_reference.py);
    None where there is none, and no check is made."""
    return np.load(path).astype(np.float32) if path.is_file() else None
