"""Screen grabs as images, shared by the Linux and Windows backends."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["bgra_to_bgr"]


def bgra_to_bgr(raw: bytes, height: int, width: int) -> NDArray[np.uint8]:
    """A screen grab's BGRA bytes as a contiguous (height, width, 3) BGR image.

    Through OpenCV rather than a NumPy slice: dropping one byte in four is a
    strided copy NumPy does in 10 ms for a 1080p frame and OpenCV in 0.6 --
    which was three quarters of the time a whole grab took.
    """
    import cv2

    pixels = np.frombuffer(raw, dtype=np.uint8)
    if pixels.size != height * width * 4:
        raise ValueError(f"{pixels.size} bytes is not a {width}x{height} BGRA image")
    return cv2.cvtColor(pixels.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
