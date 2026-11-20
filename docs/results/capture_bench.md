# Capturing the game window

**Date:** 2026-09-23
**Game:** ETS2 1.61.1.0, Linux native, X11, 1920×1080 windowed
**Machine:** i7-10750H, RTX 2060 6 GB
**Code:** `pal/linux.py` (`LinuxScreenCapture`), `tools/capture_bench.py`

Perception starts with pictures, and the Linux capture was a stub. It is now
the game window found by title with Xlib and grabbed with mss, which reads
the screen through MIT-SHM.

## Result

`tools/capture_bench.py --seconds 6`, the game in front, the truck standing on
the autostrada:

| | |
|---|---|
| grabs | 153 per second |
| a grab, median / p95 / max | 4.2 / 11.4 / 38 ms |
| interval between grabs, median / p95 | 6.8 / 10.2 ms |
| new pictures | **80 per second** -- the game's own frame rate |

Every frame the game draws is captured, twice over. The Phase 5 target was
30--60 frames per second; the limit is now the game, not the capture.

"New pictures" counts grabs that differ from the one before, by a checksum of
a stripe across the middle of the frame. A capture faster than the game
returns the same picture twice, and it is the rate of *new* pictures that
perception can use.

## What it cost to get there

The first version managed 45 frames a second. Timing the parts of one grab:

| | median |
|---|---|
| reading the window's position and size (two X round trips) | 0.31 ms |
| the mss grab itself | 3.44 ms |
| BGRA → BGR | **10.16 ms** |

Dropping one byte in four, as a NumPy slice and a contiguous copy, was three
quarters of the time a grab took. OpenCV does the same conversion in 0.61 ms,
and it is needed for perception anyway. With it: 114 frames a second through
the PAL, 153 in the benchmark loop.

## Notes

- It captures the *screen where the window is*, not the window's own
  drawable. A GL or Vulkan window read directly under a compositor can come
  back black; the screen cannot. The cost is that whatever covers the game is
  captured instead. That happened: a grab with a browser in front of the game
  returned the browser. A grab now refuses (`WindowNotInFrontError`) unless
  the game is the active window -- which, while the stack drives, it has to
  be anyway, or it ignores the stack's inputs.
- The window's position and size are re-read on every grab, so moving or
  resizing it mid-run is followed.
- A captured frame is the whole window: the windscreen, three mirror insets
  and the HUD with its route advisor. Which part a detector looks at is the
  next question.
