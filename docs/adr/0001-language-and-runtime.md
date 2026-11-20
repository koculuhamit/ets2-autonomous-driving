# ADR-0001: Core language and runtime

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The stack spans four very different workloads:

1. **Telemetry plugin** — must be a native shared library loaded by the game
   (`.so` on Linux, `.dll` on Windows). The SCS SDK is a C API.
2. **Perception** — CNN inference (lane detection, vehicle detection), image
   processing, camera calibration. Needs a mature ML/CV ecosystem.
3. **Control and planning** — numerical work (PID, LQR, MPC, Hybrid A*),
   iterated constantly during tuning.
4. **Map extraction** — parsing the game's `.scs` archives. The only mature
   parser (TruckLib) is a .NET library.

The project is also a portfolio piece: iteration speed and readability matter
as much as raw performance.

## Options considered

### Option A — Python core + C++ plugin (+ C# for offline map parsing)
- Widest ecosystem for perception (PyTorch, ONNX Runtime, OpenCV, NumPy).
- Fastest iteration loop for controller tuning.
- GIL and interpreter overhead in the hot path; needs care to stay inside the
  latency budget.

### Option B — Full C# / .NET
- Single language and runtime; TruckLib is native to it.
- Weak ML/CV story (ONNX Runtime exists, training does not).
- Slower iteration on the numerical/tuning work.

### Option C — Python orchestration + C++ hot path (pybind11)
- Best latency ceiling.
- Doubles the build complexity from day one, before we know where the real
  bottlenecks are.

## Decision

**Option A.** The core (`core/`, `pal/`, `adapters/`) is Python 3.11, managed
with `uv`. The SCS SDK plugin (`plugin/`) is C++ built with CMake for both
Linux and Windows from one source tree. Map extraction (`tools/map/`) may shell
out to a C# TruckLib-based tool, but only **offline** — it produces a cached
binary map artifact and is never on the runtime path.

Python 3.11 rather than the system 3.10: better performance, and `uv` pins it
per-project so the system interpreter is untouched.

## Consequences

- Perception and control share one language; a recorded run can be replayed and
  re-tuned without a build step.
- The latency budget (Phase 0) becomes a hard gate: if a Python loop cannot hold
  it, that specific loop is moved to C++ via pybind11 — measured first, ported
  second. Option C stays available incrementally.
- Three toolchains in the repo (uv, CMake, optionally .NET). CI must build all
  of them, and the .NET dependency must remain optional for anyone who only
  wants to run the driving stack.
