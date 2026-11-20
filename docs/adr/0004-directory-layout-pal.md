# ADR-0004: `pal/` instead of `platform/`

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The project plan specifies a top-level `platform/` package for the platform
abstraction layer. With the repository root on `sys.path`, a top-level package
named `platform` shadows Python's standard-library `platform` module for every
absolute import in the process — including imports made by NumPy, PyTorch and
setuptools. The failure is silent at import time and confusing to debug.

## Options considered

- Keep `platform/` and rely on never having the repo root on `sys.path`.
  Fragile: `pytest`, editable installs and `python -m` all put it there.
- Nest everything under a single `ets2ad/` package. Safe, but adds a directory
  level to every path in the plan for no other benefit.
- Rename the directory to `pal/` — the acronym the plan itself already uses.

## Decision

The platform abstraction layer lives in **`pal/`**. `pal/base.py`,
`pal/linux.py`, `pal/windows.py`, `pal/fake.py`.

## Consequences

- No stdlib shadowing, no import-order surprises.
- The project's plan was updated to say `pal/`; anyone reading older notes
  should read `platform/` as `pal/`.
- The portability rule "no OS-specific imports outside the PAL" is now checked
  against `pal/` in CI.
