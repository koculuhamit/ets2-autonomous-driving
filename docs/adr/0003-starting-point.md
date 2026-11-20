# ADR-0003: Starting point — from scratch, ETS2LA as reference only

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

ETS2LA is an existing, MIT-licensed autonomous driving stack for Euro Truck
Simulator 2. It already solves telemetry access, screen capture, map extraction
(via TruckLib) and lane following. Forking it would save weeks.

The project's stated secondary goal, however, is a portfolio piece: every
design decision must be defensible in an interview.

## Options considered

### Option A — Fork ETS2LA
- Immediate working baseline.
- Inherits a large codebase whose architecture, and whose mistakes, are not
  ours. "Which parts did you write?" becomes an awkward question.
- Its structure does not match our PAL-first, Linux+Windows parity goal.

### Option B — From scratch, ETS2LA (and other projects) as reference
- Every line is ours and every decision has an ADR behind it.
- We still read their code when stuck on a game-specific detail (shared memory
  layout, `.scs` parsing quirks, window capture) instead of rediscovering it.

### Option C — From scratch, no reference at all
- Maximum independence, materially slower; re-derives game-specific trivia that
  carries no engineering signal.

## Decision

**Option B.** Write the stack from scratch. Consult ETS2LA, the various
`scs-sdk-plugin` forks and `ets2-telemetry-server` as references for
game-specific facts only — never copy code.

Any non-trivial idea taken from a reference is attributed in a comment or in
the relevant design doc.

## Consequences

- Slower first demo; the architecture is ours end to end.
- Licence hygiene is simple: no third-party code is vendored, so the repo stays
  cleanly MIT.
- Phase 0's environment work (plugin build, shared memory, capture) is real
  engineering effort we cannot skip — but it is also the part with the most
  concrete, measurable results for the write-up.
