# ADR-0005: Vendor the SCS SDK headers

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The telemetry plugin is a native library the game loads, built against the SCS
Software SDK headers. Those headers are not packaged anywhere: they are
distributed as a zip from SCS Software's download server, versioned by hand
(`scs_sdk_1_14.zip`), with no checksum published alongside them.

CI has to build the plugin for Linux *and* Windows on every commit. A build
that starts by downloading a zip from a third-party server is a build that
breaks when that server has a bad day, and one whose inputs nobody can pin.

The SDK is MIT-licensed, so redistribution is permitted.

## Options considered

### Option A — Download in CI, cache by URL
- Repo stays small.
- Build depends on an external host, unauthenticated, with no integrity
  guarantee beyond TLS. Offline builds are impossible.

### Option B — Vendor `include/` into the repo
- 20 header files, ~100 KB. Build is hermetic and offline-capable.
- The SDK version becomes visible in the diff when it changes, which is exactly
  when the telemetry struct layout is at risk.
- Requires carrying the upstream licence and provenance, which is cheap.

### Option C — Git submodule pointing at a third-party mirror
- Depends on someone else's mirror staying honest and available. Worse than
  both other options for no benefit.

## Decision

**Option B.** `plugin/vendor/scs_sdk/include/` holds the unmodified 1.14
headers, with `LICENSE.txt`, the upstream readme, and a `README.md` recording
the source URL, SHA-256 and retrieval date.

Note that ADR-0003's "write from scratch, never copy code" rule is about the
autonomous driving stack. These are the interface headers of the thing we are
integrating with, the way a C library's headers are: there is no alternative
implementation to write.

## Consequences

- The plugin builds offline and in CI with no network step.
- Upgrading the SDK is a visible, reviewable commit rather than a silent change
  in what CI downloaded.
- The struct sizes asserted in these headers are part of the ABI between the
  plugin and the game. A layout test guards them, and re-running it is part of
  the upgrade procedure documented in `plugin/vendor/scs_sdk/README.md`.
- Only `include/` is vendored. The zip's examples and Visual Studio projects
  are not: `plugin/` is one CMake tree that targets both platforms.
