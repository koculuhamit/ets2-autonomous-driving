# SCS SDK 1.14 (vendored)

Unmodified headers from the official SCS Software Telemetry/Input SDK.

| | |
|---|---|
| Version | 1.14 |
| Source | <https://download.eurotrucksimulator2.com/scs_sdk_1_14.zip> |
| SHA-256 | `c6c1f7376b7324994d9f9c567f3c4141fbbf305b6bf803bc4cfeef2437b2023a` |
| Retrieved | 2026-09-20 |
| Licence | MIT — see [LICENSE.txt](LICENSE.txt) |

Only `include/` is vendored, plus the SDK's own readme, which documents the
plugin loading rules, threading model and version negotiation that the code in
`plugin/src/` depends on.

**Do not edit these files.** Upgrading means replacing the directory wholesale
and re-running the layout tests: the struct sizes the SDK asserts are part of
the ABI between us and the game.

The examples and Visual Studio project files from the zip are deliberately left
out. `plugin/` is built with CMake for both platforms from one source tree.
