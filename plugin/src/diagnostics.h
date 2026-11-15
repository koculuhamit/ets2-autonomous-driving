/**
 * @file diagnostics.h
 * @brief The plugin's own log file.
 *
 * ETS2 1.55 on Linux routes the SDK's `log()` callback nowhere we can find --
 * not into `game.log.txt`, not into Steam's console logs, verified by
 * searching both after a load that demonstrably worked. A plugin that fails
 * during initialisation would therefore leave no trace at all, and the reader
 * on the other side would only be able to report "no shared memory", which is
 * equally consistent with the game simply not running.
 *
 * So the plugin keeps a file of its own. It is the first thing to look at when
 * the shared memory block does not appear, and the PAL's error message points
 * at it.
 */
#ifndef ETS2AD_DIAGNOSTICS_H
#define ETS2AD_DIAGNOSTICS_H

#include <string>

#include "scssdk.h"

namespace ets2ad {

/** Path to the diagnostics file: `$XDG_RUNTIME_DIR/ets2ad_plugin.log`, else
 * `$TMPDIR` or `/tmp`; `%TEMP%` on Windows. */
std::string diagnostics_path();

/** Truncate the file so each plugin session starts clean. */
void reset_diagnostics();

/**
 * @brief Write one line, to both the SDK log and our own file.
 *
 * @param log The SDK's log callback, or nullptr before initialisation.
 * @param prefix Short tag identifying the subsystem, e.g. "telemetry".
 * @param message The line.
 */
void log_line(scs_log_t log, const char *prefix, const scs_log_type_t type,
              const std::string &message);

}  // namespace ets2ad

#endif  // ETS2AD_DIAGNOSTICS_H
