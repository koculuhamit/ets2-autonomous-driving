#include "diagnostics.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace ets2ad {

std::string diagnostics_path() {
#ifdef _WIN32
    const char *const base = std::getenv("TEMP");
    return std::string(base != nullptr ? base : ".") + "\\ets2ad_plugin.log";
#else
    // The user's own runtime directory (/run/user/<uid>, mode 0700) before the
    // shared /tmp, where another account could hold the name first.
    for (const char *const name : {"XDG_RUNTIME_DIR", "TMPDIR"}) {
        const char *const base = std::getenv(name);
        if (base != nullptr && *base != '\0') {
            return std::string(base) + "/ets2ad_plugin.log";
        }
    }
    return "/tmp/ets2ad_plugin.log";
#endif
}

void reset_diagnostics() {
    if (std::FILE *const file = std::fopen(diagnostics_path().c_str(), "w")) {
        std::fclose(file);
    }
}

void log_line(const scs_log_t log, const char *const prefix, const scs_log_type_t type,
              const std::string &message) {
    const std::string tagged = std::string("<ets2ad/") + prefix + "> " + message;
    if (log != nullptr) {
        log(type, tagged.c_str());
    }

    // Opened per line rather than held: the plugin is unloaded without warning
    // on "sdk unload", and a buffered handle would lose the last and most
    // interesting lines.
    const char *const severity = (type == SCS_LOG_TYPE_error)     ? "ERROR"
                                 : (type == SCS_LOG_TYPE_warning) ? "WARN "
                                                                  : "INFO ";
    if (std::FILE *const file = std::fopen(diagnostics_path().c_str(), "a")) {
        std::fprintf(file, "%s %s %s\n", severity, prefix, message.c_str());
        std::fclose(file);
    }
}

}  // namespace ets2ad
