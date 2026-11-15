/**
 * @file input_plugin.cpp
 * @brief SCS input plugin: feeds the stack's control setpoints into the game.
 *
 * This is the answer to ADR-0002. The SDK's Input API lets a plugin register
 * an input device directly with the game, so no virtual joystick exists at the
 * operating system level: no `/dev/uinput` and udev rule on Linux, no ViGEmBus
 * driver install on Windows, and nothing for a user to configure.
 *
 * The device is **semantical**, which binds its inputs straight to the game's
 * control mixes by name -- `steering`, `aforward`, `abackward` -- with no
 * binding UI and no `controls.sii` edit. Those mixes add our contribution to
 * whatever the keyboard and wheel are already producing, so manual override
 * keeps working while the stack drives. That is not a side effect worth
 * tolerating; it is the emergency path.
 *
 * Two safety properties, both structural rather than conventional:
 *
 * - **Disengaged is a flag, not a promise.** With `kCommandEngaged` clear the
 *   callback emits zeros whatever the axes hold, so nothing depends on every
 *   writer remembering to zero its own outputs before it stops.
 * - **A silent commander never keeps the throttle.** The block's counter must
 *   keep advancing; after `kCommandWatchdogSeconds` without a change the
 *   throttle and buttons go neutral and the steering is held, and after
 *   `kSafetyStopSeconds` the plugin stops the truck itself and leaves it to
 *   the driver (safety_stop.h). A stack that dies mid-corner cannot leave the
 *   throttle open, nor the truck running on with nobody at the wheel.
 *
 * The SDK does not support adding or removing devices after `scs_input_init`,
 * which is why engage and disengage are expressed in the data rather than by
 * registering and unregistering the device.
 */

#include <cstring>
#include <string>

#include <atomic>
#include <chrono>

#include "scssdk_input.h"
#include "amtrucks/scssdk_ats.h"
#include "amtrucks/scssdk_input_ats.h"
#include "eurotrucks2/scssdk_eut2.h"
#include "eurotrucks2/scssdk_input_eut2.h"

#include "ets2ad/command_layout.h"
#include "diagnostics.h"
#include "safety_stop.h"
#include "shared_memory.h"

namespace {

using namespace ets2ad;

/**
 * @brief One registered input, in the order the game will index them.
 *
 * `name` must match a game control mix exactly: the semantical device type
 * binds by name, which is what removes the configuration step. The set here
 * was taken from a fresh `controls.sii`, where a supported mix shows up as
 * `semantical.<name>?0` in its expression.
 */
struct InputDescriptor {
    const char *name;
    const char *display_name;
    scs_value_type_t type;
};

/** Axes first, in `CommandAxis` order, then buttons in `CommandButton` order.
 *  The callback relies on that correspondence, so the static assertions below
 *  pin it rather than leaving it to a comment. */
constexpr InputDescriptor kInputs[] = {
    {"steering", "Steering", SCS_VALUE_TYPE_float},
    {"aforward", "Throttle", SCS_VALUE_TYPE_float},
    {"abackward", "Brake", SCS_VALUE_TYPE_float},
    {"clutch", "Clutch", SCS_VALUE_TYPE_float},

    {"parkingbrake", "Parking Brake", SCS_VALUE_TYPE_bool},
    {"engine", "Engine", SCS_VALUE_TYPE_bool},
    {"gearup", "Gear Up", SCS_VALUE_TYPE_bool},
    {"geardown", "Gear Down", SCS_VALUE_TYPE_bool},
    {"lblinker", "Indicator Left", SCS_VALUE_TYPE_bool},
    {"rblinker", "Indicator Right", SCS_VALUE_TYPE_bool},
    {"horn", "Horn", SCS_VALUE_TYPE_bool},
    {"light", "Lights", SCS_VALUE_TYPE_bool},

    {"motorbrake", "Engine Brake", SCS_VALUE_TYPE_bool},
    {"engbrakeup", "Engine Brake Up", SCS_VALUE_TYPE_bool},
    {"engbrakedwn", "Engine Brake Down", SCS_VALUE_TYPE_bool},
    // The game's name for the hazard lights' mix; "hazard" is no mix at all,
    // and the lights never came on at a handover.
    {"flasher4way", "Hazard Lights", SCS_VALUE_TYPE_bool},
    {"activate", "Activate", SCS_VALUE_TYPE_bool},
    {"attach", "Trailer Attach", SCS_VALUE_TYPE_bool},
};

constexpr std::size_t kInputCount = sizeof(kInputs) / sizeof(kInputs[0]);

static_assert(kInputCount == static_cast<std::size_t>(kAxisCount) +
                                 static_cast<std::size_t>(kButtonCount),
              "the input table must cover every axis and button in the command block");

struct InputState {
    SharedMemory memory;
    scs_log_t log = nullptr;

    /** Last consistent copy of the command block. */
    CommandBlock command{};

    /** Whether `command` holds anything worth acting on. */
    bool command_valid = false;

    /** When `command_count` last moved, on the plugin's own clock. */
    std::chrono::steady_clock::time_point last_change_at{};

    /** Whether any command has been seen since the device became active. */
    bool ever_seen_command = false;

    /** Previous `command_count`, for the watchdog. */
    std::uint64_t last_command_count = 0;

    /** Index of the next input to emit in this frame. */
    std::size_t cursor = 0;

    /** What a quiet commander gets: held, stopped for (safety_stop.h). */
    SafetyStop safety;

    /** This frame's outputs, worked out at its first input. */
    Outputs outputs;
};

InputState g_input;

void log_input(const scs_log_type_t type, const std::string &message) {
    ets2ad::log_line(g_input.log, "input", type, message);
}

/**
 * @brief Take a consistent copy of the command block.
 *
 * The reader half of the seqlock, with the roles reversed from telemetry: here
 * the Python side writes and the plugin reads. Bounded retries, because this
 * runs on the game's main thread and a spin there would stall the frame.
 *
 * @return true if `g_input.command` now holds a consistent, valid block.
 */
bool refresh_command() {
    if (!g_input.memory.is_open()) {
        return false;
    }

    const auto *const block = static_cast<const CommandBlock *>(g_input.memory.data());
    std::atomic_ref<const std::uint32_t> sequence(block->sequence);

    constexpr int kMaxAttempts = 4;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const std::uint32_t before = sequence.load(std::memory_order_acquire);
        if (before % 2 != 0) {
            continue;  // writer is mid-update
        }
        CommandBlock candidate;
        std::memcpy(&candidate, block, sizeof(CommandBlock));
        const std::uint32_t after = sequence.load(std::memory_order_acquire);
        if (before != after) {
            continue;  // the block changed while we copied it
        }

        // A block nobody has written is all zeroes, so this rejects "no
        // commander" without needing a separate presence signal.
        if (candidate.magic != kCommandMagic ||
            candidate.layout_version != kCommandLayoutVersion ||
            candidate.block_size != sizeof(CommandBlock)) {
            return false;
        }

        g_input.command = candidate;
        return true;
    }
    return false;  // contention; the previous copy and the watchdog take over
}

/** Clamp into an axis's range, mapping NaN to neutral.
 *
 * NaN compares false against everything, so it would slip through a plain
 * min/max and reach the game as garbage. A controller emitting NaN is a bug to
 * fix upstream, but not one the vehicle should act out. */
float clamp_axis(const std::size_t axis, const float value) {
    if (value != value) {
        return 0.0f;
    }
    const float low = (axis == kAxisSteering) ? -1.0f : 0.0f;
    if (value < low) {
        return low;
    }
    return (value > 1.0f) ? 1.0f : value;
}

/** This frame's outputs, and word in the log of a change that matters. */
void work_out_outputs() {
    const auto now = std::chrono::steady_clock::now();
    const bool engaged =
        g_input.ever_seen_command && (g_input.command.flags & kCommandEngaged) != 0;
    const double silent_s = std::chrono::duration<double>(now - g_input.last_change_at).count();
    const double now_s = std::chrono::duration<double>(now.time_since_epoch()).count();
    const Source before = g_input.outputs.source;
    g_input.outputs =
        g_input.safety.step(now_s, silent_s, engaged,
                            clamp_axis(kAxisSteering, g_input.command.axes[kAxisSteering]),
                            driver_view());
    const Source after = g_input.outputs.source;
    if (after == before) {
        return;
    }
    if (after == Source::kHold) {
        log_input(SCS_LOG_TYPE_warning, "commander quiet: steering held, throttle let go");
    } else if (after == Source::kStop) {
        log_input(SCS_LOG_TYPE_warning,
                  "commander gone: stopping the truck, hazard lights on; brake or "
                  "accelerate to take over");
    } else if (before == Source::kStop) {
        log_input(SCS_LOG_TYPE_message, "the driver has the truck");
    } else if (before == Source::kHold && after == Source::kCommands) {
        log_input(SCS_LOG_TYPE_message, "commander back");
    }
}

/**
 * @brief Hand the game one input value per call, then report exhaustion.
 *
 * Called repeatedly within a frame until it returns `SCS_RESULT_not_found`.
 * Every input is emitted every frame rather than only on change: the device
 * contributes additively to the game's mixes, so a value we stop sending is a
 * value that stops being applied, and "no event" is not the same as "hold".
 */
SCSAPI_RESULT input_event_callback(scs_input_event_t *const event_info, const scs_u32_t flags,
                                   const scs_context_t /*context*/) {
    if ((flags & SCS_INPUT_EVENT_CALLBACK_FLAG_first_in_frame) != 0) {
        g_input.cursor = 0;
        g_input.command_valid = refresh_command();

        if (g_input.command_valid &&
            g_input.command.command_count != g_input.last_command_count) {
            g_input.last_command_count = g_input.command.command_count;
            g_input.last_change_at = std::chrono::steady_clock::now();
            g_input.ever_seen_command = true;
        }
        work_out_outputs();
    }

    if (g_input.cursor >= kInputCount) {
        return SCS_RESULT_not_found;
    }

    const std::size_t index = g_input.cursor++;
    const Outputs &out = g_input.outputs;

    event_info->input_index = static_cast<scs_u32_t>(index);
    if (index < kAxisCount) {
        float value = 0.0f;
        if (out.source == Source::kCommands) {
            value = clamp_axis(index, g_input.command.axes[index]);
        } else if (out.source == Source::kHold && index == kAxisSteering) {
            value = out.steering;
        } else if (out.source == Source::kStop && index == kAxisBrake) {
            value = out.brake;
        }
        event_info->value_float.value = value;
    } else {
        const std::uint32_t bit = 1u << (index - kAxisCount);
        bool pressed = false;
        if (out.source == Source::kCommands) {
            pressed = (g_input.command.buttons & bit) != 0;
        } else if (out.source == Source::kStop) {
            pressed = out.hazard && index - kAxisCount == kButtonHazard;
        }
        event_info->value_bool.value = pressed ? 1u : 0u;
    }
    return SCS_RESULT_ok;
}

/** Called when the game starts or stops polling the device. */
SCSAPI_VOID input_active_callback(const scs_u8_t active, const scs_context_t /*context*/) {
    log_input(SCS_LOG_TYPE_message, active != 0 ? "device active" : "device inactive");
    if (active == 0) {
        // Coming back is treated as a cold start: whatever the block held while
        // nobody was polling it is not something to resume from.
        g_input.ever_seen_command = false;
        g_input.command_valid = false;
        g_input.safety.reset();
        g_input.outputs = {};
    }
}

}  // namespace

// --------------------------------------------------------------------------
// SDK entry points
// --------------------------------------------------------------------------

SCSAPI_RESULT scs_input_init(const scs_u32_t version, const scs_input_init_params_t *const params) {
    if (version != SCS_INPUT_VERSION_1_00) {
        return SCS_RESULT_unsupported;
    }

    // The SDK may init/shutdown several times per process lifetime, so reset
    // explicitly rather than assuming a fresh object. SharedMemory owns a file
    // descriptor and is deliberately non-copyable, so the struct cannot simply
    // be overwritten with a default-constructed one.
    const auto *const p = static_cast<const scs_input_init_params_v100_t *>(params);
    g_input.log = p->common.log;
    g_input.command = CommandBlock{};
    g_input.command_valid = false;
    g_input.last_command_count = 0;
    g_input.cursor = 0;
    g_input.safety.reset();
    g_input.outputs = {};
    g_input.ever_seen_command = false;  // neutral until a command actually arrives
    g_input.last_change_at = std::chrono::steady_clock::time_point{};

    if (!g_input.memory.create(kCommandMemoryName, sizeof(CommandBlock))) {
        log_input(SCS_LOG_TYPE_error, "command block unavailable: " + g_input.memory.error());
        return SCS_RESULT_generic_error;
    }

    scs_input_device_input_t inputs[kInputCount];
    std::memset(inputs, 0, sizeof(inputs));
    for (std::size_t index = 0; index < kInputCount; ++index) {
        inputs[index].name = kInputs[index].name;
        inputs[index].display_name = kInputs[index].display_name;
        inputs[index].value_type = kInputs[index].type;
    }

    scs_input_device_t device;
    std::memset(&device, 0, sizeof(device));
    device.name = "ets2ad";
    device.display_name = "ETS2 Autonomous Stack";
    device.type = SCS_INPUT_DEVICE_TYPE_semantical;
    device.input_count = static_cast<scs_u32_t>(kInputCount);
    device.inputs = inputs;
    device.input_event_callback = input_event_callback;
    device.input_active_callback = input_active_callback;
    device.callback_context = nullptr;

    if (p->register_device(&device) != SCS_RESULT_ok) {
        log_input(SCS_LOG_TYPE_error, "the game refused the input device registration");
        g_input.memory.destroy();
        return SCS_RESULT_generic_error;
    }

    log_input(SCS_LOG_TYPE_message,
              "semantical input device registered with " + std::to_string(kInputCount) +
                  " inputs; commands read from '" + kCommandMemoryName + "' (" +
                  std::to_string(sizeof(CommandBlock)) + " bytes, layout v" +
                  std::to_string(kCommandLayoutVersion) + ")");
    return SCS_RESULT_ok;
}

SCSAPI_VOID scs_input_shutdown(void) {
    log_input(SCS_LOG_TYPE_message, "input shutting down");
    g_input.memory.destroy();
    g_input.log = nullptr;
    g_input.command_valid = false;
    g_input.ever_seen_command = false;
}
