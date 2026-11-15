/**
 * @file safety_stop.h
 * @brief What the plugin does when the stack goes quiet: hold, then stop.
 *
 * The stack's process died twice on Windows at 70-80 km/h -- killed from
 * outside for memory -- and the watchdog did what it was built to do: it let
 * go of everything. With nobody at the controls the truck ran on at speed,
 * left its lane and hit something. Letting go is only safe with a driver.
 *
 * And short silences, 0.2-0.9 s while the stack's process was paged out,
 * zeroed the steering each time and gave it back after: the truck swerved.
 *
 * So, by how long the command counter has stood still:
 *
 * - under `kCommandWatchdogSeconds`: the stack's commands;
 * - up to `kSafetyStopSeconds`: the steering held where it was, the throttle
 *   and every button let go -- a stack that comes back drives on;
 * - longer: the stack is gone. The truck is stopped: the steering left to the
 *   driver, the brake at `kSafetyBrake`, the hazard lights switched on. Until
 *   the driver brakes or accelerates, or switches the beacon off; then
 *   everything is theirs, whatever the stack sends, until it lets go (a
 *   frame not engaged) or the beacon is off.
 *
 * Steering does not take over: it is the driver's while stopping already,
 * and the telemetry gives only its sign -- -1, 0 or 1 -- with the stack's
 * steering in it. In a roundabout in Torino the first stop read the wheel
 * turned left, the stack's, as the driver's: it braked for one frame.
 *
 * No SDK types here, so the logic is tested on its own (tools/safety_test.cpp).
 */
#ifndef ETS2AD_SAFETY_STOP_H
#define ETS2AD_SAFETY_STOP_H

#include "ets2ad/command_layout.h"

namespace ets2ad {

/** The brake the truck is stopped with: what the stack stops it with
 *  (tools/follow.py `_brake_to_rest`). */
inline constexpr float kSafetyBrake = 0.45f;

/** How long the hazard lights' switch is held: it toggles on the press. */
inline constexpr double kHazardPressSeconds = 0.5;

/** An input the driver gives beyond this is the driver taking over. */
inline constexpr float kDriverInput = 0.1f;

/** The pedals are not looked at for this long into a stop: the telemetry's
 *  inputs are the game's, smoothed, and still carry the stack's last. */
inline constexpr double kTakeoverAfterSeconds = 0.5;

/** What the telemetry says of the truck and its driver's pedals. The game's
 *  `input_*` channels add our device's contribution to the driver's; while
 *  stopping ours is zero but for the brake. */
struct DriverView {
    bool beacon = true;
    float input_throttle = 0.0f;
    float input_brake = 0.0f;
};

/** The newest telemetry's view, from the telemetry half of the plugin
 *  (telemetry_plugin.cpp). Read without a lock: a value a frame old, or
 *  half-written float, only moves the moment the driver is seen by a frame. */
DriverView driver_view();

/** Where this frame's outputs come from. */
enum class Source {
    kCommands,  ///< the stack's command block, as it is
    kHold,      ///< the stack's steering, held; nothing else
    kStop,      ///< the safety stop's: brake, hazard lights
    kNeutral,   ///< nothing: the driver's
};

struct Outputs {
    Source source = Source::kNeutral;
    float steering = 0.0f;  ///< for `kHold`
    float brake = 0.0f;     ///< for `kStop`
    bool hazard = false;    ///< for `kStop`
};

class SafetyStop {
public:
    enum class State {
        kIdle,      ///< commands, held, or nothing, by the silence
        kStopping,  ///< stopping the truck for a stack that has gone
        kReleased,  ///< the driver took over from a stop
    };

    /**
     * @param now_s     the plugin's clock, seconds
     * @param silent_s  how long the command counter has stood still
     * @param engaged   whether the last command frame was engaged
     * @param steering  the last command frame's steering, clamped
     * @param driver    the telemetry's view of the driver
     */
    Outputs step(const double now_s, const double silent_s, const bool engaged,
                 const float steering, const DriverView &driver) {
        if (silent_s < kCommandWatchdogSeconds) {
            armed_ = true;  // the stack is publishing: a later silence may stop
        }
        if (state_ == State::kStopping) {
            if (driver_takes_over(now_s - stopping_since_, driver)) {
                state_ = State::kReleased;
                return {};
            }
            return stop(now_s);
        }
        if (state_ == State::kReleased) {
            if (engaged && driver.beacon) {
                return {};
            }
            state_ = State::kIdle;  // the stack let go, or the beacon is off
        }
        if (!engaged) {
            return {};
        }
        if (silent_s < kCommandWatchdogSeconds) {
            return {Source::kCommands};
        }
        if (silent_s < kSafetyStopSeconds) {
            return {Source::kHold, steering};
        }
        if (!armed_) {
            return {};  // stopped for this silence already, and let go
        }
        armed_ = false;
        state_ = State::kStopping;
        stopping_since_ = now_s;
        return stop(now_s);
    }

    State state() const { return state_; }

    /** Back to the start: the device went inactive. */
    void reset() {
        state_ = State::kIdle;
        armed_ = true;
    }

private:
    static bool driver_takes_over(const double stopping_s, const DriverView &driver) {
        if (!driver.beacon) {
            return true;
        }
        return stopping_s >= kTakeoverAfterSeconds &&
               (driver.input_throttle > kDriverInput ||
                driver.input_brake > kSafetyBrake + kDriverInput);
    }

    Outputs stop(const double now_s) const {
        return {Source::kStop, 0.0f, kSafetyBrake, now_s - stopping_since_ < kHazardPressSeconds};
    }

    State state_ = State::kIdle;
    bool armed_ = true;
    double stopping_since_ = 0.0;
};

}  // namespace ets2ad

#endif  // ETS2AD_SAFETY_STOP_H
