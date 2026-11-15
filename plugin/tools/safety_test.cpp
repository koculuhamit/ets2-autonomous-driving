/**
 * @file safety_test.cpp
 * @brief What the plugin does when the stack goes quiet (src/safety_stop.h),
 * checked without the game. Prints one line per check, exits non-zero on a
 * failure; tests/unit/test_plugin_safety_stop.py runs it.
 */

#include <cstdio>

#include "safety_stop.h"

using ets2ad::DriverView;
using ets2ad::kSafetyBrake;
using ets2ad::Outputs;
using ets2ad::SafetyStop;
using ets2ad::Source;

namespace {

int failures = 0;

void check(const bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

constexpr float kSteer = -0.3f;

/** Silent for `silent` seconds at `now`, engaged, with `driver`. */
Outputs at(SafetyStop &safety, const double now, const double silent,
           const DriverView &driver = {}, const bool engaged = true) {
    return safety.step(now, silent, engaged, kSteer, driver);
}

}  // namespace

int main() {
    {
        SafetyStop s;
        check(at(s, 0.0, 0.05).source == Source::kCommands, "publishing: the stack's commands");
        const Outputs held = at(s, 0.5, 0.5);
        check(held.source == Source::kHold && held.steering == kSteer,
              "half a second quiet: the steering held where it was");
        check(at(s, 0.6, 0.0).source == Source::kCommands, "back within a second: it drives on");
        check(s.state() == SafetyStop::State::kIdle, "and nothing was stopped");
    }
    {
        SafetyStop s;
        at(s, 0.0, 0.0);
        const Outputs first = at(s, 1.2, 1.2);
        check(first.source == Source::kStop && first.brake == kSafetyBrake && first.hazard,
              "quiet for over a second: braking, hazard lights pressed");
        check(at(s, 2.0, 2.0).hazard == false, "the hazard switch let go after its press");
        check(at(s, 3.0, 0.0).source == Source::kStop,
              "a stack that comes back does not take the truck back");
        check(at(s, 30.0, 30.0).source == Source::kStop, "the brake held at rest, with no driver");
    }
    {
        SafetyStop s;
        at(s, 0.0, 0.0);
        at(s, 1.2, 1.2);
        DriverView accelerating;
        accelerating.input_throttle = 0.4f;
        check(at(s, 1.3, 1.3, accelerating).source == Source::kStop,
              "the stack's last throttle, still in the input: not the driver");
        check(at(s, 1.8, 1.8, accelerating).source == Source::kNeutral,
              "the driver accelerates: theirs");
        check(at(s, 1.9, 0.0).source == Source::kNeutral,
              "and the stack, engaged again, gets nothing");
        check(at(s, 2.0, 0.0, {}, false).source == Source::kNeutral, "until it lets go");
        check(at(s, 2.1, 0.0).source == Source::kCommands, "then it may drive again");
    }
    {
        SafetyStop s;
        at(s, 0.0, 0.0);
        at(s, 1.2, 1.2);
        DriverView braking;
        braking.input_brake = kSafetyBrake;  // only ours
        check(at(s, 1.3, 1.3, braking).source == Source::kStop,
              "our own brake in the input is not the driver");
        braking.input_brake = kSafetyBrake + 0.3f;
        check(at(s, 1.4, 1.4, braking).source == Source::kStop,
              "the stack's last brake, still in the input: not the driver");
        check(at(s, 1.8, 1.8, braking).source == Source::kNeutral, "the driver brakes: theirs");
    }
    {
        SafetyStop s;
        at(s, 0.0, 0.0);
        at(s, 1.2, 1.2);
        DriverView off;
        off.beacon = false;
        check(at(s, 1.3, 1.3, off).source == Source::kNeutral, "the beacon off: the driver's");
        check(at(s, 5.0, 5.0, off).source == Source::kNeutral,
              "and a stack still gone, its last frame engaged, is not stopped for again");
        check(at(s, 6.0, 6.0).source == Source::kNeutral, "not even with the beacon back on");
    }
    {
        SafetyStop s;
        check(at(s, 0.0, 5.0, {}, false).source == Source::kNeutral,
              "a stack that let go is not stopped for, however long it is quiet");
    }
    return failures == 0 ? 0 : 1;
}
