/**
 * @file layout_dump.cpp
 * @brief Prints the shared memory layout as JSON.
 *
 * The Python reader in `core/bridge/` mirrors `telemetry_layout.h`. Two
 * descriptions of one binary format drift apart the moment someone edits one
 * of them, and the symptom -- plausible but wrong telemetry -- is miserable to
 * debug.
 *
 * So the C++ side is treated as the source of truth and made to describe
 * itself. This tool is built alongside the plugin and its output is what the
 * Python layout test asserts against, on every platform CI builds for. A
 * padding difference between GCC and MSVC shows up as a failing test rather
 * than as a truck steering into a barrier on Windows only.
 */

#include <cstddef>
#include <cstdio>
#include <cstdint>

#include "ets2ad/command_layout.h"
#include "ets2ad/telemetry_layout.h"

using namespace ets2ad;

namespace {

bool first_field = true;

void begin_struct(const char *name, std::size_t size, std::size_t alignment) {
    std::printf("%s\n    \"%s\": {\n      \"size\": %zu,\n      \"align\": %zu,\n"
                "      \"fields\": {",
                first_field ? "" : ",", name, size, alignment);
    first_field = false;
}

bool first_member = true;

void field(const char *name, std::size_t offset, std::size_t size) {
    std::printf("%s\n        \"%s\": [%zu, %zu]", first_member ? "" : ",", name, offset, size);
    first_member = false;
}

void end_struct() {
    std::printf("\n      }\n    }");
    first_member = true;
}

}  // namespace

#define FIELD(Struct, Member) field(#Member, offsetof(Struct, Member), sizeof(Struct::Member))
#define BEGIN(Struct) begin_struct(#Struct, sizeof(Struct), alignof(Struct))

int main() {
    std::printf("{\n  \"layout_version\": %u,\n  \"magic\": %u,\n"
                "  \"shared_memory_name\": \"%s\",\n  \"string_length\": %zu,\n"
                "  \"max_wheels\": %zu,\n  \"max_trailers\": %zu,\n"
                "  \"command_layout_version\": %u,\n  \"command_magic\": %u,\n"
                "  \"command_memory_name\": \"%s\",\n"
                "  \"command_watchdog_seconds\": %.3f,\n"
                "  \"command_safety_stop_seconds\": %.3f,\n  \"structs\": {",
                kLayoutVersion, kMagic, kSharedMemoryName, kStringLength, kMaxWheels,
                kMaxTrailers, kCommandLayoutVersion, kCommandMagic, kCommandMemoryName,
                kCommandWatchdogSeconds, kSafetyStopSeconds);

    BEGIN(Header);
    FIELD(Header, magic);
    FIELD(Header, layout_version);
    FIELD(Header, block_size);
    FIELD(Header, sequence);
    FIELD(Header, flags);
    FIELD(Header, game_id);
    FIELD(Header, game_version);
    FIELD(Header, sdk_version);
    FIELD(Header, write_count);
    FIELD(Header, write_time);
    FIELD(Header, _reserved);
    end_struct();

    BEGIN(CommonState);
    FIELD(CommonState, game_time);
    FIELD(CommonState, next_rest_stop);
    FIELD(CommonState, scale);
    FIELD(CommonState, _padding);
    end_struct();

    BEGIN(WheelState);
    FIELD(WheelState, steering);
    FIELD(WheelState, rotation);
    FIELD(WheelState, lift);
    FIELD(WheelState, suspension);
    FIELD(WheelState, position);
    FIELD(WheelState, radius);
    FIELD(WheelState, on_ground);
    FIELD(WheelState, steerable);
    FIELD(WheelState, powered);
    FIELD(WheelState, liftable);
    FIELD(WheelState, count);
    FIELD(WheelState, _padding);
    end_struct();

    BEGIN(TruckState);
    FIELD(TruckState, placement);
    FIELD(TruckState, linear_velocity);
    FIELD(TruckState, angular_velocity);
    FIELD(TruckState, linear_acceleration);
    FIELD(TruckState, angular_acceleration);
    FIELD(TruckState, cabin_offset_position);
    FIELD(TruckState, head_offset_position);
    FIELD(TruckState, speed);
    FIELD(TruckState, engine_rpm);
    FIELD(TruckState, engine_rpm_max);
    FIELD(TruckState, input_steering);
    FIELD(TruckState, input_throttle);
    FIELD(TruckState, input_brake);
    FIELD(TruckState, input_clutch);
    FIELD(TruckState, effective_steering);
    FIELD(TruckState, effective_throttle);
    FIELD(TruckState, effective_brake);
    FIELD(TruckState, effective_clutch);
    FIELD(TruckState, cruise_control_speed);
    FIELD(TruckState, fuel);
    FIELD(TruckState, fuel_capacity);
    FIELD(TruckState, fuel_average_consumption);
    FIELD(TruckState, fuel_range);
    FIELD(TruckState, adblue);
    FIELD(TruckState, adblue_capacity);
    FIELD(TruckState, brake_air_pressure);
    FIELD(TruckState, brake_temperature);
    FIELD(TruckState, oil_pressure);
    FIELD(TruckState, oil_temperature);
    FIELD(TruckState, water_temperature);
    FIELD(TruckState, battery_voltage);
    FIELD(TruckState, wear_engine);
    FIELD(TruckState, wear_transmission);
    FIELD(TruckState, wear_cabin);
    FIELD(TruckState, wear_chassis);
    FIELD(TruckState, wear_wheels);
    FIELD(TruckState, odometer);
    FIELD(TruckState, hook_position);
    FIELD(TruckState, gear);
    FIELD(TruckState, gear_displayed);
    FIELD(TruckState, forward_gear_count);
    FIELD(TruckState, reverse_gear_count);
    FIELD(TruckState, retarder_level);
    FIELD(TruckState, retarder_step_count);
    FIELD(TruckState, engine_enabled);
    FIELD(TruckState, electric_enabled);
    FIELD(TruckState, parking_brake);
    FIELD(TruckState, motor_brake);
    FIELD(TruckState, indicator_left);
    FIELD(TruckState, indicator_right);
    FIELD(TruckState, light_parking);
    FIELD(TruckState, light_low_beam);
    FIELD(TruckState, light_high_beam);
    FIELD(TruckState, light_brake);
    FIELD(TruckState, light_reverse);
    FIELD(TruckState, light_beacon);
    FIELD(TruckState, wipers);
    FIELD(TruckState, cruise_control_active);
    FIELD(TruckState, hazard_warning);
    FIELD(TruckState, _padding);
    FIELD(TruckState, wheels);
    FIELD(TruckState, brand_id);
    FIELD(TruckState, model_id);
    FIELD(TruckState, _tail_padding);
    end_struct();

    BEGIN(TrailerState);
    FIELD(TrailerState, placement);
    FIELD(TrailerState, linear_velocity);
    FIELD(TrailerState, angular_velocity);
    FIELD(TrailerState, linear_acceleration);
    FIELD(TrailerState, angular_acceleration);
    FIELD(TrailerState, hook_position);
    FIELD(TrailerState, cargo_damage);
    FIELD(TrailerState, wear_chassis);
    FIELD(TrailerState, wear_wheels);
    FIELD(TrailerState, axle_count);
    FIELD(TrailerState, connected);
    FIELD(TrailerState, _padding);
    FIELD(TrailerState, wheels);
    FIELD(TrailerState, id);
    FIELD(TrailerState, cargo_accessory_id);
    FIELD(TrailerState, _tail_padding);
    end_struct();

    BEGIN(JobState);
    FIELD(JobState, cargo_mass);
    FIELD(JobState, cargo_unit_mass);
    FIELD(JobState, cargo_unit_count);
    FIELD(JobState, income);
    FIELD(JobState, delivery_time);
    FIELD(JobState, planned_distance);
    FIELD(JobState, cargo_loaded);
    FIELD(JobState, special_job);
    FIELD(JobState, job_active);
    FIELD(JobState, _padding);
    FIELD(JobState, _padding2);
    FIELD(JobState, cargo_id);
    FIELD(JobState, cargo);
    FIELD(JobState, source_city);
    FIELD(JobState, source_company);
    FIELD(JobState, destination_city);
    FIELD(JobState, destination_company);
    end_struct();

    BEGIN(NavigationState);
    FIELD(NavigationState, distance);
    FIELD(NavigationState, time);
    FIELD(NavigationState, speed_limit);
    FIELD(NavigationState, _padding);
    end_struct();

    BEGIN(CommandBlock);
    FIELD(CommandBlock, magic);
    FIELD(CommandBlock, layout_version);
    FIELD(CommandBlock, block_size);
    FIELD(CommandBlock, sequence);
    FIELD(CommandBlock, command_count);
    FIELD(CommandBlock, command_time);
    FIELD(CommandBlock, flags);
    FIELD(CommandBlock, buttons);
    FIELD(CommandBlock, axes);
    FIELD(CommandBlock, _padding);
    FIELD(CommandBlock, _reserved);
    end_struct();

    BEGIN(TelemetryBlock);
    FIELD(TelemetryBlock, header);
    FIELD(TelemetryBlock, common);
    FIELD(TelemetryBlock, truck);
    FIELD(TelemetryBlock, trailers);
    FIELD(TelemetryBlock, job);
    FIELD(TelemetryBlock, navigation);
    end_struct();

    std::printf("\n  }\n}\n");
    return 0;
}
