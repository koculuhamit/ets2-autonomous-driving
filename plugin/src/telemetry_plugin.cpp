/**
 * @file telemetry_plugin.cpp
 * @brief SCS telemetry plugin: publishes the game's telemetry to shared memory.
 *
 * The game loads this library from `bin/<platform>/plugins/` and calls into it
 * from its main thread. The contract the SDK imposes is strict: callbacks must
 * be quick, must not block, and must not call back into the game outside the
 * call that triggered them. So the plugin does the least possible work per
 * callback -- store a value into a staging copy -- and publishes once per
 * frame.
 *
 * Publishing uses the seqlock in `Header::sequence`. The staging copy exists
 * so that a reader never observes a frame half-assembled: channel callbacks
 * arrive one at a time throughout the frame, and copying the finished result
 * under an odd sequence number is both cheaper and safer than holding the lock
 * open across the whole frame.
 *
 * Channels are bound by address. Each registration carries a pointer to the
 * destination field as its context, so there is one small callback per value
 * type instead of one per channel, and adding a channel is a single table row.
 */

#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "scssdk_telemetry.h"
#include "common/scssdk_telemetry_common_channels.h"
#include "common/scssdk_telemetry_common_configs.h"
#include "common/scssdk_telemetry_job_common_channels.h"
#include "common/scssdk_telemetry_trailer_common_channels.h"
#include "common/scssdk_telemetry_truck_common_channels.h"
#include "eurotrucks2/scssdk_eut2.h"
#include "eurotrucks2/scssdk_telemetry_eut2.h"
#include "amtrucks/scssdk_ats.h"
#include "amtrucks/scssdk_telemetry_ats.h"

#include "ets2ad/telemetry_layout.h"
#include "diagnostics.h"
#include "safety_stop.h"
#include "shared_memory.h"

namespace {

using namespace ets2ad;

// --------------------------------------------------------------------------
// Plugin state
// --------------------------------------------------------------------------

/** Everything the plugin owns. A single instance; the SDK is single-threaded
 *  and the C entry points have nowhere else to put it. */
struct PluginState {
    SharedMemory memory;
    TelemetryBlock staging{};
    scs_log_t log = nullptr;
    scs_telemetry_register_for_channel_t register_channel = nullptr;
    std::chrono::steady_clock::time_point started_at;
    bool initialised = false;
};

PluginState g_state;

void log_line(const scs_log_type_t type, const std::string &message) {
    ets2ad::log_line(g_state.log, "telemetry", type, message);
}

/** Copy a game string into a fixed-size field, always NUL-terminated. */
void store_string(char *destination, std::size_t capacity, const scs_string_t source) {
    if (source == nullptr) {
        destination[0] = '\0';
        return;
    }
    std::strncpy(destination, source, capacity - 1);
    destination[capacity - 1] = '\0';
}

// --------------------------------------------------------------------------
// Publishing
// --------------------------------------------------------------------------

/**
 * @brief Copy the staging block into shared memory under the seqlock.
 *
 * Release/acquire ordering on the sequence counter is what makes this work: a
 * reader that sees an even, unchanged sequence before and after its copy is
 * guaranteed to have seen the payload writes that preceded the final store.
 */
void publish() {
    if (!g_state.memory.is_open()) {
        return;
    }

    auto *const block = static_cast<TelemetryBlock *>(g_state.memory.data());
    std::atomic_ref<std::uint32_t> sequence(block->header.sequence);

    const std::uint32_t start = sequence.load(std::memory_order_relaxed);
    sequence.store(start + 1, std::memory_order_release);  // odd: write in progress

    const auto now = std::chrono::steady_clock::now();
    Header &header = g_state.staging.header;
    header.magic = kMagic;
    header.layout_version = kLayoutVersion;
    header.block_size = static_cast<std::uint32_t>(sizeof(TelemetryBlock));
    header.write_count += 1;
    header.write_time = std::chrono::duration<double>(now - g_state.started_at).count();
    header.sequence = start + 1;  // keep the staging copy consistent with the block

    // Copy around the sequence counter: it belongs to the reader-facing block
    // and must not be clobbered part-way through a write.
    constexpr std::size_t kSequenceOffset = offsetof(Header, sequence);
    constexpr std::size_t kAfterSequence = kSequenceOffset + sizeof(std::uint32_t);
    std::memcpy(block, &g_state.staging, kSequenceOffset);
    std::memcpy(reinterpret_cast<char *>(block) + kAfterSequence,
                reinterpret_cast<const char *>(&g_state.staging) + kAfterSequence,
                sizeof(TelemetryBlock) - kAfterSequence);

    sequence.store(start + 2, std::memory_order_release);  // even: consistent
}

// --------------------------------------------------------------------------
// Channel callbacks
// --------------------------------------------------------------------------
//
// One callback per value type. The context is the address of the destination
// field, which keeps the binding table declarative.

SCSAPI_VOID channel_float(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                          const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_float) {
        *static_cast<float *>(context) = value->value_float.value;
    }
}

SCSAPI_VOID channel_s32(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                        const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_s32) {
        *static_cast<std::int32_t *>(context) = value->value_s32.value;
    }
}

SCSAPI_VOID channel_u32(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                        const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_u32) {
        *static_cast<std::uint32_t *>(context) = value->value_u32.value;
    }
}

SCSAPI_VOID channel_bool(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                         const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_bool) {
        *static_cast<std::uint8_t *>(context) = value->value_bool.value != 0 ? 1u : 0u;
    }
}

SCSAPI_VOID channel_fvector(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                            const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_fvector) {
        auto *const destination = static_cast<Vector3f *>(context);
        destination->x = value->value_fvector.x;
        destination->y = value->value_fvector.y;
        destination->z = value->value_fvector.z;
    }
}

SCSAPI_VOID channel_dplacement(const scs_string_t, const scs_u32_t, const scs_value_t *const value,
                               const scs_context_t context) {
    if (value != nullptr && value->type == SCS_VALUE_TYPE_dplacement) {
        auto *const destination = static_cast<Placement *>(context);
        destination->position.x = value->value_dplacement.position.x;
        destination->position.y = value->value_dplacement.position.y;
        destination->position.z = value->value_dplacement.position.z;
        destination->orientation.heading = value->value_dplacement.orientation.heading;
        destination->orientation.pitch = value->value_dplacement.orientation.pitch;
        destination->orientation.roll = value->value_dplacement.orientation.roll;
    }
}

// --------------------------------------------------------------------------
// Channel bindings
// --------------------------------------------------------------------------

struct ChannelBinding {
    const char *name;
    scs_u32_t index;
    scs_value_type_t type;
    void *target;
};

/** Register one channel, logging but tolerating failure.
 *
 * A channel the running game version does not provide is a normal outcome, not
 * an error: the SDK's own contract says removal is backwards compatible and
 * the plugin should assume a default. The field simply stays zero. */
void register_channel(const ChannelBinding &binding) {
    scs_telemetry_channel_callback_t callback = nullptr;
    switch (binding.type) {
        case SCS_VALUE_TYPE_float: callback = channel_float; break;
        case SCS_VALUE_TYPE_s32: callback = channel_s32; break;
        case SCS_VALUE_TYPE_u32: callback = channel_u32; break;
        case SCS_VALUE_TYPE_bool: callback = channel_bool; break;
        case SCS_VALUE_TYPE_fvector: callback = channel_fvector; break;
        case SCS_VALUE_TYPE_dplacement: callback = channel_dplacement; break;
        default:
            log_line(SCS_LOG_TYPE_error,
                     std::string("unsupported value type for channel ") + binding.name);
            return;
    }

    const scs_result_t result =
        g_state.register_channel(binding.name, binding.index, binding.type,
                                 SCS_TELEMETRY_CHANNEL_FLAG_none, callback, binding.target);
    if (result != SCS_RESULT_ok) {
        log_line(SCS_LOG_TYPE_warning,
                 std::string("channel unavailable in this game version: ") + binding.name);
    }
}

void register_all_channels() {
    TelemetryBlock &b = g_state.staging;

    const ChannelBinding bindings[] = {
        // Common
        {SCS_TELEMETRY_CHANNEL_game_time, SCS_U32_NIL, SCS_VALUE_TYPE_u32, &b.common.game_time},
        {SCS_TELEMETRY_CHANNEL_local_scale, SCS_U32_NIL, SCS_VALUE_TYPE_float, &b.common.scale},
        {SCS_TELEMETRY_CHANNEL_next_rest_stop, SCS_U32_NIL, SCS_VALUE_TYPE_s32,
         &b.common.next_rest_stop},

        // Truck: pose and motion
        {SCS_TELEMETRY_TRUCK_CHANNEL_world_placement, SCS_U32_NIL, SCS_VALUE_TYPE_dplacement,
         &b.truck.placement},
        {SCS_TELEMETRY_TRUCK_CHANNEL_local_linear_velocity, SCS_U32_NIL, SCS_VALUE_TYPE_fvector,
         &b.truck.linear_velocity},
        {SCS_TELEMETRY_TRUCK_CHANNEL_local_angular_velocity, SCS_U32_NIL, SCS_VALUE_TYPE_fvector,
         &b.truck.angular_velocity},
        {SCS_TELEMETRY_TRUCK_CHANNEL_local_linear_acceleration, SCS_U32_NIL,
         SCS_VALUE_TYPE_fvector, &b.truck.linear_acceleration},
        {SCS_TELEMETRY_TRUCK_CHANNEL_local_angular_acceleration, SCS_U32_NIL,
         SCS_VALUE_TYPE_fvector, &b.truck.angular_acceleration},
        {SCS_TELEMETRY_TRUCK_CHANNEL_cabin_offset, SCS_U32_NIL, SCS_VALUE_TYPE_fvector,
         &b.truck.cabin_offset_position},
        {SCS_TELEMETRY_TRUCK_CHANNEL_head_offset, SCS_U32_NIL, SCS_VALUE_TYPE_fvector,
         &b.truck.head_offset_position},
        {SCS_TELEMETRY_TRUCK_CHANNEL_speed, SCS_U32_NIL, SCS_VALUE_TYPE_float, &b.truck.speed},

        // Truck: powertrain
        {SCS_TELEMETRY_TRUCK_CHANNEL_engine_rpm, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.engine_rpm},
        {SCS_TELEMETRY_TRUCK_CHANNEL_engine_gear, SCS_U32_NIL, SCS_VALUE_TYPE_s32, &b.truck.gear},
        {SCS_TELEMETRY_TRUCK_CHANNEL_displayed_gear, SCS_U32_NIL, SCS_VALUE_TYPE_s32,
         &b.truck.gear_displayed},

        // Truck: driver input versus what the game applied
        {SCS_TELEMETRY_TRUCK_CHANNEL_input_steering, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.input_steering},
        {SCS_TELEMETRY_TRUCK_CHANNEL_input_throttle, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.input_throttle},
        {SCS_TELEMETRY_TRUCK_CHANNEL_input_brake, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.input_brake},
        {SCS_TELEMETRY_TRUCK_CHANNEL_input_clutch, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.input_clutch},
        {SCS_TELEMETRY_TRUCK_CHANNEL_effective_steering, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.effective_steering},
        {SCS_TELEMETRY_TRUCK_CHANNEL_effective_throttle, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.effective_throttle},
        {SCS_TELEMETRY_TRUCK_CHANNEL_effective_brake, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.effective_brake},
        {SCS_TELEMETRY_TRUCK_CHANNEL_effective_clutch, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.effective_clutch},
        {SCS_TELEMETRY_TRUCK_CHANNEL_cruise_control, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.cruise_control_speed},

        // Truck: brakes and assists
        {SCS_TELEMETRY_TRUCK_CHANNEL_parking_brake, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.parking_brake},
        {SCS_TELEMETRY_TRUCK_CHANNEL_motor_brake, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.motor_brake},
        {SCS_TELEMETRY_TRUCK_CHANNEL_retarder_level, SCS_U32_NIL, SCS_VALUE_TYPE_u32,
         &b.truck.retarder_level},
        {SCS_TELEMETRY_TRUCK_CHANNEL_brake_air_pressure, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.brake_air_pressure},
        {SCS_TELEMETRY_TRUCK_CHANNEL_brake_temperature, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.brake_temperature},

        // Truck: consumables and condition
        {SCS_TELEMETRY_TRUCK_CHANNEL_fuel, SCS_U32_NIL, SCS_VALUE_TYPE_float, &b.truck.fuel},
        {SCS_TELEMETRY_TRUCK_CHANNEL_fuel_average_consumption, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.fuel_average_consumption},
        {SCS_TELEMETRY_TRUCK_CHANNEL_fuel_range, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.fuel_range},
        {SCS_TELEMETRY_TRUCK_CHANNEL_adblue, SCS_U32_NIL, SCS_VALUE_TYPE_float, &b.truck.adblue},
        {SCS_TELEMETRY_TRUCK_CHANNEL_oil_pressure, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.oil_pressure},
        {SCS_TELEMETRY_TRUCK_CHANNEL_oil_temperature, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.oil_temperature},
        {SCS_TELEMETRY_TRUCK_CHANNEL_water_temperature, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.water_temperature},
        {SCS_TELEMETRY_TRUCK_CHANNEL_battery_voltage, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.battery_voltage},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wear_engine, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.wear_engine},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wear_transmission, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.wear_transmission},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wear_cabin, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.wear_cabin},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wear_chassis, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.wear_chassis},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wear_wheels, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.wear_wheels},
        {SCS_TELEMETRY_TRUCK_CHANNEL_odometer, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.truck.odometer},

        // Truck: electrical and lights
        {SCS_TELEMETRY_TRUCK_CHANNEL_engine_enabled, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.engine_enabled},
        {SCS_TELEMETRY_TRUCK_CHANNEL_electric_enabled, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.electric_enabled},
        {SCS_TELEMETRY_TRUCK_CHANNEL_lblinker, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.indicator_left},
        {SCS_TELEMETRY_TRUCK_CHANNEL_rblinker, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.indicator_right},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_parking, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_parking},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_low_beam, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_low_beam},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_high_beam, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_high_beam},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_beacon, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_beacon},
        {SCS_TELEMETRY_TRUCK_CHANNEL_hazard_warning, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.hazard_warning},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_brake, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_brake},
        {SCS_TELEMETRY_TRUCK_CHANNEL_light_reverse, SCS_U32_NIL, SCS_VALUE_TYPE_bool,
         &b.truck.light_reverse},
        {SCS_TELEMETRY_TRUCK_CHANNEL_wipers, SCS_U32_NIL, SCS_VALUE_TYPE_bool, &b.truck.wipers},

        // Navigation: the game's own route oracle
        {SCS_TELEMETRY_TRUCK_CHANNEL_navigation_distance, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.navigation.distance},
        {SCS_TELEMETRY_TRUCK_CHANNEL_navigation_time, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.navigation.time},
        {SCS_TELEMETRY_TRUCK_CHANNEL_navigation_speed_limit, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.navigation.speed_limit},

        // Job
        {SCS_TELEMETRY_JOB_CHANNEL_cargo_damage, SCS_U32_NIL, SCS_VALUE_TYPE_float,
         &b.trailers[0].cargo_damage},
    };

    for (const ChannelBinding &binding : bindings) {
        register_channel(binding);
    }

    // Per-wheel channels are indexed. Registering each index against its own
    // array slot keeps the type-dispatch table above unchanged.
    for (scs_u32_t wheel = 0; wheel < static_cast<scs_u32_t>(kMaxWheels); ++wheel) {
        const ChannelBinding wheel_bindings[] = {
            {SCS_TELEMETRY_TRUCK_CHANNEL_wheel_steering, wheel, SCS_VALUE_TYPE_float,
             &b.truck.wheels.steering[wheel]},
            {SCS_TELEMETRY_TRUCK_CHANNEL_wheel_rotation, wheel, SCS_VALUE_TYPE_float,
             &b.truck.wheels.rotation[wheel]},
            {SCS_TELEMETRY_TRUCK_CHANNEL_wheel_lift, wheel, SCS_VALUE_TYPE_float,
             &b.truck.wheels.lift[wheel]},
            {SCS_TELEMETRY_TRUCK_CHANNEL_wheel_susp_deflection, wheel, SCS_VALUE_TYPE_float,
             &b.truck.wheels.suspension[wheel]},
            {SCS_TELEMETRY_TRUCK_CHANNEL_wheel_on_ground, wheel, SCS_VALUE_TYPE_bool,
             &b.truck.wheels.on_ground[wheel]},
        };
        for (const ChannelBinding &binding : wheel_bindings) {
            register_channel(binding);
        }
    }

    // Trailer channels share their names across trailer indices via the
    // SCS_TELEMETRY_TRAILER_CHANNEL_ prefix, which already encodes the index.
    for (std::size_t trailer = 0; trailer < kMaxTrailers; ++trailer) {
        const std::string prefix = "trailer." + std::to_string(trailer) + ".";
        const struct {
            const char *suffix;
            scs_value_type_t type;
            void *target;
        } trailer_bindings[] = {
            {"connected", SCS_VALUE_TYPE_bool, &b.trailers[trailer].connected},
            {"world.placement", SCS_VALUE_TYPE_dplacement, &b.trailers[trailer].placement},
            {"velocity.linear", SCS_VALUE_TYPE_fvector, &b.trailers[trailer].linear_velocity},
            {"velocity.angular", SCS_VALUE_TYPE_fvector, &b.trailers[trailer].angular_velocity},
            {"acceleration.linear", SCS_VALUE_TYPE_fvector,
             &b.trailers[trailer].linear_acceleration},
            {"acceleration.angular", SCS_VALUE_TYPE_fvector,
             &b.trailers[trailer].angular_acceleration},
            {"wear.chassis", SCS_VALUE_TYPE_float, &b.trailers[trailer].wear_chassis},
            {"wear.wheels", SCS_VALUE_TYPE_float, &b.trailers[trailer].wear_wheels},
        };
        for (const auto &binding : trailer_bindings) {
            const std::string name = prefix + binding.suffix;
            register_channel({name.c_str(), SCS_U32_NIL, binding.type, binding.target});
        }
    }
}

// --------------------------------------------------------------------------
// Event callbacks
// --------------------------------------------------------------------------

SCSAPI_VOID event_frame_end(const scs_event_t, const void *const, const scs_context_t) {
    publish();
}

SCSAPI_VOID event_paused(const scs_event_t event, const void *const, const scs_context_t) {
    if (event == SCS_TELEMETRY_EVENT_paused) {
        g_state.staging.header.flags |= kFlagPaused;
    } else {
        g_state.staging.header.flags &= ~kFlagPaused;
    }
    // Publish immediately: a consumer holding the throttle open must learn
    // about a pause without waiting for a frame that is not coming.
    publish();
}

/**
 * @brief Apply an indexed per-wheel configuration attribute.
 *
 * `wheel_position` is the important one: with the wheel positions in the
 * vehicle frame, the wheelbase is measured rather than assumed, which is what
 * Phase 3's bicycle model is fitted against.
 *
 * @return true if the attribute was a per-wheel one and was consumed.
 */
bool apply_wheel_attribute(WheelState &wheels, const scs_named_value_t &attribute) {
    const scs_u32_t index = attribute.index;
    if (index == SCS_U32_NIL || index >= static_cast<scs_u32_t>(kMaxWheels)) {
        return false;
    }
    const char *const name = attribute.name;
    const scs_value_t &value = attribute.value;

    if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_position) == 0 &&
        value.type == SCS_VALUE_TYPE_fvector) {
        wheels.position[index].x = value.value_fvector.x;
        wheels.position[index].y = value.value_fvector.y;
        wheels.position[index].z = value.value_fvector.z;
        return true;
    }
    if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_radius) == 0 &&
        value.type == SCS_VALUE_TYPE_float) {
        wheels.radius[index] = value.value_float.value;
        return true;
    }
    if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_steerable) == 0 &&
        value.type == SCS_VALUE_TYPE_bool) {
        wheels.steerable[index] = value.value_bool.value != 0 ? 1u : 0u;
        return true;
    }
    if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_powered) == 0 &&
        value.type == SCS_VALUE_TYPE_bool) {
        wheels.powered[index] = value.value_bool.value != 0 ? 1u : 0u;
        return true;
    }
    if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_liftable) == 0 &&
        value.type == SCS_VALUE_TYPE_bool) {
        wheels.liftable[index] = value.value_bool.value != 0 ? 1u : 0u;
        return true;
    }
    return false;
}

/**
 * @brief Semi-static configuration: truck model, trailer geometry, active job.
 *
 * These arrive as attribute lists rather than channels, and they change when
 * the player switches truck or takes a job -- which is also when the vehicle
 * model the controllers rely on changes underneath them.
 */
SCSAPI_VOID event_configuration(const scs_event_t, const void *const event_info,
                                const scs_context_t) {
    const auto *const info = static_cast<const scs_telemetry_configuration_t *>(event_info);
    if (info == nullptr || info->id == nullptr) {
        return;
    }

    TelemetryBlock &b = g_state.staging;
    const bool is_truck = std::strcmp(info->id, SCS_TELEMETRY_CONFIG_truck) == 0;
    const bool is_job = std::strcmp(info->id, SCS_TELEMETRY_CONFIG_job) == 0;

    std::size_t trailer_index = kMaxTrailers;
    if (std::strncmp(info->id, "trailer", 7) == 0) {
        trailer_index = (info->id[7] == '.') ? static_cast<std::size_t>(info->id[8] - '0') : 0;
        if (trailer_index >= kMaxTrailers) {
            return;
        }
    }

    for (const scs_named_value_t *attribute = info->attributes;
         attribute != nullptr && attribute->name != nullptr; ++attribute) {
        const char *const name = attribute->name;
        const scs_value_t &value = attribute->value;

        if (is_truck) {
            if (apply_wheel_attribute(b.truck.wheels, *attribute)) {
                continue;
            }
            if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_brand_id) == 0 &&
                value.type == SCS_VALUE_TYPE_string) {
                store_string(b.truck.brand_id, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.truck.model_id, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_rpm_limit) == 0 &&
                       value.type == SCS_VALUE_TYPE_float) {
                b.truck.engine_rpm_max = value.value_float.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_forward_gear_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.truck.forward_gear_count = static_cast<std::int32_t>(value.value_u32.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_reverse_gear_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.truck.reverse_gear_count = static_cast<std::int32_t>(value.value_u32.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_retarder_step_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.truck.retarder_step_count = value.value_u32.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_fuel_capacity) == 0 &&
                       value.type == SCS_VALUE_TYPE_float) {
                b.truck.fuel_capacity = value.value_float.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_adblue_capacity) == 0 &&
                       value.type == SCS_VALUE_TYPE_float) {
                b.truck.adblue_capacity = value.value_float.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.truck.wheels.count = value.value_u32.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_hook_position) == 0 &&
                       value.type == SCS_VALUE_TYPE_fvector) {
                b.truck.hook_position[0] = value.value_fvector.x;
                b.truck.hook_position[1] = value.value_fvector.y;
                b.truck.hook_position[2] = value.value_fvector.z;
            }
        } else if (trailer_index < kMaxTrailers) {
            TrailerState &trailer = b.trailers[trailer_index];
            if (apply_wheel_attribute(trailer.wheels, *attribute)) {
                continue;
            }
            if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_id) == 0 &&
                value.type == SCS_VALUE_TYPE_string) {
                store_string(trailer.id, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo_accessory_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(trailer.cargo_accessory_id, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_hook_position) == 0 &&
                       value.type == SCS_VALUE_TYPE_fvector) {
                trailer.hook_position[0] = value.value_fvector.x;
                trailer.hook_position[1] = value.value_fvector.y;
                trailer.hook_position[2] = value.value_fvector.z;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_wheel_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                trailer.wheels.count = value.value_u32.value;
                trailer.axle_count = value.value_u32.value / 2;
            }
        } else if (is_job) {
            if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo_id) == 0 &&
                value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.cargo_id, kStringLength, value.value_string.value);
                b.job.job_active = (b.job.cargo_id[0] != '\0') ? 1u : 0u;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.cargo, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_source_city_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.source_city, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_source_company_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.source_company, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_destination_city_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.destination_city, kStringLength, value.value_string.value);
            } else if (std::strcmp(name,
                                   SCS_TELEMETRY_CONFIG_ATTRIBUTE_destination_company_id) == 0 &&
                       value.type == SCS_VALUE_TYPE_string) {
                store_string(b.job.destination_company, kStringLength, value.value_string.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo_mass) == 0 &&
                       value.type == SCS_VALUE_TYPE_float) {
                b.job.cargo_mass = value.value_float.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo_unit_count) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.job.cargo_unit_count = value.value_u32.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_cargo_unit_mass) == 0 &&
                       value.type == SCS_VALUE_TYPE_float) {
                b.job.cargo_unit_mass = value.value_float.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_income) == 0 &&
                       value.type == SCS_VALUE_TYPE_u64) {
                b.job.income = static_cast<std::uint32_t>(value.value_u64.value);
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_delivery_time) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.job.delivery_time = value.value_u32.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_planned_distance_km) == 0 &&
                       value.type == SCS_VALUE_TYPE_u32) {
                b.job.planned_distance = value.value_u32.value;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_is_cargo_loaded) == 0 &&
                       value.type == SCS_VALUE_TYPE_bool) {
                b.job.cargo_loaded = value.value_bool.value != 0 ? 1u : 0u;
            } else if (std::strcmp(name, SCS_TELEMETRY_CONFIG_ATTRIBUTE_special_job) == 0 &&
                       value.type == SCS_VALUE_TYPE_bool) {
                b.job.special_job = value.value_bool.value != 0 ? 1u : 0u;
            }
        }
    }

    publish();
}

}  // namespace

ets2ad::DriverView ets2ad::driver_view() {
    const auto &truck = g_state.staging.truck;
    DriverView view;
    view.beacon = truck.light_beacon != 0;
    view.input_throttle = truck.input_throttle;
    view.input_brake = truck.input_brake;
    return view;
}

// --------------------------------------------------------------------------
// SDK entry points
// --------------------------------------------------------------------------

SCSAPI_RESULT scs_telemetry_init(const scs_u32_t version,
                                 const scs_telemetry_init_params_t *const params) {
    // Only the 1.01 shape is claimed. Returning unsupported for anything else
    // lets the game negotiate downwards rather than us guessing at a layout.
    if (version != SCS_TELEMETRY_VERSION_1_01) {
        return SCS_RESULT_unsupported;
    }

    ets2ad::reset_diagnostics();

    const auto *const p = static_cast<const scs_telemetry_init_params_v101_t *>(params);
    g_state.log = p->common.log;
    g_state.register_channel = p->register_for_channel;
    g_state.started_at = std::chrono::steady_clock::now();
    g_state.staging = TelemetryBlock{};

    const bool is_ets2 = std::strcmp(p->common.game_id, SCS_GAME_ID_EUT2) == 0;
    g_state.staging.header.game_id = is_ets2 ? kGameEts2 : kGameAts;
    g_state.staging.header.game_version = static_cast<std::uint32_t>(p->common.game_version);
    g_state.staging.header.sdk_version = version;
    g_state.staging.header.flags = kFlagTelemetryActive;

    if (!g_state.memory.create(kSharedMemoryName, sizeof(TelemetryBlock))) {
        log_line(SCS_LOG_TYPE_error, "shared memory unavailable: " + g_state.memory.error());
        return SCS_RESULT_generic_error;
    }

    const bool events_ok =
        p->register_for_event(SCS_TELEMETRY_EVENT_frame_end, event_frame_end, nullptr) ==
            SCS_RESULT_ok &&
        p->register_for_event(SCS_TELEMETRY_EVENT_paused, event_paused, nullptr) ==
            SCS_RESULT_ok &&
        p->register_for_event(SCS_TELEMETRY_EVENT_started, event_paused, nullptr) ==
            SCS_RESULT_ok &&
        p->register_for_event(SCS_TELEMETRY_EVENT_configuration, event_configuration, nullptr) ==
            SCS_RESULT_ok;
    if (!events_ok) {
        log_line(SCS_LOG_TYPE_error, "could not register the required telemetry events");
        g_state.memory.destroy();
        return SCS_RESULT_generic_error;
    }

    register_all_channels();
    publish();

    g_state.initialised = true;
    log_line(SCS_LOG_TYPE_message,
             "telemetry published to shared memory '" + std::string(kSharedMemoryName) + "' (" +
                 std::to_string(sizeof(TelemetryBlock)) + " bytes, layout v" +
                 std::to_string(kLayoutVersion) + ")");
    return SCS_RESULT_ok;
}

SCSAPI_VOID scs_telemetry_shutdown(void) {
    // The SDK may init/shutdown several times per process lifetime, so this
    // has to leave the plugin ready to start again.
    log_line(SCS_LOG_TYPE_message,
             "shutting down after " + std::to_string(g_state.staging.header.write_count) +
                 " published frames");
    g_state.memory.destroy();
    g_state.log = nullptr;
    g_state.register_channel = nullptr;
    g_state.initialised = false;
}
