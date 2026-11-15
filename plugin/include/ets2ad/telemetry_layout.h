/**
 * @file telemetry_layout.h
 * @brief Layout of the shared memory block the plugin publishes.
 *
 * This is the ABI between the plugin and everything that reads it: the Python
 * stack on Linux and Windows, and any tool that opens the block directly. Three
 * properties have to hold, and all three are enforced rather than hoped for.
 *
 * **Identical on every platform.** Fields are ordered widest-first and every
 * gap is an explicit `_padding` member, so the compiler never gets to insert
 * padding of its own. Sizes are asserted at compile time; `layout_dump` emits
 * every offset so the Python mirror is checked against the real thing instead
 * of against a transcription of it.
 *
 * **Readable without locking.** The game writes from its main thread while a
 * reader in another process reads whenever it likes. A seqlock covers this:
 * the writer makes `sequence` odd, writes, then makes it even. A reader that
 * sees an odd value, or a different value before and after, retries. No shared
 * mutex, no blocking the game's main thread, and a torn read is detectable
 * rather than silent.
 *
 * **Versioned.** `layout_version` changes whenever a field moves. Readers check
 * it and refuse to guess.
 *
 * Any change to this file is a breaking change to the reader. Bump
 * `kLayoutVersion`, update `core/bridge/`, and run the layout test.
 */
#ifndef ETS2AD_TELEMETRY_LAYOUT_H
#define ETS2AD_TELEMETRY_LAYOUT_H

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ets2ad {

/** "ET2A" little-endian. A reader that does not see this is looking at the
 *  wrong block, or at one that was never initialised. */
inline constexpr std::uint32_t kMagic = 0x41325445u;

/** Incremented whenever any field in this file moves, changes type or is
 *  removed. Adding a field inside existing reserved space does not count. */
inline constexpr std::uint32_t kLayoutVersion = 1u;

/** Name of the shared memory object. On Linux it appears as
 *  `/dev/shm/ets2ad_telemetry`; on Windows it is a `CreateFileMapping` name. */
inline constexpr char kSharedMemoryName[] = "ets2ad_telemetry";

/** Fixed capacity for text fields, including the terminating NUL. Game strings
 *  are truncated to fit; none of them drive control decisions. */
inline constexpr std::size_t kStringLength = 64;

/** Maximum wheels reported per vehicle. The SDK's own limit for trucks. */
inline constexpr std::size_t kMaxWheels = 16;

/** Trailers published. ETS2 supports multi-trailer combinations; the first is
 *  the one that matters for hitch angle and parking. */
inline constexpr std::size_t kMaxTrailers = 3;

// --------------------------------------------------------------------------
// Primitives
// --------------------------------------------------------------------------

/** Single-precision 3-vector, used for velocities and accelerations. */
struct Vector3f {
    float x;
    float y;
    float z;
};
static_assert(sizeof(Vector3f) == 12, "Vector3f must be 12 bytes");

/** Double-precision 3-vector. World positions only: ETS2's world is large
 *  enough that float loses centimetre resolution at the far edges. */
struct Vector3d {
    double x;
    double y;
    double z;
};
static_assert(sizeof(Vector3d) == 24, "Vector3d must be 24 bytes");

/**
 * @brief Orientation as the game reports it.
 *
 * Units are *rotations*, not radians: 0.0 to 1.0 covers a full turn. Heading
 * grows counter-clockwise, with 0 pointing north. Converting this into a
 * conventional right-handed yaw is the reader's job and is done in exactly one
 * place, `core/bridge/`, so the convention is decided once.
 */
struct Orientation {
    float heading;
    float pitch;
    float roll;
};
static_assert(sizeof(Orientation) == 12, "Orientation must be 12 bytes");

/** World position plus orientation. */
struct Placement {
    Vector3d position;        ///< metres, game world frame
    Orientation orientation;  ///< rotations, see Orientation
    std::uint32_t _padding;   ///< explicit: keeps the 8-byte alignment visible
};
static_assert(sizeof(Placement) == 40, "Placement must be 40 bytes");

// --------------------------------------------------------------------------
// Header
// --------------------------------------------------------------------------

/** Bits in Header::flags. */
enum HeaderFlag : std::uint32_t {
    kFlagTelemetryActive = 1u << 0,  ///< the game has telemetry switched on
    kFlagPaused = 1u << 1,           ///< game paused or in a menu
    kFlagTimestampsValid = 1u << 2,  ///< frame timing fields are populated
};

/** Game identifiers. The plugin builds for both; the reader refuses ATS. */
enum GameId : std::uint32_t {
    kGameUnknown = 0u,
    kGameEts2 = 1u,
    kGameAts = 2u,
};

/**
 * @brief Fixed-size block header. Never moves, never changes size.
 *
 * A reader validates `magic` and `layout_version` before trusting anything
 * else, which is what makes a stale plugin binary produce a clear error rather
 * than nonsense telemetry.
 */
struct Header {
    std::uint32_t magic;           ///< kMagic
    std::uint32_t layout_version;  ///< kLayoutVersion
    std::uint32_t block_size;      ///< sizeof(TelemetryBlock)

    /**
     * Seqlock counter. Odd while a write is in progress.
     *
     * Plain `uint32_t` on purpose: this block is a wire format read by C++,
     * Python and anything else that opens it, so it has to stay trivially
     * copyable. The writer accesses it through `std::atomic_ref`, which gives
     * the ordering guarantees without changing the layout.
     */
    std::uint32_t sequence;

    std::uint32_t flags;         ///< HeaderFlag bits
    std::uint32_t game_id;       ///< GameId
    std::uint32_t game_version;  ///< major << 16 | minor
    std::uint32_t sdk_version;   ///< major << 16 | minor

    std::uint64_t write_count;  ///< total publishes since plugin init

    /** Seconds on the plugin's steady clock at the last publish. Monotonic,
     *  unaffected by the game's own clock or by pausing. */
    double write_time;

    std::uint64_t _reserved[4];  ///< room to grow without moving anything
};
static_assert(sizeof(Header) == 80, "Header must be 80 bytes");
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free,
              "the seqlock requires a lock-free 32-bit atomic; a mutex in "
              "shared memory would not be readable from another process");

// --------------------------------------------------------------------------
// Sections
// --------------------------------------------------------------------------

/** Game-wide state that is not about any particular vehicle. */
struct CommonState {
    std::uint32_t game_time;      ///< in-game minutes since epoch
    std::int32_t next_rest_stop;  ///< in-game minutes until rest is required

    /**
     * Distance and time scaling the game applies. The map is not 1:1 and the
     * factor varies by region, so odometer and ETA deltas must be divided by
     * it before they mean anything in metres or seconds.
     */
    float scale;

    float _padding;
};
static_assert(sizeof(CommonState) == 16, "CommonState must be 16 bytes");

/**
 * @brief Per-wheel data. `count` says how many entries are meaningful.
 *
 * The first block is dynamic, updated every frame from channels. The second is
 * static geometry delivered by configuration events when the vehicle changes.
 * Keeping them together is deliberate: `position` and `radius` are what turn
 * the wheel channels into something physical, and separating them would mean
 * two lookups at every use.
 *
 * `position` in particular is the wheelbase. Phase 3 measures a bicycle model
 * against the real vehicle rather than guessing its geometry, and this is
 * where that geometry comes from.
 */
struct WheelState {
    // Dynamic, per frame.
    float steering[kMaxWheels];    ///< rotations, steered wheels only
    float rotation[kMaxWheels];    ///< rotations, accumulated
    float lift[kMaxWheels];        ///< 0 = down, 1 = fully lifted
    float suspension[kMaxWheels];  ///< deflection, metres

    // Static, from configuration events.
    Vector3f position[kMaxWheels];  ///< vehicle frame, metres
    float radius[kMaxWheels];       ///< metres

    std::uint8_t on_ground[kMaxWheels];
    std::uint8_t steerable[kMaxWheels];
    std::uint8_t powered[kMaxWheels];
    std::uint8_t liftable[kMaxWheels];

    std::uint32_t count;
    std::uint32_t _padding[3];
};
static_assert(sizeof(WheelState) == 592, "WheelState must be 592 bytes");

/**
 * @brief Truck state.
 *
 * `input_*` is what the driver (or our virtual device) asked for; `effective_*`
 * is what the game applied after its own filtering, assists and gearbox logic.
 * System identification in Phase 3 needs both: the gap between them *is* the
 * actuator model.
 */
struct TruckState {
    Placement placement;  ///< 40

    Vector3f linear_velocity;       ///< m/s, vehicle frame
    Vector3f angular_velocity;      ///< rotations/s, vehicle frame
    Vector3f linear_acceleration;   ///< m/s^2, vehicle frame
    Vector3f angular_acceleration;  ///< rotations/s^2, vehicle frame
    Vector3f cabin_offset_position;
    Vector3f head_offset_position;

    float speed;  ///< m/s, negative when reversing

    float engine_rpm;
    float engine_rpm_max;

    float input_steering;
    float input_throttle;
    float input_brake;
    float input_clutch;

    float effective_steering;
    float effective_throttle;
    float effective_brake;
    float effective_clutch;

    float cruise_control_speed;  ///< m/s, 0 when off

    float fuel;
    float fuel_capacity;
    float fuel_average_consumption;
    float fuel_range;
    float adblue;
    float adblue_capacity;

    float brake_air_pressure;
    float brake_temperature;
    float oil_pressure;
    float oil_temperature;
    float water_temperature;
    float battery_voltage;

    float wear_engine;
    float wear_transmission;
    float wear_cabin;
    float wear_chassis;
    float wear_wheels;

    float odometer;  ///< km, already scaled by the game

    /** Wheelbase-relevant geometry, from the truck configuration event. */
    float hook_position[3];  ///< trailer hook, vehicle frame, metres

    std::int32_t gear;            ///< effective; negative = reverse, 0 = neutral
    std::int32_t gear_displayed;  ///< what the dashboard shows
    std::int32_t forward_gear_count;
    std::int32_t reverse_gear_count;

    std::int32_t retarder_level;
    std::uint32_t retarder_step_count;

    std::uint8_t engine_enabled;
    std::uint8_t electric_enabled;
    std::uint8_t parking_brake;
    std::uint8_t motor_brake;

    std::uint8_t indicator_left;
    std::uint8_t indicator_right;
    std::uint8_t light_parking;
    std::uint8_t light_low_beam;

    std::uint8_t light_high_beam;
    std::uint8_t light_brake;
    std::uint8_t light_reverse;
    std::uint8_t light_beacon;

    std::uint8_t wipers;
    std::uint8_t cruise_control_active;
    /** Hazard lights switched on: a drive switches off what its hand-over
     *  switched on. Taken from what was padding, so nothing moves. */
    std::uint8_t hazard_warning;
    std::uint8_t _padding;

    WheelState wheels;

    char brand_id[kStringLength];
    char model_id[kStringLength];

    /** Explicit tail padding. The struct is 8-byte aligned because of
     *  Placement, and leaving the compiler to add this silently is exactly the
     *  kind of thing that differs between toolchains. */
    std::uint32_t _tail_padding;
};
static_assert(sizeof(TruckState) == 1008, "TruckState must be 1008 bytes");
static_assert(offsetof(TruckState, wheels) == 284, "TruckState::wheels moved");
static_assert(offsetof(TruckState, brand_id) == 876, "TruckState::brand_id moved");

/** One trailer in the combination. */
struct TrailerState {
    Placement placement;  ///< 40

    Vector3f linear_velocity;
    Vector3f angular_velocity;
    Vector3f linear_acceleration;
    Vector3f angular_acceleration;

    /**
     * Hook position in the *trailer's* frame. With the truck's hook position
     * and both placements, this is what makes the hitch angle computable --
     * the single most important quantity for reverse parking (Phase 8).
     */
    float hook_position[3];

    float cargo_damage;
    float wear_chassis;
    float wear_wheels;

    std::uint32_t axle_count;
    std::uint8_t connected;
    std::uint8_t _padding[3];

    WheelState wheels;

    char id[kStringLength];
    char cargo_accessory_id[kStringLength];
    std::uint32_t _tail_padding[2];  ///< explicit, see TruckState::_tail_padding
};
static_assert(sizeof(TrailerState) == 848, "TrailerState must be 848 bytes");
static_assert(offsetof(TrailerState, wheels) == 120, "TrailerState::wheels moved");
static_assert(offsetof(TrailerState, id) == 712, "TrailerState::id moved");

/** The active job, or zeroes when there is none. */
struct JobState {
    float cargo_mass;
    float cargo_unit_mass;
    std::uint32_t cargo_unit_count;
    std::uint32_t income;

    std::uint32_t delivery_time;      ///< in-game minutes, absolute
    std::uint32_t planned_distance;   ///< km
    std::uint8_t cargo_loaded;
    std::uint8_t special_job;
    std::uint8_t job_active;
    std::uint8_t _padding;
    std::uint32_t _padding2;

    char cargo_id[kStringLength];
    char cargo[kStringLength];
    char source_city[kStringLength];
    char source_company[kStringLength];
    char destination_city[kStringLength];
    char destination_company[kStringLength];
};
static_assert(sizeof(JobState) == 416, "JobState must be 416 bytes");

/** The game's own navigation output, used as a route oracle in Phase 4. */
struct NavigationState {
    float distance;     ///< metres to the next waypoint
    float time;         ///< seconds, at current speed
    float speed_limit;  ///< m/s, 0 when unknown
    float _padding;
};
static_assert(sizeof(NavigationState) == 16, "NavigationState must be 16 bytes");

// --------------------------------------------------------------------------
// Block
// --------------------------------------------------------------------------

/**
 * @brief The complete shared memory block.
 *
 * Read protocol:
 * @code
 *   do {
 *       s1 = header.sequence;            // acquire
 *       if (s1 & 1) continue;            // write in progress
 *       copy(block);
 *       s2 = header.sequence;            // acquire
 *   } while (s1 != s2);
 * @endcode
 */
struct TelemetryBlock {
    Header header;
    CommonState common;
    TruckState truck;
    TrailerState trailers[kMaxTrailers];
    JobState job;
    NavigationState navigation;
};

static_assert(sizeof(TelemetryBlock) == 80 + 16 + 1008 + 3 * 848 + 416 + 16,
              "TelemetryBlock size changed: bump kLayoutVersion and update "
              "core/bridge/ before relaxing this assertion");
static_assert(alignof(TelemetryBlock) == 8, "TelemetryBlock must be 8-byte aligned");
static_assert(sizeof(TelemetryBlock) == 4080, "TelemetryBlock must be 4080 bytes");
static_assert(offsetof(TelemetryBlock, header) == 0, "the header must come first");
static_assert(offsetof(TelemetryBlock, truck) == 96, "TelemetryBlock::truck moved");
static_assert(offsetof(TelemetryBlock, trailers) == 1104, "TelemetryBlock::trailers moved");

}  // namespace ets2ad

#endif  // ETS2AD_TELEMETRY_LAYOUT_H
