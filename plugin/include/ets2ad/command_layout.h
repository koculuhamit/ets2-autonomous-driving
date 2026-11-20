/**
 * @file command_layout.h
 * @brief Layout of the command block, written by the stack and read by the game.
 *
 * The mirror image of `telemetry_layout.h`: same conventions, opposite
 * direction. The stack writes control setpoints here, and the plugin's input
 * callback hands them to the game once per frame.
 *
 * The seqlock runs the other way round -- the Python side is the writer and
 * the plugin is the reader -- but the protocol is identical, and so are the
 * rules about explicit padding and versioning.
 *
 * Two safety properties are built into the format rather than layered on top:
 *
 * **Absence means neutral.** A block that was never written is all zeroes, so
 * its magic does not match and the plugin ignores it. There is no state in
 * which a missing or malformed commander produces anything but zero output.
 *
 * **Silence means neutral.** `command_count` must keep advancing. The plugin
 * watches it and releases every control if it stands still, so a stack that
 * crashes mid-corner does not leave the throttle where it was. The game has no
 * idea our process died; this is the only thing that notices.
 */
#ifndef ETS2AD_COMMAND_LAYOUT_H
#define ETS2AD_COMMAND_LAYOUT_H

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ets2ad {

/** "ET2C" little-endian. */
inline constexpr std::uint32_t kCommandMagic = 0x43325445u;

/** Bumped whenever any field below moves. */
inline constexpr std::uint32_t kCommandLayoutVersion = 1u;

/** `/dev/shm/ets2ad_command` on Linux; a `Local\` mapping on Windows. */
inline constexpr char kCommandMemoryName[] = "ets2ad_command";

/**
 * @brief How long the command counter may stand still before controls are released.
 *
 * Measured on the plugin's own steady clock. An earlier version counted input
 * callbacks instead, on the reasoning that the two processes share no clock --
 * which is true for comparing *timestamps* and irrelevant here, because this
 * measures elapsed time inside one process.
 *
 * The frame count was wrong in a way worth remembering: the input callback runs
 * once per rendered frame, so its duration moved with the frame rate. Twelve
 * frames was 200 ms at 60 FPS, 50 ms at 240, and 400 ms at 30 -- measured at
 * 132 ms on a machine running uncapped. A safety timeout that shortens when the
 * GPU is idle and lengthens exactly when the game is struggling is the opposite
 * of what it should do.
 *
 * 200 ms is long enough to ride out a scheduling hiccup in the Python process
 * and short enough that a dead commander cannot hold the throttle open through
 * a corner.
 */
inline constexpr double kCommandWatchdogSeconds = 0.2;

/**
 * @brief How long the command counter may stand still before the plugin
 * stops the truck itself (src/safety_stop.h); until then the steering is
 * held where it was.
 *
 * The stack's process stood still for up to 0.96 s while it was paged out
 * on Windows, and came back; a process killed for memory never did, and the
 * truck ran on driverless at 80 km/h. tools/follow.py ends its drive when its
 * own loop stood still this long (`SAFETY_STOP_S`): the truck is being
 * stopped already.
 */
inline constexpr double kSafetyStopSeconds = 1.0;

/** Continuous inputs, in the order they appear in `CommandBlock::axes`. */
enum CommandAxis : std::uint32_t {
    kAxisSteering = 0,  ///< [-1, 1], negative is left
    kAxisThrottle = 1,  ///< [0, 1]
    kAxisBrake = 2,     ///< [0, 1]
    kAxisClutch = 3,    ///< [0, 1]
    kAxisCount = 4,
};

/** Discrete inputs, as bit positions in `CommandBlock::buttons`. */
enum CommandButton : std::uint32_t {
    kButtonParkingBrake = 0,
    kButtonEngine = 1,
    kButtonGearUp = 2,
    kButtonGearDown = 3,
    kButtonIndicatorLeft = 4,
    kButtonIndicatorRight = 5,
    kButtonHorn = 6,
    kButtonLights = 7,

    /** Engine brake. The service brake has a dead zone below roughly 0.055 and
     *  then jumps straight to 1.8 m/s^2, so there is no gentle deceleration to
     *  be had from it; this truck has no retarder either. Engine braking is
     *  what is left. See docs/results/sysid.md. */
    kButtonMotorBrake = 8,
    kButtonEngineBrakeUp = 9,
    kButtonEngineBrakeDown = 10,

    /** Hazard lights: switched on, with two blasts of the horn, when the
     *  stack hands the truck back to the driver. */
    kButtonHazard = 11,

    /** The game's "activate" (Enter): held at a gas pump, it fills the tank. */
    kButtonActivate = 12,

    /** The game's "attach" (T): hitches the trailer the fifth wheel is
     *  under, and unhitches the one hitched. */
    kButtonAttach = 13,

    /** The retarder lever, a step up or down per press: the gentle
     *  deceleration between coasting and the service brake's floor, on the
     *  trucks that have one (`retarder_step_count`). */
    kButtonRetarderUp = 14,
    kButtonRetarderDown = 15,

    kButtonCount = 16,
};

/* Adding buttons does not move any field: `CommandBlock::buttons` is a 32-bit
 * mask with room for 32 of them, so the layout version stays where it is. */

/** Bits in `CommandBlock::flags`. */
enum CommandFlag : std::uint32_t {
    /**
     * The stack is driving. When clear, the plugin emits neutral regardless of
     * what the axes say, so disengaging is one flag rather than a promise that
     * every writer remembered to zero its outputs.
     */
    kCommandEngaged = 1u << 0,
};

/**
 * @brief The command block.
 *
 * Read protocol in the plugin is the same seqlock dance as telemetry, with the
 * roles swapped: sample the counter, copy, sample again, retry on a change.
 */
struct CommandBlock {
    std::uint32_t magic;           ///< kCommandMagic
    std::uint32_t layout_version;  ///< kCommandLayoutVersion
    std::uint32_t block_size;      ///< sizeof(CommandBlock)

    /** Seqlock counter, odd while the writer is mid-update. Plain `uint32_t`
     *  for the same reason as in the telemetry header: this is a wire format
     *  and has to stay trivially copyable. */
    std::uint32_t sequence;

    /** Incremented on every publish. The plugin's watchdog watches this and
     *  not a timestamp, because the two processes share no clock. */
    std::uint64_t command_count;

    /** Writer's steady-clock seconds at publish. Diagnostic only: useful for
     *  measuring command latency against telemetry, never for control. */
    double command_time;

    std::uint32_t flags;    ///< CommandFlag bits
    std::uint32_t buttons;  ///< CommandButton bit positions

    float axes[kAxisCount];

    /** Two words, not one: `_reserved` is 8-byte aligned, and a single
     *  word here would leave four bytes of padding the compiler inserts
     *  rather than we declare. */
    std::uint32_t _padding[2];

    std::uint64_t _reserved[4];
};

static_assert(sizeof(CommandBlock) == 96, "CommandBlock must be 96 bytes");
static_assert(alignof(CommandBlock) == 8, "CommandBlock must be 8-byte aligned");
static_assert(offsetof(CommandBlock, magic) == 0, "magic must come first");
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free,
              "the seqlock requires a lock-free 32-bit atomic");

}  // namespace ets2ad

#endif  // ETS2AD_COMMAND_LAYOUT_H
