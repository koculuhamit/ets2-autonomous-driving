# SCS telemetry plugin -- one source tree, two platforms.
#
# The Linux .so and the Windows .dll are built from identical sources with the
# same layout header. That is not a convenience: the shared memory block is an
# ABI, and the only way to know GCC and MSVC agree on it is to build both and
# compare what `layout_dump` prints.

cmake_minimum_required(VERSION 3.16)
project(ets2ad_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)            # std::atomic_ref for the seqlock
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The C++ runtime goes inside the DLL (/MT): a player's machine needs no
# Visual C++ Redistributable, and no Build Tools, for the game to load it.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_library(scs_sdk INTERFACE)
target_include_directories(scs_sdk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/scs_sdk/include")

add_library(ets2ad_layout INTERFACE)
target_include_directories(ets2ad_layout INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Warnings are errors throughout.
if(MSVC)
    set(ETS2AD_WARNINGS /W4 /WX)
    # C4996 flags getenv, fopen and strncpy as "unsafe" in favour of Microsoft
    # extensions. They are standard C++, used correctly -- strncpy with an
    # explicit terminator, fopen with a checked result -- and the alternatives
    # exist only on this platform. Rewriting portable code per-toolchain to
    # quiet a vendor preference would cost more than it buys.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    set(ETS2AD_WARNINGS -Wall -Wextra -Wpedantic -Werror)
endif()

# Padding is a wire-format concern, so -Wpadded is applied only where the wire
# format lives. Turning it on globally would flag every internal struct, where
# padding is the compiler's business and nobody's problem.
if(MSVC)
    set(ETS2AD_LAYOUT_WARNINGS ${ETS2AD_WARNINGS} /w14820)  # C4820: padding added
else()
    set(ETS2AD_LAYOUT_WARNINGS ${ETS2AD_WARNINGS} -Wpadded)
endif()

# --------------------------------------------------------------------------
# The plugin
# --------------------------------------------------------------------------

add_library(ets2ad_telemetry SHARED
    src/telemetry_plugin.cpp
    src/input_plugin.cpp
    src/shared_memory.cpp
    src/diagnostics.cpp
)
target_link_libraries(ets2ad_telemetry PRIVATE scs_sdk ets2ad_layout)
target_include_directories(ets2ad_telemetry PRIVATE src)
target_compile_options(ets2ad_telemetry PRIVATE ${ETS2AD_WARNINGS})

# The game looks for a plain file name in plugins/; no "lib" prefix on Linux.
set_target_properties(ets2ad_telemetry PROPERTIES
    PREFIX ""
    OUTPUT_NAME "ets2ad_telemetry"
)

if(WIN32)
    # MSVC exports nothing without being told: the SDK's SCSAPI macros carry no
    # __declspec(dllexport), so the DLL would load and expose no entry points.
    target_sources(ets2ad_telemetry PRIVATE exports.def)
endif()

if(UNIX AND NOT APPLE)
    # librt carries shm_open on glibc older than 2.34; harmless to link after.
    target_link_libraries(ets2ad_telemetry PRIVATE rt)

    # Only the two SDK entry points leave the library. Everything else stays
    # internal, so the game cannot collide with our symbols and vice versa.
    #
    # The version script does this on its own. Adding -fvisibility=hidden on
    # top hides the entry points before the linker script ever sees them, and
    # the result is a library the game loads and silently ignores.
    target_link_options(ets2ad_telemetry PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()

# --------------------------------------------------------------------------
# Layout description, consumed by the Python layout test
# --------------------------------------------------------------------------

# layout_dump includes the layout header and nothing else of ours, so it is
# also where the padding check runs.
add_executable(layout_dump tools/layout_dump.cpp)
target_link_libraries(layout_dump PRIVATE ets2ad_layout)
target_compile_options(layout_dump PRIVATE ${ETS2AD_LAYOUT_WARNINGS})

# What the plugin does when the stack goes quiet, tested without the game
# (tests/unit/test_plugin_safety_stop.py runs it).
add_executable(safety_test tools/safety_test.cpp)
target_link_libraries(safety_test PRIVATE ets2ad_layout)
target_include_directories(safety_test PRIVATE src)
target_compile_options(safety_test PRIVATE ${ETS2AD_WARNINGS})

# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------

install(TARGETS ets2ad_telemetry LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(TARGETS layout_dump RUNTIME DESTINATION .)
