/**
 * @file shared_memory.h
 * @brief Creating and mapping the telemetry block, on either platform.
 *
 * The only platform-specific code in the plugin. Everything else -- the
 * layout, the seqlock, the channel bindings -- is shared, which is what makes
 * the two builds behave identically rather than merely compile.
 */
#ifndef ETS2AD_SHARED_MEMORY_H
#define ETS2AD_SHARED_MEMORY_H

#include <cstddef>
#include <string>

namespace ets2ad {

/**
 * @brief An anonymous, named block of memory readable by other processes.
 *
 * The plugin is the sole writer and creates the block; readers open it by
 * name. Zero-initialised on creation, so a reader that connects before the
 * first publish sees a zero magic rather than stale contents.
 */
class SharedMemory {
public:
    SharedMemory() = default;
    ~SharedMemory();

    SharedMemory(const SharedMemory &) = delete;
    SharedMemory &operator=(const SharedMemory &) = delete;

    /**
     * @brief Create (or replace) the named block and map it.
     *
     * @param name Object name, without a leading slash.
     * @param size Size in bytes.
     * @return true on success. On failure, `error()` explains why.
     */
    bool create(const std::string &name, std::size_t size);

    /** Unmap and release. Safe to call when not open. */
    void destroy();

    void *data() const { return data_; }
    std::size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

    /** Human-readable reason the last `create()` failed. */
    const std::string &error() const { return error_; }

private:
    void *data_ = nullptr;
    std::size_t size_ = 0;
    std::string name_;
    std::string error_;

#ifdef _WIN32
    void *mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace ets2ad

#endif  // ETS2AD_SHARED_MEMORY_H
