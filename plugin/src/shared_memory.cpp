#include "shared_memory.h"

#include <cstring>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace ets2ad {

SharedMemory::~SharedMemory() { destroy(); }

#ifdef _WIN32

bool SharedMemory::create(const std::string &name, std::size_t size) {
    destroy();

    // "Local\" keeps the object in the caller's session. A "Global\" name
    // would need SeCreateGlobalPrivilege, i.e. the game running elevated,
    // which is not something to ask of a user.
    const std::string object_name = "Local\\" + name;

    mapping_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                  static_cast<DWORD>(size), object_name.c_str());
    if (mapping_ == nullptr) {
        error_ = "CreateFileMapping failed, error " + std::to_string(GetLastError());
        return false;
    }

    data_ = MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (data_ == nullptr) {
        error_ = "MapViewOfFile failed, error " + std::to_string(GetLastError());
        CloseHandle(mapping_);
        mapping_ = nullptr;
        return false;
    }

    std::memset(data_, 0, size);
    size_ = size;
    name_ = name;
    error_.clear();
    return true;
}

void SharedMemory::destroy() {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_ != nullptr) {
        CloseHandle(mapping_);
        mapping_ = nullptr;
    }
    size_ = 0;
}

#else

bool SharedMemory::create(const std::string &name, std::size_t size) {
    destroy();

    const std::string object_name = "/" + name;

    // Unlink any leftover from a crashed run before creating. Reusing an
    // existing object would inherit its size, and a stale one from an older
    // layout version is exactly the case that must not silently half-work.
    shm_unlink(object_name.c_str());

    fd_ = shm_open(object_name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd_ < 0) {
        error_ = "shm_open(" + object_name + ") failed: " + std::strerror(errno);
        return false;
    }

    if (ftruncate(fd_, static_cast<off_t>(size)) != 0) {
        error_ = "ftruncate failed: " + std::string(std::strerror(errno));
        close(fd_);
        fd_ = -1;
        shm_unlink(object_name.c_str());
        return false;
    }

    data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        error_ = "mmap failed: " + std::string(std::strerror(errno));
        data_ = nullptr;
        close(fd_);
        fd_ = -1;
        shm_unlink(object_name.c_str());
        return false;
    }

    std::memset(data_, 0, size);
    size_ = size;
    name_ = name;
    error_.clear();
    return true;
}

void SharedMemory::destroy() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    if (!name_.empty()) {
        // Readers keep working until they unmap; the name goes away now so a
        // reader started later cannot attach to a block nobody is writing.
        shm_unlink(("/" + name_).c_str());
        name_.clear();
    }
    size_ = 0;
}

#endif

}  // namespace ets2ad
