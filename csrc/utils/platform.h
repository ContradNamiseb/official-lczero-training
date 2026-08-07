#pragma once

// Portability shims for building the data loader natively on Windows.
//
// The loader was written against POSIX. Only a handful of things are
// genuinely absent under MSVC; they are collected here rather than scattered
// through the tree as #ifdefs. See docs/directml_training_port.md.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#if defined(_MSC_VER) && !defined(_SSIZE_T_DEFINED)
#define _SSIZE_T_DEFINED
// MSVC does not provide the POSIX ssize_t. It is the signed counterpart of
// size_t, which is exactly std::ptrdiff_t on every platform this targets.
// Declared at global scope, as POSIX does, so existing uses compile
// unchanged.
using ssize_t = std::ptrdiff_t;
#endif

namespace lczero {

// A read-only file supporting positioned reads.
//
// Deliberately not seek-then-read: several chunk-loading threads read from
// the same archive concurrently (see chunk_loading_threads in the loader
// config), so a shared file pointer would race. POSIX pread() and Win32
// ReadFile() with an OVERLAPPED offset both read at an explicit offset
// without touching any shared position, which is what makes that safe.
//
// Offsets are int64_t rather than off_t: MSVC's off_t is a 32-bit long,
// which would silently break archives past 2 GiB.
class PositionedFile {
 public:
  PositionedFile() = default;
  explicit PositionedFile(const std::filesystem::path& path);
  ~PositionedFile();

  PositionedFile(const PositionedFile&) = delete;
  PositionedFile& operator=(const PositionedFile&) = delete;
  PositionedFile(PositionedFile&& other) noexcept;
  PositionedFile& operator=(PositionedFile&& other) noexcept;

  bool is_open() const;

  // Reads exactly `size` bytes starting at `offset`. Returns false on a
  // short read, end of file, or error.
  bool ReadExact(int64_t offset, void* buffer, size_t size) const;

  // Returns false if the handle was open and failed to close cleanly.
  bool Close();

  // Human-readable description of the last failure on this thread.
  static std::string LastErrorMessage();

 private:
  // void* rather than HANDLE so this header never pulls in windows.h, whose
  // macros (min/max, ERROR, ...) collide with abseil and the STL.
  void* handle_ = nullptr;
  int fd_ = -1;
};

// True once no other handle to `path` is open for writing.
//
// This exists because Windows has no equivalent of inotify's IN_CLOSE_WRITE.
// ReadDirectoryChangesW reports that a file was added or modified, never that
// the process writing it has finished -- so acting on those notifications
// directly would hand the loader a half-written chunk. Opening with no share
// mode succeeds only when every other handle is gone, which is the standard
// way to ask "is the writer done?".
//
// Always true on POSIX, where IN_CLOSE_WRITE answers the question directly
// and this is never called.
bool IsFileClosedByWriters(const std::filesystem::path& path);

}  // namespace lczero
