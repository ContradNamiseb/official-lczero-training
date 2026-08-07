#include "utils/platform.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace lczero {

namespace {
#ifdef _WIN32
// A single ReadFile cannot be asked for more than a DWORD of bytes; cap well
// below that so the arithmetic stays obviously safe.
constexpr size_t kMaxSingleRead = 1u << 30;
#endif
}  // namespace

PositionedFile::PositionedFile(const std::filesystem::path& path) {
#ifdef _WIN32
  // Share everything: the writer dropping new chunks into the training
  // directory must not be blocked by the loader holding files open.
  // Inheritance is off by default, matching the POSIX side's O_CLOEXEC.
  HANDLE handle = ::CreateFileW(
      path.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  handle_ = (handle == INVALID_HANDLE_VALUE) ? nullptr : handle;
#else
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

PositionedFile::~PositionedFile() { Close(); }

PositionedFile::PositionedFile(PositionedFile&& other) noexcept
    : handle_(other.handle_), fd_(other.fd_) {
  other.handle_ = nullptr;
  other.fd_ = -1;
}

PositionedFile& PositionedFile::operator=(PositionedFile&& other) noexcept {
  if (this != &other) {
    Close();
    handle_ = other.handle_;
    fd_ = other.fd_;
    other.handle_ = nullptr;
    other.fd_ = -1;
  }
  return *this;
}

bool PositionedFile::is_open() const {
#ifdef _WIN32
  return handle_ != nullptr;
#else
  return fd_ >= 0;
#endif
}

bool PositionedFile::ReadExact(int64_t offset, void* buffer,
                               size_t size) const {
  if (!is_open()) return false;
  char* out = static_cast<char*>(buffer);
  size_t done = 0;
  while (done < size) {
#ifdef _WIN32
    // Passing an OVERLAPPED to a synchronous handle performs a positioned
    // read and completes inline -- this is the pread() equivalent, and it
    // leaves the handle's file pointer untouched.
    OVERLAPPED overlapped = {};
    const uint64_t position = static_cast<uint64_t>(offset) + done;
    overlapped.Offset = static_cast<DWORD>(position & 0xFFFFFFFFull);
    overlapped.OffsetHigh = static_cast<DWORD>(position >> 32);
    const DWORD want =
        static_cast<DWORD>(std::min<size_t>(size - done, kMaxSingleRead));
    DWORD got = 0;
    if (!::ReadFile(handle_, out + done, want, &got, &overlapped)) return false;
    if (got == 0) return false;  // End of file before `size` bytes.
    done += got;
#else
    const ssize_t got =
        ::pread(fd_, out + done, size - done, offset + static_cast<int64_t>(done));
    if (got <= 0) return false;
    done += static_cast<size_t>(got);
#endif
  }
  return true;
}

bool PositionedFile::Close() {
  bool ok = true;
#ifdef _WIN32
  if (handle_ != nullptr) ok = ::CloseHandle(handle_) != 0;
  handle_ = nullptr;
#else
  if (fd_ >= 0) ok = ::close(fd_) == 0;
  fd_ = -1;
#endif
  return ok;
}

bool IsFileClosedByWriters(const std::filesystem::path& path) {
#ifdef _WIN32
  // Share reads so two loader threads testing the same file do not report
  // each other as writers, but deny write sharing: if anything still holds
  // the file open for writing this fails with ERROR_SHARING_VIOLATION.
  HANDLE handle =
      ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return false;
  ::CloseHandle(handle);
  return true;
#else
  (void)path;
  return true;
#endif
}

std::string PositionedFile::LastErrorMessage() {
#ifdef _WIN32
  const DWORD code = ::GetLastError();
  char* text = nullptr;
  const DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<char*>(&text), 0, nullptr);
  std::string message =
      length ? std::string(text, length) : std::string("unknown error");
  if (text) ::LocalFree(text);
  while (!message.empty() &&
         (message.back() == '\n' || message.back() == '\r')) {
    message.pop_back();
  }
  return message + " (" + std::to_string(code) + ")";
#else
  return std::strerror(errno);
#endif
}

}  // namespace lczero
