#include "loader/stages/file_path_provider.h"

#include <absl/cleanup/cleanup.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>

#ifndef _WIN32
#include <sys/epoll.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"

#include "utils/platform.h"
#include "utils/trace.h"

#ifdef _WIN32
// After the absl headers: windows.h defines an ERROR macro that would
// otherwise collide with absl::LogSeverity::kError.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef ERROR
#endif

namespace lczero {
namespace training {

namespace {

bool ShouldSkipName(std::string_view name) {
  return !name.empty() && name.front() == '.';
}

bool ShouldSkipPathEntry(const FilePathProvider::Path& path) {
  return ShouldSkipName(path.filename().string());
}

#ifdef _WIN32
// ReadDirectoryChangesW reports paths relative to the watched root, so a
// notification can name something several directories deep. inotify never
// watched a hidden directory in the first place, so skip a change if any
// component of the relative path is hidden, not just the leaf.
bool ShouldSkipRelativePath(const FilePathProvider::Path& relative) {
  for (const auto& component : relative) {
    if (ShouldSkipName(component.string())) return true;
  }
  return false;
}
#endif

}  // namespace

FilePathProvider::FilePathProvider(const FilePathProviderConfig& config)
    : SingleOutputStage<File>(config.output()),
      directory_(config.directory()),
      producer_(output_queue()->CreateProducer()),
      load_metric_updater_() {
  LOG(INFO) << "Initializing FilePathProvider for directory: "
            << config.directory();
#ifndef _WIN32
  inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
  CHECK_NE(inotify_fd_, -1)
      << "Failed to initialize inotify: " << strerror(errno);
#endif
}

FilePathProvider::~FilePathProvider() {
  LOG(INFO) << "FilePathProvider shutting down.";
  Stop();
#ifdef _WIN32
  if (directory_handle_ != nullptr) {
    // Cancel any outstanding overlapped read before releasing the buffer it
    // is writing into.
    ::CancelIoEx(directory_handle_, static_cast<OVERLAPPED*>(overlapped_));
    ::CloseHandle(directory_handle_);
    directory_handle_ = nullptr;
  }
  if (overlapped_ != nullptr) {
    auto* overlapped = static_cast<OVERLAPPED*>(overlapped_);
    if (overlapped->hEvent != nullptr) ::CloseHandle(overlapped->hEvent);
    delete overlapped;
    overlapped_ = nullptr;
  }
#else
  if (inotify_fd_ != -1) close(inotify_fd_);
#endif
  LOG(INFO) << "FilePathProvider shutdown complete.";
}

void FilePathProvider::SetInputs(absl::Span<QueueBase* const> inputs) {
  if (!inputs.empty()) {
    throw std::runtime_error(
        "FilePathProvider expects no inputs, but received " +
        std::to_string(inputs.size()));
  }
}

void FilePathProvider::Start() {
  LOG(INFO) << "Starting FilePathProvider monitoring thread.";
  thread_pool_.Enqueue(
      [this](std::stop_token stop_token) { Worker(stop_token); });
}

void FilePathProvider::Stop() {
  if (stop_source_.stop_requested()) return;
  LOG(INFO) << "Stopping FilePathProvider.";
  LOG(INFO) << "Stopping all watches...";
#ifdef _WIN32
  // A single recursive handle covers the whole tree; cancelling the pending
  // read is what actually stops it. The handle itself is closed in the
  // destructor, after the worker thread has been joined.
  if (directory_handle_ != nullptr && read_pending_) {
    ::CancelIoEx(directory_handle_, static_cast<OVERLAPPED*>(overlapped_));
  }
#else
  for (const auto& [wd, path] : watch_descriptors_) {
    inotify_rm_watch(inotify_fd_, wd);
  }
  watch_descriptors_.clear();
#endif
  stop_source_.request_stop();
  thread_pool_.Shutdown();
  producer_.Close();
}

StageMetricProto FilePathProvider::FlushMetrics() {
  StageMetricProto stage_metric;
  auto load_metrics = load_metric_updater_.FlushMetrics();
  load_metrics.set_name("load");
  *stage_metric.add_load_metrics() = std::move(load_metrics);
  *stage_metric.add_queue_metrics() =
      MetricsFromQueue("output", *output_queue());
  return stage_metric;
}

void FilePathProvider::AddDirectory(const Path& directory,
                                    std::stop_token stop_token) {
  ScanDirectoryWithWatch(directory, stop_token);

#ifdef _WIN32
  LOG(INFO) << "FilePathProvider registered " << directory
            << "; watching the subtree through one recursive handle.";
#else
  LOG(INFO) << "FilePathProvider registered " << directory
            << "; active watch descriptors: " << watch_descriptors_.size();
#endif

  // Signal that initial scan is complete
  LOG(INFO) << "FilePathProvider initial scan complete";
  producer_.Put(
      {{.filepath = Path{}, .message_type = MessageType::kInitialScanComplete}},
      stop_token);
}

void FilePathProvider::ScanDirectoryWithWatch(const Path& directory,
                                              std::stop_token stop_token) {
  // Step 1: Set up watch first.
#ifdef _WIN32
  // Nothing to do: StartWatch() opened one ReadDirectoryChangesW handle over
  // the whole subtree before this scan began, so every subdirectory reached
  // here is already covered.
#else
  int wd = inotify_add_watch(inotify_fd_, directory.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << directory << ": "
                   << strerror(errno);
  watch_descriptors_[wd] = directory;
#endif

  // Step 2: Scan directory non-recursively, remembering files and subdirs
  std::vector<Path> files;
  std::vector<Path> subdirectories;
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(directory, ec);
  CHECK(!ec) << "Failed to iterate directory " << directory << ": "
             << ec.message();

  for (const auto& entry : iterator) {
    const Path entry_path = entry.path();
    if (ShouldSkipPathEntry(entry_path)) continue;

    if (entry.is_regular_file(ec) && !ec) {
      files.push_back(entry_path);
    } else if (entry.is_directory(ec) && !ec) {
      subdirectories.push_back(entry_path);
    }
  }

  const size_t initial_file_count = files.size();
  const size_t subdirectory_count = subdirectories.size();
  LOG(INFO) << "FilePathProvider scanned " << directory << " discovering "
            << initial_file_count << " file(s) and " << subdirectory_count
            << " subdirectory(ies) before watch reconciliation.";

  // Send notifications for discovered files
  constexpr size_t kBatchSize = 10000;
  std::vector<File> batch;
  batch.reserve(kBatchSize);

  auto flush_batch = [&]() {
    if (batch.empty()) return;
    producer_.Put(batch, stop_token);
    batch.clear();
  };

  for (const auto& filepath : files) {
#ifdef _WIN32
    // Remember what the scan emitted so the reconciliation pass in Worker()
    // does not emit it a second time if the watcher also reported it.
    scanned_.insert(filepath.string());
#endif
    batch.push_back(
        {.filepath = filepath.string(), .message_type = MessageType::kFile});
    if (batch.size() >= kBatchSize) flush_batch();
  }

  if (initial_file_count > 0) {
    LOG(INFO) << "FilePathProvider enqueued " << initial_file_count
              << " file(s) from initial scan of " << directory;
  }

  // Step 3: Read from watch descriptor, skipping already discovered files
#ifdef _WIN32
  // Handled once for the whole tree by the reconciliation pass in Worker():
  // ReadDirectoryChangesW delivers a single ordered stream for every
  // subdirectory, so there is nothing per-directory to drain here.
#else
  ProcessWatchEventsForNewItems(files);
#endif

  // Step 4: Clean the files vector to save memory
  files.clear();

  // Step 5: Recursively call for subdirectories
  for (const auto& subdir : subdirectories) {
    if (stop_token.stop_requested()) return;
    ScanDirectoryWithWatch(subdir, stop_token);
  }

  // Flush any remaining files
  flush_batch();
}

#ifndef _WIN32

void FilePathProvider::ProcessWatchEventsForNewItems(
    const std::vector<Path>& known_files) {
  // Create a set for fast lookup of already discovered files
  absl::flat_hash_set<std::string> known_file_set;
  for (const auto& file : known_files) {
    known_file_set.insert(file.string());
  }

  // Process any events that may have occurred during scanning
  std::array<char, 4096> buffer;
  std::vector<File> new_files;

  while (true) {
    ssize_t length = read(inotify_fd_, buffer.data(), buffer.size());
    if (length <= 0) break;  // No more events to process

    ssize_t offset = 0;
    while (offset < length) {
      const struct inotify_event* event =
          reinterpret_cast<const struct inotify_event*>(buffer.data() + offset);

      const bool skip_entry = event->len > 0 && ShouldSkipName(event->name);

      // Only process file creation/write events, skip already known files
      if ((event->mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0 &&
          event->len > 0 && !skip_entry) {
        const Path directory(watch_descriptors_.at(event->wd));
        Path filepath = directory / event->name;
        std::string filepath_string = filepath.string();

        // Only add if we haven't seen this file before
        if (!known_file_set.contains(filepath_string)) {
          new_files.push_back({.filepath = std::move(filepath_string),
                               .message_type = MessageType::kFile});
        }
      }

      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  // Send notifications for any new files discovered through watch events
  if (!new_files.empty()) {
    LOG(INFO) << "FilePathProvider observed " << new_files.size()
              << " new file(s) while reconciling race events.";
    producer_.Put(new_files);
  }
}

void FilePathProvider::AddWatchRecursive(const Path& path) {
  // Add watch for current directory
  int wd = inotify_add_watch(inotify_fd_, path.c_str(),
                             IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE |
                                 IN_DELETE | IN_DELETE_SELF | IN_MOVE);
  CHECK_NE(wd, -1) << "Failed to add inotify watch for " << path << ": "
                   << strerror(errno);
  watch_descriptors_[wd] = path;

  // Recursively add watches for subdirectories
  std::error_code ec;
  auto iterator = std::filesystem::directory_iterator(path, ec);
  CHECK(!ec) << "Failed to iterate directory " << path << ": " << ec.message();

  for (const auto& entry : iterator) {
    const Path entry_path = entry.path();
    if (ShouldSkipPathEntry(entry_path)) continue;
    if (!entry.is_directory(ec) || ec) continue;
    AddWatchRecursive(entry_path);
  }
}

void FilePathProvider::RemoveWatchRecursive(const Path& base) {
  absl::erase_if(watch_descriptors_, [&](const auto& pair) {
    const auto& [wd, path] = pair;
    const auto mismatch_iter = absl::c_mismatch(base, path).first;
    // If path is not a subdirectory (or equal) of base, skip.
    if (mismatch_iter != base.end()) return false;
    inotify_rm_watch(inotify_fd_, wd);
    return true;
  });
}

#endif  // !_WIN32

#ifdef _WIN32

void FilePathProvider::StartWatch() {
  // FILE_FLAG_BACKUP_SEMANTICS is required to obtain a handle to a directory
  // at all; FILE_FLAG_OVERLAPPED lets the worker poll for changes without
  // blocking, which is what keeps Stop() responsive.
  HANDLE handle = ::CreateFileW(
      directory_.c_str(), FILE_LIST_DIRECTORY,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
      nullptr);
  CHECK(handle != INVALID_HANDLE_VALUE)
      << "Failed to open directory for watching " << directory_ << ": "
      << PositionedFile::LastErrorMessage();
  directory_handle_ = handle;

  auto* overlapped = new OVERLAPPED{};
  overlapped->hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  CHECK(overlapped->hEvent != nullptr)
      << "Failed to create watch event: "
      << PositionedFile::LastErrorMessage();
  overlapped_ = overlapped;

  // 64 KiB is the largest buffer ReadDirectoryChangesW fills reliably.
  notify_buffer_.assign(64 * 1024, 0);
  QueueRead();
}

bool FilePathProvider::QueueRead() {
  auto* overlapped = static_cast<OVERLAPPED*>(overlapped_);
  ::ResetEvent(overlapped->hEvent);
  DWORD returned = 0;
  const BOOL ok = ::ReadDirectoryChangesW(
      directory_handle_, notify_buffer_.data(),
      static_cast<DWORD>(notify_buffer_.size()),
      TRUE,  // Recursive: one handle covers the entire tree, so unlike the
             // inotify path there are no per-directory watches to maintain.
      FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME |
          FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_SIZE,
      &returned, overlapped, nullptr);
  read_pending_ = (ok != FALSE);
  if (!read_pending_) {
    LOG(WARNING) << "ReadDirectoryChangesW failed for " << directory_ << ": "
                 << PositionedFile::LastErrorMessage();
  }
  return read_pending_;
}

void FilePathProvider::ProcessDirectoryChanges(std::stop_token stop_token) {
  LCTRACE_FUNCTION_SCOPE;
  if (!read_pending_ && !QueueRead()) return;

  auto* overlapped = static_cast<OVERLAPPED*>(overlapped_);
  DWORD bytes = 0;
  if (!::GetOverlappedResult(directory_handle_, overlapped, &bytes,
                             FALSE /* do not wait */)) {
    const DWORD code = ::GetLastError();
    // Nothing has changed yet; this is the common case.
    if (code == ERROR_IO_INCOMPLETE) return;
    read_pending_ = false;
    if (code != ERROR_OPERATION_ABORTED) {
      LOG(WARNING) << "Directory watch failed for " << directory_ << ": "
                   << PositionedFile::LastErrorMessage();
    }
    return;
  }
  read_pending_ = false;

  if (bytes == 0) {
    // The kernel's change buffer overran and events were dropped. Requeue
    // first so the window while rescanning is still covered.
    LOG(WARNING) << "Directory change buffer overflowed for " << directory_
                 << "; rescanning to resynchronize.";
    QueueRead();
    ScanDirectoryWithWatch(directory_, stop_token);
    return;
  }

  size_t offset = 0;
  while (offset + sizeof(FILE_NOTIFY_INFORMATION) <= notify_buffer_.size()) {
    const auto* info = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(
        notify_buffer_.data() + offset);
    const std::wstring relative(info->FileName,
                                info->FileNameLength / sizeof(wchar_t));
    const Path relative_path(relative);
    const Path full = directory_ / relative_path;

    if (!ShouldSkipRelativePath(relative_path)) {
      switch (info->Action) {
        case FILE_ACTION_ADDED:
        case FILE_ACTION_MODIFIED:
        case FILE_ACTION_RENAMED_NEW_NAME: {
          std::error_code ec;
          if (std::filesystem::is_directory(full, ec) && !ec) {
            // The recursive watch already covers a new subdirectory, but
            // files may have landed in it before this notification arrived.
            ScanDirectoryWithWatch(full, stop_token);
          } else if (!scanned_.contains(full.string()) &&
                     std::find(pending_.begin(), pending_.end(), full) ==
                         pending_.end()) {
            // Not emitted yet: Windows has no "writer closed the file"
            // notification, so hold it until FlushCompletedFiles can prove
            // it is complete.
            pending_.push_back(full);
          }
          break;
        }
        case FILE_ACTION_REMOVED:
        case FILE_ACTION_RENAMED_OLD_NAME:
          std::erase(pending_, full);
          break;
        default:
          break;
      }
    }

    if (info->NextEntryOffset == 0) break;
    offset += info->NextEntryOffset;
  }
  QueueRead();
}

void FilePathProvider::FlushCompletedFiles(Queue<File>::Producer& producer,
                                           std::stop_token stop_token) {
  if (pending_.empty()) return;
  std::vector<File> ready;
  std::vector<Path> still_pending;
  still_pending.reserve(pending_.size());

  for (const auto& path : pending_) {
    std::error_code ec;
    // Dropped between notification and now; nothing to emit.
    if (!std::filesystem::is_regular_file(path, ec) || ec) continue;
    if (!IsFileClosedByWriters(path)) {
      still_pending.push_back(path);
      continue;
    }
    ready.push_back({.filepath = path, .message_type = MessageType::kFile});
  }

  pending_.swap(still_pending);
  if (!ready.empty()) producer.Put(ready, stop_token);
}

void FilePathProvider::Worker(std::stop_token stop_token) {
  // Open the watch before scanning, so anything created during the scan
  // still arrives as a change event instead of being missed outright.
  StartWatch();
  AddDirectory(directory_, stop_token);

  // Reconcile the race window between opening the watch and finishing the
  // scan, then drop the scan's bookkeeping. This is the Win32 counterpart of
  // ProcessWatchEventsForNewItems.
  ProcessDirectoryChanges(stop_token);
  scanned_.clear();

  while (!stop_token.stop_requested()) {
    {
      LoadMetricPauser pauser(load_metric_updater_);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (stop_token.stop_requested()) {
        pauser.DoNotResume();
        break;
      }
    }
    ProcessDirectoryChanges(stop_token);
    FlushCompletedFiles(producer_, stop_token);
  }
}

#else

void FilePathProvider::Worker(std::stop_token stop_token) {
  // Perform directory scanning in background thread
  AddDirectory(directory_, stop_token);

  int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  CHECK_NE(epoll_fd, -1) << "Failed to create epoll fd: " << strerror(errno);
  absl::Cleanup epoll_cleanup([epoll_fd]() { close(epoll_fd); });

  struct epoll_event event;
  event.events = EPOLLIN;
  event.data.fd = inotify_fd_;
  CHECK_EQ(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd_, &event), 0)
      << "Failed to add inotify fd to epoll: " << strerror(errno);

  while (!stop_token.stop_requested()) {
    {
      LoadMetricPauser pauser(load_metric_updater_);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (stop_token.stop_requested()) {
        pauser.DoNotResume();
        break;
      }
    }

    struct epoll_event event;
    int nfds = epoll_wait(epoll_fd, &event, 1, 0);  // Non-blocking check
    CHECK_NE(nfds, -1) << "epoll_wait failed: " << strerror(errno);
    if (nfds == 0) continue;  // No events.

    do {
      assert(nfds == 1 && event.data.fd == inotify_fd_);
      ProcessInotifyEvents(producer_, stop_token);
      nfds = epoll_wait(epoll_fd, &event, 1, 0);
    } while (nfds > 0);
  }
}

void FilePathProvider::ProcessInotifyEvents(Queue<File>::Producer& producer,
                                            std::stop_token stop_token) {
  LCTRACE_FUNCTION_SCOPE;
  constexpr size_t kNotifyBatchSize = 10000;
  std::vector<File> files;
  std::array<char, 4096> buffer;

  auto flush_batch = [&]() {
    if (files.empty()) return;
    producer.Put(files, stop_token);
    files.clear();
  };

  while (true) {
    ssize_t length = read(inotify_fd_, buffer.data(), buffer.size());
    if (length <= 0) break;  // No more events to process

    ssize_t offset = 0;
    while (offset < length) {
      const struct inotify_event* event =
          reinterpret_cast<const struct inotify_event*>(buffer.data() + offset);
      auto file = ProcessInotifyEvent(*event, stop_token);
      if (file) files.push_back(*file);
      if (files.size() >= kNotifyBatchSize) flush_batch();
      offset += sizeof(struct inotify_event) + event->len;
    }
  }

  flush_batch();  // Flush any remaining files in the batch
}

auto FilePathProvider::ProcessInotifyEvent(const struct inotify_event& event,
                                           std::stop_token stop_token)
    -> std::optional<File> {
  if (event.mask & IN_IGNORED) return std::nullopt;

  const Path directory(watch_descriptors_.at(event.wd));
  const bool has_name = event.len > 0 && event.name[0] != '\0';
  const bool skip_entry = has_name && ShouldSkipName(event.name);
  const Path filepath = has_name ? directory / event.name : directory;

  // Handle different event types
  if ((event.mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0 && has_name &&
      !skip_entry) {
    // File finished writing or moved into directory
    return File{.filepath = filepath, .message_type = MessageType::kFile};
  }

  constexpr uint32_t kDirCreateMask = IN_CREATE | IN_ISDIR;
  constexpr uint32_t kDirDeleteMask = IN_DELETE | IN_ISDIR;
  if ((event.mask & kDirCreateMask) == kDirCreateMask) {
    if (!has_name || skip_entry) return std::nullopt;
    ScanDirectoryWithWatch(filepath, stop_token);
  } else if ((event.mask & kDirDeleteMask) == kDirDeleteMask) {
    if (!has_name || skip_entry) return std::nullopt;
    // Directory deleted - remove all watches for it and subdirectories
    RemoveWatchRecursive(filepath);
  } else if (event.mask & IN_DELETE_SELF) {
    RemoveWatchRecursive(directory);
  }

  return std::nullopt;
}

#endif  // _WIN32

}  // namespace training
}  // namespace lczero
