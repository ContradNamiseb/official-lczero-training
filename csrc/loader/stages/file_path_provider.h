#pragma once

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>

#ifndef _WIN32
#include <sys/inotify.h>
#endif

#include <filesystem>
#include <functional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/metrics/printer.h"
#include "utils/metrics/statistics_metric.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Message types for FilePathProvider output.
enum class FilePathProviderMessageType {
  kFile,                // File discovered (initial scan or inotify)
  kInitialScanComplete  // Initial scan is complete (empty filepath)
};

// Output type for FilePathProvider.
struct FilePathProviderFile {
  std::filesystem::path filepath;
  FilePathProviderMessageType message_type;
};

// This class watches for new files in a directory (recursively) and notifies
// registered observers when new files are either closed after writing or
// renamed into.
// Uses background thread to monitor the directory.
class FilePathProvider : public SingleOutputStage<FilePathProviderFile> {
 public:
  using Path = std::filesystem::path;
  using MessageType = FilePathProviderMessageType;
  using File = FilePathProviderFile;

  explicit FilePathProvider(const FilePathProviderConfig& config);
  ~FilePathProvider();

  // Starts monitoring the directory
  void Start() override;

  // Closes the output queue, signaling completion
  void Stop() override;

  // Returns current metrics and clears them.
  StageMetricProto FlushMetrics() override;

  // FilePathProvider has no inputs.
  void SetInputs(absl::Span<QueueBase* const> inputs) override;

 private:
  // Starts monitoring the directory.
  void AddDirectory(const Path& directory, std::stop_token stop_token);

  void Worker(std::stop_token stop_token);
  void ScanDirectoryWithWatch(const Path& directory,
                              std::stop_token stop_token);

#ifdef _WIN32
  // Win32 watches the entire tree through a single recursive handle, so
  // there is no per-directory descriptor map and nothing to add or remove
  // as subdirectories come and go.
  void StartWatch();
  // Drains any completed ReadDirectoryChangesW results into pending_.
  void ProcessDirectoryChanges(std::stop_token stop_token);
  // Issues the next overlapped ReadDirectoryChangesW call.
  bool QueueRead();
  // Emits every pending file whose writer has finished. See
  // IsFileClosedByWriters in utils/platform.h for why this is needed.
  void FlushCompletedFiles(Queue<File>::Producer& producer,
                           std::stop_token stop_token);

  // void*, not HANDLE/OVERLAPPED*, so this header never pulls in windows.h.
  void* directory_handle_ = nullptr;  // HANDLE
  void* overlapped_ = nullptr;        // OVERLAPPED*, owns its hEvent
  std::vector<unsigned char> notify_buffer_;
  bool read_pending_ = false;
  // Files seen changing but not yet proven complete, in discovery order so
  // the emitted order stays stable.
  std::vector<Path> pending_;
  // Files emitted by the initial scan. Used only to reconcile the race
  // window between opening the watch and finishing the scan, then cleared --
  // this mirrors ProcessWatchEventsForNewItems on the POSIX side.
  absl::flat_hash_set<std::string> scanned_;
#else
  void AddWatchRecursive(const Path& path);
  void RemoveWatchRecursive(const Path& path);
  void ProcessWatchEventsForNewItems(const std::vector<Path>& known_files);
  void ProcessInotifyEvents(Queue<File>::Producer& producer,
                            std::stop_token stop_token);
  std::optional<File> ProcessInotifyEvent(const struct inotify_event& event,
                                          std::stop_token stop_token);

  int inotify_fd_;
  // Watch descriptor to directory path.
  absl::flat_hash_map<int, Path> watch_descriptors_;
#endif

  Path directory_;  // Directory to monitor
  Queue<File>::Producer producer_;

  LoadMetricUpdater load_metric_updater_;
  std::stop_source stop_source_;
  ThreadPool thread_pool_{1, ThreadPoolOptions{}, stop_source_};
};

}  // namespace training
}  // namespace lczero
