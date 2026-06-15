#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "common/frame.h"
#include "common/thread_safe_queue.h"

namespace port_ai_gateway {

struct FrameSynchronizerConfig {
    int64_t max_timestamp_diff_us = 33000;
    int max_queue_size = 30;
    int sync_timeout_ms = 100;
};

class FrameSynchronizer {
public:
    FrameSynchronizer();
    ~FrameSynchronizer();

    bool Init(const FrameSynchronizerConfig& config);
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    void PushVisibleFrame(FramePtr frame);
    void PushInfraredFrame(FramePtr frame);

    StereoFramePairPtr GetSyncedFrame(int timeout_ms = 100);
    bool HasSyncedFrame() const;

    void SetOutputQueue(std::shared_ptr<ThreadSafeQueue<StereoFramePairPtr>> queue) {
        output_queue_ = queue;
    }

    void SetVisibleQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue);
    void SetInfraredQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue);

private:
    void SyncLoop();
    StereoFramePairPtr TrySync();

    FrameSynchronizerConfig config_;

    std::deque<FramePtr> visible_queue_;
    std::deque<FramePtr> infrared_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::atomic<bool> running_{false};
    std::thread sync_thread_;

    std::shared_ptr<ThreadSafeQueue<FramePtr>> visible_input_;
    std::shared_ptr<ThreadSafeQueue<FramePtr>> infrared_input_;
    std::shared_ptr<ThreadSafeQueue<StereoFramePairPtr>> output_queue_;

    std::atomic<int64_t> last_sync_pts_{0};
    std::atomic<int> sync_count_{0};
    std::atomic<int> drop_count_{0};
};

}
