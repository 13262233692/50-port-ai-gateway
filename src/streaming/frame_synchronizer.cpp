#include "streaming/frame_synchronizer.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <algorithm>
#include <cmath>

namespace port_ai_gateway {

FrameSynchronizer::FrameSynchronizer()
    : running_{false}
    , last_sync_pts_{0}
    , sync_count_{0}
    , drop_count_{0} {
}

FrameSynchronizer::~FrameSynchronizer() {
    Stop();
}

bool FrameSynchronizer::Init(const FrameSynchronizerConfig& config) {
    config_ = config;
    if (!output_queue_) {
        output_queue_ = std::make_shared<ThreadSafeQueue<StereoFramePairPtr>>(10);
    }
    return true;
}

bool FrameSynchronizer::Start() {
    if (running_.load()) {
        LOG_WARN << "FrameSynchronizer already running";
        return true;
    }

    running_.store(true);
    sync_thread_ = std::thread(&FrameSynchronizer::SyncLoop, this);

    LOG_INFO << "FrameSynchronizer started, max_diff="
             << config_.max_timestamp_diff_us << "us";

    return true;
}

void FrameSynchronizer::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    queue_cv_.notify_all();

    if (output_queue_) {
        output_queue_->Stop();
    }

    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }

    LOG_INFO << "FrameSynchronizer stopped, synced=" << sync_count_.load()
             << ", dropped=" << drop_count_.load();
}

void FrameSynchronizer::PushVisibleFrame(FramePtr frame) {
    if (!frame) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(visible_queue_.size()) >= config_.max_queue_size) {
        visible_queue_.pop_front();
        drop_count_++;
    }
    visible_queue_.push_back(frame);
    queue_cv_.notify_one();
}

void FrameSynchronizer::PushInfraredFrame(FramePtr frame) {
    if (!frame) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(infrared_queue_.size()) >= config_.max_queue_size) {
        infrared_queue_.pop_front();
        drop_count_++;
    }
    infrared_queue_.push_back(frame);
    queue_cv_.notify_one();
}

StereoFramePairPtr FrameSynchronizer::GetSyncedFrame(int timeout_ms) {
    if (!output_queue_) {
        return nullptr;
    }
    StereoFramePairPtr pair;
    if (output_queue_->WaitPop(pair, timeout_ms)) {
        return pair;
    }
    return nullptr;
}

bool FrameSynchronizer::HasSyncedFrame() const {
    return output_queue_ && !output_queue_->Empty();
}

void FrameSynchronizer::SetVisibleQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue) {
    visible_input_ = queue;
}

void FrameSynchronizer::SetInfraredQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue) {
    infrared_input_ = queue;
}

void FrameSynchronizer::SyncLoop() {
    while (running_.load()) {
        if (visible_input_) {
            FramePtr frame;
            while (visible_input_->TryPop(frame)) {
                PushVisibleFrame(frame);
            }
        }

        if (infrared_input_) {
            FramePtr frame;
            while (infrared_input_->TryPop(frame)) {
                PushInfraredFrame(frame);
            }
        }

        StereoFramePairPtr synced = TrySync();
        if (synced && synced->is_synced) {
            if (output_queue_) {
                if (!output_queue_->TryPush(synced)) {
                    LOG_WARN << "Sync output queue full";
                }
            }
            continue;
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.sync_timeout_ms),
            [this]() {
                return !running_.load() ||
                       (!visible_queue_.empty() && !infrared_queue_.empty());
            });
    }
}

StereoFramePairPtr FrameSynchronizer::TrySync() {
    auto pair = std::make_shared<StereoFramePair>();

    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (visible_queue_.empty() || infrared_queue_.empty()) {
        return pair;
    }

    FramePtr best_visible;
    FramePtr best_infrared;
    int64_t min_diff = INT64_MAX;

    for (const auto& vis : visible_queue_) {
        for (const auto& ir : infrared_queue_) {
            int64_t diff = std::abs(vis->timestamp_us - ir->timestamp_us);
            if (diff < min_diff) {
                min_diff = diff;
                best_visible = vis;
                best_infrared = ir;
            }
        }
    }

    if (!best_visible || !best_infrared) {
        return pair;
    }

    if (min_diff > config_.max_timestamp_diff_us) {
        if (visible_queue_.size() > (size_t)config_.max_queue_size / 2 &&
            visible_queue_.front()->timestamp_us < infrared_queue_.front()->timestamp_us) {
            visible_queue_.pop_front();
            drop_count_++;
        } else if (infrared_queue_.size() > (size_t)config_.max_queue_size / 2 &&
                   infrared_queue_.front()->timestamp_us < visible_queue_.front()->timestamp_us) {
            infrared_queue_.pop_front();
            drop_count_++;
        }
        return pair;
    }

    while (!visible_queue_.empty() && visible_queue_.front() != best_visible) {
        visible_queue_.pop_front();
        drop_count_++;
    }
    while (!infrared_queue_.empty() && infrared_queue_.front() != best_infrared) {
        infrared_queue_.pop_front();
        drop_count_++;
    }

    if (!visible_queue_.empty()) visible_queue_.pop_front();
    if (!infrared_queue_.empty()) infrared_queue_.pop_front();

    pair->visible_frame = best_visible;
    pair->infrared_frame = best_infrared;
    pair->sync_timestamp_us = (best_visible->timestamp_us + best_infrared->timestamp_us) / 2;
    pair->is_synced = true;

    sync_count_++;
    last_sync_pts_ = pair->sync_timestamp_us;

    return pair;
}

}
