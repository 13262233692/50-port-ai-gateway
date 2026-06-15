#pragma once

#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <vector>
#include <functional>

#include "common/frame.h"
#include "common/thread_safe_queue.h"
#include "streaming/rtsp_reader.h"
#include "streaming/frame_synchronizer.h"
#include "inference/corner_slot_detector.h"

namespace port_ai_gateway {

struct GatewayConfig {
    RtspConfig visible_camera;
    RtspConfig infrared_camera;
    FrameSynchronizerConfig sync_config;
    CornerSlotDetectorConfig detector_config;
    bool use_visible_only = false;
    bool enable_sync = true;
    int result_callback_interval_ms = 100;
};

using ResultCallback = std::function<void(const CornerSlotDetectionResultPtr&)>;

class GatewayPipeline {
public:
    GatewayPipeline();
    ~GatewayPipeline();

    bool Init(const GatewayConfig& config);
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    void SetResultCallback(ResultCallback callback) {
        result_callback_ = callback;
    }

    CornerSlotDetectionResultPtr GetLatestResult(int timeout_ms = 100);

    std::shared_ptr<RtspReader> GetVisibleReader() { return visible_reader_; }
    std::shared_ptr<RtspReader> GetInfraredReader() { return infrared_reader_; }
    std::shared_ptr<FrameSynchronizer> GetSynchronizer() { return synchronizer_; }
    std::shared_ptr<CornerSlotDetector> GetDetector() { return detector_; }

    struct Stats {
        int visible_fps = 0;
        int infrared_fps = 0;
        int sync_fps = 0;
        int detect_fps = 0;
        float avg_inference_ms = 0;
        float avg_total_ms = 0;
        int total_frames_processed = 0;
        int total_detections = 0;
    };

    Stats GetStats() const;

private:
    void ResultThread();
    void StatsThread();

    GatewayConfig config_;
    std::atomic<bool> running_{false};

    std::shared_ptr<RtspReader> visible_reader_;
    std::shared_ptr<RtspReader> infrared_reader_;
    std::shared_ptr<FrameSynchronizer> synchronizer_;
    std::shared_ptr<CornerSlotDetector> detector_;

    std::shared_ptr<ThreadSafeQueue<FramePtr>> visible_queue_;
    std::shared_ptr<ThreadSafeQueue<FramePtr>> infrared_queue_;
    std::shared_ptr<ThreadSafeQueue<StereoFramePairPtr>> sync_queue_;
    std::shared_ptr<ThreadSafeQueue<FramePtr>> detect_input_queue_;
    std::shared_ptr<ThreadSafeQueue<CornerSlotDetectionResultPtr>> result_queue_;

    ResultCallback result_callback_;
    std::thread result_thread_;
    std::thread stats_thread_;

    mutable std::mutex stats_mutex_;
    Stats stats_;

    FpsCounter visible_fps_{30};
    FpsCounter infrared_fps_{30};
    FpsCounter sync_fps_{30};
};

}
