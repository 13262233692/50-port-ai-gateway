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
#include "tracking/corner_slot_tracker.h"
#include "safety/swing_energy_analyzer.h"
#include "safety/plc_command_manager.h"

namespace port_ai_gateway {

struct GatewayConfig {
    RtspConfig visible_camera;
    RtspConfig infrared_camera;
    FrameSynchronizerConfig sync_config;
    CornerSlotDetectorConfig detector_config;
    TrackerConfig tracker_config;
    EnergyAnalyzerConfig energy_config;
    PlcManagerConfig plc_config;

    bool use_visible_only = false;
    bool enable_sync = true;
    bool enable_tracking = true;
    bool enable_energy_analysis = true;
    bool enable_plc_output = true;

    int result_callback_interval_ms = 100;
};

using ResultCallback = std::function<void(const CornerSlotDetectionResultPtr&)>;
using TrackingCallback = std::function<void(const std::vector<CornerSlotTrackPtr>&)>;
using EnergyStatusCallback = std::function<void(const std::vector<TrackEnergyStatus>&)>;
using PlcCommandCallback = std::function<void(const PlcCommand&)>;

class GatewayPipeline {
public:
    GatewayPipeline();
    ~GatewayPipeline();

    bool Init(const GatewayConfig& config);
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    void SetResultCallback(ResultCallback callback) {
        result_callback_ = std::move(callback);
    }
    void SetTrackingCallback(TrackingCallback callback) {
        tracking_callback_ = std::move(callback);
    }
    void SetEnergyStatusCallback(EnergyStatusCallback callback) {
        energy_callback_ = std::move(callback);
    }
    void SetPlcCommandCallback(PlcCommandCallback callback) {
        plc_callback_ = std::move(callback);
    }

    CornerSlotDetectionResultPtr GetLatestResult(int timeout_ms = 100);

    std::shared_ptr<RtspReader> GetVisibleReader() { return visible_reader_; }
    std::shared_ptr<RtspReader> GetInfraredReader() { return infrared_reader_; }
    std::shared_ptr<FrameSynchronizer> GetSynchronizer() { return synchronizer_; }
    std::shared_ptr<CornerSlotDetector> GetDetector() { return detector_; }
    std::shared_ptr<CornerSlotTracker> GetTracker() { return tracker_; }
    std::shared_ptr<SwingEnergyAnalyzer> GetEnergyAnalyzer() { return energy_analyzer_; }
    std::shared_ptr<PlcCommandManager> GetPlcManager() { return plc_manager_; }

    struct Stats {
        int visible_fps = 0;
        int infrared_fps = 0;
        int sync_fps = 0;
        int detect_fps = 0;
        float avg_inference_ms = 0;
        float avg_total_ms = 0;
        int total_frames_processed = 0;
        int total_detections = 0;

        int active_tracks = 0;
        int total_tracks = 0;

        int highest_safety_level = 0;
        int total_plc_commands = 0;
        int pending_plc_commands = 0;
        bool emergency_active = false;
    };

    Stats GetStats() const;

private:
    void ResultThread();
    void StatsThread();
    void SafetyThread();

    void ProcessDetectionResult(const CornerSlotDetectionResultPtr& result);

    GatewayConfig config_;
    std::atomic<bool> running_{false};

    std::shared_ptr<RtspReader> visible_reader_;
    std::shared_ptr<RtspReader> infrared_reader_;
    std::shared_ptr<FrameSynchronizer> synchronizer_;
    std::shared_ptr<CornerSlotDetector> detector_;
    std::shared_ptr<CornerSlotTracker> tracker_;
    std::shared_ptr<SwingEnergyAnalyzer> energy_analyzer_;
    std::shared_ptr<PlcCommandManager> plc_manager_;

    std::shared_ptr<ThreadSafeQueue<FramePtr>> visible_queue_;
    std::shared_ptr<ThreadSafeQueue<FramePtr>> infrared_queue_;
    std::shared_ptr<ThreadSafeQueue<StereoFramePairPtr>> sync_queue_;
    std::shared_ptr<ThreadSafeQueue<FramePtr>> detect_input_queue_;
    std::shared_ptr<ThreadSafeQueue<CornerSlotDetectionResultPtr>> result_queue_;

    ResultCallback result_callback_;
    TrackingCallback tracking_callback_;
    EnergyStatusCallback energy_callback_;
    PlcCommandCallback plc_callback_;

    std::thread result_thread_;
    std::thread stats_thread_;
    std::thread safety_thread_;

    mutable std::mutex stats_mutex_;
    Stats stats_;

    FpsCounter visible_fps_{30};
    FpsCounter infrared_fps_{30};
    FpsCounter sync_fps_{30};
};

}
