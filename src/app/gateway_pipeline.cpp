#include "app/gateway_pipeline.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <chrono>

namespace port_ai_gateway {

GatewayPipeline::GatewayPipeline()
    : running_{false} {
}

GatewayPipeline::~GatewayPipeline() {
    Stop();
}

bool GatewayPipeline::Init(const GatewayConfig& config) {
    if (running_.load()) {
        Stop();
    }

    config_ = config;

    visible_queue_ = std::make_shared<ThreadSafeQueue<FramePtr>>(20);
    infrared_queue_ = std::make_shared<ThreadSafeQueue<FramePtr>>(20);
    sync_queue_ = std::make_shared<ThreadSafeQueue<StereoFramePairPtr>>(10);
    detect_input_queue_ = std::make_shared<ThreadSafeQueue<FramePtr>>(10);
    result_queue_ = std::make_shared<ThreadSafeQueue<CornerSlotDetectionResultPtr>>(10);

    visible_reader_ = std::make_shared<RtspReader>(CameraType::VISIBLE, 0);
    if (!visible_reader_->Init(config_.visible_camera)) {
        LOG_ERROR << "Failed to initialize visible camera reader";
        return false;
    }
    visible_reader_->SetOutputQueue(visible_queue_);

    if (!config_.use_visible_only) {
        infrared_reader_ = std::make_shared<RtspReader>(CameraType::INFRARED, 1);
        if (!infrared_reader_->Init(config_.infrared_camera)) {
            LOG_ERROR << "Failed to initialize infrared camera reader";
            return false;
        }
        infrared_reader_->SetOutputQueue(infrared_queue_);
    }

    if (config_.enable_sync && !config_.use_visible_only) {
        synchronizer_ = std::make_shared<FrameSynchronizer>();
        if (!synchronizer_->Init(config_.sync_config)) {
            LOG_ERROR << "Failed to initialize frame synchronizer";
            return false;
        }
        synchronizer_->SetVisibleQueue(visible_queue_);
        synchronizer_->SetInfraredQueue(infrared_queue_);
        synchronizer_->SetOutputQueue(sync_queue_);
    }

    detector_ = std::make_shared<CornerSlotDetector>();
    if (!detector_->Init(config_.detector_config)) {
        LOG_ERROR << "Failed to initialize corner slot detector";
        return false;
    }
    detector_->SetInputQueue(detect_input_queue_);
    detector_->SetOutputQueue(result_queue_);

    if (config_.enable_tracking) {
        tracker_ = std::make_shared<CornerSlotTracker>();
        if (!tracker_->Init(config_.tracker_config)) {
            LOG_ERROR << "Failed to initialize corner slot tracker";
            return false;
        }
        LOG_INFO << "Tracking module initialized: max_tracks="
                 << config_.tracker_config.max_tracks;
    }

    if (config_.enable_energy_analysis) {
        energy_analyzer_ = std::make_shared<SwingEnergyAnalyzer>();
        if (!energy_analyzer_->Init(config_.energy_config)) {
            LOG_ERROR << "Failed to initialize swing energy analyzer";
            return false;
        }
        LOG_INFO << "Energy analyzer initialized: mass="
                 << config_.energy_config.spreader_mass_kg
                 << "kg, rope=" << config_.energy_config.rope_length_m << "m";
    }

    if (config_.enable_plc_output) {
        plc_manager_ = std::make_shared<PlcCommandManager>();
        if (!plc_manager_->Init(config_.plc_config)) {
            LOG_ERROR << "Failed to initialize PLC command manager";
            return false;
        }
        if (plc_callback_) {
            plc_manager_->SetCommandCallback(plc_callback_);
        }
        LOG_INFO << "PLC command manager initialized: "
                 << "estop=" << config_.plc_config.enable_emergency_stop
                 << ", damping=" << config_.plc_config.enable_swing_damping;
    }

    stats_ = {};
    LOG_INFO << "GatewayPipeline initialized successfully: "
             << "tracking=" << config_.enable_tracking
             << ", energy=" << config_.enable_energy_analysis
             << ", plc=" << config_.enable_plc_output;
    return true;
}

bool GatewayPipeline::Start() {
    if (running_.load()) {
        LOG_WARN << "GatewayPipeline already running";
        return true;
    }

    if (!visible_reader_->Start()) {
        LOG_ERROR << "Failed to start visible camera reader";
        return false;
    }

    if (!config_.use_visible_only && infrared_reader_) {
        if (!infrared_reader_->Start()) {
            LOG_ERROR << "Failed to start infrared camera reader";
            visible_reader_->Stop();
            return false;
        }
    }

    if (config_.enable_sync && synchronizer_ && !config_.use_visible_only) {
        if (!synchronizer_->Start()) {
            LOG_ERROR << "Failed to start frame synchronizer";
            visible_reader_->Stop();
            if (infrared_reader_) infrared_reader_->Stop();
            return false;
        }
    }

    if (!detector_->Start()) {
        LOG_ERROR << "Failed to start corner slot detector";
        visible_reader_->Stop();
        if (infrared_reader_) infrared_reader_->Stop();
        if (synchronizer_) synchronizer_->Stop();
        return false;
    }

    if (plc_manager_) {
        plc_manager_->Start();
    }

    running_.store(true);
    result_thread_ = std::thread(&GatewayPipeline::ResultThread, this);
    stats_thread_ = std::thread(&GatewayPipeline::StatsThread, this);
    safety_thread_ = std::thread(&GatewayPipeline::SafetyThread, this);

    LOG_INFO << "GatewayPipeline started successfully";
    return true;
}

void GatewayPipeline::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (result_queue_) {
        result_queue_->Stop();
    }
    if (detect_input_queue_) {
        detect_input_queue_->Stop();
    }
    if (sync_queue_) {
        sync_queue_->Stop();
    }
    if (visible_queue_) {
        visible_queue_->Stop();
    }
    if (infrared_queue_) {
        infrared_queue_->Stop();
    }

    if (result_thread_.joinable()) {
        result_thread_.join();
    }
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }
    if (safety_thread_.joinable()) {
        safety_thread_.join();
    }

    if (detector_) {
        detector_->Stop();
    }
    if (synchronizer_) {
        synchronizer_->Stop();
    }
    if (visible_reader_) {
        visible_reader_->Stop();
    }
    if (infrared_reader_) {
        infrared_reader_->Stop();
    }
    if (plc_manager_) {
        plc_manager_->Stop();
    }

    if (tracker_) {
        tracker_->Release();
    }
    if (energy_analyzer_) {
        energy_analyzer_->Release();
    }

    LOG_INFO << "GatewayPipeline stopped";
}

CornerSlotDetectionResultPtr GatewayPipeline::GetLatestResult(int timeout_ms) {
    if (!result_queue_) {
        return nullptr;
    }
    CornerSlotDetectionResultPtr result;
    if (result_queue_->WaitPop(result, timeout_ms)) {
        return result;
    }
    return nullptr;
}

GatewayPipeline::Stats GatewayPipeline::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats stats = stats_;
    stats.visible_fps = static_cast<int>(visible_fps_.GetFps());
    stats.infrared_fps = static_cast<int>(infrared_fps_.GetFps());
    stats.sync_fps = static_cast<int>(sync_fps_.GetFps());
    stats.detect_fps = detector_ ? static_cast<int>(detector_->GetFps()) : 0;

    if (tracker_) {
        auto tstats = tracker_->GetStats();
        stats.active_tracks = tstats.active_tracks;
        stats.total_tracks = tstats.total_tracks;
    }

    if (plc_manager_) {
        auto pstats = plc_manager_->GetStats();
        stats.highest_safety_level = static_cast<int>(pstats.highest_active_level);
        stats.total_plc_commands = static_cast<int>(pstats.total_commands_issued);
        stats.pending_plc_commands = pstats.pending_count;
        stats.emergency_active = plc_manager_->IsEmergencyActive();
    }

    return stats;
}

void GatewayPipeline::ProcessDetectionResult(const CornerSlotDetectionResultPtr& result) {
    if (!result) return;

    if (tracker_ && config_.enable_tracking) {
        tracker_->Update(result->corner_slots, result->timestamp_us);

        auto tracks = tracker_->GetConfirmedTracks();

        if (tracking_callback_) {
            tracking_callback_(tracks);
        }

        if (energy_analyzer_ && config_.enable_energy_analysis) {
            for (auto& track : tracks) {
                Vec3 pos = track->Position();
                Vec3 vel = track->Velocity();
                energy_analyzer_->UpdateTrack(track->Id(), pos, vel,
                                               result->timestamp_us);
            }

            if (energy_callback_) {
                auto statuses = energy_analyzer_->GetAllStatuses();
                energy_callback_(statuses);
            }

            if (plc_manager_ && config_.enable_plc_output) {
                for (auto& track : tracks) {
                    PlcCommand cmd;
                    if (energy_analyzer_->ShouldTriggerDamping(track->Id(), cmd)) {
                        plc_manager_->SubmitCommand(cmd);
                    }
                }
            }
        }
    }

    if (result_callback_) {
        result_callback_(result);
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_frames_processed++;
    stats_.total_detections += result->corner_slots.size();
    stats_.avg_inference_ms = result->inference_time_ms;
    stats_.avg_total_ms = result->total_time_ms;
}

void GatewayPipeline::ResultThread() {
    while (running_.load()) {
        FramePtr frame;

        if (config_.enable_sync && !config_.use_visible_only && synchronizer_) {
            StereoFramePairPtr pair;
            if (sync_queue_->WaitPop(pair, 100)) {
                if (pair && pair->is_synced && pair->visible_frame) {
                    frame = pair->visible_frame;
                    sync_fps_.Tick();
                }
            }
        } else {
            if (visible_queue_->WaitPop(frame, 100)) {
                visible_fps_.Tick();
            }
        }

        if (frame) {
            detect_input_queue_->TryPush(frame);
        }

        CornerSlotDetectionResultPtr result;
        while (result_queue_->TryPop(result)) {
            ProcessDetectionResult(result);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void GatewayPipeline::StatsThread() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        Stats stats = GetStats();
        int avail_ctx = detector_ ? detector_->GetEngine()
                                      ? detector_->GetEngine()->GetAvailableContexts()
                                      : 0;
        int total_ctx = detector_ ? detector_->GetEngine()
                                      ? detector_->GetEngine()->GetContextPoolSize()
                                      : 0;

        const char* safety_names[] = {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"};
        int safety_idx = stats.highest_safety_level;
        if (safety_idx < 0 || safety_idx > 4) safety_idx = 0;

        LOG_INFO << "STATS | Vis:" << stats.visible_fps << "fps "
                 << "IR:" << stats.infrared_fps << "fps "
                 << "Sync:" << stats.sync_fps << "fps "
                 << "Det:" << stats.detect_fps << "fps | "
                 << "CtxPool:" << avail_ctx << "/" << total_ctx << " | "
                 << "Tracks:" << stats.active_tracks << "active/"
                 << stats.total_tracks << "total | "
                 << "Safety:" << safety_names[safety_idx];

        if (stats.emergency_active) {
            LOG_FATAL << "!!! EMERGENCY STOP ACTIVE !!! PLC E-STOP command issued.";
        }

        if (avail_ctx == 0 && total_ctx > 0) {
            LOG_WARN << "CONTEXT POOL EXHAUSTED! All " << total_ctx
                     << " execution contexts are in use.";
        }

        if (stats.total_plc_commands > 0 && stats.total_plc_commands % 10 == 0) {
            LOG_INFO << "PLC STATS: total=" << stats.total_plc_commands
                     << ", pending=" << stats.pending_plc_commands;
        }
    }
}

void GatewayPipeline::SafetyThread() {
    LOG_INFO << "Safety monitoring thread started";

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (!energy_analyzer_ || !plc_manager_ || !tracker_) {
            continue;
        }

        auto statuses = energy_analyzer_->GetAllStatuses();
        SafetyLevel highest = SafetyLevel::SAFE;
        for (auto& s : statuses) {
            if (s.safety_level > highest) {
                highest = s.safety_level;
            }
        }

        if (highest > stats_.highest_safety_level) {
            const char* names[] = {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"};
            int idx = static_cast<int>(highest);
            if (idx >= 2) {
                LOG_WARN << "[SAFETY] Safety level RAISED to " << names[idx]
                         << " - " << statuses.size() << " tracks monitored";
            }
        }
    }

    LOG_INFO << "Safety monitoring thread stopped";
}

}
