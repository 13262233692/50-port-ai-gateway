#include "app/gateway_pipeline.h"
#include "common/logger.h"
#include "common/time_util.h"

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

    LOG_INFO << "GatewayPipeline initialized successfully";
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

    running_.store(true);
    result_thread_ = std::thread(&GatewayPipeline::ResultThread, this);
    stats_thread_ = std::thread(&GatewayPipeline::StatsThread, this);

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
    stats.detect_fps = static_cast<int>(detector_->GetFps());
    return stats;
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

        if (result_callback_) {
            CornerSlotDetectionResultPtr result;
            while (result_queue_->TryPop(result)) {
                if (result_callback_) {
                    result_callback_(result);
                }

                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.total_frames_processed++;
                stats_.total_detections += result->corner_slots.size();
                stats_.avg_inference_ms = result->inference_time_ms;
                stats_.avg_total_ms = result->total_time_ms;
            }
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

        LOG_INFO << "STATS | Visible: " << stats.visible_fps << "fps, "
                 << "IR: " << stats.infrared_fps << "fps, "
                 << "Sync: " << stats.sync_fps << "fps, "
                 << "Detect: " << stats.detect_fps << "fps, "
                 << "CtxPool: " << avail_ctx << "/" << total_ctx
                 << " (avail/total), "
                 << "Processed: " << stats.total_frames_processed
                 << ", Detections: " << stats.total_detections;

        if (avail_ctx == 0 && total_ctx > 0) {
            LOG_WARN << "CONTEXT POOL EXHAUSTED! All " << total_ctx
                     << " execution contexts are in use. Consider increasing "
                        "num_execution_contexts.";
        }
    }
}

}
