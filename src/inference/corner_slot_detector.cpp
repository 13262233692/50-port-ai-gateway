#include "inference/corner_slot_detector.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <algorithm>
#include <cmath>

namespace port_ai_gateway {

CornerSlotDetector::CornerSlotDetector()
    : running_{false}
    , num_workers_(0)
    , fps_counter_(30)
    , total_detections_{0}
    , dropped_frames_{0} {
}

CornerSlotDetector::~CornerSlotDetector() {
    Stop();
}

bool CornerSlotDetector::Init(const CornerSlotDetectorConfig& config) {
    if (running_.load()) {
        Stop();
    }

    config_ = config;

    if (!input_queue_) {
        input_queue_ = std::make_shared<ThreadSafeQueue<FramePtr>>(config_.input_queue_size);
    }
    if (!output_queue_) {
        output_queue_ = std::make_shared<ThreadSafeQueue<CornerSlotDetectionResultPtr>>(
            config_.output_queue_size);
    }

    engine_ = std::make_shared<TrtEngine>();
    if (!engine_->Init(config_.trt_config)) {
        LOG_ERROR << "Failed to initialize TensorRT engine";
        return false;
    }

    num_workers_ = config_.trt_config.num_execution_contexts;

    preprocessors_.resize(num_workers_);
    for (int i = 0; i < num_workers_; i++) {
        preprocessors_[i] = std::make_shared<CudaPreprocessor>();
        if (!preprocessors_[i]->Init(config_.preprocess_config)) {
            LOG_ERROR << "Failed to initialize CUDA preprocessor for worker " << i;
            return false;
        }
    }

    cuda_streams_.resize(num_workers_, nullptr);
    for (int i = 0; i < num_workers_; i++) {
        cudaError_t err = cudaStreamCreate(&cuda_streams_[i]);
        if (err != cudaSuccess) {
            LOG_ERROR << "Failed to create CUDA stream " << i
                      << ": " << cudaGetErrorString(err);
            return false;
        }
    }

    LOG_INFO << "CornerSlotDetector initialized: "
             << num_workers_ << " workers, contexts="
             << engine_->GetContextPoolSize()
             << ", stream_per_worker=" << (cuda_streams_.size() == (size_t)num_workers_);
    return true;
}

bool CornerSlotDetector::Start() {
    if (running_.load()) {
        LOG_WARN << "CornerSlotDetector already running";
        return true;
    }

    if (!engine_ || preprocessors_.empty()) {
        LOG_ERROR << "CornerSlotDetector not initialized";
        return false;
    }

    running_.store(true);
    worker_threads_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; i++) {
        worker_threads_.emplace_back(&CornerSlotDetector::WorkerThread, this, i);
    }

    LOG_INFO << "CornerSlotDetector started with " << num_workers_ << " workers";
    return true;
}

void CornerSlotDetector::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (input_queue_) {
        input_queue_->Stop();
    }
    if (output_queue_) {
        output_queue_->Stop();
    }

    for (auto& t : worker_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    worker_threads_.clear();

    for (auto& s : cuda_streams_) {
        if (s) {
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
    cuda_streams_.clear();

    if (engine_) {
        engine_->Release();
    }
    for (auto& p : preprocessors_) {
        if (p) {
            p->Release();
        }
    }
    preprocessors_.clear();

    LOG_INFO << "CornerSlotDetector stopped, total_detections="
             << total_detections_.load()
             << ", dropped_frames=" << dropped_frames_.load();
}

bool CornerSlotDetector::Detect(const FramePtr& frame,
                                 CornerSlotDetectionResult& result,
                                 cudaStream_t stream) {
    if (!engine_ || preprocessors_.empty() || !frame) {
        return false;
    }

    ScopedTimer total_timer;
    result.timestamp_us = frame->timestamp_us;
    result.camera_id = frame->camera_id;
    result.camera_type = frame->camera_type;

    cudaStream_t use_stream = stream;
    int stream_idx = -1;
    if (!use_stream) {
        static thread_local int tls_worker_id = 0;
        stream_idx = tls_worker_id % std::max(1, (int)cuda_streams_.size());
        use_stream = cuda_streams_[stream_idx];
    }

    int worker_id = stream_idx >= 0 ? stream_idx : 0;
    worker_id = worker_id % std::max(1, (int)preprocessors_.size());

    auto ctx = engine_->AcquireContext(5000);
    if (!ctx) {
        LOG_WARN << "Detector: Context pool exhausted (camera=" << frame->camera_id << ")";
        dropped_frames_++;
        return false;
    }

    int ctx_id = ctx->Id();
    LetterboxInfo letterbox_info;
    float* input_buffer = static_cast<float*>(ctx->InputBuffer());

    ScopedTimer preprocess_timer;
    if (!preprocessors_[worker_id]->Process(frame, input_buffer, letterbox_info, use_stream)) {
        LOG_ERROR << "Context " << ctx_id << ": Preprocessing FAILED";
        engine_->ReleaseContext(std::move(ctx));
        return false;
    }
    result.preprocess_time_ms = preprocess_timer.ElapsedMillis();

    InferenceResultPtr trt_result = engine_->RunInferenceWithContext(
        ctx, frame, letterbox_info, use_stream);

    if (!trt_result) {
        LOG_ERROR << "Context " << ctx_id << ": Inference FAILED";
        engine_->ReleaseContext(std::move(ctx));
        return false;
    }
    result.inference_time_ms = trt_result->inference_time_ms;

    engine_->ReleaseContext(std::move(ctx));

    ScopedTimer postprocess_timer;
    std::vector<Detection> detections = trt_result->detections;

    auto it = std::remove_if(detections.begin(), detections.end(),
        [this](const Detection& det) {
            return det.confidence < config_.confidence_threshold;
        });
    detections.erase(it, detections.end());

    Nms(detections, config_.nms_threshold);

    if (static_cast<int>(detections.size()) > config_.max_detections) {
        detections.resize(config_.max_detections);
    }

    result.corner_slots.clear();
    for (const auto& det : detections) {
        CornerSlot3D slot_3d;
        if (ConvertTo3D(det, frame, slot_3d)) {
            slot_3d.confidence = det.confidence;
            result.corner_slots.push_back(slot_3d);
        }
    }

    total_detections_ += static_cast<int>(result.corner_slots.size());
    result.postprocess_time_ms = postprocess_timer.ElapsedMillis();
    result.total_time_ms = total_timer.ElapsedMillis();

    fps_counter_.Tick();

    return true;
}

void CornerSlotDetector::PushFrame(const FramePtr& frame) {
    if (input_queue_) {
        if (!input_queue_->TryPush(frame)) {
            dropped_frames_++;
        }
    }
}

CornerSlotDetectionResultPtr CornerSlotDetector::GetResult(int timeout_ms) {
    if (!output_queue_) {
        return nullptr;
    }
    CornerSlotDetectionResultPtr result;
    if (output_queue_->WaitPop(result, timeout_ms)) {
        return result;
    }
    return nullptr;
}

void CornerSlotDetector::WorkerThread(int worker_id) {
    static thread_local int tls_worker_id = worker_id;

    LOG_DEBUG << "Worker " << worker_id << " started";

    while (running_.load()) {
        FramePtr frame;
        if (!input_queue_->WaitPop(frame, 100)) {
            continue;
        }
        if (!frame) {
            continue;
        }

        auto result = std::make_shared<CornerSlotDetectionResult>();
        cudaStream_t use_stream = (worker_id < (int)cuda_streams_.size())
                                      ? cuda_streams_[worker_id]
                                      : nullptr;

        if (Detect(frame, *result, use_stream)) {
            if (!output_queue_->TryPush(result)) {
                dropped_frames_++;
            }
        }
    }

    LOG_DEBUG << "Worker " << worker_id << " stopped";
}

void CornerSlotDetector::Nms(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) {
        return;
    }

    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> keep;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) {
            continue;
        }
        keep.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) {
                continue;
            }
            float iou = CalculateIou(detections[i], detections[j]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    detections = std::move(keep);
}

float CornerSlotDetector::CalculateIou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float intersection = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    if (union_area <= 0) {
        return 0;
    }
    return intersection / union_area;
}

bool CornerSlotDetector::ConvertTo3D(const Detection& det, const FramePtr& frame,
                                      CornerSlot3D& slot_3d) {
    float cx = (det.x1 + det.x2) / 2.0f;
    float cy = (det.y1 + det.y2) / 2.0f;
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;

    const float focal_length = 500.0f;
    const float real_slot_width = 0.12f;

    if (w > 0) {
        float distance = (focal_length * real_slot_width) / w;
        slot_3d.z = distance;

        int img_width = frame->size_2d.width;
        int img_height = frame->size_2d.height;

        slot_3d.x = (cx - img_width / 2.0f) * distance / focal_length;
        slot_3d.y = (cy - img_height / 2.0f) * distance / focal_length;
        slot_3d.width = w * distance / focal_length;
        slot_3d.height = h * distance / focal_length;
        slot_3d.depth = 0.05f;
        slot_3d.yaw = 0.0f;

        slot_3d.image_points.reserve(4);
        slot_3d.image_points.emplace_back(det.x1, det.y1);
        slot_3d.image_points.emplace_back(det.x2, det.y1);
        slot_3d.image_points.emplace_back(det.x2, det.y2);
        slot_3d.image_points.emplace_back(det.x1, det.y2);
        return true;
    }
    return false;
}

}
