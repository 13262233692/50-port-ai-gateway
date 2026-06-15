#include "inference/corner_slot_detector.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <algorithm>
#include <cmath>

namespace port_ai_gateway {

CornerSlotDetector::CornerSlotDetector()
    : running_{false}
    , fps_counter_(30) {
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

    preprocessor_ = std::make_shared<CudaPreprocessor>();
    if (!preprocessor_->Init(config_.preprocess_config)) {
        LOG_ERROR << "Failed to initialize CUDA preprocessor";
        return false;
    }

    cudaError_t err = cudaStreamCreate(&cuda_stream_);
    if (err != cudaSuccess) {
        LOG_ERROR << "Failed to create CUDA stream: " << cudaGetErrorString(err);
        return false;
    }

    LOG_INFO << "CornerSlotDetector initialized";
    return true;
}

bool CornerSlotDetector::Start() {
    if (running_.load()) {
        LOG_WARN << "CornerSlotDetector already running";
        return true;
    }

    if (!engine_ || !preprocessor_) {
        LOG_ERROR << "CornerSlotDetector not initialized";
        return false;
    }

    running_.store(true);
    worker_thread_ = std::thread(&CornerSlotDetector::WorkerThread, this);

    LOG_INFO << "CornerSlotDetector started";
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

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }

    if (engine_) {
        engine_->Release();
    }
    if (preprocessor_) {
        preprocessor_->Release();
    }

    LOG_INFO << "CornerSlotDetector stopped";
}

bool CornerSlotDetector::Detect(const FramePtr& frame,
                                 CornerSlotDetectionResult& result,
                                 cudaStream_t stream) {
    if (!engine_ || !preprocessor_ || !frame) {
        return false;
    }

    std::lock_guard<std::mutex> lock(detector_mutex_);

    ScopedTimer total_timer;
    result.timestamp_us = frame->timestamp_us;
    result.camera_id = frame->camera_id;
    result.camera_type = frame->camera_type;

    cudaStream_t use_stream = stream ? stream : cuda_stream_;

    LetterboxInfo letterbox_info;
    float* input_buffer = static_cast<float*>(engine_->GetInputBuffer());

    ScopedTimer preprocess_timer;
    if (!preprocessor_->Process(frame, input_buffer, letterbox_info, use_stream)) {
        LOG_ERROR << "Preprocessing failed";
        return false;
    }
    result.preprocess_time_ms = preprocess_timer.ElapsedMillis();

    InferenceResultPtr trt_result = engine_->RunInference(frame, letterbox_info, use_stream);
    if (!trt_result) {
        LOG_ERROR << "Inference failed";
        return false;
    }
    result.inference_time_ms = trt_result->inference_time_ms;

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

    result.postprocess_time_ms = postprocess_timer.ElapsedMillis();
    result.total_time_ms = total_timer.ElapsedMillis();

    fps_counter_.Tick();

    return true;
}

void CornerSlotDetector::PushFrame(const FramePtr& frame) {
    if (input_queue_) {
        input_queue_->TryPush(frame);
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

void CornerSlotDetector::WorkerThread() {
    while (running_.load()) {
        FramePtr frame;
        if (!input_queue_->WaitPop(frame, 100)) {
            continue;
        }

        if (!frame) {
            continue;
        }

        auto result = std::make_shared<CornerSlotDetectionResult>();
        if (Detect(frame, *result, cuda_stream_)) {
            output_queue_->TryPush(result);
        }
    }
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

    float focal_length = 500.0f;
    float real_slot_width = 0.12f;
    float real_slot_height = 0.08f;

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

        slot_3d.image_points.push_back({det.x1, det.y1});
        slot_3d.image_points.push_back({det.x2, det.y1});
        slot_3d.image_points.push_back({det.x2, det.y2});
        slot_3d.image_points.push_back({det.x1, det.y2});

        return true;
    }

    return false;
}

}
