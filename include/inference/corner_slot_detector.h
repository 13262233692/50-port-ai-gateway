#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>

#include "common/frame.h"
#include "common/thread_safe_queue.h"
#include "inference/trt_engine.h"
#include "inference/cuda_preprocessor.h"

namespace port_ai_gateway {

struct CornerSlotDetectorConfig {
    TrtEngineConfig trt_config;
    PreprocessConfig preprocess_config;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.45f;
    int max_detections = 100;
    int input_queue_size = 10;
    int output_queue_size = 10;
    int num_preprocess_streams = 1;
};

struct CornerSlot3D {
    int id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float depth = 0.0f;
    float confidence = 0.0f;
    float yaw = 0.0f;
    std::vector<std::pair<float, float>> image_points;
};

struct CornerSlotDetectionResult {
    int64_t timestamp_us = 0;
    int camera_id = 0;
    CameraType camera_type = CameraType::VISIBLE;
    std::vector<CornerSlot3D> corner_slots;
    float total_time_ms = 0;
    float preprocess_time_ms = 0;
    float inference_time_ms = 0;
    float postprocess_time_ms = 0;
};

using CornerSlotDetectionResultPtr = std::shared_ptr<CornerSlotDetectionResult>;

class CornerSlotDetector {
public:
    CornerSlotDetector();
    ~CornerSlotDetector();

    bool Init(const CornerSlotDetectorConfig& config);
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    bool Detect(const FramePtr& frame,
                CornerSlotDetectionResult& result,
                cudaStream_t stream = nullptr);

    void PushFrame(const FramePtr& frame);
    CornerSlotDetectionResultPtr GetResult(int timeout_ms = 100);

    void SetInputQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue) {
        input_queue_ = queue;
    }
    void SetOutputQueue(std::shared_ptr<ThreadSafeQueue<CornerSlotDetectionResultPtr>> queue) {
        output_queue_ = queue;
    }

    std::shared_ptr<TrtEngine> GetEngine() const { return engine_; }

    float GetFps() const { return fps_counter_.GetFps(); }

private:
    void WorkerThread(int worker_id);
    void Nms(std::vector<Detection>& detections, float iou_threshold);
    float CalculateIou(const Detection& a, const Detection& b);
    bool ConvertTo3D(const Detection& det, const FramePtr& frame, CornerSlot3D& slot_3d);

    CornerSlotDetectorConfig config_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> worker_threads_;
    int num_workers_ = 0;

    std::shared_ptr<TrtEngine> engine_;
    std::vector<std::shared_ptr<CudaPreprocessor>> preprocessors_;

    std::shared_ptr<ThreadSafeQueue<FramePtr>> input_queue_;
    std::shared_ptr<ThreadSafeQueue<CornerSlotDetectionResultPtr>> output_queue_;

    std::vector<cudaStream_t> cuda_streams_;

    FpsCounter fps_counter_;
    std::atomic<int> total_detections_{0};
    std::atomic<int> dropped_frames_{0};
};

}
