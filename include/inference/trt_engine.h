#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "common/frame.h"
#include "inference/cuda_preprocessor.h"

namespace port_ai_gateway {

struct TrtEngineConfig {
    std::string engine_path;
    std::string input_blob_name = "input";
    std::vector<std::string> output_blob_names;
    int max_batch_size = 1;
    bool use_fp16 = true;
    int gpu_device_id = 0;
    size_t workspace_size = 1 << 30;
};

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
    std::vector<std::pair<float, float>> mask_contour;
};

struct InferenceResult {
    int64_t timestamp_us = 0;
    int camera_id = 0;
    CameraType camera_type = CameraType::VISIBLE;
    std::vector<Detection> detections;
    float inference_time_ms = 0;
    float preprocess_time_ms = 0;
    float postprocess_time_ms = 0;
};

using InferenceResultPtr = std::shared_ptr<InferenceResult>;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void SetLevel(Severity level) { level_ = level; }

private:
    Severity level_ = Severity::kWARNING;
};

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    bool Init(const TrtEngineConfig& config);
    void Release();

    bool LoadEngine(const std::string& engine_path);
    bool DeserializeEngine(const void* data, size_t size);

    InferenceResultPtr RunInference(const FramePtr& frame,
                                     const LetterboxInfo& letterbox_info,
                                     cudaStream_t stream = nullptr);

    const std::vector<int>& GetInputDims() const { return input_dims_; }
    const std::vector<std::vector<int>>& GetOutputDims() const { return output_dims_; }

    int GetInputWidth() const { return input_dims_.size() >= 3 ? input_dims_[2] : 0; }
    int GetInputHeight() const { return input_dims_.size() >= 3 ? input_dims_[1] : 0; }

    bool IsValid() const { return engine_ != nullptr && context_ != nullptr; }

    void* GetInputBuffer() const {
        return input_index_ >= 0 ? buffers_[input_index_] : nullptr;
    }

    void* GetOutputBuffer(int index = 0) const {
        if (index < 0 || index >= static_cast<int>(output_indices_.size())) {
            return nullptr;
        }
        return buffers_[output_indices_[index]];
    }

    size_t GetInputBufferSize() const {
        return input_index_ >= 0 ? buffer_sizes_[input_index_] : 0;
    }

    size_t GetOutputBufferSize(int index = 0) const {
        if (index < 0 || index >= static_cast<int>(output_indices_.size())) {
            return 0;
        }
        return buffer_sizes_[output_indices_[index]];
    }

    static bool SaveEngine(const std::string& path, const void* data, size_t size);

private:
    bool AllocateBuffers();
    void FreeBuffers();
    bool Infer(void** buffers, cudaStream_t stream);

    TrtEngineConfig config_;
    Logger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::vector<int> input_dims_;
    std::vector<std::vector<int>> output_dims_;
    std::vector<std::string> output_names_;

    std::vector<void*> buffers_;
    std::vector<size_t> buffer_sizes_;

    bool inited_ = false;
    std::mutex mutex_;

    int input_index_ = -1;
    std::vector<int> output_indices_;
};

}
