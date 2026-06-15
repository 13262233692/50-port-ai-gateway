#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>

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
    int num_execution_contexts = 4;
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
    int context_id = -1;
};

using InferenceResultPtr = std::shared_ptr<InferenceResult>;

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void SetLevel(Severity level) { level_ = level; }

private:
    Severity level_ = Severity::kWARNING;
};

struct TrtBindingSet {
    std::vector<void*> buffers;
    std::vector<size_t> buffer_sizes;
    std::vector<void*> host_output_buffers;
    std::atomic<bool> in_use{false};

    TrtBindingSet() = default;
    ~TrtBindingSet() = default;

    TrtBindingSet(const TrtBindingSet&) = delete;
    TrtBindingSet& operator=(const TrtBindingSet&) = delete;
};

class TrtExecutionContext {
public:
    TrtExecutionContext();
    ~TrtExecutionContext();

    TrtExecutionContext(const TrtExecutionContext&) = delete;
    TrtExecutionContext& operator=(const TrtExecutionContext&) = delete;

    bool Init(nvinfer1::ICudaEngine* engine, int context_id);
    void Release();

    bool EnqueueV2(cudaStream_t stream);

    nvinfer1::IExecutionContext* Context() const { return context_; }
    TrtBindingSet* Bindings() { return &bindings_; }
    int Id() const { return id_; }

    void* InputBuffer() const { return bindings_.buffers[input_index_]; }
    void* OutputBuffer(int idx = 0) const;
    size_t InputBufferSize() const;
    size_t OutputBufferSize(int idx = 0) const;

    const std::vector<int>& InputDims() const { return input_dims_; }
    const std::vector<std::vector<int>>& OutputDims() const { return output_dims_; }

    bool CopyOutputsToHost(cudaStream_t stream);
    const float* HostOutput(int idx = 0) const;

    size_t TotalBufferSize() const;

    void SetInputTensorAddress(void* ptr) {
        if (input_index_ >= 0) bindings_.buffers[input_index_] = ptr;
    }

private:
    bool AllocateBuffers(nvinfer1::ICudaEngine* engine);
    void FreeBuffers();

    nvinfer1::IExecutionContext* context_ = nullptr;
    TrtBindingSet bindings_;
    int id_ = -1;
    int input_index_ = -1;
    std::vector<int> output_indices_;
    std::vector<int> input_dims_;
    std::vector<std::vector<int>> output_dims_;
};

using TrtExecutionContextPtr = std::shared_ptr<TrtExecutionContext>;

class TrtContextPool {
public:
    TrtContextPool();
    ~TrtContextPool();

    bool Init(nvinfer1::ICudaEngine* engine, int num_contexts);
    void Shutdown();

    TrtExecutionContextPtr Acquire(int timeout_ms = 5000);
    void Release(TrtExecutionContextPtr ctx);

    int PoolSize() const { return static_cast<int>(pool_.size()); }
    int AvailableCount() const;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<TrtExecutionContextPtr> pool_;
    std::atomic<bool> shutdown_{false};
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

    bool IsValid() const { return engine_ != nullptr && context_pool_.PoolSize() > 0; }
    nvinfer1::ICudaEngine* Engine() const { return engine_; }

    static bool SaveEngine(const std::string& path, const void* data, size_t size);

    const TrtEngineConfig& Config() const { return config_; }

    TrtExecutionContextPtr AcquireContext(int timeout_ms = 5000);
    void ReleaseContext(TrtExecutionContextPtr ctx);

    InferenceResultPtr RunInferenceWithContext(
        TrtExecutionContextPtr ctx,
        const FramePtr& frame,
        const LetterboxInfo& letterbox_info,
        cudaStream_t stream = nullptr);

    int GetContextPoolSize() const { return context_pool_.PoolSize(); }
    int GetAvailableContexts() const { return context_pool_.AvailableCount(); }

private:
    TrtEngineConfig config_;
    TrtLogger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;

    std::vector<int> input_dims_;
    std::vector<std::vector<int>> output_dims_;
    std::vector<std::string> output_names_;

    int input_index_ = -1;
    std::vector<int> output_indices_;

    TrtContextPool context_pool_;

    bool inited_ = false;
};

}
