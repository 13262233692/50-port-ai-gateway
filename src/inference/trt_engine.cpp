#include "inference/trt_engine.h"
#include "common/logger.h"
#include "common/time_util.h"
#include "common/gpu_health_monitor.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <cassert>

namespace port_ai_gateway {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > level_) return;
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG_FATAL << "[TRT] " << msg;
            break;
        case Severity::kERROR:
            LOG_ERROR << "[TRT] " << msg;
            break;
        case Severity::kWARNING:
            LOG_WARN << "[TRT] " << msg;
            break;
        case Severity::kINFO:
            LOG_INFO << "[TRT] " << msg;
            break;
        case Severity::kVERBOSE:
            LOG_DEBUG << "[TRT] " << msg;
            break;
        default:
            break;
    }
}

TrtExecutionContext::TrtExecutionContext()
    : context_(nullptr)
    , id_(-1)
    , input_index_(-1) {}

TrtExecutionContext::~TrtExecutionContext() {
    Release();
}

bool TrtExecutionContext::Init(nvinfer1::ICudaEngine* engine, int context_id) {
    if (!engine) {
        LOG_ERROR << "Engine is null";
        return false;
    }

    id_ = context_id;

    context_ = engine->createExecutionContext();
    if (!context_) {
        LOG_ERROR << "Context " << id_ << ": Failed to create IExecutionContext";
        return false;
    }

    int nb_bindings = engine->getNbBindings();
    for (int i = 0; i < nb_bindings; i++) {
        if (engine->bindingIsInput(i)) {
            input_index_ = i;
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            input_dims_.clear();
            for (int j = 0; j < dims.nbDims; j++) {
                input_dims_.push_back(dims.d[j]);
            }
        } else {
            output_indices_.push_back(i);
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            std::vector<int> dim_vec;
            for (int j = 0; j < dims.nbDims; j++) {
                dim_vec.push_back(dims.d[j]);
            }
            output_dims_.push_back(dim_vec);
        }
    }

    if (!AllocateBuffers(engine)) {
        LOG_ERROR << "Context " << id_ << ": Failed to allocate binding buffers";
        return false;
    }

    LOG_INFO << "Context " << id_ << ": Created successfully, "
             << nb_bindings << " bindings, input dims: [";
    for (size_t i = 0; i < input_dims_.size(); i++) {
        if (i > 0) LOG_INFO << ", ";
        LOG_INFO << input_dims_[i];
    }
    LOG_INFO << "], total GPU mem: "
             << (TotalBufferSize() / 1024.0 / 1024.0) << " MB";

    return true;
}

void TrtExecutionContext::Release() {
    FreeBuffers();

    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }

    input_index_ = -1;
    output_indices_.clear();
    input_dims_.clear();
    output_dims_.clear();
    id_ = -1;
}

bool TrtExecutionContext::AllocateBuffers(nvinfer1::ICudaEngine* engine) {
    int nb_bindings = engine->getNbBindings();
    bindings_.buffers.resize(nb_bindings, nullptr);
    bindings_.buffer_sizes.resize(nb_bindings, 0);
    bindings_.host_output_buffers.resize(output_indices_.size(), nullptr);

    for (int i = 0; i < nb_bindings; i++) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        size_t element_size = 4;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:  element_size = 4; break;
            case nvinfer1::DataType::kHALF:   element_size = 2; break;
            case nvinfer1::DataType::kINT8:   element_size = 1; break;
            case nvinfer1::DataType::kINT32:  element_size = 4; break;
            case nvinfer1::DataType::kBOOL:   element_size = 1; break;
            default: element_size = 4; break;
        }

        size_t size = element_size;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= static_cast<size_t>(dims.d[j] > 0 ? dims.d[j] : 1);
        }

        cudaError_t err = cudaMalloc(&bindings_.buffers[i], size);
        if (err != cudaSuccess) {
            LOG_ERROR << "Context " << id_
                      << ": Failed to cudaMalloc binding " << i
                      << " size=" << size << ": " << cudaGetErrorString(err);
            FreeBuffers();
            return false;
        }
        bindings_.buffer_sizes[i] = size;
        cudaMemset(bindings_.buffers[i], 0, size);
    }

    for (size_t i = 0; i < output_indices_.size(); i++) {
        int binding_idx = output_indices_[i];
        size_t size = bindings_.buffer_sizes[binding_idx];
        cudaError_t err = cudaHostAlloc(&bindings_.host_output_buffers[i],
                                         size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            LOG_ERROR << "Context " << id_
                      << ": Failed to cudaHostAlloc host output " << i
                      << ": " << cudaGetErrorString(err);
            FreeBuffers();
            return false;
        }
        memset(bindings_.host_output_buffers[i], 0, size);
    }

    return true;
}

void TrtExecutionContext::FreeBuffers() {
    for (size_t i = 0; i < bindings_.buffers.size(); i++) {
        if (bindings_.buffers[i] != nullptr) {
            cudaFree(bindings_.buffers[i]);
            bindings_.buffers[i] = nullptr;
        }
    }
    bindings_.buffers.clear();
    bindings_.buffer_sizes.clear();

    for (size_t i = 0; i < bindings_.host_output_buffers.size(); i++) {
        if (bindings_.host_output_buffers[i] != nullptr) {
            cudaFreeHost(bindings_.host_output_buffers[i]);
            bindings_.host_output_buffers[i] = nullptr;
        }
    }
    bindings_.host_output_buffers.clear();
}

size_t TrtExecutionContext::TotalBufferSize() const {
    size_t total = 0;
    for (size_t s : bindings_.buffer_sizes) {
        total += s;
    }
    return total;
}

void* TrtExecutionContext::OutputBuffer(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(output_indices_.size())) {
        return nullptr;
    }
    return bindings_.buffers[output_indices_[idx]];
}

size_t TrtExecutionContext::InputBufferSize() const {
    return input_index_ >= 0 ? bindings_.buffer_sizes[input_index_] : 0;
}

size_t TrtExecutionContext::OutputBufferSize(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(output_indices_.size())) {
        return 0;
    }
    return bindings_.buffer_sizes[output_indices_[idx]];
}

bool TrtExecutionContext::EnqueueV2(cudaStream_t stream) {
    if (!context_ || bindings_.buffers.empty()) {
        LOG_ERROR << "Context " << id_ << ": EnqueueV2 called with invalid state";
        return false;
    }

    PORT_AI_ASSERT(input_index_ >= 0, "input_index_ must be valid");
    PORT_AI_ASSERT(bindings_.buffers[input_index_] != nullptr,
                   "Input binding buffer is NULL for ctx=" << id_);
    PORT_AI_ASSERT(GpuHealthMonitor::Instance().ValidateDevicePointer(
                       bindings_.buffers[input_index_],
                       bindings_.buffer_sizes[input_index_]),
                   "Input binding is NOT valid GPU memory ctx=" << id_);

    for (size_t i = 0; i < output_indices_.size(); i++) {
        int bidx = output_indices_[i];
        PORT_AI_ASSERT(bindings_.buffers[bidx] != nullptr,
                       "Output binding " << i << " is NULL ctx=" << id_);
        PORT_AI_ASSERT(GpuHealthMonitor::Instance().ValidateDevicePointer(
                           bindings_.buffers[bidx],
                           bindings_.buffer_sizes[bidx]),
                       "Output binding " << i << " NOT valid GPU mem ctx=" << id_);
    }

    assert(input_index_ >= 0 && bindings_.buffers[input_index_] != nullptr);
    return context_->enqueueV2(bindings_.buffers.data(), stream, nullptr);
}

bool TrtExecutionContext::CopyOutputsToHost(cudaStream_t stream) {
    for (size_t i = 0; i < output_indices_.size(); i++) {
        int binding_idx = output_indices_[i];
        cudaError_t err = cudaMemcpyAsync(
            bindings_.host_output_buffers[i],
            bindings_.buffers[binding_idx],
            bindings_.buffer_sizes[binding_idx],
            cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            LOG_ERROR << "Context " << id_
                      << ": CopyOutputsToHost failed: " << cudaGetErrorString(err);
            return false;
        }
    }
    return true;
}

const float* TrtExecutionContext::HostOutput(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(bindings_.host_output_buffers.size())) {
        return nullptr;
    }
    return static_cast<const float*>(bindings_.host_output_buffers[idx]);
}

TrtContextPool::TrtContextPool() = default;

TrtContextPool::~TrtContextPool() {
    Shutdown();
}

bool TrtContextPool::Init(nvinfer1::ICudaEngine* engine, int num_contexts) {
    if (!engine || num_contexts <= 0) {
        LOG_ERROR << "Invalid engine or num_contexts=" << num_contexts;
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    for (int i = 0; i < num_contexts; i++) {
        auto ctx = std::make_shared<TrtExecutionContext>();
        if (!ctx->Init(engine, i)) {
            LOG_ERROR << "Failed to init context " << i;
            Shutdown();
            return false;
        }
        pool_.push(ctx);
    }

    shutdown_.store(false);
    LOG_INFO << "TrtContextPool initialized: " << num_contexts
             << " contexts ready";
    return true;
}

void TrtContextPool::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_.store(true);

    while (!pool_.empty()) {
        auto ctx = pool_.front();
        pool_.pop();
        ctx->Release();
    }
    cv_.notify_all();
}

TrtExecutionContextPtr TrtContextPool::Acquire(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
        [this]() {
            return shutdown_.load() || !pool_.empty();
        })) {
        LOG_WARN << "TrtContextPool::Acquire timeout (" << timeout_ms << "ms)";
        return nullptr;
    }

    if (shutdown_.load()) {
        return nullptr;
    }

    auto ctx = pool_.front();
    pool_.pop();
    ctx->Bindings()->in_use.store(true);
    return ctx;
}

void TrtContextPool::Release(TrtExecutionContextPtr ctx) {
    if (!ctx) return;
    ctx->Bindings()->in_use.store(false);

    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
        ctx->Release();
        return;
    }
    pool_.push(ctx);
    cv_.notify_one();
}

int TrtContextPool::AvailableCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(pool_.size());
}

TrtEngine::TrtEngine()
    : runtime_(nullptr)
    , engine_(nullptr)
    , input_index_(-1)
    , inited_(false) {}

TrtEngine::~TrtEngine() {
    Release();
}

bool TrtEngine::Init(const TrtEngineConfig& config) {
    if (inited_) {
        Release();
    }

    config_ = config;
    logger_.SetLevel(nvinfer1::ILogger::Severity::kWARNING);

    cudaSetDevice(config_.gpu_device_id);

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        LOG_ERROR << "Failed to create TensorRT runtime";
        return false;
    }

    if (!config_.engine_path.empty()) {
        if (!LoadEngine(config_.engine_path)) {
            LOG_ERROR << "Failed to load engine: " << config_.engine_path;
            return false;
        }
    }

    inited_ = true;
    LOG_INFO << "TrtEngine initialized, engine=" << config_.engine_path
             << ", contexts=" << config_.num_execution_contexts;
    return true;
}

void TrtEngine::Release() {
    context_pool_.Shutdown();

    int nb_bindings = 0;
    input_dims_.clear();
    output_dims_.clear();
    output_names_.clear();
    output_indices_.clear();
    input_index_ = -1;

    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }

    inited_ = false;
}

bool TrtEngine::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR << "Failed to open engine file: " << engine_path;
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        LOG_ERROR << "Failed to read engine file";
        return false;
    }
    file.close();

    return DeserializeEngine(data.data(), size);
}

bool TrtEngine::DeserializeEngine(const void* data, size_t size) {
    engine_ = runtime_->deserializeCudaEngine(data, size);
    if (!engine_) {
        LOG_ERROR << "Failed to deserialize CUDA engine";
        return false;
    }

    int nb_bindings = engine_->getNbBindings();
    for (int i = 0; i < nb_bindings; i++) {
        if (engine_->bindingIsInput(i)) {
            input_index_ = i;
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            input_dims_.clear();
            for (int j = 0; j < dims.nbDims; j++) {
                input_dims_.push_back(dims.d[j]);
            }
        } else {
            output_indices_.push_back(i);
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            std::vector<int> dim_vec;
            for (int j = 0; j < dims.nbDims; j++) {
                dim_vec.push_back(dims.d[j]);
            }
            output_dims_.push_back(dim_vec);
            output_names_.push_back(engine_->getBindingName(i));
        }
    }

    LOG_INFO << "Binding layout decoded: " << nb_bindings << " bindings, "
             << config_.num_execution_contexts << " contexts";

    // ---------- [CRITICAL] GPU Memory Budget Check ----------
    // 根因复盘：Activation Buffer 被多线程共享会触发显存踩踏与硬件死锁。
    // 修复措施：每个 Context 拥有全套独立 GPU 缓冲，必须确保总预算 < 70% 总显存。
    {
        size_t per_ctx_bytes = 0;
        for (size_t b = 0; b < bindings_buffer_sizes.size(); b++) {
            per_ctx_bytes += bindings_buffer_sizes[b];
        }

        size_t estimated_total = per_ctx_bytes * config_.num_execution_contexts;
        estimated_total += (256 * 1024 * 1024);

        auto snap = GpuHealthMonitor::Instance().Snapshot(config_.gpu_device_id);
        if (snap.valid) {
            size_t safe_budget = static_cast<size_t>(snap.total_bytes * 0.70);

            LOG_INFO << "[GPU-MEM-BUDGET] Context count=" << config_.num_execution_contexts
                     << ", Per-ctx=" << (per_ctx_bytes / 1024.0 / 1024.0) << "MB"
                     << ", Estimated total=" << (estimated_total / 1024.0 / 1024.0) << "MB"
                     << ", GPU total=" << (snap.total_bytes / 1024.0 / 1024.0) << "MB"
                     << ", Safe budget(70%)=" << (safe_budget / 1024.0 / 1024.0) << "MB";

            PORT_AI_ASSERT(estimated_total <= safe_budget,
                           "GPU memory budget EXCEEDED! est="
                           << (estimated_total / 1024.0 / 1024.0)
                           << "MB > safe=" << (safe_budget / 1024.0 / 1024.0)
                           << "MB. Reduce num_execution_contexts or model size.");

            if (snap.free_bytes < estimated_total) {
                LOG_WARN << "[GPU-MEM-BUDGET] Free memory ("
                         << (snap.free_bytes / 1024.0 / 1024.0)
                         << "MB) is below estimated requirement ("
                         << (estimated_total / 1024.0 / 1024.0)
                         << "MB) - OOM is HIGHLY likely!";
            }
        } else {
            LOG_WARN << "[GPU-MEM-BUDGET] Unable to snapshot GPU memory; skipping budget check.";
        }
    }

    if (!context_pool_.Init(engine_, config_.num_execution_contexts)) {
        LOG_ERROR << "Failed to initialize context pool";
        return false;
    }

    LOG_INFO << "Engine deserialized: " << nb_bindings << " bindings, "
             << config_.num_execution_contexts << " execution contexts";
    return true;
}

InferenceResultPtr TrtEngine::RunInference(const FramePtr& frame,
                                            const LetterboxInfo& letterbox_info,
                                            cudaStream_t stream) {
    auto ctx = AcquireContext(5000);
    if (!ctx) {
        LOG_ERROR << "RunInference: Failed to acquire context (pool exhausted)";
        return nullptr;
    }
    auto result = RunInferenceWithContext(ctx, frame, letterbox_info, stream);
    ReleaseContext(std::move(ctx));
    return result;
}

bool TrtEngine::SaveEngine(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(static_cast<const char*>(data), size);
    file.close();
    return true;
}

TrtExecutionContextPtr TrtEngine::AcquireContext(int timeout_ms) {
    return context_pool_.Acquire(timeout_ms);
}

void TrtEngine::ReleaseContext(TrtExecutionContextPtr ctx) {
    context_pool_.Release(std::move(ctx));
}

InferenceResultPtr TrtEngine::RunInferenceWithContext(
    TrtExecutionContextPtr ctx,
    const FramePtr& frame,
    const LetterboxInfo& letterbox_info,
    cudaStream_t stream) {

    if (!inited_ || !frame || !engine_ || !ctx) {
        return nullptr;
    }

    int ctx_id = ctx->Id();
    ScopedTimer total_timer;
    auto result = std::make_shared<InferenceResult>();
    result->timestamp_us = frame->timestamp_us;
    result->camera_id = frame->camera_id;
    result->camera_type = frame->camera_type;
    result->context_id = ctx_id;

    cudaStream_t use_stream = stream ? stream : nullptr;

    ScopedTimer infer_timer;
    bool infer_ok = ctx->EnqueueV2(use_stream);
    if (!infer_ok) {
        LOG_ERROR << "Context " << ctx_id << ": enqueueV2 FAILED";
        return nullptr;
    }

    if (use_stream != nullptr) {
        cudaStreamSynchronize(use_stream);
    } else {
        cudaDeviceSynchronize();
    }
    result->inference_time_ms = infer_timer.ElapsedMillis();

    ScopedTimer postprocess_timer;

    ctx->CopyOutputsToHost(use_stream);
    if (use_stream != nullptr) {
        cudaStreamSynchronize(use_stream);
    } else {
        cudaDeviceSynchronize();
    }

    const auto& out_dims = ctx->OutputDims();
    if (!out_dims.empty() && out_dims[0].size() >= 2) {
        int num_detections = 0;
        if (out_dims[0].size() == 3) {
            num_detections = out_dims[0][1];
        } else if (out_dims[0].size() == 2) {
            num_detections = out_dims[0][0];
        }

        result->detections.reserve(num_detections);

        const float* output_data = ctx->HostOutput(0);
        size_t out_size = ctx->OutputBufferSize(0) / sizeof(float);
        if (output_data == nullptr || out_size == 0) {
            LOG_WARN << "Context " << ctx_id << ": No output data";
            return result;
        }

        int cols = out_dims[0].size() >= 3 ? out_dims[0][2] : 5;

        for (int i = 0; i < num_detections; i++) {
            int base = i * cols;
            if (static_cast<size_t>(base + 4) >= out_size) break;

            Detection det;
            if (cols >= 5) {
                float cx = output_data[base + 0];
                float cy = output_data[base + 1];
                float w = output_data[base + 2];
                float h = output_data[base + 3];
                det.confidence = output_data[base + 4];

                det.x1 = cx - w / 2.0f;
                det.y1 = cy - h / 2.0f;
                det.x2 = cx + w / 2.0f;
                det.y2 = cy + h / 2.0f;

                det.x1 = (det.x1 - letterbox_info.pad_left) / letterbox_info.scale;
                det.y1 = (det.y1 - letterbox_info.pad_top) / letterbox_info.scale;
                det.x2 = (det.x2 - letterbox_info.pad_left) / letterbox_info.scale;
                det.y2 = (det.y2 - letterbox_info.pad_top) / letterbox_info.scale;

                int orig_w = letterbox_info.original_width;
                int orig_h = letterbox_info.original_height;
                det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(orig_w)));
                det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(orig_h)));
                det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(orig_w)));
                det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(orig_h)));

                det.class_id = 0;
                det.class_name = "corner_slot";

                if (det.confidence > 0.01f) {
                    result->detections.push_back(det);
                }
            }
        }
    }

    result->postprocess_time_ms = postprocess_timer.ElapsedMillis();
    return result;
}

}
