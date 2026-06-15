#include "inference/trt_engine.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace port_ai_gateway {

void Logger::log(Severity severity, const char* msg) noexcept {
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

TrtEngine::TrtEngine()
    : runtime_(nullptr)
    , engine_(nullptr)
    , context_(nullptr)
    , inited_(false)
    , input_index_(-1) {
}

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
    LOG_INFO << "TrtEngine initialized with engine: " << config_.engine_path;
    return true;
}

void TrtEngine::Release() {
    std::lock_guard<std::mutex> lock(mutex_);

    FreeBuffers();

    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }

    input_dims_.clear();
    output_dims_.clear();
    output_names_.clear();
    output_indices_.clear();
    buffers_.clear();
    buffer_sizes_.clear();

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

    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR << "Failed to create execution context";
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

    if (!AllocateBuffers()) {
        LOG_ERROR << "Failed to allocate buffers";
        return false;
    }

    LOG_INFO << "Engine deserialized successfully, "
             << nb_bindings << " bindings, "
             << "input dims: [";
    for (size_t i = 0; i < input_dims_.size(); i++) {
        if (i > 0) LOG_INFO << ", ";
        LOG_INFO << input_dims_[i];
    }
    LOG_INFO << "]";

    return true;
}

bool TrtEngine::AllocateBuffers() {
    int nb_bindings = engine_->getNbBindings();
    buffers_.resize(nb_bindings, nullptr);
    buffer_sizes_.resize(nb_bindings, 0);

    for (int i = 0; i < nb_bindings; i++) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine_->getBindingDataType(i);

        size_t element_size = 0;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                element_size = 4;
                break;
            case nvinfer1::DataType::kHALF:
                element_size = 2;
                break;
            case nvinfer1::DataType::kINT8:
                element_size = 1;
                break;
            case nvinfer1::DataType::kINT32:
                element_size = 4;
                break;
            case nvinfer1::DataType::kBOOL:
                element_size = 1;
                break;
            default:
                element_size = 4;
                break;
        }

        size_t size = element_size;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }

        cudaError_t err = cudaMalloc(&buffers_[i], size);
        if (err != cudaSuccess) {
            LOG_ERROR << "Failed to allocate buffer " << i << ": " << cudaGetErrorString(err);
            FreeBuffers();
            return false;
        }
        buffer_sizes_[i] = size;
    }

    return true;
}

void TrtEngine::FreeBuffers() {
    for (size_t i = 0; i < buffers_.size(); i++) {
        if (buffers_[i] != nullptr) {
            cudaFree(buffers_[i]);
            buffers_[i] = nullptr;
        }
    }
    buffers_.clear();
    buffer_sizes_.clear();
}

bool TrtEngine::Infer(void** buffers, cudaStream_t stream) {
    if (!context_) {
        return false;
    }

    bool success = false;
    if (stream != nullptr) {
        success = context_->enqueueV2(buffers, stream, nullptr);
    } else {
        success = context_->executeV2(buffers);
    }

    return success;
}

InferenceResultPtr TrtEngine::RunInference(const FramePtr& frame,
                                            const LetterboxInfo& letterbox_info,
                                            cudaStream_t stream) {
    if (!inited_ || !frame || !IsValid()) {
        return nullptr;
    }

    ScopedTimer total_timer;
    auto result = std::make_shared<InferenceResult>();
    result->timestamp_us = frame->timestamp_us;
    result->camera_id = frame->camera_id;
    result->camera_type = frame->camera_type;

    ScopedTimer preprocess_timer;
    float* input_data = static_cast<float*>(buffers_[input_index_]);
    result->preprocess_time_ms = preprocess_timer.ElapsedMillis();

    ScopedTimer infer_timer;
    if (!Infer(buffers_.data(), stream)) {
        LOG_ERROR << "TensorRT inference failed";
        return nullptr;
    }

    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    result->inference_time_ms = infer_timer.ElapsedMillis();

    ScopedTimer postprocess_timer;

    if (!output_dims_.empty() && output_dims_[0].size() >= 2) {
        int num_detections = 0;
        if (output_dims_[0].size() == 3) {
            num_detections = output_dims_[0][1];
        } else if (output_dims_[0].size() == 2) {
            num_detections = output_dims_[0][0];
        }

        result->detections.reserve(num_detections);

        std::vector<float> output_data(buffer_sizes_[output_indices_[0]] / sizeof(float));
        cudaMemcpy(output_data.data(), buffers_[output_indices_[0]],
                   buffer_sizes_[output_indices_[0]],
                   cudaMemcpyDeviceToHost);

        int num_outputs = static_cast<int>(output_dims_.size());
        if (num_outputs >= 1 && output_dims_[0].size() >= 2) {
            int cols = output_dims_[0].size() >= 3 ? output_dims_[0][2] : 5;

            for (int i = 0; i < num_detections; i++) {
                Detection det;
                if (cols >= 5) {
                    float cx = output_data[i * cols + 0];
                    float cy = output_data[i * cols + 1];
                    float w = output_data[i * cols + 2];
                    float h = output_data[i * cols + 3];
                    det.confidence = output_data[i * cols + 4];

                    det.x1 = cx - w / 2.0f;
                    det.y1 = cy - h / 2.0f;
                    det.x2 = cx + w / 2.0f;
                    det.y2 = cy + h / 2.0f;

                    det.x1 = (det.x1 - letterbox_info.pad_left) / letterbox_info.scale;
                    det.y1 = (det.y1 - letterbox_info.pad_top) / letterbox_info.scale;
                    det.x2 = (det.x2 - letterbox_info.pad_left) / letterbox_info.scale;
                    det.y2 = (det.y2 - letterbox_info.pad_top) / letterbox_info.scale;

                    det.x1 = std::max(0.0f, std::min(det.x1,
                                     static_cast<float>(letterbox_info.original_width)));
                    det.y1 = std::max(0.0f, std::min(det.y1,
                                     static_cast<float>(letterbox_info.original_height)));
                    det.x2 = std::max(0.0f, std::min(det.x2,
                                     static_cast<float>(letterbox_info.original_width)));
                    det.y2 = std::max(0.0f, std::min(det.y2,
                                     static_cast<float>(letterbox_info.original_height)));

                    det.class_id = 0;
                    det.class_name = "corner_slot";

                    if (det.confidence > 0.25f) {
                        result->detections.push_back(det);
                    }
                }
            }
        }
    }

    result->postprocess_time_ms = postprocess_timer.ElapsedMillis();

    return result;
}

bool TrtEngine::SaveEngine(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(static_cast<const char*>(data), size);
    file.close();
    return true;
}

}
