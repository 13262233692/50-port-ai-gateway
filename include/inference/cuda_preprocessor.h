#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "common/frame.h"

namespace port_ai_gateway {

struct PreprocessConfig {
    int input_width = 640;
    int input_height = 640;
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;
    float std_r = 0.229f;
    float std_g = 0.224f;
    float std_b = 0.225f;
    bool normalize = true;
    bool bgr_to_rgb = false;
    bool letterbox = true;
    float letterbox_color = 114.0f;
    bool use_float16 = false;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    int original_width = 0;
    int original_height = 0;
};

class CudaPreprocessor {
public:
    CudaPreprocessor();
    ~CudaPreprocessor();

    bool Init(const PreprocessConfig& config);
    void Release();

    bool Process(const FramePtr& input_frame,
                 float* output_gpu_data,
                 LetterboxInfo& letterbox_info,
                 cudaStream_t stream = nullptr);

    const PreprocessConfig& GetConfig() const { return config_; }

    static int GetOutputSize(const PreprocessConfig& config) {
        return config.input_width * config.input_height * 3;
    }

private:
    PreprocessConfig config_;
    bool inited_ = false;

    uint8_t* resize_buffer_ = nullptr;
    size_t resize_buffer_size_ = 0;
};

}
