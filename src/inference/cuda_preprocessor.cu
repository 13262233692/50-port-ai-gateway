#include "inference/cuda_preprocessor.h"
#include "common/logger.h"
#include "common/gpu_memory.h"

#include <cuda_runtime.h>

namespace port_ai_gateway {

__global__ void nv12_to_rgb_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int width, int height, int y_pitch, int uv_pitch,
    uint8_t* __restrict__ rgb,
    int rgb_pitch) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int y_idx = y * y_pitch + x;
        int uv_x = (x >> 1);
        int uv_y = (y >> 1);
        int uv_idx = uv_y * uv_pitch + uv_x * 2;

        float y_val = static_cast<float>(y_plane[y_idx]);
        float u_val = static_cast<float>(uv_plane[uv_idx]);
        float v_val = static_cast<float>(uv_plane[uv_idx + 1]);

        y_val = 1.164f * (y_val - 16.0f);
        u_val -= 128.0f;
        v_val -= 128.0f;

        float r = y_val + 1.596f * v_val;
        float g = y_val - 0.392f * u_val - 0.813f * v_val;
        float b = y_val + 2.017f * u_val;

        r = min(max(r, 0.0f), 255.0f);
        g = min(max(g, 0.0f), 255.0f);
        b = min(max(b, 0.0f), 255.0f);

        int rgb_idx = y * rgb_pitch + x * 3;
        rgb[rgb_idx] = static_cast<uint8_t>(r);
        rgb[rgb_idx + 1] = static_cast<uint8_t>(g);
        rgb[rgb_idx + 2] = static_cast<uint8_t>(b);
    }
}

__global__ void resize_bilinear_kernel(
    const uint8_t* __restrict__ src,
    int src_width, int src_height, int src_pitch,
    uint8_t* __restrict__ dst,
    int dst_width, int dst_height, int dst_pitch,
    int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height) {
        float scale_x = static_cast<float>(src_width) / dst_width;
        float scale_y = static_cast<float>(src_height) / dst_height;

        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;

        src_x = min(max(src_x, 0.0f), src_width - 1.0f);
        src_y = min(max(src_y, 0.0f), src_height - 1.0f);

        int x0 = static_cast<int>(floor(src_x));
        int y0 = static_cast<int>(floor(src_y));
        int x1 = min(x0 + 1, src_width - 1);
        int y1 = min(y0 + 1, src_height - 1);

        float fx = src_x - x0;
        float fy = src_y - y0;

        for (int c = 0; c < channels; c++) {
            float v00 = static_cast<float>(src[y0 * src_pitch + x0 * channels + c]);
            float v10 = static_cast<float>(src[y0 * src_pitch + x1 * channels + c]);
            float v01 = static_cast<float>(src[y1 * src_pitch + x0 * channels + c]);
            float v11 = static_cast<float>(src[y1 * src_pitch + x1 * channels + c]);

            float v0 = v00 * (1 - fx) + v10 * fx;
            float v1 = v01 * (1 - fx) + v11 * fx;
            float v = v0 * (1 - fy) + v1 * fy;

            dst[y * dst_pitch + x * channels + c] = static_cast<uint8_t>(min(max(v, 0.0f), 255.0f));
        }
    }
}

__global__ void letterbox_kernel(
    const uint8_t* __restrict__ src,
    int src_width, int src_height, int src_pitch,
    uint8_t* __restrict__ dst,
    int dst_width, int dst_height, int dst_pitch,
    int pad_left, int pad_top,
    float pad_value,
    int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height) {
        int src_x = x - pad_left;
        int src_y = y - pad_top;

        if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
            int src_idx = src_y * src_pitch + src_x * channels;
            int dst_idx = y * dst_pitch + x * channels;
            for (int c = 0; c < channels; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        } else {
            int dst_idx = y * dst_pitch + x * channels;
            for (int c = 0; c < channels; c++) {
                dst[dst_idx + c] = static_cast<uint8_t>(pad_value);
            }
        }
    }
}

__global__ void normalize_float_kernel(
    const uint8_t* __restrict__ src,
    int width, int height, int pitch,
    float* __restrict__ dst,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    bool bgr_to_rgb) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 3;
        int out_idx = y * width + x;

        float r = static_cast<float>(src[idx]) / 255.0f;
        float g = static_cast<float>(src[idx + 1]) / 255.0f;
        float b = static_cast<float>(src[idx + 2]) / 255.0f;

        if (bgr_to_rgb) {
            float tmp = r;
            r = b;
            b = tmp;
        }

        r = (r - mean_r) / std_r;
        g = (g - mean_g) / std_g;
        b = (b - mean_b) / std_b;

        dst[out_idx] = r;
        dst[out_idx + width * height] = g;
        dst[out_idx + width * height * 2] = b;
    }
}

static void launch_nv12_to_rgb(
    const uint8_t* y_plane, const uint8_t* uv_plane,
    int width, int height, int y_pitch, int uv_pitch,
    uint8_t* rgb, int rgb_pitch,
    cudaStream_t stream) {

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    nv12_to_rgb_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, width, height, y_pitch, uv_pitch, rgb, rgb_pitch);
}

static void launch_resize_bilinear(
    const uint8_t* src, int src_w, int src_h, int src_pitch,
    uint8_t* dst, int dst_w, int dst_h, int dst_pitch,
    int channels, cudaStream_t stream) {

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, src_pitch, dst, dst_w, dst_h, dst_pitch, channels);
}

static void launch_letterbox(
    const uint8_t* src, int src_w, int src_h, int src_pitch,
    uint8_t* dst, int dst_w, int dst_h, int dst_pitch,
    int pad_left, int pad_top, float pad_value,
    int channels, cudaStream_t stream) {

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    letterbox_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, src_pitch, dst, dst_w, dst_h, dst_pitch,
        pad_left, pad_top, pad_value, channels);
}

static void launch_normalize(
    const uint8_t* src, int width, int height, int pitch,
    float* dst,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    bool bgr_to_rgb, cudaStream_t stream) {

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    normalize_float_kernel<<<grid, block, 0, stream>>>(
        src, width, height, pitch, dst,
        mean_r, mean_g, mean_b, std_r, std_g, std_b, bgr_to_rgb);
}

CudaPreprocessor::CudaPreprocessor()
    : inited_(false)
    , resize_buffer_(nullptr)
    , resize_buffer_size_(0) {
}

CudaPreprocessor::~CudaPreprocessor() {
    Release();
}

bool CudaPreprocessor::Init(const PreprocessConfig& config) {
    if (inited_) {
        Release();
    }

    config_ = config;

    size_t max_size = config.input_width * config.input_height * 3 * 4;
    cudaError_t err = cudaMalloc(&resize_buffer_, max_size);
    if (err != cudaSuccess) {
        LOG_ERROR << "Failed to alloc resize buffer: " << cudaGetErrorString(err);
        return false;
    }
    resize_buffer_size_ = max_size;

    inited_ = true;
    LOG_INFO << "CudaPreprocessor initialized: " << config.input_width << "x" << config.input_height;
    return true;
}

void CudaPreprocessor::Release() {
    if (resize_buffer_) {
        cudaFree(resize_buffer_);
        resize_buffer_ = nullptr;
    }
    resize_buffer_size_ = 0;
    inited_ = false;
}

bool CudaPreprocessor::Process(const FramePtr& input_frame,
                                float* output_gpu_data,
                                LetterboxInfo& letterbox_info,
                                cudaStream_t stream) {
    if (!inited_ || !input_frame || !output_gpu_data) {
        return false;
    }

    int src_w = input_frame->size_2d.width;
    int src_h = input_frame->size_2d.height;
    int dst_w = config_.input_width;
    int dst_h = config_.input_height;

    letterbox_info.original_width = src_w;
    letterbox_info.original_height = src_h;

    uint8_t* rgb_buffer = resize_buffer_;
    int rgb_pitch = src_w * 3;

    if (input_frame->format == PixelFormat::NV12) {
        const uint8_t* y_plane = input_frame->data;
        const uint8_t* uv_plane = input_frame->data + src_w * src_h;

        launch_nv12_to_rgb(
            y_plane, uv_plane,
            src_w, src_h, src_w, src_w,
            rgb_buffer, rgb_pitch,
            stream);
    } else if (input_frame->format == PixelFormat::RGB) {
        if (input_frame->is_gpu_memory) {
            cudaMemcpyAsync(rgb_buffer, input_frame->data, src_w * src_h * 3,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(rgb_buffer, input_frame->data, src_w * src_h * 3,
                            cudaMemcpyHostToDevice, stream);
        }
    } else {
        LOG_ERROR << "Unsupported pixel format";
        return false;
    }

    uint8_t* letterbox_buffer = resize_buffer_ + src_w * src_h * 3;
    int letterbox_pitch = dst_w * 3;

    if (config_.letterbox) {
        float scale_w = static_cast<float>(dst_w) / src_w;
        float scale_h = static_cast<float>(dst_h) / src_h;
        float scale = min(scale_w, scale_h);

        int new_w = static_cast<int>(src_w * scale);
        int new_h = static_cast<int>(src_h * scale);

        letterbox_info.scale = scale;
        letterbox_info.pad_left = (dst_w - new_w) / 2;
        letterbox_info.pad_top = (dst_h - new_h) / 2;

        uint8_t* resized_buffer = letterbox_buffer;
        int resized_pitch = new_w * 3;

        launch_resize_bilinear(
            rgb_buffer, src_w, src_h, rgb_pitch,
            resized_buffer, new_w, new_h, resized_pitch,
            3, stream);

        launch_letterbox(
            resized_buffer, new_w, new_h, resized_pitch,
            letterbox_buffer, dst_w, dst_h, letterbox_pitch,
            letterbox_info.pad_left, letterbox_info.pad_top,
            config_.letterbox_color, 3, stream);
    } else {
        letterbox_info.scale = 1.0f;
        letterbox_info.pad_left = 0;
        letterbox_info.pad_top = 0;

        launch_resize_bilinear(
            rgb_buffer, src_w, src_h, rgb_pitch,
            letterbox_buffer, dst_w, dst_h, letterbox_pitch,
            3, stream);
    }

    if (config_.normalize) {
        launch_normalize(
            letterbox_buffer, dst_w, dst_h, letterbox_pitch,
            output_gpu_data,
            config_.mean_r, config_.mean_g, config_.mean_b,
            config_.std_r, config_.std_g, config_.std_b,
            config_.bgr_to_rgb, stream);
    } else {
        cudaMemcpyAsync(output_gpu_data, letterbox_buffer, dst_w * dst_h * 3 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    return true;
}

}
