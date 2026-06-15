#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <functional>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st* cudaStream_t;
#endif

namespace port_ai_gateway {

enum class PixelFormat {
    UNKNOWN = 0,
    RGB = 1,
    BGR = 2,
    NV12 = 3,
    YUV420P = 4,
    GRAY = 5
};

enum class CameraType {
    VISIBLE = 0,
    INFRARED = 1
};

struct FrameSize {
    int width = 0;
    int height = 0;
};

struct Frame {
    uint8_t* data = nullptr;
    size_t size = 0;
    FrameSize size_2d;
    PixelFormat format = PixelFormat::UNKNOWN;
    int64_t pts = 0;
    int64_t timestamp_us = 0;
    CameraType camera_type = CameraType::VISIBLE;
    int camera_id = 0;
    bool is_gpu_memory = false;
    cudaStream_t stream = nullptr;

    Frame() = default;

    ~Frame() {
        Release();
    }

    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;

    Frame(Frame&& other) noexcept
        : data(other.data)
        , size(other.size)
        , size_2d(other.size_2d)
        , format(other.format)
        , pts(other.pts)
        , timestamp_us(other.timestamp_us)
        , camera_type(other.camera_type)
        , camera_id(other.camera_id)
        , is_gpu_memory(other.is_gpu_memory)
        , stream(other.stream) {
        other.data = nullptr;
        other.size = 0;
    }

    Frame& operator=(Frame&& other) noexcept {
        if (this != &other) {
            Release();
            data = other.data;
            size = other.size;
            size_2d = other.size_2d;
            format = other.format;
            pts = other.pts;
            timestamp_us = other.timestamp_us;
            camera_type = other.camera_type;
            camera_id = other.camera_id;
            is_gpu_memory = other.is_gpu_memory;
            stream = other.stream;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void Release() {
        if (data != nullptr) {
            if (is_gpu_memory) {
#ifdef __CUDACC__
                cudaFree(data);
#else
                free(data);
#endif
            } else {
                free(data);
            }
            data = nullptr;
            size = 0;
        }
    }
};

using FramePtr = std::shared_ptr<Frame>;

struct StereoFramePair {
    FramePtr visible_frame;
    FramePtr infrared_frame;
    int64_t sync_timestamp_us = 0;
    bool is_synced = false;
};

using StereoFramePairPtr = std::shared_ptr<StereoFramePair>;

}
