#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/time.h>
}

#include "common/frame.h"
#include "common/thread_safe_queue.h"

namespace port_ai_gateway {

struct RtspConfig {
    std::string url;
    std::string transport = "tcp";
    int timeout_ms = 5000;
    int max_reconnect = 5;
    int reconnect_interval_ms = 3000;
    bool use_hw_decode = true;
    std::string hw_device_type = "cuda";
};

class RtspReader {
public:
    explicit RtspReader(CameraType camera_type, int camera_id);
    ~RtspReader();

    bool Init(const RtspConfig& config);
    bool Start();
    void Stop();
    bool IsRunning() const { return running_.load(); }

    FramePtr GetFrame(int timeout_ms = 100);
    bool HasFrame() const { return frame_queue_ && !frame_queue_->Empty(); }

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    int GetFps() const { return fps_; }

    CameraType GetCameraType() const { return camera_type_; }
    int GetCameraId() const { return camera_id_; }

    void SetOutputQueue(std::shared_ptr<ThreadSafeQueue<FramePtr>> queue) {
        frame_queue_ = queue;
    }

private:
    void ReadLoop();
    bool OpenStream();
    void CloseStream();
    bool Reconnect();

    bool InitHardwareDecoder();
    AVCodecContext* CreateDecoderContext();

    CameraType camera_type_;
    int camera_id_;
    RtspConfig config_;

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    int video_stream_index_ = -1;

    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;
    AVRational time_base_;

    std::atomic<bool> running_{false};
    std::atomic<bool> inited_{false};
    std::thread read_thread_;

    std::shared_ptr<ThreadSafeQueue<FramePtr>> frame_queue_;

    int reconnect_count_ = 0;
    std::mutex reconnect_mutex_;
};

}
