#include "streaming/rtsp_reader.h"
#include "common/logger.h"
#include "common/time_util.h"
#include "common/gpu_memory.h"

#include <cstring>
#include <chrono>
#include <thread>

namespace port_ai_gateway {

static enum AVPixelFormat GetHwFormat(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return AV_PIX_FMT_CUDA;
        }
    }
    LOG_WARN << "No CUDA pixel format not found, using sw format";
    return AV_PIX_FMT_NONE;
}

RtspReader::RtspReader(CameraType camera_type, int camera_id)
    : camera_type_(camera_type)
    , camera_id_(camera_id)
    , fmt_ctx_(nullptr)
    , codec_ctx_(nullptr)
    , hw_device_ctx_(nullptr)
    , video_stream_index_(-1)
    , width_(0)
    , height_(0)
    , fps_(0)
    , reconnect_count_(0) {
}

RtspReader::~RtspReader() {
    Stop();
    CloseStream();
}

bool RtspReader::Init(const RtspConfig& config) {
    if (inited_.load()) {
        return true;
    }
    config_ = config;

    if (!frame_queue_) {
        frame_queue_ = std::make_shared<ThreadSafeQueue<FramePtr>>(10);
    }

    inited_.store(true);
    return true;
}

bool RtspReader::Start() {
    if (!inited_.load()) {
        LOG_ERROR << "RtspReader not initialized";
        return false;
    }

    if (running_.load()) {
        LOG_WARN << "RtspReader already running";
        return true;
    }

    if (!OpenStream()) {
        LOG_ERROR << "Failed to open RTSP stream: " << config_.url;
        return false;
    }

    running_.store(true);
    read_thread_ = std::thread(&RtspReader::ReadLoop, this);

    LOG_INFO << "RtspReader started for camera " << camera_id_
             << " (" << (camera_type_ == CameraType::VISIBLE ? "VISIBLE" : "INFRARED")
             << ": " << config_.url;

    return true;
}

void RtspReader::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (frame_queue_) {
        frame_queue_->Stop();
    }

    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    CloseStream();
    LOG_INFO << "RtspReader stopped for camera " << camera_id_;
}

FramePtr RtspReader::GetFrame(int timeout_ms) {
    if (!frame_queue_) {
        return nullptr;
    }
    FramePtr frame;
    if (frame_queue_->WaitPop(frame, timeout_ms)) {
        return frame;
    }
    return nullptr;
}

bool RtspReader::OpenStream() {
    CloseStream();

    AVDictionary* options = nullptr;

    av_dict_set(&options, "rtsp_transport", config_.transport.c_str(), 0);
    av_dict_set(&options, "stimeout", std::to_string(config_.timeout_ms * 1000).c_str(), 0);
    av_dict_set(&options, "max_delay", "500000", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);

    int ret = avformat_open_input(&fmt_ctx_, config_.url.c_str(), nullptr, &options);
    av_dict_free(&options);

    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR << "Failed to open input: " << errbuf;
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        LOG_ERROR << "Failed to find stream info";
        CloseStream();
        return false;
    }

    video_stream_index_ = -1;
    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index_ = i;
            break;
        }
    }

    if (video_stream_index_ < 0) {
        LOG_ERROR << "No video stream found";
        CloseStream();
        return false;
    }

    AVStream* video_stream = fmt_ctx_->streams[video_stream_index_];
    AVCodecParameters* codecpar = video_stream->codecpar;

    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        LOG_ERROR << "Unsupported codec";
        CloseStream();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR << "Failed to alloc codec context";
        CloseStream();
        return false;
    }

    avcodec_parameters_to_context(codec_ctx_, codecpar);

    if (video_stream->avg_frame_rate.den != 0) {
        fps_ = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    }
    time_base_ = video_stream->time_base;
    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;

    if (config_.use_hw_decode) {
        if (!InitHardwareDecoder()) {
            LOG_WARN << "Hardware decoder init failed, falling back to software";
        }
    }

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        LOG_ERROR << "Failed to open codec";
        CloseStream();
        return false;
    }

    LOG_INFO << "Stream opened: " << width_ << "x" << height_
             << " @" << fps_ << "fps, codec: " << codec->name
             << ", hw_decode: " << (config_.use_hw_decode ? "yes" : "no");

    return true;
}

bool RtspReader::InitHardwareDecoder() {
#ifdef USE_NVDEC
    AVBufferRef* hw_device_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                     nullptr, nullptr, 0);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_WARN << "Failed to create CUDA HW device: " << errbuf;
        return false;
    }

    hw_device_ctx_ = hw_device_ctx;
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format = GetHwFormat;

    LOG_INFO << "NVDEC hardware decoder initialized";
    return true;
#else
    LOG_WARN << "NVDEC not compiled out";
    return false;
#endif
}

void RtspReader::CloseStream() {
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_index_ = -1;
}

void RtspReader::ReadLoop() {
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();

    if (!pkt || !frame || !sw_frame) {
        LOG_ERROR << "Failed to allocate frames";
        av_packet_free(&pkt);
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        running_.store(false);
        return;
    }

    while (running_.load()) {
        int ret = av_read_frame(fmt_ctx_, pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                LOG_WARN << "Stream ended or interrupted, attempting reconnect";
                Reconnect();
                av_packet_unref(pkt);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_WARN << "Read frame error: " << errbuf;
            Reconnect();
            av_packet_unref(pkt);
            continue;
        }

        if (pkt->stream_index != video_stream_index_) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt);
        av_packet_unref(pkt);

        if (ret < 0) {
            LOG_WARN << "Send packet error";
            continue;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                LOG_WARN << "Receive frame error";
                break;
            }

            AVFrame* output_frame = frame;

            if (frame->format == AV_PIX_FMT_CUDA) {
                ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                if (ret < 0) {
                    LOG_WARN << "Failed to transfer frame from GPU";
                    av_frame_unref(frame);
                    continue;
                }
                output_frame = sw_frame;
            }

            FramePtr out_frame = std::make_shared<Frame>();
            out_frame->camera_type = camera_type_;
            out_frame->camera_id = camera_id_;
            out_frame->size_2d.width = output_frame->width;
            out_frame->size_2d.height = output_frame->height;
            out_frame->pts = output_frame->pts;
            out_frame->timestamp_us = TimeUtil::NowMicros();

            int width = output_frame->width;
            int height = output_frame->height;

            if (frame->format == AV_PIX_FMT_CUDA && output_frame == frame) {
                out_frame->format = PixelFormat::NV12;
                out_frame->is_gpu_memory = true;

                size_t y_size = width * height;
                size_t uv_size = width * height / 2;
                size_t total_size = y_size + uv_size;

                uint8_t* gpu_buf = nullptr;
                cudaError_t cuda_err = cudaMalloc(&gpu_buf, total_size);
                if (cuda_err != cudaSuccess) {
                    LOG_WARN << "cudaMalloc failed: " << cudaGetErrorString(cuda_err);
                    av_frame_unref(frame);
                    continue;
                }

                cudaMemcpy(gpu_buf, output_frame->data[0], y_size, cudaMemcpyDeviceToDevice);
                cudaMemcpy(gpu_buf + y_size, output_frame->data[1], uv_size, cudaMemcpyDeviceToDevice);

                out_frame->data = gpu_buf;
                out_frame->size = total_size;
            } else {
                out_frame->format = PixelFormat::YUV420P;
                out_frame->is_gpu_memory = false;

                size_t y_size = width * height;
                size_t uv_size = y_size / 4;
                size_t total_size = y_size + uv_size * 2;

                uint8_t* buf = static_cast<uint8_t*>(malloc(total_size));
                out_frame->data = buf;
                out_frame->size = total_size;

                memcpy(buf, output_frame->data[0], y_size);
                memcpy(buf + y_size, output_frame->data[1], uv_size);
                memcpy(buf + y_size + uv_size, output_frame->data[2], uv_size);
            }

            if (frame_queue_) {
                if (!frame_queue_->TryPush(out_frame)) {
                    LOG_WARN << "Frame queue full, dropping frame";
                }
            }

            av_frame_unref(frame);
            if (output_frame == sw_frame) {
                av_frame_unref(sw_frame);
            }
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
}

bool RtspReader::Reconnect() {
    std::lock_guard<std::mutex> lock(reconnect_mutex_);

    if (!running_.load()) {
        return false;
    }

    if (reconnect_count_ >= config_.max_reconnect) {
        LOG_ERROR << "Max reconnection attempts reached";
        return false;
    }

    reconnect_count_++;
    LOG_INFO << "Reconnecting... attempt " << reconnect_count_
             << "/" << config_.max_reconnect;

    CloseStream();

    std::this_thread::sleep_for(
        std::chrono::milliseconds(config_.reconnect_interval_ms));

    if (!OpenStream()) {
        LOG_ERROR << "Reconnection failed";
        return false;
    }

    reconnect_count_ = 0;
    LOG_INFO << "Reconnected successfully";
    return true;
}

}
