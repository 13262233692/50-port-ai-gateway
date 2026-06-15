#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>

#include "common/logger.h"
#include "common/time_util.h"
#include "app/gateway_pipeline.h"

using namespace port_ai_gateway;

std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    LOG_INFO << "Received signal " << sig << ", shutting down...";
    g_running.store(false);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --visible <url>      Visible camera RTSP URL" << std::endl;
    std::cout << "  --infrared <url>     Infrared camera RTSP URL" << std::endl;
    std::cout << "  --engine <path>      TensorRT engine file path" << std::endl;
    std::cout << "  --no-sync            Disable frame synchronization" << std::endl;
    std::cout << "  --visible-only       Use only visible camera" << std::endl;
    std::cout << "  --help               Print this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string visible_url = "rtsp://192.168.1.100:554/visible";
    std::string infrared_url = "rtsp://192.168.1.100:554/infrared";
    std::string engine_path = "corner_slot.trt";
    bool enable_sync = true;
    bool visible_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--visible" && i + 1 < argc) {
            visible_url = argv[++i];
        } else if (arg == "--infrared" && i + 1 < argc) {
            infrared_url = argv[++i];
        } else if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--no-sync") {
            enable_sync = false;
        } else if (arg == "--visible-only") {
            visible_only = true;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    Logger::Instance().SetLevel(LogLevel::INFO);

    LOG_INFO << "========================================";
    LOG_INFO << "  Port AI Gateway - Starting...";
    LOG_INFO << "========================================";
    LOG_INFO << "Visible Camera URL: " << visible_url;
    if (!visible_only) {
        LOG_INFO << "Infrared Camera URL: " << infrared_url;
    }
    LOG_INFO << "TensorRT Engine: " << engine_path;
    LOG_INFO << "Frame Sync: " << (enable_sync ? "Enabled" : "Disabled");
    LOG_INFO << "Mode: " << (visible_only ? "Visible Only" : "Stereo");

    GatewayConfig config;

    config.visible_camera.url = visible_url;
    config.visible_camera.transport = "tcp";
    config.visible_camera.timeout_ms = 5000;
    config.visible_camera.use_hw_decode = true;
    config.visible_camera.max_reconnect = 10;
    config.visible_camera.reconnect_interval_ms = 3000;

    config.infrared_camera.url = infrared_url;
    config.infrared_camera.transport = "tcp";
    config.infrared_camera.timeout_ms = 5000;
    config.infrared_camera.use_hw_decode = true;
    config.infrared_camera.max_reconnect = 10;
    config.infrared_camera.reconnect_interval_ms = 3000;

    config.sync_config.max_timestamp_diff_us = 33000;
    config.sync_config.max_queue_size = 30;
    config.sync_config.sync_timeout_ms = 100;

    config.detector_config.trt_config.engine_path = engine_path;
    config.detector_config.trt_config.max_batch_size = 1;
    config.detector_config.trt_config.use_fp16 = true;
    config.detector_config.trt_config.gpu_device_id = 0;

    config.detector_config.preprocess_config.input_width = 640;
    config.detector_config.preprocess_config.input_height = 640;
    config.detector_config.preprocess_config.normalize = true;
    config.detector_config.preprocess_config.bgr_to_rgb = true;
    config.detector_config.preprocess_config.letterbox = true;
    config.detector_config.preprocess_config.letterbox_color = 114.0f;

    config.detector_config.confidence_threshold = 0.5f;
    config.detector_config.nms_threshold = 0.45f;
    config.detector_config.max_detections = 100;

    config.enable_sync = enable_sync;
    config.use_visible_only = visible_only;

    GatewayPipeline pipeline;

    pipeline.SetResultCallback(
        [](const CornerSlotDetectionResultPtr& result) {
            if (!result) return;

            LOG_INFO << "Detection result - time: " << result->total_time_ms
                     << "ms, preprocess: " << result->preprocess_time_ms
                     << "ms, inference: " << result->inference_time_ms
                     << "ms, postprocess: " << result->postprocess_time_ms
                     << "ms, slots: " << result->corner_slots.size();

            for (size_t i = 0; i < result->corner_slots.size(); i++) {
                const auto& slot = result->corner_slots[i];
                LOG_INFO << "  Corner Slot " << i
                         << ": conf=" << slot.confidence
                         << ", pos=(" << slot.x << ", " << slot.y << ", " << slot.z << ")"
                         << ", size=(" << slot.width << "x" << slot.height << "x" << slot.depth << ")";
            }
        });

    if (!pipeline.Init(config)) {
        LOG_ERROR << "Failed to initialize gateway pipeline";
        return -1;
    }

    if (!pipeline.Start()) {
        LOG_ERROR << "Failed to start gateway pipeline";
        return -1;
    }

    LOG_INFO << "Gateway pipeline running. Press Ctrl+C to stop.";

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    LOG_INFO << "Stopping gateway pipeline...";
    pipeline.Stop();

    LOG_INFO << "Gateway pipeline stopped. Exiting.";
    return 0;
}
