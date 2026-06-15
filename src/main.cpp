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
    config.detector_config.trt_config.num_execution_contexts = 4;

    config.detector_config.preprocess_config.input_width = 640;
    config.detector_config.preprocess_config.input_height = 640;
    config.detector_config.preprocess_config.normalize = true;
    config.detector_config.preprocess_config.bgr_to_rgb = true;
    config.detector_config.preprocess_config.letterbox = true;
    config.detector_config.preprocess_config.letterbox_color = 114.0f;

    config.detector_config.confidence_threshold = 0.5f;
    config.detector_config.nms_threshold = 0.45f;
    config.detector_config.max_detections = 100;
    config.detector_config.num_preprocess_streams = 4;

    config.enable_sync = enable_sync;
    config.use_visible_only = visible_only;

    config.enable_tracking = true;
    config.enable_energy_analysis = true;
    config.enable_plc_output = true;

    config.tracker_config.max_age = 60;
    config.tracker_config.n_init = 3;
    config.tracker_config.max_matching_distance = 2.0f;
    config.tracker_config.kf_config.process_noise_pos = 0.01f;
    config.tracker_config.kf_config.process_noise_vel = 0.5f;
    config.tracker_config.kf_config.measurement_noise = 0.1f;

    config.energy_config.spreader_mass_kg = 35000.0f;
    config.energy_config.rope_length_m = 15.0f;
    config.energy_config.energy_warning_threshold_j = 5000.0f;
    config.energy_config.energy_danger_threshold_j = 15000.0f;
    config.energy_config.energy_critical_threshold_j = 30000.0f;
    config.energy_config.speed_warning_m_s = 1.5f;
    config.energy_config.speed_danger_m_s = 2.5f;
    config.energy_config.speed_critical_m_s = 4.0f;
    config.energy_config.amplitude_warning_m = 0.5f;
    config.energy_config.amplitude_danger_m = 1.2f;
    config.energy_config.amplitude_critical_m = 2.0f;

    config.plc_config.enable_emergency_stop = true;
    config.plc_config.enable_speed_limit = true;
    config.plc_config.enable_swing_damping = true;
    config.plc_config.min_interval_between_same_type_ms = 500;

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

    pipeline.SetTrackingCallback(
        [](const std::vector<CornerSlotTrackPtr>& tracks) {
            if (tracks.empty()) return;
            LOG_DEBUG << "[Tracker] " << tracks.size() << " active tracks";
            for (auto& t : tracks) {
                auto pos = t->Position();
                auto vel = t->Velocity();
                LOG_DEBUG << "  Track " << t->Id()
                          << ": pos=(" << pos.x << "," << pos.y << "," << pos.z << ")"
                          << " |v|=" << vel.Norm() << "m/s"
                          << " hits=" << t->HitStreak();
            }
        });

    pipeline.SetEnergyStatusCallback(
        [](const std::vector<TrackEnergyStatus>& statuses) {
            for (auto& s : statuses) {
                if (s.safety_level >= SafetyLevel::WARNING) {
                    const char* names[] = {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"};
                    int idx = static_cast<int>(s.safety_level);
                    LOG_WARN << "[ENERGY] Track " << s.track_id
                             << " Level=" << names[idx]
                             << " E=" << s.current_energy_j << "J"
                             << " V=" << s.current_speed_m_s << "m/s"
                             << " Amp=" << s.current_amplitude_m << "m"
                             << " Period=" << s.swing_period_s << "s"
                             << " Cycles=" << s.completed_cycles;
                }
            }
        });

    pipeline.SetPlcCommandCallback(
        [](const PlcCommand& cmd) {
            const char* type_names[] = {
                "NONE", "SPEED_LIMIT", "SWING_DAMPING",
                "EMERGENCY_STOP", "HOIST_HOLD", "TROLLEY_SLOW", "ACK"
            };
            const char* prio_names[] = {"LOW", "NORMAL", "HIGH", "CRITICAL"};
            int t = static_cast<int>(cmd.type);
            int p = static_cast<int>(cmd.priority);

            if (cmd.type == PlcCommandType::EMERGENCY_STOP) {
                LOG_FATAL << "[PLC-ESTOP] EMERGENCY STOP issued! ID=" << cmd.command_id
                          << " track=" << cmd.track_id
                          << " energy=" << cmd.swing_energy_joules << "J"
                          << " reason: " << cmd.reason;
            } else if (cmd.priority >= PlcCommandPriority::HIGH) {
                LOG_ERROR << "[PLC] " << type_names[t]
                          << " prio=" << prio_names[p]
                          << " ID=" << cmd.command_id
                          << " speed_limit=" << (cmd.speed_limit_ratio * 100) << "%"
                          << " damping=" << cmd.damping_gain
                          << " track=" << cmd.track_id
                          << " reason: " << cmd.reason;
            } else {
                LOG_INFO << "[PLC] " << type_names[t]
                         << " prio=" << prio_names[p]
                         << " ID=" << cmd.command_id
                         << " speed_limit=" << (cmd.speed_limit_ratio * 100) << "%";
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
