#pragma once

#include <cstddef>
#include <string>
#include <cuda_runtime.h>

#include "common/logger.h"

namespace port_ai_gateway {

#define PORT_AI_ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            LOG_FATAL << "ASSERT FAILED: " #cond \
                      << " | " << __VA_ARGS__ \
                      << " [" << __FILE__ << ":" << __LINE__ << "]"; \
            std::abort(); \
        } \
    } while (0)

#define PORT_AI_CHECK_CUDA(err, ...) \
    do { \
        cudaError_t _e = (err); \
        if (_e != cudaSuccess) { \
            LOG_FATAL << "CUDA ERROR: " << cudaGetErrorString(_e) \
                      << " (" << static_cast<int>(_e) << ")" \
                      << " | " << __VA_ARGS__ \
                      << " [" << __FILE__ << ":" << __LINE__ << "]"; \
            std::abort(); \
        } \
    } while (0)

#define PORT_AI_CHECK_CUDA_WARN(err, ...) \
    do { \
        cudaError_t _e = (err); \
        if (_e != cudaSuccess) { \
            LOG_WARN << "CUDA WARNING: " << cudaGetErrorString(_e) \
                     << " (" << static_cast<int>(_e) << ")" \
                     << " | " << __VA_ARGS__ \
                     << " [" << __FILE__ << ":" << __LINE__ << "]"; \
        } \
    } while (0)

struct GpuMemorySnapshot {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    size_t used_bytes = 0;
    double used_percent = 0.0;
    int device_id = -1;
    bool valid = false;
};

class GpuHealthMonitor {
public:
    static GpuHealthMonitor& Instance();

    GpuMemorySnapshot Snapshot(int device_id = -1);

    bool CheckMemoryPressure(double threshold_percent = 90.0);

    bool ValidateDevicePointer(const void* ptr, size_t expected_min_size = 0);

    bool ValidateHostPointer(const void* ptr, size_t expected_min_size = 0);

    void LogSnapshot(const char* tag = nullptr);

    bool IsGpuHealthy(int device_id = -1);

    struct HealthStatus {
        bool memory_ok = true;
        bool device_ok = true;
        bool last_cuda_ok = true;
        std::string error_message;
    };

    HealthStatus CheckAll(int device_id = -1);

private:
    GpuHealthMonitor() = default;
    ~GpuHealthMonitor() = default;
};

class ScopedGpuHealthCheck {
public:
    explicit ScopedGpuHealthCheck(const char* name = nullptr,
                                  double memory_threshold = 90.0)
        : name_(name)
        , start_snapshot_(GpuHealthMonitor::Instance().Snapshot()) {}

    ~ScopedGpuHealthCheck() {
        GpuMemorySnapshot end = GpuHealthMonitor::Instance().Snapshot();
        if (!end.valid || !start_snapshot_.valid) return;

        int64_t leak = static_cast<int64_t>(end.used_bytes) -
                        static_cast<int64_t>(start_snapshot_.used_bytes);

        if (std::abs(leak) > 10 * 1024 * 1024) {
            LOG_WARN << "[GPU-Health] " << (name_ ? name_ : "(anon)")
                     << ": Δused=" << (leak / 1024.0 / 1024.0) << "MB"
                     << ", start=" << (start_snapshot_.used_percent) << "%"
                     << ", end=" << end.used_percent << "%";
        }
    }

private:
    const char* name_;
    GpuMemorySnapshot start_snapshot_;
};

#define SCOPED_GPU_HEALTH_CHECK(tag) \
    ::port_ai_gateway::ScopedGpuHealthCheck _gpu_health_scope_##__LINE__(tag)

}
