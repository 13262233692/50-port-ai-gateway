#include "common/gpu_health_monitor.h"
#include "common/logger.h"

#include <cuda_runtime.h>
#include <sstream>

namespace port_ai_gateway {

GpuHealthMonitor& GpuHealthMonitor::Instance() {
    static GpuHealthMonitor instance;
    return instance;
}

GpuMemorySnapshot GpuHealthMonitor::Snapshot(int device_id) {
    GpuMemorySnapshot s;
    int current_dev = -1;

    if (cudaGetDevice(&current_dev) != cudaSuccess) {
        s.valid = false;
        return s;
    }

    if (device_id >= 0 && device_id != current_dev) {
        if (cudaSetDevice(device_id) != cudaSuccess) {
            s.valid = false;
            return s;
        }
    }
    s.device_id = (device_id >= 0) ? device_id : current_dev;

    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        s.valid = false;
        if (device_id >= 0 && device_id != current_dev) {
            cudaSetDevice(current_dev);
        }
        return s;
    }

    s.free_bytes = free;
    s.total_bytes = total;
    s.used_bytes = total - free;
    s.used_percent = total > 0 ? (100.0 * s.used_bytes / total) : 0.0;
    s.valid = true;

    if (device_id >= 0 && device_id != current_dev) {
        cudaSetDevice(current_dev);
    }

    return s;
}

bool GpuHealthMonitor::CheckMemoryPressure(double threshold_percent) {
    auto s = Snapshot();
    if (!s.valid) return false;
    return s.used_percent > threshold_percent;
}

bool GpuHealthMonitor::ValidateDevicePointer(const void* ptr, size_t expected_min_size) {
    if (!ptr) return false;

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        return false;
    }

#if (CUDART_VERSION >= 10000)
    if (attrs.type != cudaMemoryTypeDevice) {
        return false;
    }
#else
    if (attrs.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
#endif

    if (expected_min_size > 0) {
        if (static_cast<size_t>(
#if (CUDART_VERSION >= 10000)
                attrs.allocationSize
#else
                0
#endif
            ) < expected_min_size) {
            return false;
        }
    }

    return true;
}

bool GpuHealthMonitor::ValidateHostPointer(const void* ptr, size_t) {
    if (!ptr) return false;

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        return true;
    }
    (void)attrs;
    return true;
}

void GpuHealthMonitor::LogSnapshot(const char* tag) {
    auto s = Snapshot();
    std::ostringstream oss;
    if (!s.valid) {
        LOG_WARN << "[GPU-Health] " << (tag ? tag : "") << " SNAPSHOT FAILED";
        return;
    }
    LOG_INFO << "[GPU-Health] " << (tag ? tag : "")
             << " Dev" << s.device_id << ": "
             << "Used=" << (s.used_bytes / 1024.0 / 1024.0) << "MB/"
             << (s.total_bytes / 1024.0 / 1024.0) << "MB ("
             << s.used_percent << "%), "
             << "Free=" << (s.free_bytes / 1024.0 / 1024.0) << "MB";
}

bool GpuHealthMonitor::IsGpuHealthy(int device_id) {
    auto s = Snapshot(device_id);
    if (!s.valid) return false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR << "[GPU-Health] Pending CUDA error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

GpuHealthMonitor::HealthStatus GpuHealthMonitor::CheckAll(int device_id) {
    HealthStatus status;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        status.last_cuda_ok = false;
        status.error_message = std::string("CUDA last error: ") + cudaGetErrorString(err);
        return status;
    }

    auto s = Snapshot(device_id);
    if (!s.valid) {
        status.device_ok = false;
        status.error_message = "Failed to get memory snapshot";
        return status;
    }

    if (s.used_percent > 95.0) {
        status.memory_ok = false;
        std::ostringstream oss;
        oss << "Memory pressure critical: " << s.used_percent << "% used";
        status.error_message = oss.str();
    }

    return status;
}

}
