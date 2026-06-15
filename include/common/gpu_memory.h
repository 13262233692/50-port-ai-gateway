#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace port_ai_gateway {

class GpuMemoryManager {
public:
    static GpuMemoryManager& Instance() {
        static GpuMemoryManager instance;
        return instance;
    }

    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    static cudaError_t AllocDevice(void** ptr, size_t size);
    static cudaError_t AllocHost(void** ptr, size_t size, unsigned int flags = cudaHostAllocDefault);
    static cudaError_t AllocPinned(void** ptr, size_t size);
    static cudaError_t FreeDevice(void* ptr);
    static cudaError_t FreeHost(void* ptr);

    static cudaError_t MemcpyHtoD(void* dst, const void* src, size_t size,
                                  cudaStream_t stream = nullptr);
    static cudaError_t MemcpyDtoH(void* dst, const void* src, size_t size,
                                  cudaStream_t stream = nullptr);
    static cudaError_t MemcpyDtoD(void* dst, const void* src, size_t size,
                                  cudaStream_t stream = nullptr);

    static cudaError_t MemsetD(void* ptr, int value, size_t size,
                               cudaStream_t stream = nullptr);

    int GetDeviceCount() const;
    int GetCurrentDevice() const;
    bool SetDevice(int device_id);

    size_t GetTotalMemory(int device_id = -1) const;
    size_t GetFreeMemory(int device_id = -1) const;

private:
    GpuMemoryManager() = default;
    ~GpuMemoryManager() = default;
};

class GpuBuffer {
public:
    GpuBuffer();
    explicit GpuBuffer(size_t size);
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    bool Allocate(size_t size);
    void Release();

    void* Data() const { return data_; }
    size_t Size() const { return size_; }
    bool IsValid() const { return data_ != nullptr && size_ > 0; }

    void CopyFromHost(const void* host_data, size_t size, cudaStream_t stream = nullptr);
    void CopyToHost(void* host_data, size_t size, cudaStream_t stream = nullptr) const;

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

class PinnedBuffer {
public:
    PinnedBuffer();
    explicit PinnedBuffer(size_t size);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    bool Allocate(size_t size);
    void Release();

    void* Data() const { return data_; }
    size_t Size() const { return size_; }
    bool IsValid() const { return data_ != nullptr && size_ > 0; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

}
