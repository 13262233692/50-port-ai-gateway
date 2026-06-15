#include "common/gpu_memory.h"
#include <stdexcept>
#include <cuda_runtime.h>

namespace port_ai_gateway {

cudaError_t GpuMemoryManager::AllocDevice(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t GpuMemoryManager::AllocHost(void** ptr, size_t size, unsigned int flags) {
    return cudaHostAlloc(ptr, size, flags);
}

cudaError_t GpuMemoryManager::AllocPinned(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocPortable);
}

cudaError_t GpuMemoryManager::FreeDevice(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    return cudaFree(ptr);
}

cudaError_t GpuMemoryManager::FreeHost(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    return cudaFreeHost(ptr);
}

cudaError_t GpuMemoryManager::MemcpyHtoD(void* dst, const void* src, size_t size,
                                          cudaStream_t stream) {
    if (stream != nullptr) {
        return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    }
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

cudaError_t GpuMemoryManager::MemcpyDtoH(void* dst, const void* src, size_t size,
                                          cudaStream_t stream) {
    if (stream != nullptr) {
        return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    }
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

cudaError_t GpuMemoryManager::MemcpyDtoD(void* dst, const void* src, size_t size,
                                          cudaStream_t stream) {
    if (stream != nullptr) {
        return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

cudaError_t GpuMemoryManager::MemsetD(void* ptr, int value, size_t size,
                                         cudaStream_t stream) {
    if (stream != nullptr) {
        return cudaMemsetAsync(ptr, value, size, stream);
    }
    return cudaMemset(ptr, value, size);
}

int GpuMemoryManager::GetDeviceCount() const {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

int GpuMemoryManager::GetCurrentDevice() const {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return -1;
    }
    return device;
}

bool GpuMemoryManager::SetDevice(int device_id) {
    return cudaSetDevice(device_id) == cudaSuccess;
}

size_t GpuMemoryManager::GetTotalMemory(int device_id) const {
    int current_device = GetCurrentDevice();
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    size_t total = 0;
    size_t free = 0;
    cudaMemGetInfo(&free, &total);
    if (device_id >= 0 && device_id != current_device) {
        cudaSetDevice(current_device);
    }
    return total;
}

size_t GpuMemoryManager::GetFreeMemory(int device_id) const {
    int current_device = GetCurrentDevice();
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    size_t total = 0;
    size_t free = 0;
    cudaMemGetInfo(&free, &total);
    if (device_id >= 0 && device_id != current_device) {
        cudaSetDevice(current_device);
    }
    return free;
}

GpuBuffer::GpuBuffer() : data_(nullptr), size_(0) {}

GpuBuffer::GpuBuffer(size_t size) : data_(nullptr), size_(0) {
    Allocate(size);
}

GpuBuffer::~GpuBuffer() {
    Release();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
        Release();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool GpuBuffer::Allocate(size_t size) {
    Release();
    if (size == 0) {
        return false;
    }
    cudaError_t err = cudaMalloc(&data_, size);
    if (err != cudaSuccess) {
        data_ = nullptr;
        return false;
    }
    size_ = size;
    return true;
}

void GpuBuffer::Release() {
    if (data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }
    size_ = 0;
}

void GpuBuffer::CopyFromHost(const void* host_data, size_t size, cudaStream_t stream) {
    if (data_ == nullptr || host_data == nullptr || size == 0) {
        return;
    }
    size_t copy_size = (size < size_) ? size : size_;
    if (stream != nullptr) {
        cudaMemcpyAsync(data_, host_data, copy_size, cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(data_, host_data, copy_size, cudaMemcpyHostToDevice);
    }
}

void GpuBuffer::CopyToHost(void* host_data, size_t size, cudaStream_t stream) const {
    if (data_ == nullptr || host_data == nullptr || size == 0) {
        return;
    }
    size_t copy_size = (size < size_) ? size : size_;
    if (stream != nullptr) {
        cudaMemcpyAsync(host_data, data_, copy_size, cudaMemcpyDeviceToHost, stream);
    } else {
        cudaMemcpy(host_data, data_, copy_size, cudaMemcpyDeviceToHost);
    }
}

PinnedBuffer::PinnedBuffer() : data_(nullptr), size_(0) {}

PinnedBuffer::PinnedBuffer(size_t size) : data_(nullptr), size_(0) {
    Allocate(size);
}

PinnedBuffer::~PinnedBuffer() {
    Release();
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        Release();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool PinnedBuffer::Allocate(size_t size) {
    Release();
    if (size == 0) {
        return false;
    }
    cudaError_t err = cudaHostAlloc(&data_, size, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        data_ = nullptr;
        return false;
    }
    size_ = size;
    return true;
}

void PinnedBuffer::Release() {
    if (data_ != nullptr) {
        cudaFreeHost(data_);
        data_ = nullptr;
    }
    size_ = 0;
}

}
