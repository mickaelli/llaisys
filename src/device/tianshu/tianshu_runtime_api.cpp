#include "../runtime_api.hpp"

// Tianshu (InnGrit BI-CUDA) is source-compatible with CUDA
// We use its runtime headers and map them to our LLAISYS API
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::tianshu {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

void streamSynchronize(llaisysStream_t stream) {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
}

void *mallocDevice(size_t size) {
    void *ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tianshu cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    return ptr;
}

void freeDevice(void *ptr) {
    cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tianshu cudaMallocHost failed: ") + cudaGetErrorString(err));
    }
    return ptr;
}

void freeHost(void *ptr) {
    cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    }
    cudaError_t err = cudaMemcpy(dst, src, size, cuda_kind);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tianshu cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    }
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_kind, reinterpret_cast<cudaStream_t>(stream));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tianshu cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::tianshu
