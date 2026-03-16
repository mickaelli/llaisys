#include "nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_err));
    }

    cublasStatus_t status = cublasCreate(&_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed with status " + std::to_string(static_cast<int>(status)));
    }
}

Resource::~Resource() {
    if (_cublas_handle) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}

// Global cuBLAS handle cache: one handle per CUDA device.
static std::mutex s_cublas_mutex;
static std::unordered_map<int, cublasHandle_t> s_cublas_handles;

static void destroyAllCublasHandles() {
    std::lock_guard<std::mutex> lock(s_cublas_mutex);
    for (auto& [dev, handle] : s_cublas_handles) {
        if (handle) cublasDestroy(handle);
    }
    s_cublas_handles.clear();
}

cublasHandle_t getCublasHandle() {
    int device_id = 0;
    cudaGetDevice(&device_id);

    std::lock_guard<std::mutex> lock(s_cublas_mutex);
    auto it = s_cublas_handles.find(device_id);
    if (it != s_cublas_handles.end()) {
        return it->second;
    }

    cublasHandle_t handle = nullptr;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed with status " + std::to_string(static_cast<int>(status)));
    }
    s_cublas_handles[device_id] = handle;

    // Register cleanup on first creation
    static bool registered = false;
    if (!registered) {
        std::atexit(destroyAllCublasHandles);
        registered = true;
    }

    return handle;
}

} // namespace llaisys::device::nvidia
