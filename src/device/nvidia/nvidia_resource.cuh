#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas_handle = nullptr;

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t getCublasHandle() const { return _cublas_handle; }
};

/// Get a cuBLAS handle for the currently active CUDA device.
/// Lazily creates one handle per device and caches it.
cublasHandle_t getCublasHandle();

} // namespace llaisys::device::nvidia
