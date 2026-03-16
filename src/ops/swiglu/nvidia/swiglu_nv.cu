#include "swiglu_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(llaisys::fp16_t v) { return __half2float(*reinterpret_cast<const __half *>(&v)); }
__device__ inline float to_float(llaisys::bf16_t v) { return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&v)); }
__device__ inline llaisys::fp16_t from_float_fp16(float v) {
    __half hv = __float2half_rn(v);
    return llaisys::fp16_t{*reinterpret_cast<const uint16_t *>(&hv)};
}
__device__ inline llaisys::bf16_t from_float_bf16(float v) {
    __nv_bfloat16 hv = __float2bfloat16_rn(v);
    return llaisys::bf16_t{*reinterpret_cast<const uint16_t *>(&hv)};
}

// Vectorized SwiGLU kernel: each thread processes 4 float elements
__global__ void swiglu_f32_vec4_kernel(float *out, const float *gate, const float *up, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 g = *reinterpret_cast<const float4 *>(gate + idx);
        float4 u = *reinterpret_cast<const float4 *>(up + idx);
        float4 o;
        o.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        o.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        o.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        o.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        *reinterpret_cast<float4 *>(out + idx) = o;
    } else {
        // Handle tail elements
        for (size_t i = idx; i < n && i < idx + 4; ++i) {
            float g = gate[i];
            float u = up[i];
            out[i] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// Generic scalar kernel for half types
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = to_float(gate[idx]);
    float u = to_float(up[idx]);
    float activated = g / (1.0f + expf(-g));
    float res = activated * u;
    if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        out[idx] = from_float_fp16(res);
    } else {
        out[idx] = from_float_bf16(res);
    }
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype) {
    if (numel == 0) return;

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        // Use vectorized kernel processing 4 elements per thread
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>(((numel + 3) / 4 + block.x - 1) / block.x));
        swiglu_f32_vec4_kernel<<<grid, block>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                                                reinterpret_cast<const float *>(up), numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((numel + block.x - 1) / block.x));
        swiglu_kernel<<<grid, block>>>(reinterpret_cast<llaisys::fp16_t *>(out),
                                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((numel + block.x - 1) / block.x));
        swiglu_kernel<<<grid, block>>>(reinterpret_cast<llaisys::bf16_t *>(out),
                                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for CUDA swiglu");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA swiglu launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
