#include "rope_nv.hpp"

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

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                             size_t seq_len, size_t n_head, size_t head_dim) {
    size_t s = blockIdx.x;
    size_t h = blockIdx.y;
    size_t d = threadIdx.x;
    size_t half_dim = head_dim / 2;
    if (s >= seq_len || h >= n_head || d >= half_dim) return;

    double m = static_cast<double>(pos_ids[s]);
    // Use __powf for faster computation on GPU (single-precision fast path)
    double freq = 1.0 / pow(static_cast<double>(theta), 2.0 * d / static_cast<double>(head_dim));
    double angle = m * freq;

    // Use __sincosf for simultaneous sin/cos computation
    float sin_angle, cos_angle;
    sincosf(static_cast<float>(angle), &sin_angle, &cos_angle);

    size_t head_offset = s * n_head * head_dim + h * head_dim;
    size_t idx1 = head_offset + d;
    size_t idx2 = head_offset + d + half_dim;

    float x1 = to_float(in[idx1]);
    float x2 = to_float(in[idx2]);

    float o1 = x1 * cos_angle - x2 * sin_angle;
    float o2 = x1 * sin_angle + x2 * cos_angle;

    if constexpr (std::is_same_v<T, float>) {
        out[idx1] = o1;
        out[idx2] = o2;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        out[idx1] = from_float_fp16(o1);
        out[idx2] = from_float_fp16(o2);
    } else {
        out[idx1] = from_float_bf16(o1);
        out[idx2] = from_float_bf16(o2);
    }
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          size_t seq_len, size_t n_head, size_t head_dim, llaisysDataType_t type) {
    if (seq_len == 0 || n_head == 0 || head_dim == 0) return;
    dim3 grid(static_cast<unsigned int>(seq_len), static_cast<unsigned int>(n_head));
    // Ensure block size is multiple of warp size (32)
    unsigned int half_dim = static_cast<unsigned int>(head_dim / 2);
    unsigned int block_size = ((half_dim + 31) / 32) * 32; // round up to warp multiple
    if (block_size > 1024) block_size = 1024;
    if (block_size == 0) block_size = 32;
    dim3 block(block_size);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<grid, block>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<grid, block>>>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<grid, block>>>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_head, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA rope");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA rope launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
