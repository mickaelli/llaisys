#include "argmax_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
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

struct ValIdx {
    float val;
    int64_t idx;
};

// Warp-level argmax reduction
__device__ inline ValIdx warp_reduce_argmax(ValIdx vi) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, vi.val, offset);
        int64_t other_idx = __shfl_down_sync(0xffffffff, vi.idx, offset);
        if (other_val > vi.val) {
            vi.val = other_val;
            vi.idx = other_idx;
        }
    }
    return vi;
}

// Parallel argmax kernel using block-level reduction
// Shared memory: float[num_warps] for values, int64_t[num_warps] for indices
template <typename T>
__global__ void argmax_kernel(int64_t *out_idx, T *out_val, const T *vals, size_t n) {
    extern __shared__ char smem_raw[];
    int num_warps = (blockDim.x + 31) >> 5;
    float *s_vals = reinterpret_cast<float *>(smem_raw);
    int64_t *s_idxs = reinterpret_cast<int64_t *>(s_vals + num_warps);

    // Each thread finds local best across its strided elements
    ValIdx best{-INFINITY, -1};
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > best.val) {
            best.val = v;
            best.idx = static_cast<int64_t>(i);
        }
    }

    // Warp-level reduction
    best = warp_reduce_argmax(best);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_vals[warp] = best.val;
        s_idxs[warp] = best.idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp == 0) {
        ValIdx final_best{-INFINITY, -1};
        if (lane < num_warps) {
            final_best.val = s_vals[lane];
            final_best.idx = s_idxs[lane];
        }
        final_best = warp_reduce_argmax(final_best);

        if (lane == 0) {
            *out_idx = final_best.idx;
            if constexpr (std::is_same_v<T, float>) {
                *out_val = final_best.val;
            } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                *out_val = from_float_fp16(final_best.val);
            } else {
                *out_val = from_float_bf16(final_best.val);
            }
        }
    }
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    if (numel == 0) {
        throw std::runtime_error("Argmax on empty tensor");
    }
    
    // Use 256 threads for parallel reduction
    unsigned int block_size = 256;
    unsigned int num_warps = (block_size + 31) / 32;
    size_t smem = num_warps * (sizeof(float) + sizeof(int64_t));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<1, block_size, smem>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<1, block_size, smem>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<1, block_size, smem>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA argmax");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA argmax launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
