#include "rms_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// Warp-level sum reduction using shuffle
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void rms_kernel(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) return;
    const T *row_in = in + row * cols;
    T *row_out = out + row * cols;

    // Phase 1: Compute sum of squares using warp shuffle reduction
    extern __shared__ float sdata[];
    int num_warps = (blockDim.x + 31) >> 5;

    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = to_float(row_in[j]);
        sum += v * v;
    }

    // Warp-level reduction first
    sum = warp_reduce_sum(sum);

    // Write warp results to shared memory
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    // Final reduction in first warp
    if (warp == 0) {
        float warp_sum = (lane < num_warps) ? sdata[lane] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0) {
            sdata[0] = rsqrtf(warp_sum / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();

    float scale = sdata[0];

    // Phase 2: Scale and write output
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(row_in[j]);
        float w = to_float(weight[j]);
        float outv = val * scale * w;
        if constexpr (std::is_same_v<T, float>) {
            row_out[j] = outv;
        } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
            row_out[j] = from_float_fp16(outv);
        } else {
            row_out[j] = from_float_bf16(outv);
        }
    }
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              size_t rows, size_t cols, llaisysDataType_t dtype) {
    if (rows == 0 || cols == 0) return;
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(256);
    unsigned int num_warps = (block.x + 31) / 32;
    size_t smem = num_warps * sizeof(float);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_kernel<<<grid, block, smem>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                          reinterpret_cast<const float *>(weight), eps, rows, cols);
        break;
    case LLAISYS_DTYPE_F16:
        rms_kernel<<<grid, block, smem>>>(reinterpret_cast<llaisys::fp16_t *>(out),
                                          reinterpret_cast<const llaisys::fp16_t *>(in),
                                          reinterpret_cast<const llaisys::fp16_t *>(weight), eps, rows, cols);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_kernel<<<grid, block, smem>>>(reinterpret_cast<llaisys::bf16_t *>(out),
                                          reinterpret_cast<const llaisys::bf16_t *>(in),
                                          reinterpret_cast<const llaisys::bf16_t *>(weight), eps, rows, cols);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA rms_norm");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA rms_norm launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
