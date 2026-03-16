#include "self_attention_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

// Warp-level reduction for sum
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using shared memory + warp reduction
__device__ float block_reduce_sum(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (threadIdx.x < static_cast<unsigned>(num_warps)) ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val; // only thread 0 has the final result
}

__device__ float block_reduce_max(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    val = (threadIdx.x < static_cast<unsigned>(num_warps)) ? shared[threadIdx.x] : -INFINITY;
    if (warp == 0) val = warp_reduce_max(val);
    return val;
}

// Each block handles one (qpos, head) pair.
// blockDim.x threads collaborate on computing dot products, softmax, and output.
template <typename T>
__global__ void self_attn_kernel(T *out, const T *q, const T *k, const T *v, float scale,
                                  size_t seq_q, size_t seq_k,
                                  size_t nhead, size_t nkv_head, size_t head_dim) {
    size_t idx = blockIdx.x;
    size_t h = idx % nhead;
    size_t qpos = idx / nhead;
    if (qpos >= seq_q || h >= nhead) return;
    size_t group_size = nhead / nkv_head;
    size_t kv_h = h / group_size;

    // Shared memory layout:
    // [0 .. seq_k-1]: scores array
    // [seq_k .. seq_k + num_warps - 1]: warp reduction scratch
    extern __shared__ float smem[];
    float *scores = smem;
    float *warp_scratch = smem + seq_k;

    // Phase 1: Compute Q·K dot products in parallel
    // Each thread computes partial dot for multiple K positions
    const T *q_row = q + qpos * nhead * head_dim + h * head_dim;

    for (size_t j = 0; j < seq_k; ++j) {
        float dot = 0.0f;
        if (j > qpos + (seq_k - seq_q)) {
            // Causal mask: future positions
            if (threadIdx.x == 0) scores[j] = -INFINITY;
        } else {
            const T *k_row = k + j * nkv_head * head_dim + kv_h * head_dim;
            // Parallelize dot product across threads
            for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                dot += to_float(q_row[d]) * to_float(k_row[d]);
            }
            // Reduce across threads in block
            dot = block_reduce_sum(dot, warp_scratch);
            if (threadIdx.x == 0) {
                scores[j] = dot * scale;
            }
        }
        __syncthreads();
    }

    // Phase 2: Parallel softmax
    // 2a: Find max score
    float local_max = -INFINITY;
    for (size_t j = threadIdx.x; j < seq_k; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    float max_score = block_reduce_max(local_max, warp_scratch);
    max_score = __shfl_sync(0xffffffff, max_score, 0); // broadcast from thread 0
    // Broadcast max_score to all threads
    if (threadIdx.x == 0) warp_scratch[0] = max_score;
    __syncthreads();
    max_score = warp_scratch[0];

    // 2b: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < seq_k; j += blockDim.x) {
        float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    __syncthreads();
    float sum_exp = block_reduce_sum(local_sum, warp_scratch);
    // Broadcast sum
    if (threadIdx.x == 0) warp_scratch[0] = sum_exp;
    __syncthreads();
    float inv_sum = 1.0f / warp_scratch[0];

    // 2c: Normalize scores
    for (size_t j = threadIdx.x; j < seq_k; j += blockDim.x) {
        scores[j] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Compute output = softmax(scores) * V
    // Each thread handles a subset of head_dim
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < seq_k; ++j) {
            size_t v_idx = j * nkv_head * head_dim + kv_h * head_dim + d;
            acc += scores[j] * to_float(v[v_idx]);
        }
        size_t out_idx = qpos * nhead * head_dim + h * head_dim + d;
        if constexpr (std::is_same_v<T, float>) {
            out[out_idx] = acc;
        } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
            out[out_idx] = from_float_fp16(acc);
        } else {
            out[out_idx] = from_float_bf16(acc);
        }
    }
}

} // namespace

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    size_t seq_q, size_t seq_k, size_t nhead, size_t nkv_head, size_t head_dim, llaisysDataType_t dtype) {
    if (seq_q == 0 || seq_k == 0 || nhead == 0 || head_dim == 0) return;
    
    // One block per (qpos, head)
    dim3 grid(static_cast<unsigned int>(seq_q * nhead));
    // Use enough threads for parallel dot product and V aggregation
    unsigned int block_size = 256;
    if (head_dim <= 64) block_size = 128;
    if (head_dim <= 32) block_size = 64;
    dim3 block(block_size);
    
    // Shared memory: scores[seq_k] + warp_scratch[num_warps]
    unsigned int num_warps = (block_size + 31) / 32;
    size_t smem = (seq_k + num_warps) * sizeof(float);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attn_kernel<<<grid, block, smem>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
                                                reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                                                scale, seq_q, seq_k, nhead, nkv_head, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attn_kernel<<<grid, block, smem>>>(reinterpret_cast<llaisys::fp16_t *>(out),
                                                reinterpret_cast<const llaisys::fp16_t *>(q),
                                                reinterpret_cast<const llaisys::fp16_t *>(k),
                                                reinterpret_cast<const llaisys::fp16_t *>(v), scale, seq_q, seq_k, nhead,
                                                nkv_head, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attn_kernel<<<grid, block, smem>>>(reinterpret_cast<llaisys::bf16_t *>(out),
                                                reinterpret_cast<const llaisys::bf16_t *>(q),
                                                reinterpret_cast<const llaisys::bf16_t *>(k),
                                                reinterpret_cast<const llaisys::bf16_t *>(v), scale, seq_q, seq_k, nhead,
                                                nkv_head, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA self_attention");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA self_attention launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
