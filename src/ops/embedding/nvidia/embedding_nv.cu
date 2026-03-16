#include "embedding_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

// Vectorized embedding copy using float4 (128-bit loads/stores)
__global__ void embedding_f32_vec4_kernel(float *out, const int64_t *index, const float *weight,
                                          size_t num_tokens, size_t embedding_dim, size_t vocab_size) {
    size_t token = blockIdx.x;
    if (token >= num_tokens) return;
    int64_t row = index[token];
    if (row < 0 || static_cast<size_t>(row) >= vocab_size) return;

    const float *src = weight + static_cast<size_t>(row) * embedding_dim;
    float *dst = out + token * embedding_dim;

    // Process 4 floats at a time
    size_t vec4_count = embedding_dim / 4;
    for (size_t i = threadIdx.x; i < vec4_count; i += blockDim.x) {
        reinterpret_cast<float4 *>(dst)[i] = reinterpret_cast<const float4 *>(src)[i];
    }
    // Handle remainder
    size_t remainder_start = vec4_count * 4;
    for (size_t d = remainder_start + threadIdx.x; d < embedding_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Vectorized embedding for 16-bit types using uint32 (two 16-bit elements)
template <typename T>
__global__ void embedding_16bit_vec_kernel(T *out, const int64_t *index, const T *weight,
                                            size_t num_tokens, size_t embedding_dim, size_t vocab_size) {
    size_t token = blockIdx.x;
    if (token >= num_tokens) return;
    int64_t row = index[token];
    if (row < 0 || static_cast<size_t>(row) >= vocab_size) return;

    const T *src = weight + static_cast<size_t>(row) * embedding_dim;
    T *dst = out + token * embedding_dim;

    // Process 4 half elements at a time (64 bits = int2 = 4 x fp16/bf16)
    size_t vec4_count = embedding_dim / 4;
    const int2 *src_v = reinterpret_cast<const int2 *>(src);
    int2 *dst_v = reinterpret_cast<int2 *>(dst);
    for (size_t i = threadIdx.x; i < vec4_count; i += blockDim.x) {
        dst_v[i] = src_v[i];
    }
    // Handle remainder
    size_t remainder_start = vec4_count * 4;
    for (size_t d = remainder_start + threadIdx.x; d < embedding_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_tokens,
               size_t embedding_dim, size_t vocab_size, llaisysDataType_t type) {
    if (num_tokens == 0 || embedding_dim == 0) {
        return;
    }
    dim3 grid(static_cast<unsigned int>(num_tokens));
    dim3 block(256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_f32_vec4_kernel<<<grid, block>>>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                                                   reinterpret_cast<const float *>(weight), num_tokens, embedding_dim, vocab_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_16bit_vec_kernel<<<grid, block>>>(reinterpret_cast<llaisys::fp16_t *>(out),
                                                     reinterpret_cast<const int64_t *>(index),
                                                     reinterpret_cast<const llaisys::fp16_t *>(weight), num_tokens, embedding_dim,
                                                     vocab_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_16bit_vec_kernel<<<grid, block>>>(reinterpret_cast<llaisys::bf16_t *>(out),
                                                     reinterpret_cast<const int64_t *>(index),
                                                     reinterpret_cast<const llaisys::bf16_t *>(weight), num_tokens, embedding_dim,
                                                     vocab_size);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA embedding");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA embedding launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
