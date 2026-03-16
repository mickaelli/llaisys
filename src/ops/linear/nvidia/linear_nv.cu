#include "linear_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../../device/nvidia/nvidia_resource.cuh"

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

// Small kernel to add bias to each row of the output matrix.
// out[m, n] += bias[n], for m in [0, M), n in [0, N)
template <typename T>
__device__ inline float to_float(T v);

template <>
__device__ inline float to_float<float>(float v) { return v; }

template <>
__device__ inline float to_float<llaisys::fp16_t>(llaisys::fp16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

template <>
__device__ inline float to_float<llaisys::bf16_t>(llaisys::bf16_t v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&v));
}

template <typename T>
__device__ inline T from_float(float v);

template <>
__device__ inline float from_float<float>(float v) { return v; }

template <>
__device__ inline llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    __half hv = __float2half_rn(v);
    return llaisys::fp16_t{*reinterpret_cast<const uint16_t *>(&hv)};
}

template <>
__device__ inline llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    __nv_bfloat16 hv = __float2bfloat16_rn(v);
    return llaisys::bf16_t{*reinterpret_cast<const uint16_t *>(&hv)};
}

template <typename T>
__global__ void bias_add_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = blockIdx.y;
    if (m >= M || n >= N) return;
    float val = to_float(out[m * N + n]) + to_float(bias[n]);
    out[m * N + n] = from_float<T>(val);
}

template <typename T>
void add_bias(T *out, const T *bias, size_t M, size_t N) {
    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((N + block.x - 1) / block.x), static_cast<unsigned int>(M));
    bias_add_kernel<<<grid, block>>>(out, bias, M, N);
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const void *bias,
            size_t M, size_t N, size_t K, llaisysDataType_t dtype) {
    if (M == 0 || N == 0 || K == 0) return;

    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();

    // cuBLAS is column-major. We need to compute: out = in * weight^T
    // which is C[M,N] = A[M,K] * B[N,K]^T
    //
    // In column-major terms, this is equivalent to:
    //   C^T[N,M] = B[N,K] * A^T[K,M]
    // So we call: cublas(OP_T, OP_N, N, M, K, B, K, A, K, C, N)
    //
    // But our data is row-major:
    //   A row-major [M,K] = A^T column-major [K,M], leading dim = K
    //   B row-major [N,K] = B^T column-major [K,N], leading dim = K  
    //   C row-major [M,N] = C^T column-major [N,M], leading dim = N
    //
    // We want C = A * B^T in row-major.
    // C^T = (A * B^T)^T = B * A^T in column-major.
    // So: cublas(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, weight, K, in, K, beta, out, N)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        status = cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                             &alpha,
                             reinterpret_cast<const float *>(weight), static_cast<int>(K),
                             reinterpret_cast<const float *>(in), static_cast<int>(K),
                             &beta,
                             reinterpret_cast<float *>(out), static_cast<int>(N));
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSgemm failed with status " + std::to_string(static_cast<int>(status)));
        }
        if (bias) {
            add_bias(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), M, N);
        }
        break;

    case LLAISYS_DTYPE_F16:
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                              &alpha,
                              weight, CUDA_R_16F, static_cast<int>(K),
                              in, CUDA_R_16F, static_cast<int>(K),
                              &beta,
                              out, CUDA_R_16F, static_cast<int>(N),
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx (F16) failed with status " + std::to_string(static_cast<int>(status)));
        }
        if (bias) {
            add_bias(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(bias), M, N);
        }
        break;

    case LLAISYS_DTYPE_BF16:
        status = cublasGemmEx(handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                              &alpha,
                              weight, CUDA_R_16BF, static_cast<int>(K),
                              in, CUDA_R_16BF, static_cast<int>(K),
                              &beta,
                              out, CUDA_R_16BF, static_cast<int>(N),
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx (BF16) failed with status " + std::to_string(static_cast<int>(status)));
        }
        if (bias) {
            add_bias(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(bias), M, N);
        }
        break;

    default:
        throw std::runtime_error("Unsupported dtype for CUDA linear");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA linear kernel error: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
