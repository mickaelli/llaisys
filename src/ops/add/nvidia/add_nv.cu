#include "add_nv.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

#define CUDA_CHECK(EXPR)                                                                                          \
	do {                                                                                                          \
		cudaError_t err__ = (EXPR);                                                                               \
		if (err__ != cudaSuccess) {                                                                               \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));                   \
		}                                                                                                         \
	} while (0)

__device__ inline __half to_half(llaisys::fp16_t v) {
	return *reinterpret_cast<const __half *>(&v);
}

__device__ inline llaisys::fp16_t from_half(__half v) {
	return llaisys::fp16_t{*reinterpret_cast<const uint16_t *>(&v)};
}

__device__ inline __nv_bfloat16 to_bf16(llaisys::bf16_t v) {
	return *reinterpret_cast<const __nv_bfloat16 *>(&v);
}

__device__ inline llaisys::bf16_t from_bf16(__nv_bfloat16 v) {
	return llaisys::bf16_t{*reinterpret_cast<const uint16_t *>(&v)};
}

__global__ void add_f32_kernel(float *c, const float *a, const float *b, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

__global__ void add_fp16_kernel(llaisys::fp16_t *c, const llaisys::fp16_t *a, const llaisys::fp16_t *b, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		__half ha = to_half(a[idx]);
		__half hb = to_half(b[idx]);
		float sum = __half2float(ha) + __half2float(hb);
		__half hc = __float2half_rn(sum);
		c[idx] = from_half(hc);
	}
}

__global__ void add_bf16_kernel(llaisys::bf16_t *c, const llaisys::bf16_t *a, const llaisys::bf16_t *b, size_t n) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		__nv_bfloat16 ha = to_bf16(a[idx]);
		__nv_bfloat16 hb = to_bf16(b[idx]);
		float sum = __bfloat162float(ha) + __bfloat162float(hb);
		__nv_bfloat16 hc = __float2bfloat16_rn(sum);
		c[idx] = from_bf16(hc);
	}
}

} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
	if (numel == 0) {
		return;
	}

	const int threads = 256;
	const int blocks = static_cast<int>((numel + threads - 1) / threads);

	switch (type) {
	case LLAISYS_DTYPE_F32:
		add_f32_kernel<<<blocks, threads>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
											reinterpret_cast<const float *>(b), numel);
		break;
	case LLAISYS_DTYPE_F16:
		add_fp16_kernel<<<blocks, threads>>>(reinterpret_cast<llaisys::fp16_t *>(c),
											 reinterpret_cast<const llaisys::fp16_t *>(a),
											 reinterpret_cast<const llaisys::fp16_t *>(b), numel);
		break;
	case LLAISYS_DTYPE_BF16:
		add_bf16_kernel<<<blocks, threads>>>(reinterpret_cast<llaisys::bf16_t *>(c),
											 reinterpret_cast<const llaisys::bf16_t *>(a),
											 reinterpret_cast<const llaisys::bf16_t *>(b), numel);
		break;
	default:
		throw std::runtime_error("Unsupported data type for NVIDIA add");
	}

	CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
