#include"linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template<typename T>
void linear_(T*out,const T*in,const T*weight, const T *bias, size_t M, size_t N, size_t K){
    //if constexpr(std::is_same_v<T,llaisys::bf16_t>)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float x = llaisys::utils::cast<float>(in[m * K + k]);
                float w = llaisys::utils::cast<float>(weight[n * K + k]);
                sum += x*w;
            }
            if (bias) sum += llaisys::utils::cast<float>(bias[n]);
            out[m*N+n] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const void *bias,
            size_t M, size_t N, size_t K, llaisysDataType_t dtype) {
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in), 
                        reinterpret_cast<const float *>(weight), 
                        reinterpret_cast<const float *>(bias), 
                        M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in), 
                        reinterpret_cast<const llaisys::bf16_t *>(weight), 
                        reinterpret_cast<const llaisys::bf16_t *>(bias),
                        M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in), 
                        reinterpret_cast<const llaisys::fp16_t *>(weight), 
                        reinterpret_cast<const llaisys::fp16_t *>(bias),
                        M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}