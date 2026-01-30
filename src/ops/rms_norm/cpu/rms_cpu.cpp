#include "rms_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
   for (size_t i = 0; i < rows; i++) {
        const T *row_in = in + i * cols;
        T *row_out = out + i * cols;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / cols + eps);
        float scale = 1.0f / rms;

        for (size_t j = 0; j < cols; j++) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            
            row_out[j] = llaisys::utils::cast<T>(val * scale * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              size_t rows, size_t cols, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_(reinterpret_cast<float *>(out), 
                             reinterpret_cast<const float *>(in), 
                             reinterpret_cast<const float *>(weight), 
                             eps, rows, cols);
    case LLAISYS_DTYPE_BF16:
        return rms_(reinterpret_cast<llaisys::bf16_t *>(out), 
                             reinterpret_cast<const llaisys::bf16_t *>(in), 
                             reinterpret_cast<const llaisys::bf16_t *>(weight), 
                             eps, rows, cols);
    case LLAISYS_DTYPE_F16:
        return rms_(reinterpret_cast<llaisys::fp16_t *>(out), 
                             reinterpret_cast<const llaisys::fp16_t *>(in), 
                             reinterpret_cast<const llaisys::fp16_t *>(weight), 
                             eps, rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
