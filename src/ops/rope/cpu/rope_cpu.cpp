#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < n_head; h++) {
            float m = static_cast<float>(pos_ids[s]);
            for (size_t d = 0; d < head_dim / 2; d++) {
                float freq = 1.0f / (std::pow(theta, 2.0f * d / head_dim));
                float angle = m * freq;
                float cos_angle = std::cos(angle);
                float sin_angle = std::sin(angle);
                size_t index = s * n_head * head_dim + h * head_dim + 2 * d;

                float x1 = llaisys::utils::cast<float>(in[index]);
                float x2 = llaisys::utils::cast<float>(in[index + 1]);
                out[index] = llaisys::utils::cast<T>(x1 * cos_angle - x2 * sin_angle);
                out[index + 1] = llaisys::utils::cast<T>(x1 * sin_angle + x2 * cos_angle);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
             size_t seq_len, size_t n_head, size_t head_dim, llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out), 
                             reinterpret_cast<const float *>(in), 
                             reinterpret_cast<const int64_t *>(pos_ids), 
                             theta, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
                             reinterpret_cast<const llaisys::bf16_t *>(in), 
                             reinterpret_cast<const int64_t *>(pos_ids), 
                             theta, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
                             reinterpret_cast<const llaisys::fp16_t *>(in), 
                             reinterpret_cast<const int64_t *>(pos_ids), 
                             theta, seq_len, n_head, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
