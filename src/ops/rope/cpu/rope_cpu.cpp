#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < n_head; h++) {
            double m = static_cast<double>(pos_ids[s]);
            size_t head_offset = s * n_head * head_dim + h * head_dim;

            for (size_t d = 0; d < half_dim; d++) {
                double freq = 1.0 / std::pow(static_cast<double>(theta), 2.0 * d / head_dim);
                double angle = m * freq;

                float cos_angle = static_cast<float>(std::cos(angle));
                float sin_angle = static_cast<float>(std::sin(angle));

                size_t idx1 = head_offset + d;
                size_t idx2 = head_offset + d + half_dim;

                float x1 = llaisys::utils::cast<float>(in[idx1]);
                float x2 = llaisys::utils::cast<float>(in[idx2]);

                out[idx1] = llaisys::utils::cast<T>(x1 * cos_angle - x2 * sin_angle);
                out[idx2] = llaisys::utils::cast<T>(x1 * sin_angle + x2 * cos_angle);
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