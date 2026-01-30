#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v, 
                    float scale, size_t seq_q, size_t seq_k, 
                    size_t nhead, size_t nkv_head, size_t head_dim) {
    
    size_t group_size = nhead / nkv_head;

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / group_size;
        for (size_t i = 0; i < seq_q; i++) {
            std::vector<float> scores(seq_k);
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < seq_k; j++) {
                if (j > i + (seq_k - seq_q)) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                } else {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        size_t q_idx = i * nhead * head_dim + h * head_dim + d;
                        size_t k_idx = j * nkv_head * head_dim + kv_h * head_dim + d;
                
                        float q_val = llaisys::utils::cast<float>(q[q_idx]);
                        float k_val = llaisys::utils::cast<float>(k[k_idx]);
                        dot += q_val * k_val;
                    }
                    scores[j] = dot * scale;
                }
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < seq_k; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j < seq_k; j++) {
                scores[j] /= sum_exp;
            }

            for (size_t d = 0; d < head_dim; d++) {
                float val_acc = 0.0f;
                for (size_t j = 0; j < seq_k; j++) {
                    size_t v_idx = j * nkv_head * head_dim + kv_h * head_dim + d;
                    float v_val = llaisys::utils::cast<float>(v[v_idx]);
                    val_acc += scores[j] * v_val;
                }
                
                size_t out_idx = i * nhead * head_dim + h * head_dim + d;
                out[out_idx] = llaisys::utils::cast<T>(val_acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, 
                    float scale, size_t seq_q, size_t seq_k, 
                    size_t nhead, size_t nkv_head, size_t head_dim, 
                    llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out), 
                             reinterpret_cast<const float *>(q), 
                             reinterpret_cast<const float *>(k), 
                             reinterpret_cast<const float *>(v),
                             scale, seq_q, seq_k, nhead, nkv_head, head_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out), 
                             reinterpret_cast<const llaisys::bf16_t *>(q), 
                             reinterpret_cast<const llaisys::bf16_t *>(k), 
                             reinterpret_cast<const llaisys::bf16_t *>(v),
                             scale, seq_q, seq_k, nhead, nkv_head, head_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out), 
                             reinterpret_cast<const llaisys::fp16_t *>(q), 
                             reinterpret_cast<const llaisys::fp16_t *>(k), 
                             reinterpret_cast<const llaisys::fp16_t *>(v),
                             scale, seq_q, seq_k, nhead, nkv_head, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu