#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v, 
                    float scale, size_t seq_q, size_t seq_k, 
                    size_t nhead, size_t nkv_head, size_t head_dim) {
    // TODO: implement self attention kernel
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
