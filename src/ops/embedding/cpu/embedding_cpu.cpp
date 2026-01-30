#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include<cstring>
#include<cstdint>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight,size_t num_tokens, size_t embedding_dim, size_t vocab_size) {
    for(size_t idx = 0; idx<num_tokens;idx++){
        int64_t row_id = index[idx];
        if (row_id < 0 || (size_t)row_id >= vocab_size) 
            continue;
        const T *src = weight + row_id * embedding_dim;
        T *dst = out + idx * embedding_dim;
        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
    size_t num_tokens, size_t embedding_dim, size_t vocab_size, llaisysDataType_t type) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),idx_ptr, reinterpret_cast<const float *>(weight), 
                    num_tokens, embedding_dim, vocab_size);

    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,reinterpret_cast<const llaisys::bf16_t *>(weight),
                    num_tokens, embedding_dim, vocab_size);

    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),idx_ptr, reinterpret_cast<const llaisys::fp16_t *>(weight),
                    num_tokens, embedding_dim, vocab_size);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
