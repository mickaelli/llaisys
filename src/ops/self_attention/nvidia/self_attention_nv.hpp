#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    size_t seq_q, size_t seq_k, size_t nhead, size_t nkv_head, size_t head_dim,
                    llaisysDataType_t dtype);
}
