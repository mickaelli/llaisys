#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          size_t seq_len, size_t n_head, size_t head_dim, llaisysDataType_t type);
}
