#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t num_tokens, size_t embedding_dim, size_t vocab_size, llaisysDataType_t type);
}
