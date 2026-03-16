#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void sampling(std::byte *out_idx,
              const std::byte *logits,
              llaisysDataType_t dtype,
              size_t batch,
              size_t vocab,
              int top_k,
              float top_p,
              float temperature,
              uint64_t seed);
}