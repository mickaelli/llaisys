#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops {
void sampling(tensor_t out_idx, tensor_t logits, int top_k, float top_p, float temperature, uint64_t seed);
}