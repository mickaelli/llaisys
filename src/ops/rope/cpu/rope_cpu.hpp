#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t batch, size_t seq_len, size_t head_dim, llaisysDataType_t type);
}
