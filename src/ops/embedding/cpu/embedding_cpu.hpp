#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t num_tokens, size_t embedding_dim, size_t vocab_size, llaisysDataType_t dtype);
}