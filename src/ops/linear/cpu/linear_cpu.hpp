#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const void *bias, size_t M, size_t N, size_t K, llaisysDataType_t dtype);
}