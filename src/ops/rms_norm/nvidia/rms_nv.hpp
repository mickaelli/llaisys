#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              size_t rows, size_t cols, llaisysDataType_t dtype);
}
