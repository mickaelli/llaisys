#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype);
}
