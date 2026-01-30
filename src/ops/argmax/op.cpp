#include "op.hpp"
#include "../../core/llaisys_core.hpp" 
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // Only support contiguous inputs with same shape for now.
    ASSERT(vals->numel()>=1 && max_idx->numel()==max_val->numel(), "Argmax: all tensors must have compatible number of elements.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Add: all tensors must be contiguous.");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx tensor must be of type int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(max_val->deviceType(), max_val->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
