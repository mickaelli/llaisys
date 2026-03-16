#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_TIANSHU_API)
#include "nvidia/rms_nv.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {

    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->shape() == in->shape(), "RMS Norm: output and input shapes must be identical");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1D");
    ASSERT(weight->shape()[0] == in->shape().back(), "RMS Norm: weight dim must match input feature dim");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Add: all tensors must be contiguous.");
    size_t cols = in->shape().back();
    size_t rows = in->numel() / cols;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, rows, cols, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, rows, cols, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, rows, cols, out->dtype());
#endif
#ifdef ENABLE_TIANSHU_API
    case LLAISYS_DEVICE_TIANSHU:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, rows, cols, out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops

