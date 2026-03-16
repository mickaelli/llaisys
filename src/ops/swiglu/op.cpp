#include "op.hpp"
#include "../../core/llaisys_core.hpp" 
#include "../../utils/check.hpp"
#include "cpu/swiglu_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_TIANSHU_API)
#include "nvidia/swiglu_nv.hpp"
#endif
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out,gate,up);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->shape() == gate->shape() && gate->shape() == up->shape(), "Swiglu: all tensors must have the same shape.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "Swiglu: all tensors must be contiguous.");   

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->numel(),out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->numel(),out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->numel(), out->dtype());
#endif
#ifdef ENABLE_TIANSHU_API
    case LLAISYS_DEVICE_TIANSHU:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->numel(), out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
