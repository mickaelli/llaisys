#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_TIANSHU_API)
#include "nvidia/linear_nv.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    if (bias) { 
        ASSERT(bias->deviceType() == out->deviceType(), "Bias device mismatch");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Bias must be contiguous");
        ASSERT(bias->numel() == weight->shape()[0], "Bias shape mismatch");
    }
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: Input and Weight feature dim (K) mismatch");
    ASSERT(out->shape()[0] == M, "Linear: Output batch size (M) mismatch");
    ASSERT(out->shape()[1] == N, "Linear: Output feature dim (N) mismatch");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Add: all tensors must be contiguous.");
    const void* bias_data = bias ? bias->data() : nullptr;
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
            return cpu::linear(out->data(), in->data(), weight->data(),bias_data,M,N,K, out->dtype());
        }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),bias_data,M,N,K, out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, M, N, K, out->dtype());
#endif
#ifdef ENABLE_TIANSHU_API
    case LLAISYS_DEVICE_TIANSHU:
        // Tianshu (BI-CUDA) is source-compatible, reusing nvidia implementation
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, M, N, K, out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
