#include "op.hpp"
#include "../../core/llaisys_core.hpp" 
#include "cpu/embedding_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_TIANSHU_API)
#include "nvidia/embedding_nv.hpp"
#endif
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Only support contiguous inputs with same shape for now.
    ASSERT(index->numel()>=1 && out->numel()==index->numel()*weight->shape()[1], "Embedding: all tensors must have compatible number of elements.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    if (index->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1], weight->shape()[0], out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (index->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1], weight->shape()[0], out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1],
                                  weight->shape()[0], out->dtype());
#endif
#ifdef ENABLE_TIANSHU_API
    case LLAISYS_DEVICE_TIANSHU:
        return nvidia::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1],
                                  weight->shape()[0], out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;

}
}
} // namespace llaisys::ops
