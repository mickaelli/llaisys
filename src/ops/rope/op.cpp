#include "op.hpp"
#include "../../core/llaisys_core.hpp" 
#include "../../utils/check.hpp"
#include "cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out,in,pos_ids);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids tensor must be of type int64.");
    ASSERT(pos_ids->ndim() == 2, "RoPE: pos_ids must be 2D [Batch, SeqLen]");
    size_t head_dim = in->shape().back();
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even for pairing");
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [Batch, SeqLen, HeadDim] for this exercise");
    ASSERT(in->shape()[0] == pos_ids->shape()[0], "RoPE: Batch size mismatch");
    ASSERT(in->shape()[1] == pos_ids->shape()[1], "RoPE: SeqLen mismatch");
    ASSERT(out->shape() == in->shape(), "RoPE: Output shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    size_t batch = in->shape()[0];
    size_t seq_len = in->shape()[1];

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, batch, seq_len, head_dim, out->dtype());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, batch, seq_len, head_dim, out->dtype());
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
