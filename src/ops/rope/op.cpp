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
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [SeqLen]");
    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [SeqLen, NHead, HeadDim]"); 
    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even for pairing");
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: PosIds length mismatch with SeqLen");
    ASSERT(out->shape() == in->shape(), "RoPE: Output shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim, out->dtype());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim, out->dtype());
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
