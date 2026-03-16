#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../ops/argmax/op.hpp"
#include "cpu/sampling_cpu.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace llaisys::ops {
void sampling(tensor_t out_idx, tensor_t logits, int top_k, float top_p, float temperature, uint64_t seed) {
    ASSERT(logits->numel() > 0, "Sampling: logits tensor is empty.");
    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64, "Sampling: output tensor must be int64.");
    ASSERT(logits->isContiguous() && out_idx->isContiguous(), "Sampling: tensors must be contiguous.");
    ASSERT(out_idx->deviceType() == logits->deviceType() || out_idx->deviceType() == LLAISYS_DEVICE_CPU,
           "Sampling: output must be on CPU or match logits device.");

    size_t batch = 1;
    size_t vocab = 0;
    if (logits->shape().size() == 1) {
        vocab = logits->shape()[0];
    } else if (logits->shape().size() == 2) {
        batch = logits->shape()[0];
        vocab = logits->shape()[1];
    } else {
        ASSERT(false, "Sampling: logits must be 1D or 2D.");
    }

    ASSERT(out_idx->numel() == batch, "Sampling: output shape mismatch.");

    const int k = (top_k <= 0) ? static_cast<int>(vocab) : top_k;
    const float p = (top_p <= 0.0f) ? 1.0f : top_p;
    const float temp = (temperature <= 0.0f) ? 1.0f : temperature;

    // Fast greedy path for single-row logits
    if (k <= 1 && batch == 1) {
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, out_idx->deviceType(), out_idx->deviceId());
        tensor_t max_val = Tensor::create({1}, logits->dtype(), logits->deviceType(), logits->deviceId());
        ops::argmax(max_idx, max_val, logits->view({vocab}));
        if (out_idx->deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(out_idx->data(), max_idx->data(), sizeof(int64_t));
        } else {
            core::context().runtime().api()->memcpy_sync(
                out_idx->data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2D);
        }
        return;
    }

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::sampling(out_idx->data(), logits->data(), logits->dtype(), batch, vocab, k, p, temp, seed);
    }

    core::context().setDevice(logits->deviceType(), logits->deviceId());
    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sampling(out_idx->data(), logits->data(), logits->dtype(), batch, vocab, k, p, temp, seed);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        const size_t elem_size = llaisys::utils::dsize(logits->dtype());
        std::vector<std::byte> host_logits(logits->numel() * elem_size);
        core::context().runtime().api()->memcpy_sync(
            host_logits.data(), logits->data(), host_logits.size(), LLAISYS_MEMCPY_D2H);

        std::vector<int64_t> host_idx(batch);
        cpu::sampling(reinterpret_cast<std::byte *>(host_idx.data()), host_logits.data(), logits->dtype(), batch, vocab, k, p, temp, seed);

        if (out_idx->deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(out_idx->data(), host_idx.data(), batch * sizeof(int64_t));
        } else {
            core::context().runtime().api()->memcpy_sync(
                out_idx->data(), host_idx.data(), batch * sizeof(int64_t), LLAISYS_MEMCPY_H2D);
        }
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops