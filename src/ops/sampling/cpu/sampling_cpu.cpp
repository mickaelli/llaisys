#include "sampling_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
int64_t greedy_argmax(const T *row, size_t vocab) {
    float best = -std::numeric_limits<float>::infinity();
    size_t best_idx = 0;
    for (size_t i = 0; i < vocab; ++i) {
        float val = llaisys::utils::cast<float>(row[i]);
        if (val > best) {
            best = val;
            best_idx = i;
        }
    }
    return static_cast<int64_t>(best_idx);
}

template <typename T>
int64_t sample_row(const T *row,
                   size_t vocab,
                   int top_k,
                   float top_p,
                   float temperature,
                   uint64_t seed,
                   size_t row_idx) {
    if (top_k <= 1) {
        return greedy_argmax(row, vocab);
    }

    const float temp = std::max(temperature, 1e-6f);
    std::vector<float> scores(vocab);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        float v = llaisys::utils::cast<float>(row[i]);
        v /= temp;
        scores[i] = v;
        if (v > max_logit) {
            max_logit = v;
        }
    }

    float prob_sum = 0.0f;
    for (size_t i = 0; i < vocab; ++i) {
        float p = std::exp(scores[i] - max_logit);
        scores[i] = p;
        prob_sum += p;
    }

    if (prob_sum <= 0.0f) {
        return greedy_argmax(row, vocab);
    }

    for (size_t i = 0; i < vocab; ++i) {
        scores[i] /= prob_sum;
    }

    struct Entry {
        float p;
        size_t idx;
    };

    std::vector<Entry> entries;
    entries.reserve(vocab);
    for (size_t i = 0; i < vocab; ++i) {
        entries.push_back(Entry{scores[i], i});
    }

    const size_t k = static_cast<size_t>(std::min<int>(top_k > 0 ? top_k : static_cast<int>(vocab), static_cast<int>(vocab)));
    std::partial_sort(entries.begin(), entries.begin() + k, entries.end(), [](const Entry &a, const Entry &b) {
        return a.p > b.p;
    });
    entries.resize(k);

    const float cutoff = (top_p <= 0.0f) ? 1.0f : std::min(top_p, 1.0f);
    float running = 0.0f;
    size_t keep = 0;
    for (; keep < entries.size(); ++keep) {
        running += entries[keep].p;
        if (running >= cutoff) {
            ++keep;
            break;
        }
    }
    if (keep == 0) {
        keep = 1;
    }
    entries.resize(keep);

    float norm = 0.0f;
    for (auto &e : entries) {
        norm += e.p;
    }
    if (norm <= 0.0f) {
        return greedy_argmax(row, vocab);
    }
    for (auto &e : entries) {
        e.p /= norm;
    }

    uint64_t row_seed = seed;
    if (row_seed == 0) {
        row_seed = static_cast<uint64_t>(std::random_device{}());
    }
    row_seed ^= static_cast<uint64_t>(row_idx + 0x9e3779b97f4a7c15ULL);
    std::mt19937_64 rng(row_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float acc = 0.0f;
    for (const auto &e : entries) {
        acc += e.p;
        if (r <= acc) {
            return static_cast<int64_t>(e.idx);
        }
    }
    return static_cast<int64_t>(entries.back().idx);
}

template <typename T>
void sampling_impl(std::byte *out_idx,
                   const std::byte *logits,
                   size_t batch,
                   size_t vocab,
                   int top_k,
                   float top_p,
                   float temperature,
                   uint64_t seed) {
    const T *typed_logits = reinterpret_cast<const T *>(logits);
    int64_t *typed_out = reinterpret_cast<int64_t *>(out_idx);
    for (size_t b = 0; b < batch; ++b) {
        typed_out[b] = sample_row(typed_logits + b * vocab, vocab, top_k, top_p, temperature, seed, b);
    }
}

void sampling(std::byte *out_idx,
              const std::byte *logits,
              llaisysDataType_t dtype,
              size_t batch,
              size_t vocab,
              int top_k,
              float top_p,
              float temperature,
              uint64_t seed) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sampling_impl<float>(out_idx, logits, batch, vocab, top_k, top_p, temperature, seed);
    case LLAISYS_DTYPE_BF16:
        return sampling_impl<llaisys::bf16_t>(out_idx, logits, batch, vocab, top_k, top_p, temperature, seed);
    case LLAISYS_DTYPE_F16:
        return sampling_impl<llaisys::fp16_t>(out_idx, logits, batch, vocab, top_k, top_p, temperature, seed);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu