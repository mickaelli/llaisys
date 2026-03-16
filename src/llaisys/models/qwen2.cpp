#include "llaisys/models/qwen2.h"
#include "../llaisys_tensor.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp" 
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/sampling/op.hpp"

#include <vector>
#include <cstring>
#include <memory>
#include <cmath>
#include <random>


using namespace llaisys;

size_t get_element_size(llaisysDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32: return 4;
        case LLAISYS_DTYPE_I64: return 8;
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16: return 2;
        default: return 1;
    }
}

llaisysTensor_t create_weight_tensor(const std::vector<size_t>& shape, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    auto tensor = Tensor::create(shape, dtype, device, device_id);
    return new LlaisysTensor{tensor};
}

tensor_t create_shared_tensor(const std::vector<size_t>& shape, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}

tensor_t wrap_ptr(llaisysTensor_t ptr) {
    if (!ptr) return nullptr;
    auto wrapper = reinterpret_cast<LlaisysTensor*>(ptr);
    return wrapper->tensor;
}

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    size_t cur_pos = 0;
    std::vector<int64_t> cached_tokens; // Track tokens currently in KV-cache
    
    llaisysDeviceType_t device_type;
    int device_id;
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    try {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device_type = device;
    model->device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

    // Validate device id against available runtimes
    try {
        core::context().setDevice(model->device_type, model->device_id);
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[C++ Exception in llaisysQwen2ModelCreate]: %s (device_type=%d, device_id=%d)\n",
                e.what(), static_cast<int>(model->device_type), model->device_id);
        delete model;
        return nullptr;
    }

    llaisysDataType_t dtype = meta->dtype;
    size_t hidden_size = meta->hs;
    size_t vocab_size = meta->voc;
    size_t n_head = meta->nh;
    size_t n_kv_head = meta->nkvh;
    size_t head_dim = meta->dh;
    size_t intermediate_size = meta->di;
    size_t max_seq = meta->maxseq;

    auto alloc = [&](const std::vector<size_t>& shape) -> llaisysTensor_t {
        return create_weight_tensor(shape, dtype, model->device_type, model->device_id);
    };

    model->weights.in_embed = alloc({vocab_size, hidden_size});
    model->weights.out_embed = alloc({vocab_size, hidden_size});
    model->weights.out_norm_w = alloc({hidden_size});
    
    model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
    
    model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_up_w   = new llaisysTensor_t[meta->nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];

    for (size_t i = 0; i < meta->nlayer; ++i) {
        model->weights.attn_norm_w[i] = alloc({hidden_size});
        
        model->weights.attn_q_w[i] = alloc({n_head * head_dim, hidden_size});
        model->weights.attn_q_b[i] = alloc({n_head * head_dim});
        
        model->weights.attn_k_w[i] = alloc({n_kv_head * head_dim, hidden_size});
        model->weights.attn_k_b[i] = alloc({n_kv_head * head_dim});
        
        model->weights.attn_v_w[i] = alloc({n_kv_head * head_dim, hidden_size});
        model->weights.attn_v_b[i] = alloc({n_kv_head * head_dim});
        
        model->weights.attn_o_w[i] = alloc({hidden_size, n_head * head_dim});

        model->weights.mlp_norm_w[i] = alloc({hidden_size});
        model->weights.mlp_gate_w[i] = alloc({intermediate_size, hidden_size});
        model->weights.mlp_up_w[i]   = alloc({intermediate_size, hidden_size});
        model->weights.mlp_down_w[i] = alloc({hidden_size, intermediate_size});

        model->k_cache.push_back(create_shared_tensor({max_seq, n_kv_head, head_dim}, dtype, model->device_type, model->device_id));
        model->v_cache.push_back(create_shared_tensor({max_seq, n_kv_head, head_dim}, dtype, model->device_type, model->device_id));
    }

    return model;
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[C++ Exception in llaisysQwen2ModelCreate]: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "\n[C++ Exception in llaisysQwen2ModelCreate]: Unknown exception\n");
        return nullptr;
    }
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;

    auto free_t = [](llaisysTensor_t t) { if (t) delete reinterpret_cast<LlaisysTensor*>(t); };
    
    free_t(model->weights.in_embed);
    free_t(model->weights.out_embed);
    free_t(model->weights.out_norm_w);
    
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        free_t(model->weights.attn_norm_w[i]);
        free_t(model->weights.attn_q_w[i]);
        free_t(model->weights.attn_q_b[i]);
        free_t(model->weights.attn_k_w[i]);
        free_t(model->weights.attn_k_b[i]);
        free_t(model->weights.attn_v_w[i]);
        free_t(model->weights.attn_v_b[i]);
        free_t(model->weights.attn_o_w[i]);
        free_t(model->weights.mlp_norm_w[i]);
        free_t(model->weights.mlp_gate_w[i]);
        free_t(model->weights.mlp_up_w[i]);
        free_t(model->weights.mlp_down_w[i]);
    }
    
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->weights;
}

void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model) {
    if (!model) return;
    model->cur_pos = 0;
    model->cached_tokens.clear();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, const LlaisysSamplingParams *sampling) {
    try {
        core::context().setDevice(model->device_type, model->device_id);
        
        // --- Prefix Matching ---
        size_t match_len = 0;
        size_t max_match = std::min(ntoken, model->cached_tokens.size());
        while (match_len < max_match && token_ids[match_len] == model->cached_tokens[match_len]) {
            match_len++;
        }

        // We only process tokens from match_len to ntoken
        // If match_len == ntoken, we already have everything except we need the last token's logits
        // BUT wait, ntoken == 1 is sampling. If token_ids[0] == cached_tokens[cur_pos-1]? No.
        
        // Implementation strategy: 
        // 1. If match_len > 0, set cur_pos = match_len. 
        // 2. Adjust cached_tokens to match_len.
        // 3. Process remaining (ntoken - match_len) tokens.
        
        // Edge case: If we match EVERYTHING (match_len == ntoken), we still need to run the model once 
        // to get the logits for the last token (sampled from the previous step). 
        // Actually, in auto-regressive, we always compute logits for the LAST token in the input.
        // So if match_len == ntoken, we skip EVERYTHING except the very last token? 
        // No, if match_len == ntoken, it means the user is asking for the next token for a prefix 
        // that is ALREADY FULLY CACHED (including the last token).
        // In that case, we should set match_len = ntoken - 1 so we re-process the last token to get its logits.
        
        if (match_len == ntoken && ntoken > 0) {
            match_len = ntoken - 1;
        }

        model->cur_pos = match_len;
        model->cached_tokens.resize(match_len);
        
        size_t effective_match_len = match_len;
        size_t seq_len = ntoken - effective_match_len;
        size_t total_len = model->cur_pos + seq_len;

        if (total_len > model->meta.maxseq) {
            fprintf(stderr, "[LLAISYS] Warning: Sequence length %zu exceeds max_seq %zu. Resetting cache.\n", total_len, model->meta.maxseq);
            model->cur_pos = 0;
            model->cached_tokens.clear();
            effective_match_len = 0;
            seq_len = std::min(ntoken, model->meta.maxseq);
            total_len = seq_len;
        }

        const int64_t * active_tokens_ptr = token_ids + (ntoken - seq_len);

        // 1. 输入 Tensor (I64)
        tensor_t tokens = create_shared_tensor({seq_len}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        tokens->load(active_tokens_ptr); 

        // 2. Embedding
        tensor_t x = create_shared_tensor({seq_len, model->meta.hs}, static_cast<llaisysDataType_t>(model->meta.dtype), model->device_type, model->device_id);
        ops::embedding(x, tokens, wrap_ptr(model->weights.in_embed));

        // 3. Layers Loop
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            tensor_t residual = x;
            
            // --- Attention ---
            tensor_t x_norm = create_shared_tensor(x->shape(), x->dtype(), x->deviceType(), x->deviceId());
            ops::rms_norm(x_norm, x, wrap_ptr(model->weights.attn_norm_w[i]), model->meta.epsilon);

            size_t q_dim = model->meta.nh * model->meta.dh;
            size_t kv_dim = model->meta.nkvh * model->meta.dh;
            
            tensor_t q = create_shared_tensor({seq_len, model->meta.nh, model->meta.dh}, x->dtype(), x->deviceType(), x->deviceId());
            tensor_t k = create_shared_tensor({seq_len, model->meta.nkvh, model->meta.dh}, x->dtype(), x->deviceType(), x->deviceId());
            tensor_t v = create_shared_tensor({seq_len, model->meta.nkvh, model->meta.dh}, x->dtype(), x->deviceType(), x->deviceId());
            
            tensor_t q_flat = q->view({seq_len, q_dim});
            tensor_t k_flat = k->view({seq_len, kv_dim});
            tensor_t v_flat = v->view({seq_len, kv_dim});
            
            ops::linear(q_flat, x_norm, wrap_ptr(model->weights.attn_q_w[i]), wrap_ptr(model->weights.attn_q_b[i]));
            ops::linear(k_flat, x_norm, wrap_ptr(model->weights.attn_k_w[i]), wrap_ptr(model->weights.attn_k_b[i]));
            ops::linear(v_flat, x_norm, wrap_ptr(model->weights.attn_v_w[i]), wrap_ptr(model->weights.attn_v_b[i]));

            // RoPE
            std::vector<int64_t> pos_vec(seq_len);
            for(size_t j=0; j<seq_len; ++j) pos_vec[j] = model->cur_pos + j;
            tensor_t pos_ids = create_shared_tensor({seq_len}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
            pos_ids->load(pos_vec.data());
            
            ops::rope(q, q, pos_ids, model->meta.theta);
            ops::rope(k, k, pos_ids, model->meta.theta);

            // KV Cache Update (device-aware memcpy)
            size_t row_size = kv_dim * get_element_size(x->dtype());
            uint8_t* k_ptr = (uint8_t*)model->k_cache[i]->data();
            uint8_t* v_ptr = (uint8_t*)model->v_cache[i]->data();
            uint8_t* src_k = (uint8_t*)k->data();
            uint8_t* src_v = (uint8_t*)v->data();

            auto kind = (model->device_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
            for(size_t s=0; s<seq_len; ++s) {
                core::context().runtime().api()->memcpy_sync(
                    k_ptr + (model->cur_pos + s) * row_size,
                    src_k + s * row_size,
                    row_size,
                    kind);
                core::context().runtime().api()->memcpy_sync(
                    v_ptr + (model->cur_pos + s) * row_size,
                    src_v + s * row_size,
                    row_size,
                    kind);
            }
            
            // Self Attention
            tensor_t k_curr = model->k_cache[i]->slice(0, 0, total_len); 
            tensor_t v_curr = model->v_cache[i]->slice(0, 0, total_len);
            
            tensor_t attn_out = create_shared_tensor({seq_len, model->meta.nh, model->meta.dh}, x->dtype(), x->deviceType(), x->deviceId());
            
            float scale = 1.0f / sqrtf(static_cast<float>(model->meta.dh));
            ops::self_attention(attn_out, q, k_curr, v_curr, scale);
            tensor_t attn_out_flat = attn_out->view({seq_len, model->meta.hs});
            ops::linear(x_norm, attn_out_flat, wrap_ptr(model->weights.attn_o_w[i]), nullptr);
            ops::add(x, x, x_norm);

            // --- MLP ---
            ops::rms_norm(x_norm, x, wrap_ptr(model->weights.mlp_norm_w[i]), model->meta.epsilon);
            
            tensor_t gate = create_shared_tensor({seq_len, model->meta.di}, x->dtype(), x->deviceType(), x->deviceId());
            tensor_t up   = create_shared_tensor({seq_len, model->meta.di}, x->dtype(), x->deviceType(), x->deviceId());
            
            ops::linear(gate, x_norm, wrap_ptr(model->weights.mlp_gate_w[i]), nullptr);
            ops::linear(up, x_norm, wrap_ptr(model->weights.mlp_up_w[i]), nullptr);
            
            ops::swiglu(gate, gate, up);
            ops::linear(x_norm, gate, wrap_ptr(model->weights.mlp_down_w[i]), nullptr);            
            ops::add(x, x, x_norm);
        }

        // 4. Final
        ops::rms_norm(x, x, wrap_ptr(model->weights.out_norm_w), model->meta.epsilon);

        tensor_t last_token_hidden = x->slice(0, seq_len - 1, seq_len);
        tensor_t logits = create_shared_tensor({1, model->meta.voc}, x->dtype(), x->deviceType(), x->deviceId());
        ops::linear(logits, last_token_hidden, wrap_ptr(model->weights.out_embed), nullptr);
        
        const int top_k = sampling ? sampling->top_k : 1;
        const float top_p = sampling ? sampling->top_p : 1.0f;
        const float temperature = sampling ? sampling->temperature : 1.0f;
        uint64_t seed = sampling ? sampling->seed : 0;
        if (seed == 0) {
            seed = static_cast<uint64_t>(std::random_device{}());
        }

        tensor_t sampled_idx = create_shared_tensor({1}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        ops::sampling(sampled_idx, logits->view({model->meta.voc}), top_k, top_p, temperature, seed);

        // Update cached tokens
        for(size_t i=0; i<seq_len; ++i) {
            model->cached_tokens.push_back(active_tokens_ptr[i]);
        }
        model->cur_pos = model->cached_tokens.size();

        int64_t next_token=-1;
        if (model->device_type == LLAISYS_DEVICE_CPU) {
            next_token = *reinterpret_cast<int64_t*>(sampled_idx->data());
        } else {
            int64_t host_idx = -1;
            core::context().runtime().api()->memcpy_sync(
                &host_idx,
                sampled_idx->data(),
                sizeof(int64_t),
                LLAISYS_MEMCPY_D2H);
            next_token = host_idx;
        }
        return next_token;
        } catch (const std::exception& e) {
        // 
        fprintf(stderr, "\n[C++ Exception in llaisysQwen2ModelInfer]: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "\n[C++ Exception]: Unknown exception occurred.\n");
        return -1;
    }
}

} // extern "C"