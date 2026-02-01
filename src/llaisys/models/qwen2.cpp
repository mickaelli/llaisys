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

#include <vector>
#include <cstring>
#include <memory>
#include <cmath>


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
    
    llaisysDeviceType_t device_type;
    int device_id;
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device_type = device;
    model->device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

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

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    try {
        size_t batch = 1;
        size_t seq_len = ntoken;
        size_t total_len = model->cur_pos + seq_len;
        
        // 1. 输入 Tensor (I64)
        tensor_t tokens = create_shared_tensor({seq_len}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        tokens->load(token_ids); 

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

            // KV Cache Update (CPU memcpy)
            if (model->device_type == LLAISYS_DEVICE_CPU) {
                size_t row_size = kv_dim * get_element_size(x->dtype());
                uint8_t* k_ptr = (uint8_t*)model->k_cache[i]->data();
                uint8_t* v_ptr = (uint8_t*)model->v_cache[i]->data();
                uint8_t* src_k = (uint8_t*)k->data();
                uint8_t* src_v = (uint8_t*)v->data();

                for(size_t s=0; s<seq_len; ++s) {
                    memcpy(k_ptr + (model->cur_pos + s) * row_size, src_k + s * row_size, row_size);
                    memcpy(v_ptr + (model->cur_pos + s) * row_size, src_v + s * row_size, row_size);
                }
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
        
        tensor_t max_val = create_shared_tensor({1}, x->dtype(), x->deviceType(), x->deviceId());
        tensor_t max_idx = create_shared_tensor({1}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        ops::argmax(max_idx, max_val, logits->view({model->meta.voc}));

        model->cur_pos += seq_len;

        int64_t next_token;
        if (model->device_type == LLAISYS_DEVICE_CPU) {
            next_token = *reinterpret_cast<int64_t*>(max_idx->data());
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