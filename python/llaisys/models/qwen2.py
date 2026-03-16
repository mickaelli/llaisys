from typing import Sequence, Optional
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
import json
from pathlib import Path
import safetensors
import torch
import ctypes
from ..libllaisys import LlaisysTensor, llaisysDataType_t, DeviceType, DataType

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),    
        ("nh", ctypes.c_size_t),    
        ("nkvh", ctypes.c_size_t),  
        ("dh", ctypes.c_size_t),    
        ("di", ctypes.c_size_t),    
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", ctypes.c_void_p),      
        ("out_embed", ctypes.c_void_p),
        ("out_norm_w", ctypes.c_void_p),
        ("attn_norm_w", ctypes.POINTER(ctypes.c_void_p)), 
        ("attn_q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_gate_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_up_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_down_w", ctypes.POINTER(ctypes.c_void_p)),
    ]


class LlaisysSamplingParams(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]

LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]

LIB_LLAISYS.llaisysQwen2ModelReset.argtypes = [ctypes.c_void_p]

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.POINTER(LlaisysSamplingParams),
]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64



class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        
        with open(self.model_path / "config.json", "r") as f:
            config = json.load(f)
        
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.BF16.value
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh 
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = 2048 
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = config.get("eos_token_id", 151643) 

        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device.value, None, 0)
        if not self.model_ptr:
            raise RuntimeError("llaisysQwen2ModelCreate returned null; see stderr for C++ side errors (likely CUDA/device init)")

        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        self.weights = self.weights_ptr.contents 

        print("Loading weights...")
        for file in sorted(self.model_path.glob("*.safetensors")):
            print(f"Processing {file.name}...")
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys(): 
        
                    tensor_pt = f.get_tensor(name)
                    data_bytes = tensor_pt.view(torch.uint16).numpy().tobytes()
                    
                    target_tensor_ptr = None
                    
                    if name == "model.embed_tokens.weight":
                        target_tensor_ptr = self.weights.in_embed
                    elif name == "lm_head.weight":
                        target_tensor_ptr = self.weights.out_embed
                    elif name == "model.norm.weight":
                        target_tensor_ptr = self.weights.out_norm_w
                    
                    elif name.startswith("model.layers."):
                        parts = name.split(".")
                        layer_idx = int(parts[2])
                        suffix = ".".join(parts[3:])
                        
                        if suffix == "input_layernorm.weight":
                            target_tensor_ptr = self.weights.attn_norm_w[layer_idx]
                        elif suffix == "post_attention_layernorm.weight":
                            target_tensor_ptr = self.weights.mlp_norm_w[layer_idx]
                        
                        elif suffix == "self_attn.q_proj.weight":
                            target_tensor_ptr = self.weights.attn_q_w[layer_idx]
                        elif suffix == "self_attn.q_proj.bias":
                            target_tensor_ptr = self.weights.attn_q_b[layer_idx]
                        elif suffix == "self_attn.k_proj.weight":
                            target_tensor_ptr = self.weights.attn_k_w[layer_idx]
                        elif suffix == "self_attn.k_proj.bias":
                            target_tensor_ptr = self.weights.attn_k_b[layer_idx]
                        elif suffix == "self_attn.v_proj.weight":
                            target_tensor_ptr = self.weights.attn_v_w[layer_idx]
                        elif suffix == "self_attn.v_proj.bias":
                            target_tensor_ptr = self.weights.attn_v_b[layer_idx]
                        elif suffix == "self_attn.o_proj.weight":
                            target_tensor_ptr = self.weights.attn_o_w[layer_idx]
                            
                        elif suffix == "mlp.gate_proj.weight":
                            target_tensor_ptr = self.weights.mlp_gate_w[layer_idx]
                        elif suffix == "mlp.up_proj.weight":
                            target_tensor_ptr = self.weights.mlp_up_w[layer_idx]
                        elif suffix == "mlp.down_proj.weight":
                            target_tensor_ptr = self.weights.mlp_down_w[layer_idx]

                    if target_tensor_ptr:
                        if target_tensor_ptr is None:
                            print(f"Warning: Tensor pointer for {name} is Null!")
                            continue
                            
                        c_tensor = LlaisysTensor(target_tensor_ptr, from_native=True)
                        c_tensor.copy_from_host(data_bytes)
                    else:
                        pass

    def __del__(self):
        if hasattr(self, "model_ptr") and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)

    def reset(self):
        if hasattr(self, "model_ptr") and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelReset(self.model_ptr)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ):
        tokens = list(inputs)

        sampling = LlaisysSamplingParams(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=0 if seed is None else seed,
        )
        sampling_ptr = ctypes.byref(sampling)

        input_ids = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, input_ids, len(tokens), sampling_ptr)
        tokens.append(next_token)
        
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
                
            input_ids = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, input_ids, len(tokens), sampling_ptr)
            tokens.append(next_token)

        return tokens

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ):
        sampling = LlaisysSamplingParams(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=0 if seed is None else seed,
        )
        sampling_ptr = ctypes.byref(sampling)

        tokens = list(inputs)
        input_ids = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, input_ids, len(tokens), sampling_ptr)
        if next_token == self.meta.end_token:
            return
        tokens.append(next_token)
        yield next_token
        
        for _ in range(max_new_tokens - 1):
            input_ids = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, input_ids, len(tokens), sampling_ptr)
            if next_token == self.meta.end_token:
                break
            tokens.append(next_token)
            yield next_token