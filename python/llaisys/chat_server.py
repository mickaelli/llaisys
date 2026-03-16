import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

import llaisys
from llaisys import DeviceType


app = FastAPI(title="LLAISYS Chat Server", version="0.1")

tokenizer = None
model = None
_model_lock = asyncio.Lock()
_model_path: Optional[str] = None
_device_name: str = "cpu"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role, e.g. user or assistant")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(128, ge=1, le=512)
    top_k: int = Field(50, ge=1, le=1024)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    temperature: float = Field(0.8, gt=0.0)
    seed: Optional[int] = Field(None, description="Optional RNG seed for reproducible sampling")


class ChatResponse(BaseModel):
    output_text: str
    tokens: List[int]
    latency_ms: float


def _resolve_device(name: str) -> DeviceType:
    return DeviceType.NVIDIA if name.lower() == "nvidia" else DeviceType.CPU


def _ensure_model() -> None:
    global tokenizer, model, _model_path, _device_name
    if tokenizer is not None and model is not None:
        return

    model_path = os.environ.get("LLAISYS_MODEL_PATH")
    if not model_path:
        raise RuntimeError("LLAISYS_MODEL_PATH not set; provide --model or set env variable.")

    device_name = os.environ.get("LLAISYS_DEVICE", "cpu").lower()
    _device_name = device_name
    device = _resolve_device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.Qwen2(model_path, device)
    _model_path = model_path


def _build_prompt(messages: List[ChatMessage]) -> str:
    assert tokenizer is not None
    return tokenizer.apply_chat_template(
        conversation=[{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        tokenize=False,
    )


@app.on_event("startup")
async def startup_event():
    try:
        _ensure_model()
    except Exception as exc:  # noqa: BLE001
        # Defer error handling to request time if startup load fails
        print(f"[LLAISYS] Startup load deferred: {exc}")


@app.get("/health", response_model=dict)
async def health():
    return {"status": "ok", "model": _model_path, "device": _device_name}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    async with _model_lock:
        try:
            _ensure_model()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        assert tokenizer is not None and model is not None

        prompt = _build_prompt(req.messages)
        input_ids = tokenizer.encode(prompt)

        start = time.perf_counter()
        tokens = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            seed=req.seed,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        output_text = tokenizer.decode(tokens, skip_special_tokens=True)

        return ChatResponse(output_text=output_text, tokens=tokens, latency_ms=latency_ms)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    try:
        _ensure_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assert tokenizer is not None and model is not None

    async def stream_generator():
        async with _model_lock:
            prompt = _build_prompt(req.messages)
            input_ids = tokenizer.encode(prompt)

            token_gen = model.stream_generate(
                input_ids,
                max_new_tokens=req.max_new_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                seed=req.seed,
            )

            for token in token_gen:
                if await request.is_disconnected():
                    break
                
                # Yield each token individually or decode it.
                # For simplicity, we stream the decoded delta.
                chunk_text = tokenizer.decode([token], skip_special_tokens=True)
                if chunk_text:
                    yield f"data: {json.dumps({'text': chunk_text, 'token': token})}\n\n"
            
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


ui_dir = Path(__file__).resolve().parent.parent.parent / "ui"
print(f"[LLAISYS] UI Directory: {ui_dir} (Exists: {ui_dir.exists()})")
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui/")
else:
    @app.get("/", include_in_schema=False)
    async def root_placeholder():
        return JSONResponse({"message": "UI not found. Create ui/index.html or hit /api/chat."})


def parse_args():
    parser = argparse.ArgumentParser(description="LLAISYS Chat Server")
    parser.add_argument("--model", required=False, help="Path to model directory (safetensors + config.json)")
    parser.add_argument("--device", choices=["cpu", "nvidia"], default="cpu", help="Target device")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload (dev only)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model:
        os.environ["LLAISYS_MODEL_PATH"] = args.model
    os.environ["LLAISYS_DEVICE"] = args.device

    # Load once before serving to surface errors early.
    _ensure_model()

    import uvicorn

    uvicorn.run(
        "llaisys.chat_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()