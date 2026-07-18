"""Simple OpenAI-compatible completions server for custom-arch HF models.

BFCL's OSS pipeline talks to a local model through an OpenAI-compatible
`/v1/completions` endpoint. Because Lumma uses a custom `nandi` architecture that
vLLM/sglang cannot load, we serve it with plain `transformers` + trust_remote_code
here, then run BFCL with `--skip-server-setup`.

Requests are processed one at a time (no batching / no concurrency) for maximum
reliability with the custom architecture. Run BFCL with `--num-threads 1`.

Usage:
    python lumma_server.py --model-path FrontiersMind/Lumma-0.6B-Tool --port 1053

    bfcl generate --model FrontiersMind/Lumma-0.6B-Tool \
        --test-category multi_turn --skip-server-setup --num-threads 1
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOKENIZER = None
MODEL_ID = None

MAX_NEW_TOKENS_CAP = 4096
# Prompts longer than this (model context minus a small margin) can't be run and
# would raise a position-embedding error, so we skip them with an empty response.
MAX_INPUT_TOKENS = 100000

# Generation is not thread-safe on a single model instance; serialize requests.
_GEN_LOCK = threading.Lock()

app = FastAPI()


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    stop_token_ids: list[int] | None = None
    skip_special_tokens: bool | None = None


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        inputs = TOKENIZER(req.prompt, return_tensors="pt").to(MODEL.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        if prompt_tokens <= MAX_INPUT_TOKENS:
            do_sample = req.temperature is not None and req.temperature > 0.0
            gen_kwargs = dict(
                max_new_tokens=min(MAX_NEW_TOKENS_CAP, req.max_tokens),
                do_sample=do_sample,
                eos_token_id=req.stop_token_ids or TOKENIZER.eos_token_id,
                pad_token_id=TOKENIZER.pad_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = req.temperature

            with _GEN_LOCK, torch.no_grad():
                out = MODEL.generate(**inputs, **gen_kwargs)

            gen_ids = out[0, prompt_tokens:]
            completion_tokens = int(gen_ids.shape[0])
            skip_special = bool(req.skip_special_tokens) if req.skip_special_tokens is not None else False
            text = TOKENIZER.decode(gen_ids, skip_special_tokens=skip_special)
            text = text.split("<|endoftext|>")[0]
    except Exception as exc:
        # Never crash the client: return an empty (scored-wrong) response instead.
        print(f"[warn] generation failed: {exc}")

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="FrontiersMind/Lumma-0.6B-Tool")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    global MODEL, TOKENIZER, MODEL_ID, MAX_INPUT_TOKENS
    MODEL_ID = args.model_path
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model_path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    MODEL.eval()

    ctx = getattr(MODEL.config, "max_position_embeddings", None)
    if ctx:
        MAX_INPUT_TOKENS = ctx - 16
        print(f"Model context length: {ctx} (skipping prompts > {MAX_INPUT_TOKENS} tokens)")

    print(f"Model ready (sequential, no batching). Serving on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
