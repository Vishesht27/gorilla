"""Minimal OpenAI-compatible completions server for custom-architecture HF models.

BFCL's OSS pipeline talks to a local model through an OpenAI-compatible
`/v1/completions` endpoint. Because Lumma uses a custom `nandi` architecture that
vLLM/sglang cannot load, we serve it with plain `transformers` + trust_remote_code
here, then run BFCL with `--skip-server-setup`.

Usage:
    python lumma_server.py --model-path FrontiersMind/Lumma-0.6B-Tool --port 1053

Then, in another shell:
    bfcl generate --model FrontiersMind/Lumma-0.6B-Tool \
      --test-category single_turn --skip-server-setup
"""

from __future__ import annotations

import argparse
import time
import threading
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Generation is not thread-safe on a single model instance; serialize requests.
_GEN_LOCK = threading.Lock()

MODEL = None
TOKENIZER = None
MODEL_ID = None

app = FastAPI()


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    # BFCL may forward these via extra_body; accepted but optional.
    stop_token_ids: list[int] | None = None
    skip_special_tokens: bool | None = None


@app.get("/v1/models")
def list_models():
    # BFCL polls this endpoint to detect server readiness.
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    inputs = TOKENIZER(req.prompt, return_tensors="pt").to(MODEL.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    do_sample = req.temperature is not None and req.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=req.max_tokens,
        do_sample=do_sample,
        eos_token_id=TOKENIZER.eos_token_id,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = req.temperature
    if req.stop_token_ids:
        gen_kwargs["eos_token_id"] = req.stop_token_ids

    with _GEN_LOCK, torch.no_grad():
        out = MODEL.generate(**inputs, **gen_kwargs)

    gen_ids = out[0, prompt_tokens:]
    completion_tokens = int(gen_ids.shape[0])
    skip_special = bool(req.skip_special_tokens) if req.skip_special_tokens is not None else False
    text = TOKENIZER.decode(gen_ids, skip_special_tokens=skip_special)
    # Lumma terminates a turn with <|endoftext|>; trim anything after it.
    text = text.split("<|endoftext|>")[0]

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

    global MODEL, TOKENIZER, MODEL_ID
    MODEL_ID = args.model_path
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model_path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    MODEL.eval()
    print(f"Model ready. Serving OpenAI-compatible endpoint on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
