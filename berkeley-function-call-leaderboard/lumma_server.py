"""Dynamic-batching OpenAI-compatible completions server for custom-arch HF models.

BFCL's OSS pipeline talks to a local model through an OpenAI-compatible
`/v1/completions` endpoint. Because Lumma uses a custom `nandi` architecture that
vLLM/sglang cannot load, we serve it with plain `transformers` + trust_remote_code
here, then run BFCL with `--skip-server-setup`.

To exploit spare GPU, a background worker merges concurrent requests into a single
batched `generate()` call (dynamic batching). Pair this with a high BFCL
`--num-threads` so the batches stay full.

Usage:
    python lumma_server.py --model-path FrontiersMind/Lumma-0.6B-Tool \
        --port 1053 --max-batch-size 16

    bfcl generate --model FrontiersMind/Lumma-0.6B-Tool \
        --test-category all --skip-server-setup --num-threads 16
"""

from __future__ import annotations

import argparse
import queue
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

MAX_BATCH_SIZE = 16
BATCH_WAIT_S = 0.01  # how long to wait accumulating a batch
MAX_NEW_TOKENS_CAP = 4096

_REQUEST_QUEUE: "queue.Queue[_Job]" = queue.Queue()

app = FastAPI()


class _Job:
    __slots__ = ("prompt", "max_tokens", "temperature", "event", "result")

    def __init__(self, prompt: str, max_tokens: int, temperature: float):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.event = threading.Event()
        self.result: dict | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    stop_token_ids: list[int] | None = None
    skip_special_tokens: bool | None = None


def _run_batch(batch: list[_Job]) -> None:
    prompts = [j.prompt for j in batch]
    enc = TOKENIZER(prompts, return_tensors="pt", padding=True).to(MODEL.device)
    input_len = enc["input_ids"].shape[1]

    temperature = batch[0].temperature
    do_sample = temperature is not None and temperature > 0.0
    max_new = min(MAX_NEW_TOKENS_CAP, max(j.max_tokens for j in batch))

    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        eos_token_id=TOKENIZER.eos_token_id,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = MODEL.generate(**enc, **gen_kwargs)

    gen = out[:, input_len:]
    eos_id = TOKENIZER.eos_token_id
    for i, job in enumerate(batch):
        row = gen[i]
        text = TOKENIZER.decode(row, skip_special_tokens=False).split("<|endoftext|>")[0]
        row_list = row.tolist()
        completion_tokens = row_list.index(eos_id) if eos_id in row_list else len(row_list)
        prompt_tokens = int(enc["attention_mask"][i].sum())
        job.result = {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        job.event.set()


def _worker_loop() -> None:
    while True:
        first = _REQUEST_QUEUE.get()
        batch = [first]
        deadline = time.time() + BATCH_WAIT_S
        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            try:
                batch.append(_REQUEST_QUEUE.get(timeout=timeout))
            except queue.Empty:
                break
        try:
            _run_batch(batch)
        except Exception as exc:  # surface the error to every waiter in the batch
            for job in batch:
                job.result = {"error": str(exc)}
                job.event.set()


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    job = _Job(req.prompt, req.max_tokens, req.temperature)
    _REQUEST_QUEUE.put(job)
    job.event.wait()

    result = job.result or {}
    if "error" in result:
        return {"error": {"message": result["error"], "type": "server_error"}}

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {"text": result["text"], "index": 0, "logprobs": None, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="FrontiersMind/Lumma-0.6B-Tool")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=int, default=10)
    args = parser.parse_args()

    global MODEL, TOKENIZER, MODEL_ID, MAX_BATCH_SIZE, BATCH_WAIT_S
    MODEL_ID = args.model_path
    MAX_BATCH_SIZE = args.max_batch_size
    BATCH_WAIT_S = args.batch_wait_ms / 1000.0
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model_path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Left padding is required for correct batched decoder-only generation.
    TOKENIZER.padding_side = "left"
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    MODEL.eval()

    worker = threading.Thread(target=_worker_loop, daemon=True)
    worker.start()

    print(
        f"Model ready. Dynamic batching (max_batch={MAX_BATCH_SIZE}, "
        f"wait={args.batch_wait_ms}ms). Serving on {args.host}:{args.port}"
    )
    # Allow enough threadpool workers to hold many concurrent BFCL requests.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
