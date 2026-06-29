# how the harness talks to each engine: an OpenAI streaming client (mini-vllm + vLLM
# share it) and an in-process transformers runner (the naive baseline, no server)
import json
import time

import httpx

from engine.config import config_from_env


async def stream_completion(client: httpx.AsyncClient, model: str, prompt: str,
                            max_tokens: int) -> tuple[float, float]:
    """Send one streaming request. Returns (time-to-first-token, total latency) in seconds."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True, "ignore_eos": True}
    start = time.perf_counter()
    ttft = None
    async with client.stream("POST", "/v1/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk["choices"][0]["text"]:
                ttft = time.perf_counter() - start    # first visible token
    latency = time.perf_counter() - start
    return (ttft if ttft is not None else latency), latency


def run_transformers(prompts: list[str], max_tokens: int, concurrency: int) -> dict:
    """Naive baseline: vanilla transformers generate() with static batching.

    Processes `concurrency` prompts at a time as one fixed batch. generate() returns
    everything at once, so there's no streaming and time-to-first-token equals the full
    latency — which is exactly the limitation continuous batching + streaming remove.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .metrics import summarize

    # same device/dtype as the other engines, so the floor baseline is a fair comparison
    cfg = config_from_env()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=cfg.dtype).to(cfg.device).eval()

    samples: list[tuple[float, float]] = []
    wall_start = time.perf_counter()
    for i in range(0, len(prompts), concurrency):
        batch = prompts[i:i + concurrency]
        enc = tok(batch, return_tensors="pt", padding=True).to(cfg.device)
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=max_tokens, min_new_tokens=max_tokens,
                           do_sample=False)
        latency = time.perf_counter() - start
        # the whole static batch finishes together, so every request sees the same latency
        samples += [(latency, latency)] * len(batch)
    wall = time.perf_counter() - wall_start

    return summarize(samples, len(prompts) * max_tokens, wall)
