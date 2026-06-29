# drives a load against a server engine (mini-vllm or vLLM) and measures it, plus
# helpers for peak GPU memory and the max-concurrency-before-OOM stress test
import asyncio
import subprocess
import time
from typing import Callable

import httpx

from .clients import stream_completion
from .metrics import summarize


async def run_server_engine(make_client: Callable[[], httpx.AsyncClient], model: str,
                            prompts: list[str], max_tokens: int, concurrency: int) -> dict:
    """Fire all prompts with at most `concurrency` in flight; return the metrics dict."""
    sem = asyncio.Semaphore(concurrency)

    async with make_client() as client:
        async def one(prompt: str):
            async with sem:
                return await stream_completion(client, model, prompt, max_tokens)

        start = time.perf_counter()
        samples = await asyncio.gather(*(one(p) for p in prompts))
        wall = time.perf_counter() - start

    return summarize(list(samples), len(prompts) * max_tokens, wall)


def gpu_mem_used_mb() -> float | None:
    # peak memory is sampled from nvidia-smi; returns None off a CUDA box (e.g. local Mac)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return max(float(x) for x in out.decode().split())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


async def max_concurrency_before_oom(make_client, model, prompt, max_tokens,
                                     start=8, step=8, ceiling=512) -> int:
    """Ramp simultaneous requests until one fails; return the last level that fully succeeded."""
    last_ok = 0
    level = start
    while level <= ceiling:
        try:
            async with make_client() as client:
                await asyncio.gather(
                    *(stream_completion(client, model, prompt, max_tokens) for _ in range(level)))
            last_ok = level
            level += step
        except Exception:
            break
    return last_ok
