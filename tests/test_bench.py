# validate the harness mechanics locally: metric math, table formatting, and the
# async load driver end-to-end against mini-vllm's own server (CPU here; CUDA on the bench box)
import asyncio

import httpx

from bench import report
from bench.metrics import summarize
from bench.runner import run_server_engine
from server.app import app


def test_summarize_math():
    samples = [(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)]
    m = summarize(samples, total_tokens=30, wall_time=3.0)
    assert m["throughput_tok_s"] == 10.0       # 30 tokens / 3 s
    assert m["latency_p50"] == 2.0
    assert m["ttft_p50"] == 0.2


def test_table_and_pct_of_vllm():
    rows = [
        {"engine": "mini-vllm", "concurrency": 8, "throughput_tok_s": 80.0},
        {"engine": "vllm", "concurrency": 8, "throughput_tok_s": 100.0},
    ]
    table = report.to_markdown(rows)
    assert "Throughput (tok/s)" in table and "mini-vllm" in table
    assert "**80.0%** of vLLM" in report.pct_of_vllm(rows)


def test_load_driver_against_mini_vllm():
    async def body():
        # run the real server lifespan (loads model, starts the step loop), drive it over ASGI
        async with app.router.lifespan_context(app):
            def make_client():
                return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://bench", timeout=None)
            prompts = ["2 + 2 =", "Hello there", "The sky is", "Once upon a"]
            return await run_server_engine(make_client, "qwen", prompts,
                                           max_tokens=4, concurrency=2)

    m = asyncio.run(body())
    assert m["throughput_tok_s"] > 0
    assert m["ttft_p50"] <= m["latency_p50"] + 1e-6
