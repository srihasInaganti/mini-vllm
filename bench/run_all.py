# entrypoint: run the workload through each engine at each concurrency, write the table.
# meant to run on a single CUDA box where mini-vllm, vLLM, and transformers all sit on
# identical hardware/dtype. see bench/README.md for the full runbook.
import argparse
import asyncio
from functools import partial
from pathlib import Path

import httpx

from engine.config import ModelConfig
from . import report, workload
from .clients import run_transformers
from .runner import gpu_mem_used_mb, max_concurrency_before_oom, run_server_engine

MODEL = ModelConfig.model_id


def _client_factory(base_url: str):
    return lambda: httpx.AsyncClient(base_url=base_url, timeout=None)


async def _bench_server(name: str, base_url: str) -> list[dict]:
    make_client = _client_factory(base_url)
    rows = []
    for c in workload.CONCURRENCIES:
        metrics = await run_server_engine(
            make_client, MODEL, workload.prompts(workload.NUM_REQUESTS), workload.MAX_TOKENS, c)
        metrics.update(engine=name, concurrency=c, peak_mem_gb=_to_gb(gpu_mem_used_mb()))
        rows.append(metrics)
    # the OOM ceiling is the memory-efficiency headline, measured once per engine
    ceiling = await max_concurrency_before_oom(
        make_client, MODEL, workload.prompts(1)[0], workload.MAX_TOKENS)
    rows[-1]["max_concurrency"] = ceiling
    return rows


def _bench_transformers() -> list[dict]:
    rows = []
    for c in workload.CONCURRENCIES:
        metrics = run_transformers(workload.prompts(workload.NUM_REQUESTS), workload.MAX_TOKENS, c)
        metrics.update(engine="transformers", concurrency=c, peak_mem_gb=_to_gb(gpu_mem_used_mb()))
        rows.append(metrics)
    return rows


def _to_gb(mb: float | None) -> float | None:
    return None if mb is None else mb / 1024.0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini-vllm-url", default="http://localhost:8000")
    ap.add_argument("--vllm-url", default="http://localhost:8001")
    ap.add_argument("--skip-transformers", action="store_true")
    ap.add_argument("--out", default="bench/results.md")
    args = ap.parse_args()

    rows: list[dict] = []
    rows += await _bench_server("mini-vllm", args.mini_vllm_url)
    rows += await _bench_server("vllm", args.vllm_url)
    if not args.skip_transformers:
        rows += _bench_transformers()

    table = report.to_markdown(rows) + "\n" + report.pct_of_vllm(rows) + "\n"
    Path(args.out).write_text(table)
    print(table)


if __name__ == "__main__":
    asyncio.run(main())
