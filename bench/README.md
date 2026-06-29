# Benchmarks

Compares **mini-vllm** against two baselines on one identical machine:

- **transformers `generate()`** — the naive floor (static batching, no paging). Shows *why* paging + continuous batching matter.
- **vLLM** — the ceiling (production engine, custom CUDA kernels). Shows how close a pure-PyTorch engine gets.

The headline is mini-vllm's throughput as a **percentage of vLLM's**, measured on the same model, prompts, hardware, dtype, and concurrency.

## Why this needs a CUDA GPU

vLLM only runs on NVIDIA GPUs, so the whole comparison has to run on one CUDA box. mini-vllm is pure PyTorch and runs there too — point it at the GPU with env vars. Running mini-vllm on a Mac and vLLM elsewhere would not be a fair comparison.

## Runbook (single CUDA box)

Same model and dtype for all three (`Qwen/Qwen2.5-0.5B-Instruct`, bf16).

1. **Start mini-vllm** (its OpenAI server), on CUDA + bf16:
   ```
   MINIVLLM_DEVICE=cuda MINIVLLM_DTYPE=bf16 uvicorn server.app:app --port 8000
   ```

2. **Start vLLM** (its OpenAI server), same model + dtype:
   ```
   vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 --port 8001
   ```

3. **Run the harness** (drives both servers, runs transformers in-process):
   ```
   python -m bench.run_all --mini-vllm-url http://localhost:8000 --vllm-url http://localhost:8001
   ```

It writes `bench/results.md` and prints the table plus the %-of-vLLM line. Copy that table into the top-level README.

## What's measured

Per engine × concurrency (1, 8, 32): throughput (tok/s), p50/p99 latency, p50/p99 time-to-first-token, peak GPU memory, and max concurrent sequences before OOM. Output length is fixed (`ignore_eos`) so token counts are identical and throughput is directly comparable. transformers has no streaming, so its time-to-first-token equals full latency — that gap is part of the point.
