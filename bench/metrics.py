# turn raw per-request timings into the numbers that go in the table
import numpy as np


def summarize(samples: list[tuple[float, float]], total_tokens: int, wall_time: float) -> dict:
    """samples: list of (ttft_seconds, latency_seconds) per request."""
    ttfts = [s[0] for s in samples]
    latencies = [s[1] for s in samples]
    return {
        # throughput is the whole batch's tokens over the wall-clock it took to serve them
        "throughput_tok_s": total_tokens / wall_time,
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p99": float(np.percentile(latencies, 99)),
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
    }
