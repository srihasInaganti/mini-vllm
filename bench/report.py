# assemble the results into the markdown table that goes in the README,
# including the headline: mini-vllm throughput as a percentage of vLLM's
_COLUMNS = [
    ("engine", "Engine"),
    ("concurrency", "Concurrency"),
    ("throughput_tok_s", "Throughput (tok/s)"),
    ("latency_p50", "Lat p50 (s)"),
    ("latency_p99", "Lat p99 (s)"),
    ("ttft_p50", "TTFT p50 (s)"),
    ("ttft_p99", "TTFT p99 (s)"),
    ("peak_mem_gb", "Peak mem (GB)"),
    ("max_concurrency", "Max conc. before OOM"),
]


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in _COLUMNS) + " |")
    return "\n".join(lines)


def pct_of_vllm(rows: list[dict]) -> str:
    """The resume line: mini-vllm throughput as a % of vLLM at each concurrency."""
    vllm = {r["concurrency"]: r["throughput_tok_s"]
            for r in rows if r["engine"] == "vllm"}
    out = ["", "### mini-vllm throughput as % of vLLM", ""]
    for r in rows:
        if r["engine"] != "mini-vllm":
            continue
        ceiling = vllm.get(r["concurrency"])
        if not ceiling:
            continue
        pct = 100.0 * r["throughput_tok_s"] / ceiling
        out.append(f"- concurrency {r['concurrency']}: **{pct:.1f}%** of vLLM")
    return "\n".join(out)
