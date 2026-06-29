# the one workload every engine runs: same prompts, same length, same sampling
# fixed length (ignore_eos) so every request emits exactly MAX_TOKENS and token counts match
MAX_TOKENS = 128
CONCURRENCIES = (1, 8, 32)
NUM_REQUESTS = 96            # enough samples for a meaningful p99

_PROMPTS = [
    "Explain how a CPU cache works.",
    "Write a short story about a lighthouse keeper.",
    "Summarize the causes of World War I.",
    "Describe the water cycle step by step.",
    "What is the difference between TCP and UDP?",
    "Give three tips for writing clean code.",
    "Explain recursion to a five year old.",
    "Outline a plan for learning to cook.",
]


def prompts(n: int) -> list[str]:
    # cycle the base prompts up to n requests so the mix is identical across engines
    return [_PROMPTS[i % len(_PROMPTS)] for i in range(n)]
