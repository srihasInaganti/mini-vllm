# how to turn a row of logits into the next token: greedy, or temperature + top_p
from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0   # 0 means greedy (just take the top token)
    top_p: float = 1.0         # 1.0 means no nucleus filtering


def sample(logits: torch.Tensor, params: SamplingParams) -> int:
    """logits: (vocab,) for one sequence -> a single token id."""
    if params.temperature == 0.0:
        return int(logits.argmax())

    # temperature flattens (>1) or sharpens (<1) the distribution before sampling
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        probs = _top_p_filter(probs, params.top_p)
    return int(torch.multinomial(probs, num_samples=1))


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    # keep the smallest set of tokens whose probabilities add up to top_p, drop the
    # long tail, then renormalize so what's left sums to 1
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # a token is kept if everything before it summed to less than top_p
    # (so the token that crosses the line is included, and the top token always stays)
    keep = (cumsum - sorted_probs) < top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    out = torch.zeros_like(probs)
    out[sorted_idx] = sorted_probs
    return out
