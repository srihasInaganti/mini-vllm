# prompt -> logits, plus two greedy loops:
#   greedy_decode       — re-runs the whole forward each step (O(T^2))
#   paged_greedy_decode — uses the paged KV cache, so each step is O(T)
import torch

from .block_manager import BlockManager, BlockTable
from .kv_cache import PagedKVCache
from .model import Qwen2Model


@torch.no_grad()
def forward_logits(model: Qwen2Model, input_ids: torch.Tensor) -> torch.Tensor:
    """token ids (B, T) -> logits (B, T, vocab)."""
    return model(input_ids)


@torch.no_grad()
def greedy_decode(model: Qwen2Model, input_ids: torch.Tensor,
                  max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Greedy = always pick the highest-scoring next token. Returns new ids only."""
    ids = input_ids.clone()
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(ids)                       # (1, T, vocab)
        # take the top token at the last position
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
    return generated


@torch.no_grad()
def paged_greedy_decode(model: Qwen2Model, input_ids: torch.Tensor,
                        max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Same greedy rule as above, but K/V are cached so past tokens aren't recomputed."""
    cfg = model.cfg
    device = input_ids.device
    # size the pool for prompt + generation, with a little slack
    max_len = input_ids.shape[1] + max_new_tokens
    num_blocks = (max_len + cfg.block_size - 1) // cfg.block_size + 2
    cache = PagedKVCache(cfg, num_blocks, cfg.block_size)
    block_table = BlockTable(BlockManager(num_blocks, cfg.block_size))

    generated: list[int] = []
    # first pass feeds the whole prompt (prefill); later passes feed one new token (decode)
    tokens = input_ids[0]
    for _ in range(max_new_tokens):
        logits = model.forward_paged(tokens, block_table, cache)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == eos_token_id:
            break
        tokens = torch.tensor([next_id], device=device)
    return generated
