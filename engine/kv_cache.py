# the physical KV pool: one K and one V tensor per layer
# each tensor is a flattened (num_blocks * block_size) view, so a flat slot id indexes it directly
from dataclasses import dataclass

import torch

from .config import ModelConfig


class PagedKVCache:
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int):
        self.block_size = block_size
        num_slots = num_blocks * block_size
        # only num_kv_heads stored (not the 14 query heads) — the GQA memory win
        shape = (num_slots, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=cfg.device, dtype=cfg.dtype) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, device=cfg.device, dtype=cfg.dtype) for _ in range(cfg.num_layers)]

    def write(self, layer: int, slot_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        # k, v: (count, kv_heads, head_dim) for the tokens being written
        self.k[layer][slot_ids] = k
        self.v[layer][slot_ids] = v

    def gather(self, layer: int, slot_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pull a sequence's stored K/V back into contiguous order: (count, kv_heads, head_dim)
        return self.k[layer][slot_ids], self.v[layer][slot_ids]


@dataclass
class PagedStep:
    """Everything attention needs to use the cache for one forward step.

    The three lists are indexed by batch position — one entry per sequence — so a
    single forward can serve a batch of sequences that each have their own history.
    """
    cache: PagedKVCache
    write_slots: list[torch.Tensor]    # per sequence: where this step's new tokens go
    gather_slots: list[torch.Tensor]   # per sequence: every token to attend over
    key_positions: list[torch.Tensor]  # per sequence: absolute position of each gathered key
