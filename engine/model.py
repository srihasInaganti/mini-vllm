# the full model wired together: embed -> 24 decoder layers -> final norm -> head
import torch
import torch.nn as nn

from .block_manager import BlockTable
from .config import ModelConfig
from .kv_cache import PagedKVCache, PagedStep
from .layers import DecoderLayer, RMSNorm
from .rope import build_cos_sin


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # each layer keeps its index so it reads/writes its own slice of the cache
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        # build the RoPE tables once, up to the longest sequence expected
        cos, sin = build_cos_sin(4096, cfg.head_dim, cfg.rope_theta, cfg.device, cfg.dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) -> logits: (B, T, vocab_size)."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, positions)
        x = self.norm(x)
        # tied head: reuse the embedding matrix instead of a separate lm_head
        return torch.matmul(x, self.embed_tokens.weight.t())

    @torch.no_grad()
    def forward_paged(self, input_ids: torch.Tensor, block_table: BlockTable,
                      cache: PagedKVCache) -> torch.Tensor:
        """input_ids: (T,) new tokens — the whole prompt on prefill, one token per decode step.

        Returns logits (1, T, vocab). K/V for these tokens are written into the cache,
        and attention runs over every token stored so far.
        """
        device = input_ids.device
        start = block_table.length             # absolute position of the first new token
        n_new = input_ids.shape[0]
        block_table.append(n_new)              # reserve blocks for the new tokens
        positions = torch.arange(start, start + n_new, device=device)

        step = PagedStep(
            cache=cache,
            write_slots=torch.tensor(block_table.slots(range(start, start + n_new)),
                                     dtype=torch.long, device=device),
            gather_slots=torch.tensor(block_table.all_slots(), dtype=torch.long, device=device),
            key_positions=torch.arange(block_table.length, device=device),
        )

        x = self.embed_tokens(input_ids).unsqueeze(0)   # (1, T, hidden)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, positions, paged=step)
        x = self.norm(x)
        return torch.matmul(x, self.embed_tokens.weight.t())
