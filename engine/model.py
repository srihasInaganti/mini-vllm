# the full model wired together: embed -> 24 decoder layers -> final norm -> head
import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import DecoderLayer, RMSNorm
from .rope import build_cos_sin


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
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
