# the building blocks: RMSNorm, attention, SwiGLU MLP, decoder layer
# no KV cache, so each forward re-attends over the whole sequence (O(T^2))
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope


class RMSNorm(nn.Module):
    """Scales each token vector to a steady size. Cheaper than LayerNorm (no
    mean-subtract, no bias), what Qwen/Llama use. Done in float32 then cast back,
    like HF, so it stays stable at any model dtype."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # divide each vector by its own root-mean-square so its size is consistent
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(in_dtype))


class Attention(nn.Module):
    """Grouped-query attention. 14 query heads, 2 KV heads (each KV head serves
    7 query heads). q/k/v have a bias, o_proj doesn't — a Qwen2 quirk."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        q_dim = cfg.num_q_heads * cfg.head_dim    # 896
        kv_dim = cfg.num_kv_heads * cfg.head_dim  # 128
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(self, x, cos, sin, positions):
        B, T, _ = x.shape
        cfg = self.cfg
        # project then split into heads: (B, T, H*d) -> (B, H, T, d)
        q = self.q_proj(x).view(B, T, cfg.num_q_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, positions)

        # copy each KV head 7 times so it lines up with the query heads it serves
        k = k.repeat_interleave(cfg.n_rep, dim=1)  # (B, num_q_heads, T, d)
        v = v.repeat_interleave(cfg.n_rep, dim=1)

        # attention scores: how much each token attends to every earlier token
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        # block a token from looking at tokens that come after it
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        # softmax in float32 for stability, then back to x's dtype
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v)                                  # (B, H, T, d)

        out = out.transpose(1, 2).reshape(B, T, -1)                  # merge heads
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x)). The gate lets the network learn
    which channels to let through — beats a plain ReLU/GELU MLP in practice."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """One transformer block. Norm before each sublayer, add the input back
    around it (pre-norm), which keeps deep stacks stable."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, positions):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
