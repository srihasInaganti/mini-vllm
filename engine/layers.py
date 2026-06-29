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

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx        # which slice of the KV cache is mine
        q_dim = cfg.num_q_heads * cfg.head_dim    # 896
        kv_dim = cfg.num_kv_heads * cfg.head_dim  # 128
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(self, x, cos, sin, positions, paged=None):
        B, T, _ = x.shape
        cfg = self.cfg
        # project then split into heads: (B, T, H*d) -> (B, H, T, d)
        q = self.q_proj(x).view(B, T, cfg.num_q_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, positions)

        if paged is None:
            # M1 path: the keys are just this sequence's own tokens
            # copy each KV head 7 times so it lines up with the query heads it serves
            k_full = k.repeat_interleave(cfg.n_rep, dim=1)
            v_full = v.repeat_interleave(cfg.n_rep, dim=1)
            # block a token from looking at tokens that come after it
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
            scores = scores + mask
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            out = torch.matmul(attn, v_full)
            out = out.transpose(1, 2).reshape(B, T, -1)              # merge heads
            return self.o_proj(out)

        # paged path: each sequence has its own history, so attend one sequence at a
        # time (the heavy projections above were already batched across the whole batch).
        outs = []
        for i in range(B):
            # stash this sequence's new K/V, then read its whole history back.
            # the cache stores (slot, kv_heads, d), so put tokens first.
            paged.cache.write(self.layer_idx, paged.write_slots[i],
                              k[i].transpose(0, 1), v[i].transpose(0, 1))
            k_all, v_all = paged.cache.gather(self.layer_idx, paged.gather_slots[i])
            ki = k_all.transpose(0, 1).repeat_interleave(cfg.n_rep, dim=0)  # (H, L, d)
            vi = v_all.transpose(0, 1).repeat_interleave(cfg.n_rep, dim=0)
            # a query at position p may attend any key whose position is <= p
            mask = torch.where(paged.key_positions[i].view(1, -1) <= positions[i].view(-1, 1),
                               0.0, float("-inf"))
            scores = torch.matmul(q[i], ki.transpose(-2, -1)) * self.scale  # (H, T, L)
            scores = scores + mask
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            outs.append(torch.matmul(attn, vi))                             # (H, T, d)

        out = torch.stack(outs, dim=0)                  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, -1)     # merge heads
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

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, positions, paged=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions, paged)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
