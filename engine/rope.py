# RoPE: rotate q and k by an angle that depends on position, so attention sees
# how far apart two tokens are. Must match HF's exact convention or parity breaks.
import torch


def build_cos_sin(seq_len: int, head_dim: int, theta: float,
                  device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables, shape (seq_len, head_dim)."""
    # each frequency pair spins at its own speed: low index fast, high index slow
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)          # (seq_len,)
    freqs = torch.outer(pos, inv_freq)                        # (seq_len, head_dim/2)
    # HF duplicates the freqs [f, f] so cos/sin line up with rotate_half below.
    emb = torch.cat([freqs, freqs], dim=-1)                   # (seq_len, head_dim)
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # split in half and rotate: [x1, x2] -> [-x2, x1]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k. q,k: (B, H, T, head_dim). positions: (T,) = where each token sits."""
    # gather the rows needed, then broadcast over batch + head dims: (1,1,T,d)
    cos = cos[positions].unsqueeze(0).unsqueeze(0)
    sin = sin[positions].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
