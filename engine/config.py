# model + runtime settings, frozen so nothing can change the shape mid-run
# numbers taken straight from Qwen2.5-0.5B-Instruct's config.json
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelConfig:
    # --- architecture (verified against the HF config.json) ---
    hidden_size: int = 896
    num_layers: int = 24
    num_q_heads: int = 14            # query heads
    num_kv_heads: int = 2           # GQA: 2 KV heads shared across the 14 q heads
    head_dim: int = 64              # 896 / 14
    intermediate_size: int = 4864   # SwiGLU inner width
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0  # NOTE: 1e6, not the usual 1e4
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True  # lm_head shares embed_tokens.weight

    # --- runtime (Milestone 1: CPU + fp32 for exact parity with transformers) ---
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    block_size: int = 16            # tokens per KV cache block

    @property
    def n_rep(self) -> int:
        # how many query heads each KV head feeds (14 / 2 = 7)
        return self.num_q_heads // self.num_kv_heads
