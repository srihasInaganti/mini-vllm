# Milestone 1 deliverable: prompt -> logits, plus a simple greedy decode loop
# no KV cache yet, so each step re-runs the whole forward pass (O(T^2), but correct)
import torch

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
