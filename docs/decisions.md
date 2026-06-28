# Design Decisions

Every milestone there were some options to consider. This md file serves as a list of
what was considered, and what I chose and why.

---

## Milestone 1

The goal of this milestone is to load Qwen2.5-0.5B-Instruct from safetensors into my own modules, run a forward
pass, and match `transformers`' greedy tokens exactly.

### Device and dtype

**Problem:** token parity is a weak test. A different device or a smaller dtype
shifts the rounding, causing an argmax to flip, and the test fails.

**Options:** CPU + fp32 / MPS + bf16 / MPS + fp16.

**Decision:** I chose to go with CPU + fp32. It's the slowest, but it takes device and precision out
of the equation. When the tokens disagree I know the bug is in my forward pass, not in float noise.

**At scale:** fp32 on CPU is worthless for a real model. It's better to use bf16 on GPU,
and instead of trying to get identical tokens, compare how close the logits are instead.

### Attention

**Problem:** I want attention I can learn from, not a mystery box.

**Options:** write QK^T / mask / softmax / V by hand, or call
`scaled_dot_product_attention`.

**Decision:** I wrote it out by hand. SDPA is faster and what you would use if you were actually shipping,
but it hides the parts I want to understand: the GQA head sharing, the fp32
softmax, and the causal mask. Putting SDPA or FlashAttention back in later is
easy once I trust the mechanics.

### KV cache

**Problem:** greedy decode re-runs the full forward pass every step, which is
wasteful since the cost grows with the square of the sequence length.

**Decision:** I skipped it. A cache built now would get thrown out the
moment I start paging the KV cache in Milestone 2, so there's no point writing it
twice.

### Weight loading

**Problem:** I need to map every HF tensor into my own modules without silently
dropping one or inventing one.

**Decision:** I named my modules to match HF's and strip the `model.` prefix off
each tensor name, then load with `strict=True` so a mismatch throws instead of
passing quietly. The one issue is that Qwen ties its embeddings, so there's no
`lm_head` weight in the files. Logits come from the hidden states times the
embedding matrix, transposed.

### The parity bug

**Problem:** my argmax and `transformers`' "greedy" kept drifting apart even with
`do_sample=False`.

**Cause:** `generate()` still applies whatever is in `generation_config.json`, and
Qwen ships a `repetition_penalty` of 1.1, so the reference was never doing pure
greedy in the first place.

**Fix:** I reset `repetition_penalty`, `temperature`, `top_p`, and `top_k` to
their no-op values on the reference so both sides run the same decode. After that
the logits match to ~3.7e-5 and the argmax agrees at every position.

### Qwen2 specifics

Checked against the actual tensors, not memory:
- 14 query heads, 2 KV heads. Each KV head feeds 7 query heads (GQA).
- RoPE theta is 1e6, not 1e4. The wrong value corrupts later positions.
- q/k/v projections have a bias. o_proj doesn't.
- embeddings tied, no lm_head.
