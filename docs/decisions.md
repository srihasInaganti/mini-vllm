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

---

## Milestone 2

The goal is to stop recomputing K/V every decode step. Each token's K/V get
stored once and looked up later, but the storage is split into fixed-size blocks
drawn from a shared pool instead of one contiguous tensor per sequence. Output
still has to match Milestone 1 token-for-token.

### Why paging instead of one contiguous tensor per sequence

A contiguous cache means reserving a slab big enough for the longest
the sequence might ever get. Real prompts don't all reach that length, so most of
each slab sits empty but reserved. And once a few sequences of different lengths
come and go, the free space left between them is chopped into pieces too small to
fit the next request even when the total free space is plenty.

To get around this I utilized paging because it cuts down the memory into uniform blocks.
A sequence grabs blocks one at a time only as it grows, and a block table maps the
sequence's logical positions to whatever physical blocks it happened to get. The
blocks don't have to be next to each other, so there are no unusable gaps. 
The waste drops from a whole slab to at most one partly filled block per sequence.

### How attention reads the cache

**Problem:** the keys and values for a sequence are now scattered across blocks
that aren't contiguous, but the attention math wants them in order.

**Options:** gather the blocks back into a contiguous tensor and run the M1
attention as-is, or attend block-by-block with a running softmax and never
reconstruct.

**Decision:** I gather and reconstruct. It reuses the exact M1 attention, so
parity is easy to trust, and the only cost is briefly materializing the
contiguous K/V. The block-by-block version is closer to a real paged kernel but
much more code and a parity risk. This is also the spot where a custom CUDA paged
kernel would slot in later. Instead of having to gather, a custom CUDA kernal could read straight from the blocks.

### Pool layout

**Problem:** where do the K/V tensors actually live.

**Decision:** one K and one V tensor per layer, each a flat
(num_blocks * block_size, num_kv_heads, head_dim) tensor. Flat so a single slot id
(block * block_size + offset) indexes it directly, which makes write and gather
one-liners. Only the 2 KV heads are stored.

### Scope

**Decision:** one sequence at a time. Batching across sequences and an eviction
policy when the pool runs out belong with the scheduler, so I kept those out and
let `allocate` just raise when blocks run dry. That keeps this milestone purely
about cache accuracy.

### At scale

The pool is sized once from a memory budget. At 100x traffic that budget, the
block size, and how full the last block runs become the levers that decide how
many sequences fit — and the pure-PyTorch gather becomes the bottleneck a CUDA
kernel would replace.

---

## Milestone 3

The goal is a scheduler loop that serves many requests at once: every step it
admits new requests, decodes one token for the whole running batch, retires
sequences that just finished, and preempts when the block pool runs out.

### Static vs continuous batching

Static batching gathers a group of requests, runs them together until they all
finish, then takes the next group. The whole group runs at the speed of its
longest sequence. When a short request finishes early its slot just sits there
doing wasted work until the rest catch up, and a request that shows up in the
middle has to wait for the entire group to drain before it can start.

Continuous batching rebuilds the batch every single decode step instead. A
sequence that hits its limit is retired that step and its blocks freed; a request
that just arrived is admitted into the next step. The batch is always full of
sequences that are actually still generating, so the work isn't wasted on
finished or padded slots. It only works because we implemented paging, which is
why the cache came first.

### Batching ragged lengths

**Problem:** in a decode step every sequence is at a different length, so their
gathers don't line up into one tensor.

**Options:** loop the attention per sequence, or pad every sequence to the batch
max length and attend in one masked batch.

**Decision:** I loop the attention. Padding would waste compute.

### Getting a new request into the batch

**Problem:** a new request's prompt has to be processed before it can decode.

**Decision:** prefill it on its own first, then it joins a decode batch where every
sequence contributes exactly one token per step. Keeping prefill and decode
separate makes the steady-state batch uniform and easy to reason about. Mixing
many-token prefills and one-token decodes into a single forward is what production
engines do, but the indexing is much fiddlier.

### Eviction policy

**Problem:** when the pool runs dry mid-step, something running has to give.

**Decision:** preempt and recompute. I free a victim's blocks and put it back on
the waiting queue; when it's readmitted its KV is rebuilt by re-running prefill
over its prompt plus the tokens it already generated. No extra memory, and
recompute is correct because a token's K/V only depend on the token and its
position. The victim is the most recently admitted sequence, and it goes to the
front of the waiting queue. Picking the newest means the oldest sequences are
never the ones knocked out, so they keep making progress and nothing starves; the
front-of-queue placement lets a preempted sequence resume as soon as memory frees.
The alternative, swapping blocks to host RAM, avoids the recompute but needs swap
space and copy logic.

### At scale

The per-sequence attention loop is the obvious bottleneck, it's Python overhead
times batch size times layers. At 100x traffic this is the first thing to go: a
varlen paged-attention kernel handles the whole ragged batch in one launch.
Recompute-on-preempt also gets expensive for long sequences, so past some length
swapping to host memory (or just a bigger pool) wins.
