# continuous-batching scheduler: every step admit new requests, decode the whole
# running batch one token, retire finished sequences, and preempt when blocks run out
from collections import deque

import torch

from .block_manager import BlockManager, BlockTable
from .config import ModelConfig
from .kv_cache import PagedKVCache
from .loader import load_model
from .model import Qwen2Model


class Sequence:
    """One request's state as it moves through waiting -> running -> finished."""

    def __init__(self, seq_id, prompt_ids: list[int], max_tokens: int, eos_id: int | None):
        self.seq_id = seq_id
        self.prompt_ids = list(prompt_ids)
        self.output_ids: list[int] = []
        self.max_tokens = max_tokens
        self.eos_id = eos_id
        self.block_table: BlockTable | None = None   # set when admitted, cleared when preempted

    @property
    def last_token(self) -> int:
        # the newest token, which the next decode step feeds back in
        return self.output_ids[-1] if self.output_ids else self.prompt_ids[-1]

    @property
    def is_finished(self) -> bool:
        if len(self.output_ids) >= self.max_tokens:
            return True
        return bool(self.output_ids) and self.output_ids[-1] == self.eos_id

    def append(self, token: int) -> None:
        self.output_ids.append(token)


class Scheduler:
    def __init__(self, model: Qwen2Model, cfg: ModelConfig, num_blocks: int):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.block_size = cfg.block_size
        self.manager = BlockManager(num_blocks, self.block_size)
        self.cache = PagedKVCache(cfg, num_blocks, self.block_size)
        self.waiting: deque[Sequence] = deque()   # FCFS queue of requests not currently running
        self.running: list[Sequence] = []         # sequences in the decode batch
        self.num_preemptions = 0                   # for tests / debugging

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _admit(self) -> None:
        # FCFS: take from the front while the next request's prompt fits. Stopping at the
        # first one that doesn't fit (instead of skipping ahead) is what prevents starvation.
        while self.waiting:
            seq = self.waiting[0]
            n_tokens = len(seq.prompt_ids) + len(seq.output_ids)
            if self._blocks_for(n_tokens) > self.manager.num_free:
                break
            self.waiting.popleft()
            self._prefill(seq)
            self.running.append(seq)

    def _prefill(self, seq: Sequence) -> None:
        # build the cache for everything known so far, then generate one token.
        # after a preemption this re-runs over prompt + already-generated tokens.
        seq.block_table = BlockTable(self.manager)
        tokens = torch.tensor(seq.prompt_ids + seq.output_ids, device=self.device)
        logits = self.model.forward_paged(tokens, seq.block_table, self.cache)
        seq.append(int(logits[0, -1].argmax()))

    def _make_room(self, active: list[Sequence]) -> None:
        # a sequence needs a fresh block this step exactly when its length is block-aligned
        def needed() -> int:
            return sum(1 for s in active if s.block_table.length % self.block_size == 0)

        # preempt the newest sequences first so the oldest keep making progress (no starvation)
        while len(active) > 1 and self.manager.num_free < needed():
            victim = active.pop()
            self.running.remove(victim)
            victim.block_table.free()
            victim.block_table = None
            self.waiting.appendleft(victim)   # resume first once memory frees up
            self.num_preemptions += 1

    def _decode(self) -> None:
        active = [s for s in self.running if not s.is_finished]
        if not active:
            return
        self._make_room(active)
        logits = self.model.forward_decode(
            [s.last_token for s in active], [s.block_table for s in active], self.cache)
        for s, row in zip(active, logits):
            s.append(int(row.argmax()))

    def _retire(self) -> list[Sequence]:
        done = [s for s in self.running if s.is_finished]
        for s in done:
            s.block_table.free()             # finished sequences hand their blocks back
            s.block_table = None
        self.running = [s for s in self.running if not s.is_finished]
        return done

    def step(self) -> list[Sequence]:
        """Run one scheduler iteration. Returns the sequences that finished this step."""
        self._admit()
        self._decode()
        return self._retire()


class LLMEngine:
    """Ties the model and scheduler together and runs requests to completion."""

    def __init__(self, model: Qwen2Model | None = None, cfg: ModelConfig | None = None,
                 num_blocks: int = 512):
        self.cfg = cfg or ModelConfig()
        self.model = model or load_model(self.cfg)
        self.scheduler = Scheduler(self.model, self.cfg, num_blocks)

    def add(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def run(self) -> dict:
        # drive the loop until every request is done; return {seq_id: output_ids}
        results = {}
        while self.scheduler.has_work:
            for seq in self.scheduler.step():
                results[seq.seq_id] = seq.output_ids
        return results
