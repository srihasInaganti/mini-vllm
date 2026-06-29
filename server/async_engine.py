# async wrapper around the scheduler: one background loop owns the model and steps it,
# while each HTTP request submits a sequence and awaits its tokens over its own queue
import asyncio

from engine.config import ModelConfig
from engine.loader import load_model
from engine.sampling import SamplingParams
from engine.scheduler import Scheduler, Sequence


class AsyncLLMEngine:
    def __init__(self, model=None, cfg: ModelConfig | None = None,
                 eos_id: int | None = None, num_blocks: int = 512):
        self.cfg = cfg or ModelConfig()
        self.model = model or load_model(self.cfg)
        self.eos_id = eos_id
        self.scheduler = Scheduler(self.model, self.cfg, num_blocks)
        self._queues: dict[int, asyncio.Queue] = {}   # seq_id -> tokens delivered to its handler
        self._emitted: dict[int, int] = {}            # seq_id -> how many tokens already delivered
        self._next_id = 0
        self._loop_task: asyncio.Task | None = None

    def start(self) -> None:
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()

    async def _run_loop(self) -> None:
        # step forever; yield to the event loop each time so handlers can register requests
        while True:
            if self.scheduler.has_work:
                finished = self.scheduler.step()
                self._fan_out(finished)
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.005)   # nothing to do; don't busy-spin

    def _fan_out(self, finished: list[Sequence]) -> None:
        # push any newly produced tokens to each request's queue, plus a None sentinel on finish
        finished_ids = {s.seq_id for s in finished}
        for seq in self.scheduler.running + finished:
            queue = self._queues.get(seq.seq_id)
            if queue is None:
                continue
            already = self._emitted.get(seq.seq_id, 0)
            for token in seq.output_ids[already:]:
                queue.put_nowait(token)
            self._emitted[seq.seq_id] = len(seq.output_ids)
            if seq.seq_id in finished_ids:
                queue.put_nowait(None)

    async def generate(self, prompt_ids: list[int], max_tokens: int, params: SamplingParams):
        """Submit a request and yield its token ids one at a time as they're produced."""
        seq_id = self._next_id
        self._next_id += 1
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[seq_id] = queue
        self._emitted[seq_id] = 0
        self.scheduler.add(Sequence(seq_id, prompt_ids, max_tokens, self.eos_id, params))
        try:
            while True:
                token = await queue.get()
                if token is None:        # sentinel: this sequence is finished
                    break
                yield token
        finally:
            self._queues.pop(seq_id, None)
            self._emitted.pop(seq_id, None)
