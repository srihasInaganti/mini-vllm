# OpenAI-compatible server: POST /v1/completions, streaming (SSE) and non-streaming
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from engine.config import ModelConfig
from engine.sampling import SamplingParams
from server.async_engine import AsyncLLMEngine


class CompletionRequest(BaseModel):
    model: str = ModelConfig.model_id
    prompt: str
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


# filled in at startup so the model loads once and the step loop starts on the event loop
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    engine = AsyncLLMEngine(cfg=cfg, eos_id=tokenizer.eos_token_id)
    engine.start()
    state.update(cfg=cfg, tokenizer=tokenizer, engine=engine)
    yield
    await engine.stop()


app = FastAPI(lifespan=lifespan)


def _finish_reason(ids: list[int], max_tokens: int, eos_id: int | None) -> str:
    # OpenAI uses "stop" when generation ended on its own, "length" when it hit the cap
    return "stop" if ids and ids[-1] == eos_id else "length"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    tokenizer = state["tokenizer"]
    engine = state["engine"]
    eos_id = tokenizer.eos_token_id
    prompt_ids = tokenizer(req.prompt).input_ids
    params = SamplingParams(temperature=req.temperature, top_p=req.top_p)

    cmpl_id = "cmpl-" + uuid.uuid4().hex
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream(engine, tokenizer, prompt_ids, req, cmpl_id, created, eos_id),
            media_type="text/event-stream",
        )

    # non-streaming: drain the whole generation, then return one body
    ids = [tok async for tok in engine.generate(prompt_ids, req.max_tokens, params)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": None,
            "finish_reason": _finish_reason(ids, req.max_tokens, eos_id),
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
            "total_tokens": len(prompt_ids) + len(ids),
        },
    }


async def _stream(engine, tokenizer, prompt_ids, req, cmpl_id, created, eos_id):
    params = SamplingParams(temperature=req.temperature, top_p=req.top_p)

    def chunk(text: str, finish_reason):
        body = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": req.model,
            "choices": [{"text": text, "index": 0, "logprobs": None,
                         "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(body)}\n\n"

    ids: list[int] = []
    prev_text = ""
    async for token in engine.generate(prompt_ids, req.max_tokens, params):
        ids.append(token)
        # decode the whole output and emit only the new suffix, so characters that
        # span several tokens come out whole instead of as broken bytes
        text = tokenizer.decode(ids, skip_special_tokens=True)
        delta, prev_text = text[len(prev_text):], text
        if delta:
            yield chunk(delta, None)

    yield chunk("", _finish_reason(ids, req.max_tokens, eos_id))
    yield "data: [DONE]\n\n"
