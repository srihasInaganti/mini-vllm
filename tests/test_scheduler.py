# Milestone 3 gate: scheduler invariants (no leak, no starvation, fairness) and
# output correctness — batched/preempted decode must match the M2 single-sequence path
import pytest
from transformers import AutoTokenizer

from engine.config import ModelConfig
from engine.generate import paged_greedy_decode
from engine.loader import load_model
from engine.scheduler import LLMEngine, Scheduler, Sequence


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig()


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg.model_id)


@pytest.fixture(scope="module")
def model(cfg):
    return load_model(cfg)


def _ids(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()


def _reference(model, tokenizer, prompt, max_tokens):
    # the M2 single-sequence path is the trusted answer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return paged_greedy_decode(model, input_ids, max_tokens, eos_token_id=tokenizer.eos_token_id)


def test_batched_decode_matches_single_sequence(model, tokenizer, cfg):
    # several sequences decoded together must each match decoding them alone
    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time,"]
    max_tokens = 12
    eos = tokenizer.eos_token_id

    engine = LLMEngine(model=model, cfg=cfg, num_blocks=256)
    for i, p in enumerate(prompts):
        engine.add(Sequence(i, _ids(tokenizer, p), max_tokens, eos))
    results = engine.run()

    for i, p in enumerate(prompts):
        assert results[i] == _reference(model, tokenizer, p, max_tokens)
    # no leak: every block came back once all sequences finished
    assert engine.scheduler.manager.num_free == engine.scheduler.manager.num_blocks


def test_short_request_is_not_blocked_by_a_long_one(model, tokenizer, cfg):
    eos = tokenizer.eos_token_id
    sched = Scheduler(model, cfg, num_blocks=256)
    sched.add(Sequence("long", _ids(tokenizer, "Once upon a time,"), 20, eos))
    sched.add(Sequence("short", _ids(tokenizer, "2 + 2 ="), 3, eos))

    finished_at = {}
    for step_idx in range(1, 200):
        for s in sched.step():
            finished_at[s.seq_id] = step_idx
        if not sched.has_work:
            break

    # the short request finishes on its own small budget, long before the long one
    assert finished_at["short"] <= 3
    assert finished_at["short"] < finished_at["long"]


def test_preemption_completes_all_requests_without_leak(model, tokenizer, cfg):
    # a pool too small to hold every sequence at once forces preemption + recompute
    eos = tokenizer.eos_token_id
    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time,", "List three colors:"]
    max_tokens = 18

    engine = LLMEngine(model=model, cfg=cfg, num_blocks=3)
    for i, p in enumerate(prompts):
        engine.add(Sequence(i, _ids(tokenizer, p), max_tokens, eos))
    results = engine.run()

    # every request completed despite the squeeze (no starvation)...
    assert set(results) == set(range(len(prompts)))
    # ...preemption actually happened...
    assert engine.scheduler.num_preemptions > 0
    # ...the pool is whole again (no leak)...
    assert engine.scheduler.manager.num_free == engine.scheduler.manager.num_blocks
    # ...and recompute-after-preemption still produced the correct tokens
    for i, p in enumerate(prompts):
        assert results[i] == _reference(model, tokenizer, p, max_tokens)
