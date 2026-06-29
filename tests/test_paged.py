# Milestone 2 gate: the paged KV cache must give the same greedy tokens as the M1 dense path
# (the dense path was already shown to match transformers, so matching it is enough)
import pytest
from transformers import AutoTokenizer

from engine.config import ModelConfig
from engine.generate import greedy_decode, paged_greedy_decode
from engine.loader import load_model

MAX_NEW = 25
PROMPTS = [
    "The capital of France is",
    "Write a one-sentence definition of recursion.",
    "2 + 2 =",
    "Once upon a time,",
    "List three primary colors:",
]


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig()


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg.model_id)


@pytest.fixture(scope="module")
def model(cfg):
    return load_model(cfg)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_paged_matches_dense(prompt, tokenizer, model, cfg):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cfg.device)
    eos = tokenizer.eos_token_id

    dense = greedy_decode(model, input_ids, MAX_NEW, eos_token_id=eos)
    paged = paged_greedy_decode(model, input_ids, MAX_NEW, eos_token_id=eos)

    assert paged == dense, (
        f"\nprompt: {prompt!r}"
        f"\n dense: {dense}"
        f"\n paged: {paged}"
    )
