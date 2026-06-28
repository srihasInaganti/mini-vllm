# Milestone 1 gate: the engine's greedy decode must match transformers token-for-token
# transformers is only allowed here as the reference, never inside the engine
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import ModelConfig
from engine.generate import greedy_decode
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
def ours(cfg):
    return load_model(cfg)


@pytest.fixture(scope="module")
def oracle(cfg):
    # reference model, same dtype/device as the engine
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.dtype
    ).to(cfg.device).eval()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_parity(prompt, tokenizer, ours, oracle, cfg):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cfg.device)
    eos = tokenizer.eos_token_id

    # turn off the sampling/penalty settings in Qwen's generation_config so the
    # reference does plain greedy, the same rule the engine uses. its default
    # repetition_penalty=1.1 would otherwise drift away from pure greedy.
    ref_full = oracle.generate(
        input_ids, max_new_tokens=MAX_NEW, do_sample=False,
        repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0,
        eos_token_id=eos, pad_token_id=eos,
    )
    ref = ref_full[0, input_ids.shape[1]:].tolist()  # continuation only

    got = greedy_decode(ours, input_ids, MAX_NEW, eos_token_id=eos)

    assert got == ref, (
        f"\nprompt: {prompt!r}"
        f"\n ours: {got}"
        f"\n ref : {ref}"
        f"\n ours text: {tokenizer.decode(got)!r}"
        f"\n ref  text: {tokenizer.decode(ref)!r}"
    )
