# load Qwen2.5-0.5B weights from safetensors into my modules
# the module names mirror HF's, so the only fixup is stripping the "model." prefix
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .config import ModelConfig
from .model import Qwen2Model


def load_model(cfg: ModelConfig | None = None) -> Qwen2Model:
    cfg = cfg or ModelConfig()
    path = snapshot_download(cfg.model_id)
    state = load_file(os.path.join(path, "model.safetensors"))

    # strip the "model." prefix; every other name already matches
    remapped = {k.removeprefix("model."): v.to(cfg.dtype) for k, v in state.items()}

    model = Qwen2Model(cfg).to(cfg.device, cfg.dtype)
    # strict=True throws if a weight is missing or made up
    missing, unexpected = model.load_state_dict(remapped, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model
