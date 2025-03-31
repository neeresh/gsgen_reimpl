from typing import Any, Dict

import torch
from torch import nn

from point_e.models.transformer import CLIPImagePointDiffusionTransformer

MODEL_CONFIGS = {"base40M-textvec": {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 1024,
        "name": "CLIPImagePointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "token_cond": True,
        "width": 512,
    }}


def model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
        config = config.copy()
        name = config.pop("name")

        if name == "CLIPImagePointDiffusionTransformer":
                return CLIPImagePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
        else:
                raise NotImplementedError(f"Model {name} not implemented")
