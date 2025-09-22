""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import jittor as jt

from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, None]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model (Jittor version, only supports loading state_dict, not JIT)"""
    if device is None:
        device = "cuda" if jt.has_cuda else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    # 只支持state_dict方式
    import pickle
    with open(model_path, 'rb') as f:
        state_dict = jt.load(f)

    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    if precision.startswith('amp') or precision == 'fp32':
        model.float()
    elif precision == 'bf16':
        convert_weights_to_lp(model, dtype=jt.bfloat16)

    # ensure image_size attr available at consistent location
    if hasattr(model, 'input_resolution'):
        model.visual.image_size = model.input_resolution.item()
    return model
