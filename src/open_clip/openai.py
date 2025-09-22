""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model (Jittor version, only supports state_dict loading)

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32'.
    device : str
        The device to put the loaded model (not used in Jittor, for compatibility)
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : Jittor nn.Module
        The CLIP model
    """
    if precision is None:
        precision = 'fp32'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    # 只支持state_dict权重加载
    import jittor as jt
    print(">>> where to load (only state) model checkpoint:", model_path)
    #* 修改2
    #state_dict = jt.load(model_path)
    state_dict = jt.load("/home/ubuntu/Codes/lyh/OVS/CLIPSelf-main/checkpoints/ViT-B-16-state.pth")
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    # Jittor模型默认在CPU或自动分配设备
    if precision.startswith('amp') or precision == 'fp32':
        #*修改 删除 float
        # model.float()
        model
        
    elif precision == 'bf16':
        convert_weights_to_lp(model, dtype=jt.bfloat16)

    # ensure image_size attr available at consistent location
    if hasattr(model, 'input_resolution'):
        model.visual.image_size = int(model.input_resolution)
    return model
