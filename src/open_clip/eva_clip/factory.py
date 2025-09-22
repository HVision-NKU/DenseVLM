import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import jittor as jt

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, CustomCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, get_cast_dtype
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model
from .transform import image_transform
from .tokenizer import HFTokenizer, tokenize
from .utils import resize_clip_pos_embed, resize_evaclip_pos_embed, resize_visual_pos_embed, resize_eva_pos_embed

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))
    for cf in config_files:
        with open(cf, "r", encoding="utf8") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
    _MODEL_CONFIGS = dict(sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0])))

_rescan_model_configs()

def list_models():
    return list(_MODEL_CONFIGS.keys())

def add_model_config(path):
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def get_tokenizer(model_name):
    config = get_model_config(model_name)
    tokenizer = HFTokenizer(config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer

def load_state_dict(checkpoint_path: str, model_key: str='model|module|state_dict', is_openai: bool=False, skip_list: list=[]):
    if is_openai:
        # 只支持Jittor权重加载
        state_dict = jt.load(checkpoint_path)
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        checkpoint = jt.load(checkpoint_path)
        for mk in model_key.split('|'):
            if isinstance(checkpoint, dict) and mk in checkpoint:
                state_dict = checkpoint[mk]
                break
            else:
                state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    for k in skip_list:
        if k in list(state_dict.keys()):
            logging.info(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]
    if os.getenv('RoPE') == '1':
        for k in list(state_dict.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k:
                del state_dict[k]
    return state_dict

def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict"):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key, is_openai=False)
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    if 'text.logit_scale' in state_dict and hasattr(model, 'logit_scale'):
        state_dict['logit_scale'] = state_dict['text.logit_scale']
        del state_dict['text.logit_scale']
    if 'visual.positional_embedding' in state_dict:
        resize_clip_pos_embed(state_dict, model)
    elif 'visual.pos_embed' in state_dict:
        resize_evaclip_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict)
    logging.info(f"incompatible_keys.missing_keys: {getattr(incompatible_keys, 'missing_keys', None)}")
    return incompatible_keys

def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, None] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_clip: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_model: str = None,
        pretrained_text_model: str = None,
        cache_dir: Optional[str] = None,
        skip_list: list  = [],
):
    model_name = model_name.replace('/', '-')
    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')
        if 'rope' in model_cfg.get('vision_cfg', {}):
            if model_cfg['vision_cfg']['rope']:
                os.environ['RoPE'] = "1"
        else:
            os.environ['RoPE'] = "0"
        if force_quick_gelu:
            model_cfg["quick_gelu"] = True
        if force_patch_dropout is not None:
            model_cfg['vision_cfg']["patch_dropout"] = force_patch_dropout
        cast_dtype = get_cast_dtype(precision)
        custom_clip = model_cfg.pop('custom_text', False) or force_custom_clip or ('hf_model_name' in model_cfg['text_cfg'])
        if custom_clip:
            if 'hf_model_name' in model_cfg.get('text_cfg', {}):
                model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
            model = CustomCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)
        pretrained_cfg = {}
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained
            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model,
                               checkpoint_path,
                               model_key="model|module|state_dict",
                               ) 
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
        if "fp16" in precision or "bf16" in precision:
            logging.info(f'convert precision to {precision}')
            if 'bf16' in precision:
                convert_weights_to_lp(model, dtype=jt.bfloat16)
            else:
                convert_weights_to_lp(model, dtype=jt.float16)
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD
    return model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, None] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_clip: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_model: str = None,
        pretrained_text_model: str = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        skip_list: list = [],
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_clip=force_custom_clip,
        force_patch_dropout=force_patch_dropout,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        pretrained_hf=pretrained_hf,
        pretrained_visual_model=pretrained_visual_model,
        pretrained_text_model=pretrained_text_model,
        cache_dir=cache_dir,
        skip_list=skip_list,
    )
    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )
    return model, preprocess_train, preprocess_val

def create_model_from_pretrained(
        model_name: str,
        pretrained: str,
        precision: str = 'fp32',
        device: Union[str, None] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_clip: bool = False,
        force_patch_dropout: Optional[float] = None,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        is_frozen: bool = False,
):
    if not is_pretrained_cfg(model_name, pretrained) and not os.path.exists(pretrained):
        raise RuntimeError(
            f'{pretrained} is not a valid pretrained cfg or checkpoint for {model_name}.'
            f' Use open_clip.list_pretrained() to find one.')
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_clip=force_custom_clip,
        force_patch_dropout=force_patch_dropout,
        cache_dir=cache_dir,
    )
    if is_frozen:
        for param in model.parameters():
            param.requires_grad = False
    if not return_transform:
        return model
    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )
    return model, preprocess
