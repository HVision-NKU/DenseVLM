from itertools import repeat
import collections.abc
import logging
import math
import numpy as np

# open CLIP
# 只保留Jittor/纯Python/numpy兼容的辅助函数和工具，去掉所有PyTorch/torchvision/分布式相关内容

def resize_clip_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose(0, 3, 1, 2)
    # Jittor: 使用nn.interpolate替换F.interpolate
    import jittor as jt
    import jittor.nn as nn
    pos_emb_img = nn.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = jt.concat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

def resize_visual_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    import jittor as jt
    import jittor.nn as nn
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose(0, 3, 1, 2)
    pos_emb_img = nn.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = jt.concat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed

def resize_evaclip_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # 兼容EVA-CLIP模型的positional embedding插值
    old_pos_embed = state_dict.get('visual.pos_embed', None)
    if old_pos_embed is None or not hasattr(model.visual, 'patch_embed'):
        return
    grid_size = to_2tuple(model.visual.patch_embed.patch_shape)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[1]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:, :extra_tokens], old_pos_embed[:, extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(pos_emb_img.shape[1])))
    logging.info('Resizing EVA-CLIP position embedding grid-size from %s to %s', old_grid_size, grid_size)
    import jittor as jt
    import jittor.nn as nn
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose(0, 3, 1, 2)
    pos_emb_img = nn.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
    if pos_emb_tok is not None:
        new_pos_embed = jt.concat([pos_emb_tok, pos_emb_img], dim=1)
    else:
        new_pos_embed = pos_emb_img
        state_dict['visual.pos_embed'] = new_pos_embed

def resize_eva_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # 兼容EVA模型的positional embedding插值
    old_pos_embed = state_dict.get('pos_embed', None)
    if old_pos_embed is None or not hasattr(model, 'patch_embed'):
        return
    grid_size = to_2tuple(model.patch_embed.patch_shape)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[1]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:, :extra_tokens], old_pos_embed[:, extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(pos_emb_img.shape[1])))
    logging.info('Resizing EVA position embedding grid-size from %s to %s', old_grid_size, grid_size)
    import jittor as jt
    import jittor.nn as nn
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose(0, 3, 1, 2)
    pos_emb_img = nn.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
    if pos_emb_tok is not None:
        new_pos_embed = jt.concat([pos_emb_tok, pos_emb_img], dim=1)
    else:
        new_pos_embed = pos_emb_img
        state_dict['pos_embed'] = new_pos_embed

def freeze_batch_norm_2d(module, module_match={}, name=''):
    # Jittor: 仅保留接口，实际可直接返回module或自定义实现
    return module

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

def is_logging(args):
    def is_global_master(args):
        return args.rank == 0
    def is_local_master(args):
        return args.local_rank == 0
    def is_master(args, local=False):
        return is_local_master(args) if local else is_global_master(args)
    return is_master