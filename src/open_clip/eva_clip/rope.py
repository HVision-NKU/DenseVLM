from math import pi
import jittor as jt
import jittor.nn as nn
import logging
import numpy as np

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set([len(t.shape) for t in tensors])
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*[list(t.shape) for t in tensors]))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([len(set(t[1])) <= 2 for t in expandable_dims]), 'invalid dimensions for broadcastable concatentation'
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[t[1] for t in expanded_dims]))
    out = []
    for t, shape in zip(tensors, expandable_shapes):
        if list(t.shape) == list(shape):
            out.append(t)
        else:
            # 手动broadcast实现
            target_shape = tuple(shape)
            t_shape = list(t.shape)
            # 先扩展维度
            while len(t_shape) < len(target_shape):
                t = t.unsqueeze(-1)
                t_shape.append(1)
            # broadcast到目标shape
            t = t.broadcast(target_shape)
            out.append(t)
    return jt.concat(out, dim = dim)

def rotate_half(x):
    # x: [..., 2*d]
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = jt.stack([-x2, x1], dim=-1)
    return x.reshape(*x.shape[:-2], -1)

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        self.ft_seq_len = ft_seq_len
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (jt.arange(0, dim, 2)[:(dim // 2)].float32() / dim))
        elif freqs_for == 'pixel':
            freqs = jt.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = jt.ones(num_freqs).float32()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = jt.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_h = jt.einsum('i,j->ij', t, freqs)
        freqs_h = jt.repeat(freqs_h, 2)
        freqs_w = jt.einsum('i,j->ij', t, freqs)
        freqs_w = jt.repeat(freqs_w, 2)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)
        self.freqs_cos = freqs.cos()
        self.freqs_sin = freqs.sin()
        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')
    def interpolate_freq(self, t_len, freq):
        if t_len == self.ft_seq_len ** 2:
            return freq
        tar_size = int(t_len ** 0.5)
        freq = freq.reshape(1, self.ft_seq_len, self.ft_seq_len, freq.shape[-1]).transpose(0, 3, 1, 2)
        freq = nn.interpolate(freq, (tar_size, tar_size), mode='bicubic', align_corners=False).reshape(-1, t_len).transpose(0, 1)
        return freq
    def execute(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t_mid = (t_mid * self.interpolate_freq(t.shape[2], self.freqs_cos)) \
            + (rotate_half(t_mid) * self.interpolate_freq(t.shape[2], self.freqs_sin))
        return jt.concat([t_left, t_mid, t_right], dim = -1)

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, patch_dropout=0.):
        super().__init__()
        self.custom_freqs = custom_freqs
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len
        self.freqs_for = freqs_for
        self.dim = dim
        self.theta = theta
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (jt.arange(0, dim, 2)[:(dim // 2)].float32() / dim))
        elif freqs_for == 'pixel':
            freqs = jt.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = jt.ones(num_freqs).float32()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = jt.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = jt.einsum('i,j->ij', t, freqs)
        freqs = jt.repeat(freqs, 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
        freqs_cos = freqs.cos().reshape(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().reshape(-1, freqs.shape[-1])
        self.patch_dropout = patch_dropout
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')
        self.flag = 0
    def execute(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.shape[0]
            batch_indices = jt.arange(batch).unsqueeze(1)
            freqs_cos = jt.repeat(self.freqs_cos, t.shape[0])
            freqs_sin = jt.repeat(self.freqs_sin, t.shape[0])
            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = freqs_cos.transpose(0, 2, 1, 3)
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = freqs_sin.transpose(0, 2, 1, 3)
            return  t * freqs_cos + rotate_half(t) * freqs_sin
        freqs_cos, freqs_sin = self.recalculate(t)
        return t * freqs_cos + rotate_half(t) * freqs_sin
    def interpolate_freq(self, t_len, freq):
        if t_len == self.ft_seq_len ** 2:
            return freq
        tar_size = int(t_len ** 0.5)
        freq = freq.reshape(1, self.ft_seq_len, self.ft_seq_len, freq.shape[-1]).transpose(0, 3, 1, 2)
        freq = nn.interpolate(freq, (tar_size, tar_size), mode='bicubic', align_corners=False).reshape(-1, t_len).transpose(0, 1)
        return freq
    def recalculate(self, x):
        x_len = x.shape[2]
        if x_len == self.ft_seq_len ** 2:
            return self.freqs_cos, self.freqs_sin
        ft_seq_len = int(x_len ** 0.5)
        if self.custom_freqs is not None:
            freqs = self.custom_freqs
        elif self.freqs_for == 'lang':
            freqs = 1. / (self.theta ** (jt.arange(0, self.dim, 2)[:(self.dim // 2)].float32() / self.dim))
        elif self.freqs_for == 'pixel':
            freqs = jt.linspace(1., self.max_freq / 2, self.dim // 2) * pi
        elif self.freqs_for == 'constant':
            freqs = jt.ones(self.num_freqs).float32()
        else:
            raise ValueError(f'unknown modality {self.freqs_for}')
        t = jt.arange(ft_seq_len) / ft_seq_len * self.pt_seq_len
        freqs = jt.einsum('i,j->ij', t, freqs)
        freqs = jt.repeat(freqs, 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
        freqs_cos = freqs.cos().reshape(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().reshape(-1, freqs.shape[-1])
        self.flag += 1
        logging.info(f'Add a new rope freq of shape: {freqs_cos.shape}')
        print(f'Add a new rope freq of shape: {freqs_cos.shape}', flush=True)
        return freqs_cos, freqs_sin
