# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import jittor as jt
import jittor.nn as nn
from .transformer import PatchDropout
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast
# torchvision.ops.roi_align 替换为 NotImplemented
roi_align = NotImplemented
from typing import Sequence

# ==== 本地实现 ==== #
from open_clip.utils import to_2tuple

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor = random_tensor.floor()
    output = x / keep_prob * random_tensor
    return output

def trunc_normal_(tensor, mean=0., std=1.):
    # Jittor: 直接用正态分布初始化
    with jt.no_grad():
        tmp = jt.normal(mean, std, tensor.shape)
        tensor.assign(tmp)
    return tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0., subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def execute(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = jt.zeros(all_head_dim)
            self.v_bias = jt.zeros(all_head_dim)
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(jt.zeros(self.num_relative_distance, num_heads))
            coords_h = jt.arange(window_size[0])
            coords_w = jt.arange(window_size[1])
            coords = jt.stack(jt.meshgrid([coords_h, coords_w]))
            coords_flatten = coords.reshape(2, -1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0)
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = jt.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.relative_position_index = relative_position_index
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope
    def execute(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln:
            q = nn.linear(x, self.q_proj.weight, self.q_bias)
            k = nn.linear(x, self.k_proj.weight, None)
            v = nn.linear(x, self.v_proj.weight, self.v_bias)
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = jt.concat([self.q_bias, jt.zeros_like(self.v_bias), self.v_bias])
            qkv = nn.linear(x, self.qkv.weight, qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope:
            if attn_mask is not None:
                attn_mask = attn_mask.cast(q.dtype)
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = jt.concat([q[:, :, :1, :], ro_q_t], -2).cast(v.dtype)
            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = jt.concat([k[:, :, :1, :], ro_k_t], -2).cast(v.dtype)
        if self.xattn:
            raise NotImplementedError('xformers attention not supported in Jittor')
        else:
            q = q * self.scale
            attn = jt.matmul(q, k.transpose(-2, -1))
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1)
                attn = attn + relative_position_bias.unsqueeze(0).cast(attn.dtype)
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.cast(attn.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = jt.where(~attn_mask[:, None, None, :], float('-inf'), attn)
            attn = nn.softmax(attn, -1)
            attn = self.attn_drop(attn)
            x = jt.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
    def proj_without_attn(self, x):
        x = nn.linear(x, self.v_proj.weight, self.v_bias)
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False, subln=False, naiveswiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if naiveswiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, subln=subln, norm_layer=norm_layer)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, subln=subln, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * jt.ones((dim)))
            self.gamma_2 = nn.Parameter(init_values * jt.ones((dim)))
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.postnorm = postnorm
    def execute(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    def forward_without_attn(self, x):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn.proj_without_attn(x)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn.proj_without_attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn.proj_without_attn(x)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn.proj_without_attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def execute(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).reshape(B, -1, self.patch_shape[0]*self.patch_shape[1]).transpose(1, 2)
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(jt.zeros(self.num_relative_distance, num_heads))
        coords_h = jt.arange(window_size[0])
        coords_w = jt.arange(window_size[1])
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = jt.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
    def execute(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)
        return relative_position_bias.permute(2, 0, 1)

class EVAVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0., use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False, use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False, pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False):
        super().__init__()
        self.image_size = img_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(jt.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(jt.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len if intp_freq else None)
        else:
            self.rope = None
        self.naiveswiglu = naiveswiglu
        dpr = [float(x) for x in jt.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None, xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight *= init_scale
            self.head.bias *= init_scale
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.grad_checkpointing = grad_checkpointing
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param /= math.sqrt(2.0 * layer_id)
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight, layer_id + 1)
    def get_cast_dtype(self):
        return self.blocks[0].mlp.fc2.weight.dtype
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.assign(0)
        elif isinstance(m, nn.LayerNorm):
            m.bias.assign(0)
            m.weight.assign(1.0)
    def get_num_layers(self):
        return len(self.blocks)
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.stop_grad()
        def _unlock(x):
            if isinstance(x, list):
                for g in x:
                    _unlock(g)
            else:
                if isinstance(x, nn.Parameter):
                    x.start_grad()
                else:
                    for p in x.parameters():
                        p.start_grad()
        for blk in self.blocks[-unlocked_groups:]:
            _unlock(blk)
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    def get_classifier(self):
        return self.head
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward_features(self, x, return_all_features=False):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = jt.repeat(self.cls_token, batch_size, 1, 1)
        x = jt.concat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)
        x = self.patch_dropout(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x
    def post_attention(self, x, return_all_features=False):
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x
    def execute(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x
    def encode_dense(self, x, keep_shape=True):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = jt.repeat(self.cls_token, batch_size, 1, 1)
        x = jt.concat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)
        x = self.patch_dropout(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.blocks[-1].forward_without_attn(x)[:, 1:]
        x = self.norm(x)
        x = self.head(x)
        assert self.fc_norm is None
        x = nn.normalize(x, dim=-1)
        if keep_shape:
            x = x.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        return x
    def extract_roi_features(self, x, normed_boxes, **kwargs):
        x = self.encode_dense(x, keep_shape=True)
        raise NotImplementedError('roi_align is not implemented in Jittor')
    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        if (h, w) == self.patch_embed.patch_shape:
            return self.pos_embed
        rescaled_positional_embedding = jt.zeros((1, 1 + h*w, self.pos_embed.shape[2]), dtype=self.pos_embed.dtype)
        rescaled_positional_embedding[0, 0] = self.pos_embed[0, 0]
        pe_2d = self.pos_embed[0, 1:].transpose(0, 1).reshape(1, -1, *self.patch_embed.patch_shape)
        pe_2d = nn.interpolate(pe_2d, size=out_size, mode='bicubic', align_corners=False).reshape(-1, h*w)
        rescaled_positional_embedding[0, 1:] = pe_2d.transpose(0, 1)
        return rescaled_positional_embedding
    def mask_pool(self, x, masks):
        feature_map = self.encode_dense(x, keep_shape=False)
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = jt.concat(masks).float().reshape(-1, feature_map.shape[1])
        feature_map = jt.repeat(feature_map, repeats=jt.array(num_masks_per_image), dim=0)
        features = (feature_map * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdims=True) + 1e-12)
        return features
    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
    def encode_rois_and_image(self, x, normed_boxes):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = jt.repeat(self.cls_token, batch_size, 1, 1)
        x = jt.concat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)
        x = self.patch_dropout(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x_image = self.head(self.post_attention(self.blocks[-1](x, rel_pos_bias=rel_pos_bias)))
        x_image = nn.normalize(x_image, dim=-1)
        x = self.blocks[-1].forward_without_attn(x)[:, 1:]
        x = self.norm(x)
        x = self.head(x)
        assert self.fc_norm is None
        x = nn.normalize(x, dim=-1)
        x = x.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        raise NotImplementedError('roi_align is not implemented in Jittor')
