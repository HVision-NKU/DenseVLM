from collections import OrderedDict
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union, Final
from functools import partial

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.checkpoint import checkpoint
import jittor as jt
from jittor import nn
from jittor import init

#* 导入roi_align
from jdet.ops import roi_align

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def execute(self, x: jt.Var):
        orig_type = x.dtype
        x = nn.layer_norm(x.to(jt.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def execute(self, x: jt.Var):
        orig_type = x.dtype
        x = nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def execute(self, x: jt.Var):
        return x * jt.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        # self.gamma = nn.Parameter(init_values * jt.ones(dim))
        self.gamma = init_values * jt.ones(dim)

    def execute(self, x):
        # return x.mul_(self.gamma) if self.inplace else x * self.gamma
        return x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def execute(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            # cls_tokens = torch.jit.annotate(jt.Var, x[:, :1])
            cls_tokens = x[:, :1]

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = jt.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = jt.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = jt.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            batch_first: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        # self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = jt.randn((dim * 3, dim)) * self.scale
        if qkv_bias:
            self.in_proj_bias = jt.zeros(dim * 3)
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = jt.log(10 * jt.ones((num_heads, 1, 1)))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = jt.ones((num_heads, 1, 1))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def execute(self, x, attn_mask: Optional[jt.Var] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = nn.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == jt.bool:
            new_attn_mask = jt.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = jt.bmm(jt.normalize(q, dim=-1), jt.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = jt.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = jt.bmm(attn, v)
        else:
            # if self.use_fsdpa:
            #     x = F.scaled_dot_product_attention(
            #         q, k, v,
            #         attn_mask=attn_mask,
            #         dropout_p=self.attn_drop.p if self.training else 0.,
            #     )
            # else:
            q = q * self.scale
            attn = jt.bmm(q, k.transpose(-1, -2))
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = jt.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = jt.randn(n_queries, d_model)
        self.attn = jt.attention.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def execute(self, x: jt.Var):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = jt.attention.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: jt.Var,
            k_x: Optional[jt.Var] = None,
            v_x: Optional[jt.Var] = None,
            attn_mask: Optional[jt.Var] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]
        
    #* 新添加，为 encode_pseudo_boxes 函数
    def proj_without_attn(self, value):
        attn_module = self.attn
        value = nn.linear(value, attn_module.in_proj_weight,
                         bias=attn_module.in_proj_bias)[..., -attn_module.embed_dim:]
        value = nn.linear(value, attn_module.out_proj.weight,
                         bias=attn_module.out_proj.bias)

        return value
    #* 新添加，为 encode_pseudo_boxes 函数
    def forward_without_attn(self, q_x):
        x = q_x + self.ls_1(self.proj_without_attn(value=self.ln_1(q_x)))    # use the maskclip-zhou style
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
    
    def execute(
            self,
            q_x: jt.Var,
            k_x: Optional[jt.Var] = None,
            v_x: Optional[jt.Var] = None,
            attn_mask: Optional[jt.Var] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def execute(self, x: jt.Var, attn_mask: Optional[jt.Var] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> jt.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    #* 新添加，为 encode_pseudo_boxes 函数
    def extract_feature_map(self, x, return_forward=False):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x)
        x_forward = self.resblocks[-1](x)
        x = self.resblocks[-1].forward_without_attn(x)

        if return_forward:
            return x, x_forward
        else:
            return x
    #* 新添加，为 encode_pseudo_boxes 函数
    def forward_without_attn(self, q_x):
        x = q_x + self.ls_1(self.proj_without_attn(value=self.ln_1(q_x)))    # use the maskclip-zhou style
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
    


    
    def execute(self, x: jt.Var, attn_mask: Optional[jt.Var] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #     # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #     x = checkpoint(r, x, None, None, attn_mask)
            # else:
            x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run trasnformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> jt.dtype:
        weight = self.resblocks[0].get_reference_weight()
        if hasattr(weight, 'int8_original_dtype'):
            return weight.int8_original_dtype
        return weight.dtype

    def execute(self, x: jt.Var, attn_mask: Optional[jt.Var] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #     # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #     x = checkpoint(r, x, None, None, attn_mask)
            # else:
            x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x


class VisionTransformer(nn.Module):
    # output_tokens: torch.jit.Final[bool]
    output_tokens: Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = scale * jt.randn(width)
        if pos_embed_type == 'learnable':
            self.positional_embedding = scale * jt.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = jt.zeros(self.grid_size[0] * self.grid_size[1] + 1, width)
            self.positional_embedding.stop_grad()
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding = jt.array(pos_embed_type).float()
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = scale * jt.randn(pool_dim, output_dim)

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.stop_grad()

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, jt.Var):
                        x.start_grad()
                    else:
                        for p in x.parameters():
                            p.start_grad()

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # init.gauss_(self.class_embedding, std=self.scale)
        # init.gauss_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     init.gauss_(block.attn.in_proj_weight, std=attn_std)
        #     init.gauss_(block.attn.out_proj.weight, std=proj_std)
        #     init.gauss_(block.mlp.c_fc.weight, std=fc_std)
        #     init.gauss_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     init.gauss_(self.text_projection, std=self.scale)
        pass

    # @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    # @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: jt.Var) -> Tuple[jt.Var, jt.Var]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    #* 新添加，为 encode_masks 函数
    def mask_pool(self, x, masks):
        # 1. 编码特征
        feature_map = self.encode_dense(x)              # [B, H*W, C] 或 [B, C, H, W] 取决于 encode_dense
        feature_map = jt.misc.normalize(feature_map, dim=-1) # 归一化

        # 2. 每张图的 mask 数量
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        # 3. 拼接所有 mask，并展平成 [bs, h*w]
        masks = jt.concat(masks).float().flatten(start_dim=-2, end_dim=-1)

        # 4. 按照 num_masks_per_image 重复 feature_map
        repeat_counts = num_masks_per_image
        
        feature_map = jt.repeat_interleave(feature_map, repeat_counts[0], dim=0)
        
        # 5. 加权平均池化
        features = (feature_map * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdims=True) + 1e-12)

        return features
    
    #* 新添加，为 encode_masks 函数
    def encode_dense(self, x, keep_shape=False):
        # shape = [*, width, grid, grid]
        x = self.conv1(x)  
        bs, _, h, w = x.shape

        # [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)  

        # [*, grid ** 2, width]
        x = x.transpose(0, 2, 1)  

        # 拼接 cls token
        class_token = self.class_embedding.astype(x.dtype) + jt.zeros(
            (x.shape[0], 1, x.shape[-1]), dtype=x.dtype
        )
        x = jt.concat([class_token, x], dim=1)  # [*, grid ** 2 + 1, width]

        # 位置编码
        if (h, w) == self.grid_size:
            pe = self.positional_embedding.astype(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # dropout & norm
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        # NLD -> LND
        x = x.transpose(1, 0, 2)  

        # transformer 提取特征
        x = self.transformer.extract_feature_map(x)

        # LND -> NLD
        x = x.transpose(1, 0, 2)  

        # attention pool
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            _, tokens = self._global_pool(x)
        else:
            _, tokens = self._global_pool(x)
            tokens = self.ln_post(tokens)

        # 可选 projection
        if self.proj is not None:
            tokens = tokens @ self.proj

        # 展平特征图 [bs, h*w, -1]
        feature_map = tokens.reshape(bs, h * w, -1)

        # normalize
        feature_map = jt.misc.normalize(feature_map, dim=-1)

        if keep_shape:
            # [bs, h, w, -1] -> [bs, -1, h, w]
            feature_map = feature_map.reshape(bs, h, w, -1).transpose(0, 3, 1, 2)

        return feature_map
    
    
    #* 新添加，为 encode_pseudo_boxes 函数
    def extract_roi_features(self, x, normed_boxes, extract_type='v2'):
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        elif extract_type == 'v2':
            return self._extract_roi_features_v2(x, normed_boxes)
        else:
            raise NotImplementedError
            # assert extract_type == 'v3'
            # return self._extract_roi_features_v3(x, normed_boxes)

    #* 新添加，为 encode_pseudo_boxes 函数
    def _extract_roi_features_v2(self, x, normed_boxes):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        # assert h == w     # TODO: support input of any shape, need to change the normed boxes to real boxes
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jt.concat(
            [self.class_embedding.to(x.dtype) + jt.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if (h, w) == self.grid_size:
            pe = self.positional_embedding.to(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.extract_feature_map(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            _, tokens = self._global_pool(x)
        else:
            _, tokens = self._global_pool(x)
            tokens = self.ln_post(tokens)

        if self.proj is not None:
            tokens = tokens @ self.proj
        
        #* 此处修改
        # tokens = F.normalize(tokens, dim=-1)   # normalize along last dimension 
        tokens = tokens.normalize(dim=-1)
        
        tokens = tokens.view(bs, h, w, -1).permute(0, 3, 1, 2)
        
        #* 此处修改，否则导入的被认为是模块而非函数
        # return roi_align(tokens, self._denormalize_boxes(normed_boxes, tokens),
        #                  (1, 1), 1.0, -1, True)[..., 0, 0]
        # 使用 jdet 的 roi_align
        roi_layer = roi_align.ROIAlign(
            output_size=(1, 1),  # 对应原来的 (1,1)
            spatial_scale=1.0,   # 对应原来的 scale
            sampling_ratio=-1
        )

        # denormalize boxes 生成实际 roi
        rois = self._denormalize_boxes(normed_boxes, tokens)  # [num_boxes, 5]

        # RoIAlign 输出 shape [num_boxes, C, 1, 1]
        roi_feats = roi_layer(tokens, rois)
        roi_feats = roi_feats.squeeze(-1).squeeze(-1)  # [num_boxes, C]

        return roi_feats
    

    #* 新添加，为 encode_pseudo_boxes 函数
    def rescale_positional_embedding(self, out_size, dtype):
        h, w = out_size
        #* jittor没有newzeros，进行改写
        # rescaled_positional_embedding = \
        #     self.positional_embedding.new_zeros(1 + h*w, self.positional_embedding.shape[1])
        rescaled_positional_embedding = jt.zeros(
            (1 + h*w, self.positional_embedding.shape[1]),
            dtype=self.positional_embedding.dtype
        )
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        #* 改写转置
        pe_2d = self.positional_embedding[1:].transpose().contiguous().view(
            1, -1, *self.grid_size)
        pe_2d = nn.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[1:] = pe_2d.transpose().contiguous()

        return rescaled_positional_embedding.to(dtype=dtype)    
       

    
    #* 新添加，为 encode_pseudo_boxes 函数
    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()     # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
     
    def execute(self, x: jt.Var):
        # import pdb; pdb.set_trace()
        #* 查看输入的x
        # print("查看execute") # [1,3,512,512,] float32
        # print("Input shape1:", x.shape)  
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # print("Input shape2:", x.shape)  
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # print("Input shape3:", x.shape)  
        
        x = x.permute(0, 2, 1)  # ape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = jt.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        
        
        print("查看x+pe的匹配")
        print(x.shape)
        print(self.positional_embedding.shape)
        
        #* 目前主要问题：此处的 tensor 形状不匹配；主要是x的形状
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

    

def text_global_pool(x, text: Optional[jt.Var] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        argmax_idx = text.argmax(dim=-1)[0]
        pooled, tokens = x[jt.arange(x.shape[0]), argmax_idx], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: Optional[int] = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_type: str = 'linear',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = jt.empty(width)
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = jt.empty(self.num_pos, width)
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self._attn_mask = None
        else:
            # self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)
            self._attn_mask = self.build_causal_mask()

        if proj_type == 'none' or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = jt.empty(width, output_dim)

        self.init_parameters()

    def init_parameters(self):
        init.gauss_(self.token_embedding.weight, std=0.02)
        init.gauss_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            init.gauss_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init.gauss_(block.attn.in_proj_weight, std=attn_std)
            init.gauss_(block.attn.out_proj.weight, std=proj_std)
            init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            init.gauss_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                init.gauss_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    init.zeros_(self.text_projection.bias)
            else:
                init.gauss_(self.text_projection, std=self.transformer.width ** -0.5)

    # @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    # @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if self.cls_emb is not None:
            no_wd.add('cls_emb')
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = jt.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: jt.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = nn.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = jt.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = jt.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def execute(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self._attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = jt.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
            batch_first: bool = True,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

        # self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        self._attn_mask = self.build_attention_mask()

        self.ln_final = norm_layer(width)
        self.text_projection = jt.empty(width, output_dim)

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init.gauss_(block.attn.in_proj_weight, std=attn_std)
            init.gauss_(block.attn.out_proj.weight, std=proj_std)
            init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            init.gauss_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            init.gauss_(block.attn.in_proj_weight, std=attn_std)
            init.gauss_(block.attn.out_proj.weight, std=proj_std)
            init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            init.gauss_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            init.gauss_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = jt.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def execute(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]
        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #     # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #     text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
            #     text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            # else:
            text_embs = resblock(text_embs, attn_mask=self._attn_mask[:seq_len, :seq_len])
            text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    # @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
