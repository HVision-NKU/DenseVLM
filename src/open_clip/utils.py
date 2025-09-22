from itertools import repeat
import collections.abc

# import torch
# from torch import nn as nn
# from torchvision.ops.misc import FrozenBatchNorm2d
import jittor as jt
from jittor import nn

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self._weight = jt.ones(num_features)
        self._bias = jt.zeros(num_features)
        self._running_mean = jt.zeros(num_features)
        self._running_var = jt.ones(num_features)

    def _load_from_state_dict(self, state_dict, prefix="", *args, **kw):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        
        old_keys = ["weight", "bias", "running_mean", "running_var"]
        for old_key in old_keys:
            new_key = f"_{old_key}"
            if old_key in state_dict:
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
        
        super()._load_from_state_dict(state_dict, prefix, *args, **kw)
        
    def execute(self, x: jt.Var) -> jt.Var:
        w = self._weight.reshape(1, -1, 1, 1)
        b = self._bias.reshape(1, -1, 1, 1)
        rv = self._running_var.reshape(1, -1, 1, 1)
        rm = self._running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._weight.shape[0]}, eps={self.eps})"


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, nn.BatchNorm2d):
        res = FrozenBatchNorm2d(module.num_features)
        # res.num_features = module.num_features
        # res.affine = module.affine
        if module.affine:
            res._weight = module.weight.clone().detach()
            res._bias = module.bias.clone().detach()
        res._running_mean = module.running_mean
        res._running_var = module.running_var
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
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

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, jt.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.copy_(old_module.weight)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype