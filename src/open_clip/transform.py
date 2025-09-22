import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from jittor import transform
from jittor.transform import resize
import jittor as jt
import jittor.nn as nn
from PIL import Image
import numpy as np
import random

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False

# --- PIL/numpy transforms ---
class Resize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        if isinstance(self.size, int):
            return img.resize((self.size, self.size), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        th, tw = self.size, self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img.crop((j, i, j + tw, i + th))

class ToTensor:
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = pic.transpose((2, 0, 1))
            return jt.array(img).float32() / 255.0
        img = np.array(pic, dtype=np.float32, copy=False)
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))
        return jt.array(img).float32() / 255.0

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]
    def __call__(self, tensor):
        return (tensor - jt.array(self.mean)) / jt.array(self.std)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

def _convert_to_rgb(image):
    return image.convert('RGB')

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            # PIL
            if hasattr(img, 'transpose'):
                from PIL import Image
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            # numpy/jt.Var
            elif hasattr(img, 'shape'):
                arr = img.numpy() if hasattr(img, 'numpy') else img
                arr = np.flip(arr, axis=-1)
                if hasattr(img, 'numpy'):
                    import jittor as jt
                    return jt.array(arr)
                else:
                    return arr
        return img

def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        image_size = image_size[0]
    normalize = Normalize(mean=mean, std=std)
    # 只保留推理/val分支
    transforms = [
        Resize(image_size),
        CenterCrop(image_size),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ]
    return Compose(transforms)

def det_image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        image_size = image_size[0]
    normalize = Normalize(mean=mean, std=std)
    transforms = [
        Resize(image_size),
        CenterCrop(image_size),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ]
    return Compose(transforms)


#* 本函数在jt的transform中没有实现
class ResizeLongest(nn.Module):
    def __init__(self, max_size, interpolation=Image.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fill = fill

    def execute(self, img):
        # print("=== Start ResizeLongest===")
        
        # --------------------------
        # Step 1: tensor -> PIL.Image
        # --------------------------
        if isinstance(img, jt.Var):
            # print(f"Input is jt.var, shape: {img.shape}, dtype: {img.dtype}")
            if img.ndim == 3:  # C,H,W
                C, H, W = img.shape
                if C == 1:
                    img_np = (img[0].numpy() * 255).astype(np.uint8)  # [H,W]
                    img_pil = Image.fromarray(img_np, mode='L')
                elif C == 3:
                    img_np = (img.numpy().transpose(1,2,0) * 255).astype(np.uint8)  # [H,W,C]
                    img_pil = Image.fromarray(img_np, mode='RGB')
                else:
                    raise ValueError(f"Unsupported channel size {C}")
            else:
                raise ValueError("Expected 3D tensor C,H,W")
        elif isinstance(img, Image.Image):
            img_pil = img
            # print(f"Input is PIL.Image, size: {img_pil.size}")
        else:
            raise TypeError(f"Unsupported type {type(img)}")

        width, height = img_pil.size
        # print(f"Original width: {width}, height: {height}")

        # --------------------------
        # Step 2: 计算缩放比例
        # --------------------------
        scale = self.max_size / float(max(width, height))
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        # print(f"Scale: {scale:.4f}, new_width: {new_width}, new_height: {new_height}")

        # --------------------------
        # Step 3: resize
        # --------------------------
        img_resized = img_pil.resize((new_width, new_height), self.interpolation)
        # print(f"After resize, type: {type(img_resized)}, size: {img_resized.size}")

        # --------------------------
        # Step 4: padding
        # --------------------------
        pad_w = self.max_size - new_width
        pad_h = self.max_size - new_height
        if pad_h > 0 or pad_w > 0:
            new_img = Image.new(img_resized.mode, (self.max_size, self.max_size), color=self.fill)
            new_img.paste(img_resized, (0,0))  # 左上角对齐
            img_resized = new_img
        # print(f"After padding, type: {type(img_resized)}, size: {img_resized.size}")

        # --------------------------
        # Step 5: PIL.Image -> tensor
        # --------------------------
        def pil_to_jt(img_pil):
            """
            Args:
                img_pil: PIL.Image
            Returns:
                jt.Var: shape [C,H,W], float32, values in [0,1]
            """
            img_np = np.array(img_pil)  # H x W x C 或 H x W
            if img_np.ndim == 2:  # 单通道
                img_np = img_np[None, :, :]  # 1 x H x W
            else:  # H x W x C -> C x H x W
                img_np = img_np.transpose(2, 0, 1)
            # 转为 jittor.Var 并归一化到 [0,1]
            img_var = jt.array(img_np, dtype=jt.float32) / 255.0
            return img_var
        img_tensor = pil_to_jt(img_resized)  # [H,W] 或 [H,W,C]
        if img_tensor.ndim == 2:  # 单通道
            img_tensor = img_tensor.unsqueeze(0)  # [1,H,W]
        else:  # [H,W,C] -> [C,H,W]
            img_tensor = img_tensor.permute(2,0,1)

        # print(f"Output tensor, shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        # print("=== End ResizeLongest ===\n")
        return img_tensor

#*第二次尝试手搓longest
# class ResizeLongest(nn.Module):
#     def __init__(self, max_size: int, interpolation=Image.BICUBIC, fill=0):
#         super().__init__()
#         if not isinstance(max_size, int):
#             raise TypeError(f"Size should be int. Got {type(max_size)}")
#         self.max_size = max_size
#         self.interpolation = interpolation
#         self.fill = fill

#     def execute(self, img):
#         """
#         Args:
#             img: PIL Image 或 jittor.Var
#         Returns:
#             img: jittor.Var，大小为 (max_size, max_size)
#         """
#         print("获取尺寸")
#         print(len(img.shape))
#         # 获取尺寸
#         if isinstance(img, jt.Var):
#             # jittor tensor: [C,H,W] 或 [H,W,C]
#             if len(img.shape) == 3:
#                 _, height, width = img.shape
#             else:
#                 raise ValueError("Expected 3D image tensor")
#         elif isinstance(img, Image.Image):
#             width, height = img.size
#         else:
#             raise TypeError(f"Unsupported type {type(img)}")

#         # 计算缩放比例
#         scale = self.max_size / float(max(height, width))
#         new_height = round(height * scale)
#         new_width = round(width * scale)

#         print("处理前的 img 形状")
#         print(img.shape)  # [1,512,683,] 是 jt var 
        
#         img = img[1:]
        
#         # 缩放
#         #* 转换1 手写
        
#         print("输入 resize 前的 img 形状")
#         print(img.size)   # (512, 1) 不应该是这样
        
#         #! 核心矛盾
#         #* jittor里的 resize 只支持 PIL 输入，原本输入的 img 是 jittor 类型
#         img = jt.transform.resize(img, [new_height, new_width], self.interpolation)
        
#         #* 转换2 手写
#         to_tensor = transform.ToTensor()
#         img = to_tensor(img)
        
#         # 右下填充到 max_size
#         pad_h = self.max_size - new_height
#         pad_w = self.max_size - new_width

#         if pad_h > 0 or pad_w > 0:
#             # jittor pad 顺序：左, 上, 右, 下
#             img = nn.pad(img, padding=[0, 0, pad_w, pad_h])

        
#         print("处理后的 img 形状")
#         print(img)

#         return img # 输出应当是jt var 

# class ResizeLongest(nn.Module):
#     def __init__(self, max_size, interpolation='bicubic', fill=0):
#         super().__init__()
#         if not isinstance(max_size, int):
#             raise TypeError(f"Size should be int. Got {type(max_size)}")
#         self.max_size = max_size
#         self.interpolation = interpolation
#         self.fill = fill

#     def execute(self, img):
#         if isinstance(img, jt.Var):
#             height, width = img.shape[1:]
#         else:
#             width, height = img.size
#         scale = self.max_size / float(max(height, width))
#         new_height, new_width = round(height * scale), round(width * scale)

#         #! 此处最好结合学长的 image transform的 longest 函数来改
#         #* 此处调用下方resize_image新函数
#         img = resize_image(img, new_height, new_width, mode='bilinear')
#         pad_h = self.max_size - new_height
#         pad_w = self.max_size - new_width
#         img = nn.pad(img, [0, 0, pad_w, pad_h], value=self.fill)

#         return img
    
# #* 新增函数用于调整 resizelongest 的问题
# #! 目前认为核心问题在这里
# def resize_image(img, new_height, new_width, mode='bilinear'):
#     """
#     img: jt.Var, shape (C,H,W) 或 (N,C,H,W)
#     new_height, new_width: int
#     mode: 'bilinear' / 'nearest'
#     """
#     # 保证 img 是 4D: (N,C,H,W)
#     if img.ndim == 3:
#         img = img[None, :, :, :]  # 1,C,H,W
#     elif img.ndim == 2:
#         img = img[None, None, :, :]  # 1,1,H,W

#     # jittor.nn.interpolate
#     img_resized = nn.interpolate(img, size=(new_height, new_width), mode=mode)

#     # 如果原来是 3D 或 2D，去掉 batch 维
#     if img_resized.shape[0] == 1:
#         img_resized = img_resized[0]
#         if img_resized.shape[0] == 1:
#             img_resized = img_resized[0]

#     return img_resized

def get_scale(old_image, new_image):
    """
    计算old_image到new_image的缩放比例。
    支持PIL.Image和ndarray/jt.Var输入。
    返回float或ndarray比例。
    """
    # 处理 old_image
    if hasattr(old_image, 'size') and isinstance(old_image.size, tuple):
        ow, oh = old_image.size
    elif hasattr(old_image, 'shape'):
        oh, ow = old_image.shape[-2:]
    else:
        raise ValueError('Unsupported image type for get_scale (old_image)')
    # 处理 new_image
    if hasattr(new_image, 'size') and isinstance(new_image.size, tuple):
        nw, nh = new_image.size
    elif hasattr(new_image, 'shape'):
        nh, nw = new_image.shape[-2:]
    else:
        raise ValueError('Unsupported image type for get_scale (new_image)')
    return nw / ow
