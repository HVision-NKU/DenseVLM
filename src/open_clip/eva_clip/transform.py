from typing import Optional, Tuple
from PIL import Image
import numpy as np
import jittor as jt
import jittor.nn as nn
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

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

def _convert_to_rgb(image):
    return image.convert('RGB')

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class ResizeMaxSize:
    def __init__(self, max_size, interpolation=Image.BICUBIC, fn='max', fill=0):
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill
    def __call__(self, img):
        width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = img.resize((new_size[1], new_size[0]), self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = Image.fromarray(np.pad(np.array(img), ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0,0)), mode='constant', constant_values=self.fill))
        return img

def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
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
    if is_train:
        return Compose([
            Resize(image_size, interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=Image.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)
