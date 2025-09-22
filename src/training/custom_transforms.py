import random
from PIL import Image

class CustomRandomResize:
    def __init__(self, scale=(0.5, 2.0), interpolation=Image.BILINEAR):
        self.min_scale, self.max_scale = min(scale), max(scale)
        self.interpolation = interpolation
    def __call__(self, img):
        if hasattr(img, 'size'):
            width, height = img.size
        else:
            raise ValueError('CustomRandomResize只支持PIL.Image输入')
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = (int(width * scale), int(height * scale))
        return img.resize(new_size, self.interpolation)

class CustomRandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    def __call__(self, img):
        if hasattr(img, 'size'):
            width, height = img.size
        else:
            raise ValueError('CustomRandomCrop只支持PIL.Image输入')
        tar_w, tar_h = self.size
        tar_w = min(tar_w, width)
        tar_h = min(tar_h, height)
        if width == tar_w and height == tar_h:
            return img
        i = random.randint(0, height - tar_h) if height > tar_h else 0
        j = random.randint(0, width - tar_w) if width > tar_w else 0
        return img.crop((j, i, j + tar_w, i + tar_h))
