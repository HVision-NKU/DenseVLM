from collections import OrderedDict
import jittor as jt
import jittor.nn as nn
from open_clip.eva_clip.utils import freeze_batch_norm_2d

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv(inplanes, planes, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.act2 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv(planes, planes * self.expansion, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion)
        self.act3 = nn.ReLU()
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv(inplanes, planes * self.expansion, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm(planes * self.expansion)
            )
    def execute(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(jt.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    def execute(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = jt.concat([x.mean(dim=0, keepdims=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].cast(x.dtype)
        # Jittor没有multi_head_attention_forward，需用自定义或简化实现
        # 这里只做结构迁移，具体实现需根据Jittor支持情况补充
        # x, _ = ...
        # return x[0]
        raise NotImplementedError("AttentionPool2d multi-head attention需用Jittor实现替换")

class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.conv1 = nn.Conv(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv(width // 2, width // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv(width // 2, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)
        self.init_parameters()
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def init_parameters(self):
        # 这里只做结构迁移，参数初始化可根据Jittor支持情况补充
        pass
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)
    def set_grad_checkpointing(self, enable=True):
        pass
    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x
    def execute(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x
