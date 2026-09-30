"""三个无需预训练权重的单通道分割网络，用相同损失比较。"""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ('tiny_unet', 'unet', 'resunet')


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels))
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if residual else None

    def forward(self, x):
        y = self.layers(x)
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return F.relu(y)


class UNet(nn.Module):
    def __init__(self, channels, residual=False):
        super().__init__()
        self.encoders = nn.ModuleList()
        previous = 1
        for width in channels:
            self.encoders.append(Block(previous, width, residual))
            previous = width
        self.decoders = nn.ModuleList([
            Block(channels[i + 1] + channels[i], channels[i], residual)
            for i in reversed(range(len(channels) - 1))])
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.output(x)


def build_model(name):
    if name == 'tiny_unet':
        return UNet((8, 16, 32))
    if name == 'unet':
        return UNet((16, 32, 64, 128))
    if name == 'resunet':
        return UNet((16, 32, 64, 128), residual=True)
    raise ValueError(f'未知模型 {name}; 可选 {MODEL_NAMES}')
