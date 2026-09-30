"""以 ImageNet 预训练 ResNet 为编码器的 U-Net 分割网络。"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

MODEL_NAMES = ('resnet18_unet', 'resnet34_unet', 'resnet50_unet')

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else (
            x.shape[-2] * 2, x.shape[-1] * 2
        )
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.layers(x)


class ResNetUNet(nn.Module):
    """保留预训练 ResNet 编码器，并用 U-Net 跳连逐级恢复分辨率。"""

    def __init__(self, encoder, encoder_channels):
        super().__init__()
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder_layers = nn.ModuleList([
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4,
        ])
        skip_channels = encoder_channels[:-1]
        decoder_channels = (256, 128, 64, 32)
        self.decoders = nn.ModuleList()
        current = encoder_channels[-1]
        for skip, output in zip(reversed(skip_channels), decoder_channels):
            self.decoders.append(DecoderBlock(current, skip, output))
            current = output
        self.final_decoder = DecoderBlock(current, 0, 16)
        self.output = nn.Conv2d(16, 1, 1)
        self.register_buffer(
            'image_mean', torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'image_std', torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # 灰度图复制为 RGB，再使用预训练权重对应的 ImageNet 归一化。
        x = (x.repeat(1, 3, 1, 1) - self.image_mean) / self.image_std
        skips = [self.stem(x)]
        x = self.pool(skips[0])
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)
        x = skips.pop()
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        x = self.final_decoder(x)
        return self.output(x)

def build_model(name, pretrained=True):
    """构建模型；训练时下载/复用预训练权重，加载 checkpoint 时可关闭。"""
    builders = {
        'resnet18_unet': (resnet18, ResNet18_Weights.DEFAULT, (64, 64, 128, 256, 512)),
        'resnet34_unet': (resnet34, ResNet34_Weights.DEFAULT, (64, 64, 128, 256, 512)),
        'resnet50_unet': (resnet50, ResNet50_Weights.DEFAULT, (64, 256, 512, 1024, 2048)),
    }
    if name in builders:
        builder, weights, channels = builders[name]
        encoder = builder(weights=weights if pretrained else None)
        return ResNetUNet(encoder, channels)
    raise ValueError(f'未知模型 {name}; 可选 {MODEL_NAMES}')
