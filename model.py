import torch
import torch.nn as nn
import torchvision.models as models


class SqueezeExcitationBlock(nn.Module):
    # SE attention - helps the decoder focus on actual crack pixels
    # and ignore the rough concrete background noise
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class DecoderBlock(nn.Module):
    # standard upsample + concat with skip, then double conv
    # SE at the end to reweight channels before passing up
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcitationBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # sometimes off by 1 pixel due to rounding, fix that here
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.se(self.conv(x))


class ResNet34CrackUNet(nn.Module):
    # ResNet34 encoder pretrained on ImageNet, U-Net style decoder
    # tried VGG16 first but ResNet34 was cleaner for skip connections
    def __init__(self, pretrained: bool = True, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # pull out the encoder blocks we need
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool   = resnet.maxpool
        self.layer1    = resnet.layer1   # 64 ch
        self.layer2    = resnet.layer2   # 128 ch
        self.layer3    = resnet.layer3   # 256 ch
        self.layer4    = resnet.layer4   # 512 ch

        # decoder goes back up, each step halves channels roughly
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  64,  32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d = self.dec4(x4, x3)
        d = self.dec3(d,  x2)
        d = self.dec2(d,  x1)
        d = self.dec1(d,  x0)

        return self.out(self.final_up(d))
