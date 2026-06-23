"""
UNet architecture for Stage-1 mask generation.
Pure feature extractor -> segmentation head. No reconstruction paths here.

Outputs raw unactivated logits [B, 1, H, W] to ensure numerical stability 
when paired with BCEWithLogitsLoss. Apply sigmoid + 0.5 threshold at inference.
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class SegUNet(nn.Module):
    """
    Segmentation-only UNet. Outputs a single-channel logit map.
    Apply sigmoid + threshold (>0.5) at inference time to get binary mask.
    """
    def __init__(self):
        super().__init__()
        self.enc1       = DoubleConv(3,   64)
        self.enc2       = DownBlock(64,  128)
        self.enc3       = DownBlock(128, 256)
        self.enc4       = DownBlock(256, 512)
        self.bottleneck = DownBlock(512, 512)
        self.dec4       = UpBlock(512, 512, 256)
        self.dec3       = UpBlock(256, 256, 128)
        self.dec2       = UpBlock(128, 128,  64)
        self.dec1       = UpBlock( 64,  64,  64)
        self.head       = nn.Conv2d(64, 1, kernel_size=1)   # logit output

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        z  = self.bottleneck(s4)
        x  = self.dec4(z,  s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)
        return self.head(x)   # [B, 1, H, W] logit — no sigmoid here

