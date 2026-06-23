import torch
import torch.nn.functional as F
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super().__init__()
      
        bottleneck = max(channels // reduction_rate, 4)

        self.excite = nn.Sequential(
           nn.Flatten(),
           nn.Linear(channels, bottleneck, bias=False),
           nn.ReLU(inplace=True),
           nn.Linear(bottleneck, channels, bias=False),
           nn.Sigmoid()
        )
        
    def forward(self, x):
        
        B,C, H, W = x.shape
        squeezed = F.avg_pool2d(x, kernel_size=(H,W))

        w = self.excite(squeezed)

        return x * w.view(B,C, 1, 1)
      
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reduction_rate):
    
        super().__init__()
        self.conv_path = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),

        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),

        )

        self.se = SEBlock(out_ch, reduction_rate=reduction_rate)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.proj = (
        nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_ch)
        )
        if in_ch != out_ch else nn.Identity()
        )
    
    def forward(self, x):
       
        residual = self.proj(x)
        out = self.se(self.conv_path(x))
    
        return self.act(out + residual)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reduction_rate):
      super().__init__()

      self.down = nn.Sequential(
         nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
         nn.InstanceNorm2d(out_ch),
         nn.LeakyReLU(0.2, inplace=True)
      )
      self.res = ResBlock(out_ch, out_ch, reduction_rate)

    def forward(self, x):
       
       return self.res(self.down(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, reduction_rate):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.res = ResBlock(out_ch + skip_ch, out_ch, reduction_rate)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        
        return self.res(x)

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # Standard convolution path without Squeeze-and-Excitation
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_ch)
            )
            if in_ch != out_ch else nn.Identity()
        )
    
    def forward(self, x):
        residual = self.proj(x)
        out = self.conv_path(x)
        return self.act(out + residual)

class SimpleEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.res = SimpleResBlock(out_ch, out_ch)

    def forward(self, x):
        return self.res(self.down(x))

class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.res = SimpleResBlock(out_ch + skip_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class SimpleWAV2(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()  

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc1 = SimpleEncoderBlock(32, 64)
        self.enc2 = SimpleEncoderBlock(64, 128)
        self.enc3 = SimpleEncoderBlock(128, 256)
        self.enc4 = SimpleEncoderBlock(256, latent_channels)

        self.dec4 = SimpleDecoderBlock(latent_channels, 256, 256)
        self.dec3 = SimpleDecoderBlock(256, 128, 128)
        self.dec2 = SimpleDecoderBlock(128, 64, 64)
        self.dec1 = SimpleDecoderBlock(64, 32, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        z = self.enc4(e3)

        d = self.dec4(z, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, s0)

        final = self.final(d)
        return final, z

class WAV2(nn.Module):
    def __init__(self, latent_channels=256, reduction_rate=16):

        super().__init__()  

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc1 = EncoderBlock(32, 64, reduction_rate)
        self.enc2 = EncoderBlock(64, 128, reduction_rate)
        self.enc3 = EncoderBlock(128, 256, reduction_rate)
        self.enc4 = EncoderBlock(256, latent_channels, reduction_rate)

        self.dec4 = DecoderBlock(latent_channels, 256, 256, reduction_rate)
        self.dec3 = DecoderBlock(256, 128, 128, reduction_rate)
        self.dec2 = DecoderBlock(128, 64, 64, reduction_rate)
        self.dec1 = DecoderBlock(64, 32, 32, reduction_rate)

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        z = self.enc4(e3)

        d = self.dec4(z, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, s0)

        final = self.final(d)

        return final, z



class DualAutoencoder(nn.Module): # because DistributedDataParallel expects one nn.Module model, cause sync across gpu's would be nightmare
    def __init__(self, latent_channels=256, reduction_rate=16):
        
        super().__init__()
        
        self.ae_obj = SimpleWAV2(latent_channels=latent_channels)
        self.ae_bg = SimpleWAV2(latent_channels=latent_channels)

    def forward(self, objs, bgs):

        rec_obj, z_obj = self.ae_obj(objs)
        rec_bg, z_bg = self.ae_bg(bgs)

        return rec_obj, rec_bg, z_obj, z_bg