import torch
import torch.nn.functional as F
import torch.nn as nn


# ==============================================================================
# PHASE 2: DUAL-STREAM VARIATIONAL AUTOENCODER (VAE)
# ==============================================================================
# According to the research paper (Section IV.B), this phase is responsible for 
# compressing images into a "Variational Latent Space" (a mathematical bottleneck).
# We use two identical autoencoders: one for the Object (Foreground) and one 
# for the Background. This prevents the watermark from blurring across object boundaries.

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    Think of this as an 'attention' mechanism. It looks at all the channels of an image
    feature and decides which ones are the most important, scaling them up and suppressing
    the useless ones.
    """
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
        B, C, H, W = x.shape
        # 'Squeeze' the spatial dimensions (H, W) into a single pixel per channel
        squeezed = F.avg_pool2d(x, kernel_size=(H,W))

        # 'Excite' the channels by calculating a weight for each one
        w = self.excite(squeezed)

        # Multiply the original features by these weights
        return x * w.view(B, C, 1, 1)
      
class ResBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Residual blocks allow the network to learn 'changes' (residuals) rather than 
    entire images from scratch at every layer. This makes deep networks easier to train.
    """
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
    """
    Encoder Block.
    This shrinks the image's height and width by half (stride=2), while increasing 
    the number of feature channels. It forces the network to learn higher-level concepts.
    """
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
    """
    Decoder Block.
    This expands the image back to its original size. 
    It also receives 'skip connections' (the skip parameter) from the Encoder.
    
    PAPER NOTE: The paper warns about the "Lazy Decoder" pathology (Section III.A),
    where the decoder relies entirely on these skip connections instead of the 
    watermarked latent bottleneck. A 'SkipDropout' (dropping skips 30% of the time)
    is recommended here to force the network to use the bottleneck.
    """
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
        # Concatenate the lower-level features (x) with the skip connection features
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

# ------------------------------------------------------------------------------
# Simple blocks are identical to the above but without Squeeze-and-Excitation (SE)
# ------------------------------------------------------------------------------
class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
    """
    A lightweight Variational Autoencoder (VAE) used for one of the streams 
    (either object or background). It shrinks the image down to a 'latent bottleneck', 
    which is where the watermark is mathematically injected.
    """
    def __init__(self, latent_channels=256):
        super().__init__()  

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling path (Encoder)
        self.enc1 = SimpleEncoderBlock(32, 64)
        self.enc2 = SimpleEncoderBlock(64, 128)
        self.enc3 = SimpleEncoderBlock(128, 256)
        self.enc4 = SimpleEncoderBlock(256, latent_channels)

        # Upsampling path (Decoder)
        self.dec4 = SimpleDecoderBlock(latent_channels, 256, 256)
        self.dec3 = SimpleDecoderBlock(256, 128, 128)
        self.dec2 = SimpleDecoderBlock(128, 64, 64)
        self.dec1 = SimpleDecoderBlock(64, 32, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. ENCODE: Shrink image and save intermediate steps (skip connections)
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 2. BOTTLENECK: The most compressed representation (Z)
        # In the paper, this is where the Chaotic Watermark is injected!
        z = self.enc4(e3)

        # 3. DECODE: Rebuild the image using the bottleneck (z) and skips (e3, e2, e1, s0)
        d = self.dec4(z, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, s0)

        final = self.final(d)
        return final, z

class WAV2(nn.Module):
    """
    A heavier Variational Autoencoder (VAE) with SE (Attention) blocks.
    Functions exactly identically to SimpleWAV2 but with more parameters for better quality.
    """
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


class DualAutoencoder(nn.Module):
    """
    The Core Dual-Stream Framework.
    Because DistributedDataParallel (DDP) expects one single model to sync across GPUs, 
    we wrap both autoencoders (Object and Background) into this one parent class.
    
    This ensures that the foreground (pet) and the background are watermarked 
    through entirely separate neural networks, preventing edge artifacts.
    """
    def __init__(self, latent_channels=256, reduction_rate=16):
        super().__init__()
        
        # Stream 1: The Object (Foreground) Autoencoder
        self.ae_obj = SimpleWAV2(latent_channels=latent_channels)
        
        # Stream 2: The Background Context Autoencoder
        self.ae_bg = SimpleWAV2(latent_channels=latent_channels)

    def forward(self, objs, bgs):
        # Process the foreground object
        rec_obj, z_obj = self.ae_obj(objs)
        
        # Process the background
        rec_bg, z_bg = self.ae_bg(bgs)

        return rec_obj, rec_bg, z_obj, z_bg