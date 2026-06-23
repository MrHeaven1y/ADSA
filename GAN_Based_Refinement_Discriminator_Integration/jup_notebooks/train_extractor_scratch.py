"""
CNN Watermark Extractor — From Scratch  (Stage 3b)
====================================================
Custom CNN backbone trained entirely from scratch.
Fully corrected architecture with dynamic identity pools,
spectral supervision, and robust forensic heads.
"""

import os
import json
import argparse
import time
import math
import torch
import hashlib
from datetime import timedelta
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SegmentTransform:
    def __init__(self, img_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], p=0.5):
        self.image_size = [img_size, img_size] if isinstance(img_size, int) else img_size
        self.mean = mean
        self.std  = std
        self.p    = p

    def __call__(self, img, mask):
        img  = TF.resize(img,  self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        if torch.rand(1) < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        img = TF.normalize(img, self.mean, self.std)
        return img, mask

class OxfordPetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None,
                 max_samples=None, cache_ram=True, max_cache_size=7500):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_list, self.mask_list = [], []
        self._extract()

        if max_samples is not None and max_samples < len(self.img_list):
            torch.manual_seed(42)
            perm = torch.randperm(len(self.img_list))[:max_samples].tolist()
            self.img_list  = [self.img_list[i]  for i in perm]
            self.mask_list = [self.mask_list[i] for i in perm]

        self.transforms     = transforms
        self.len            = len(self.img_list)
        self.cache_ram      = cache_ram
        self.max_cache_size = max_cache_size
        self.cache          = OrderedDict()

    def _extract(self):
        SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        img_files  = sorted(f for f in os.listdir(self.img_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)
        mask_files = sorted(f for f in os.listdir(self.mask_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)
        img_dict  = {os.path.splitext(f)[0]: f for f in img_files}
        mask_dict = {os.path.splitext(f)[0].replace('_mask', ''): f for f in mask_files}
        for key in sorted(set(img_dict) & set(mask_dict)):
            self.img_list.append(os.path.join(self.img_dir,  img_dict[key]))
            self.mask_list.append(os.path.join(self.mask_dir, mask_dict[key]))

    def _load(self, idx):
        if self.cache_ram and idx in self.cache:
            return self.cache[idx]
        try:
            img  = read_image(self.img_list[idx])
            mask = read_image(self.mask_list[idx])
        except RuntimeError:
            from PIL import Image as PILImage
            img  = TF.pil_to_tensor(PILImage.open(self.img_list[idx]).convert('RGB'))
            mask = TF.pil_to_tensor(PILImage.open(self.mask_list[idx]).convert('L'))
        if self.cache_ram:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[idx] = (img, mask)
        return img, mask

    def __len__(self): return self.len

    def __getitem__(self, idx):
        img, mask = self._load(idx)

        if img.shape[0] == 4: img = img[:3]
        elif img.shape[0] == 1: img = img.repeat(3, 1, 1)
        if mask.shape[0] > 1: mask = mask[0:1, :, :]

        img  = img.float().div(255.0)
        mask = mask.float()
        
        if mask.max() <= 3.0:
            mask = (mask == 1.0).float()
        else:
            mask = (mask > 127.0).float()
        
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        return img, mask

class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

def split_dataset(img_dir, mask_dir, train_tf, val_tf,
                  max_samples=None, split_size=0.85, cache_ram=True):
    dataset = OxfordPetDataset(img_dir, mask_dir, transforms=None,
                               max_samples=max_samples, cache_ram=cache_ram)
    n_train = int(split_size * len(dataset))
    n_val   = len(dataset) - n_train
    train_sub, val_sub = random_split(dataset, [n_train, n_val])
    return TransformedDataset(train_sub, train_tf), TransformedDataset(val_sub, val_tf)

# ══════════════════════════════════════════════════════════════════════════════
#  SS INJECTOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, secret_key):
    hash_hex = hashlib.sha512(secret_key.encode('utf-8')).hexdigest()
    x0 = int(hash_hex[:64], 16) / (16**64)
    y0 = int(hash_hex[64:], 16) / (16**64)
    
    if x0 == 0 or x0 == 0.5 or x0 == 1: x0 = 0.12345
    if y0 == 0 or y0 == 0.5 or y0 == 1: y0 = 0.67890

    N = latent_channels * latent_h * latent_w
    r, mu, T_thresh = 3.99, 0.99, 1.0
    
    x, y = x0, y0
    watermark_flat = torch.zeros(N, dtype=torch.float32)
    
    for i in range(N):
        x = r * x * (1.0 - x)
        y = mu * math.sin(math.pi * y)
        watermark_flat[i] = 1.0 if (x + y) > T_thresh else -1.0
        
    w = watermark_flat.view(1, latent_channels, latent_h, latent_w)

    w = w / (w.std(dim=(1,2,3), keepdim=True) + 1e-6)
    w = torch.tanh(w)
    
    return w

class AdaptiveSemanticMasker(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.filters = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        self.filters.weight.data[0, 0] = sobel_x
        self.filters.weight.data[1, 0] = sobel_y
        self.filters.weight.requires_grad = False
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

    @torch.no_grad()
    def forward(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edges = self.filters(gray)
        sobel_mag = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2 + 1e-6)
        local_mean = self.avg_pool(gray)
        local_var = self.avg_pool((gray - local_mean)**2)
        intersect = (0.4 * sobel_mag + 0.6 * local_var).clamp(0, 1)
        return F.adaptive_avg_pool2d(intersect, (28, 28))

class SSWatermarkInjector(nn.Module):
    def __init__(self, latent_channels, latent_h, latent_w):
        super().__init__()
        # Retain the chaotic watermark pools
        self.register_buffer('base_w1_pool', torch.cat([generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, f"base_obj_{i}") for i in range(256)], dim=0))
        self.register_buffer('base_w2_pool', torch.cat([generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, f"base_bg_{i}") for i in range(256)], dim=0))
        
        self.register_buffer('fake_w1_pool', torch.cat([generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, f"fake_obj_{i}") for i in range(256)], dim=0))
        self.register_buffer('fake_w2_pool', torch.cat([generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, f"fake_bg_{i}") for i in range(256)], dim=0))
        
        
        
        self.project = nn.Identity()

    def inject(self, z, w, alpha, semantic_mask):
        # 1. Variance Normalization (Independent of tensor size)
        w = w / (w.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        
        # 2. Bound Amplitudes
        w = torch.tanh(w)
        
        # 3. Gain Control
        w = w * 0.28
        
        # 4. Semantic Masking (Floor at 0.25 so the background isn't completely ignored)
        if semantic_mask is not None:
            mask = (semantic_mask * 0.75) + 0.25
            w = w * mask
            
        # 5. Inject
        delta = alpha * w
        
        # Keep the safety clamp just in case, but the math above naturally protects it now
        delta = torch.clamp(delta, min=-0.25, max=0.25) 
        
        return z + delta

class ResBlockGroupNorm(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.proj(x) + self.conv_path(x))

class ResNetEncoder(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()
        self.stem  = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True), ResBlockGroupNorm(64, 64))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(inplace=True), ResBlockGroupNorm(128, 128))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.GroupNorm(8, 256), nn.ReLU(inplace=True), ResBlockGroupNorm(256, 256))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, 1, bias=False), nn.GroupNorm(8, latent_channels), nn.ReLU(inplace=True),
            ResBlockGroupNorm(latent_channels, latent_channels), 
            ResBlockGroupNorm(latent_channels, latent_channels), 
            ResBlockGroupNorm(latent_channels, latent_channels),
        )

    def forward(self, x):
        s0 = self.stem(x); s1 = self.down1(s0); s2 = self.down2(s1); s3 = self.down3(s2)
        return self.bottleneck(s3), (s0, s1, s2, s3)

class ResNetDecoder(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(latent_channels+256, 256, 3, 1, 1, bias=False), nn.GroupNorm(8, 256), nn.ReLU(inplace=True), ResBlockGroupNorm(256, 256))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256+128, 128, 3, 1, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(inplace=True), ResBlockGroupNorm(128, 128))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128+64, 64, 3, 1, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True), ResBlockGroupNorm(64, 64))
        self.output = nn.Sequential(nn.Conv2d(64+64, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, skips):
        s0, s1, s2, s3 = skips
        x = self.up1(torch.cat([z, s3], dim=1))
        x = self.up2(torch.cat([x, s2], dim=1))
        x = self.up3(torch.cat([x, s1], dim=1))
        return self.output(torch.cat([x, s0], dim=1))

class DualAutoencoder(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()
        self.enc_obj = ResNetEncoder(latent_channels)
        self.enc_bg  = ResNetEncoder(latent_channels)
        self.shared_decoder = ResNetDecoder(latent_channels)

    def blend_skips(self, sk_obj, sk_bg, mask):
        return tuple(
            o * F.interpolate(mask, size=o.shape[2:], mode='bilinear', align_corners=False) + 
            b * (1.0 - F.interpolate(mask, size=b.shape[2:], mode='bilinear', align_corners=False)) 
            for o, b in zip(sk_obj, sk_bg)
        )

# ══════════════════════════════════════════════════════════════════════════════
#  ATTACK SIMULATION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DifferentiableJPEG(nn.Module):
    def __init__(self):
        super().__init__()
        q = torch.tensor([
            [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99],
        ], dtype=torch.float32)
        self.register_buffer('q_table', q)

    def _dct_block(self, x):
        N = 8
        n = torch.arange(N, dtype=torch.float32, device=x.device)
        k = n.unsqueeze(1)
        D = torch.cos(torch.pi * k * (2*n + 1) / (2*N))
        D[0] *= (1.0 / 2**0.5)
        D = D * (2.0/N)**0.5
        return torch.einsum('ki,bcij,lj->bckl', D, x, D)

    def _idct_block(self, X):
        N = 8
        n = torch.arange(N, dtype=torch.float32, device=X.device)
        k = n.unsqueeze(1)
        D = torch.cos(torch.pi * k * (2*n + 1) / (2*N))
        D[0] *= (1.0 / 2**0.5)
        D = D * (2.0/N)**0.5
        return torch.einsum('ik,bcij,jl->bckl', D, X, D)

    def forward(self, x, quality=75):
        B, C, H, W = x.shape
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        q = (self.q_table * scale / 100.0).clamp(1, 255)
        
        x = x * 128.0
        pH = (8 - H % 8) % 8; pW = (8 - W % 8) % 8
        x = F.pad(x, (0, pW, 0, pH), mode='reflect')
        H2, W2 = x.shape[2], x.shape[3]
        blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)
        nH, nW = blocks.shape[2], blocks.shape[3]
        blocks_4d = blocks.contiguous().view(-1, 1, 8, 8)
        dct_out   = self._dct_block(blocks_4d)
        q_exp     = q.view(1, 1, 8, 8)
        quant     = dct_out / q_exp
        quant_round = quant + (quant.round() - quant).detach()
        dequant   = quant_round * q_exp
        
        recon = self._idct_block(dequant)
        recon = recon.view(B, C, nH, nW, 8, 8)
        recon = recon.permute(0, 1, 2, 4, 3, 5).contiguous()
        out = recon.view(B, C, nH * 8, nW * 8)
        
        out = out / 128.0
        return out[:, :, :H, :W].clamp(-1, 1)

class AttackSimulationLayer(nn.Module):
    def __init__(self, p_apply=0.5, deterministic=False):
        super().__init__()
        self.p_apply = p_apply
        self.deterministic = deterministic
        self.jpeg = DifferentiableJPEG()

    def _gaussian_noise(self, x):
        sigma = 0.03 if self.deterministic else 0.01 + torch.rand(1).item() * 0.04
        return (x + torch.randn_like(x) * sigma).clamp(-1, 1)

    def _gaussian_blur(self, x):
        k = 5 if self.deterministic else [3,5,7][torch.randint(3,(1,)).item()]
        sigma = 0.3 * ((k-1)*0.5-1) + 0.8
        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k//2
        k1d = torch.exp(-coords**2/(2*sigma**2))
        k2d = (k1d/k1d.sum()).outer(k1d/k1d.sum()).view(1,1,k,k).expand(x.shape[1],1,k,k)
        return F.conv2d(x, k2d, padding=k//2, groups=x.shape[1])

    def _random_crop_resize(self, x):
        B,C,H,W = x.shape
        scale = 0.85 if self.deterministic else 0.75 + torch.rand(1).item()*0.2
        ch, cw = int(H*scale), int(W*scale)
        top, left = 0, 0
        if not self.deterministic:
            top  = torch.randint(0, H-ch+1, (1,)).item()
            left = torch.randint(0, W-cw+1, (1,)).item()
        return F.interpolate(x[:,:,top:top+ch,left:left+cw], size=(H,W), mode='bilinear', align_corners=False)

    def _brightness_contrast(self, x):
        b = 0.0 if self.deterministic else (torch.rand(1).item()-0.5)*0.4
        c = 1.0 if self.deterministic else 0.8 + torch.rand(1).item()*0.4
        return (x*c + b).clamp(-1, 1)

    def _rescale(self, x):
        scale = 0.8 if self.deterministic else 0.7 + torch.rand(1).item()*0.25
        h,w = x.shape[2], x.shape[3]
        new_h, new_w = int(h*scale), int(w*scale)
        resized = F.interpolate(x, size=(new_h,new_w), mode='bilinear', align_corners=False)
        return F.interpolate(resized, size=(h,w), mode='bilinear', align_corners=False)
    
    def _compound_jpeg(self, x):
        q1 = 50 if self.deterministic else torch.randint(30, 91, (1,)).item()
        q2 = 40 if self.deterministic else torch.randint(20, 71, (1,)).item()
        x = self.jpeg(x, quality=q1)
        x = self.jpeg(x, quality=q2)
        return x
    
    # REPLACE the forward function in AttackSimulationLayer:
    def forward(self, x, current_epoch, max_epochs=80):
        # Calculate dynamic thresholds based on total epochs
        stage1 = int(0.25 * max_epochs)    # e.g., 20 out of 80
        stage2 = int(0.4375 * max_epochs)  # e.g., 35 out of 80
        stage3 = int(0.625 * max_epochs)   # e.g., 50 out of 80

        if current_epoch < stage1: 
            return x 
            
        if current_epoch >= stage1 and (self.deterministic or torch.rand(1).item() < self.p_apply):
            q = 60 if self.deterministic else torch.randint(50, 91, (1,)).item()
            x = self.jpeg(x, quality=q)
            
        if current_epoch >= stage2 and (self.deterministic or torch.rand(1).item() < self.p_apply):
            x = self._gaussian_blur(x)
            
        if current_epoch >= stage3:
            for attack_fn in [self._gaussian_noise, self._random_crop_resize,
                              self._brightness_contrast, self._rescale]:
                if self.deterministic or torch.rand(1).item() < self.p_apply:
                    x = attack_fn(x)
            if self.deterministic or torch.rand(1).item() < 0.5:
                x = self._compound_jpeg(x)
                
        return x

class DecoderResBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class AdvancedTamperLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, seed=None):
        B, C, H, W = x.shape
        masks = torch.zeros(B, 1, H, W, device=x.device)
        tampered_x = x.clone()
        
        # Setup local generator if seed is provided for deterministic validation
        gen = None
        if seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)
        
        for i in range(B):
            h_t, w_t = H // 3, W // 3
            
            if gen is not None:
                top = torch.randint(0, H - h_t, (1,), generator=gen, device=x.device).item()
                left = torch.randint(0, W - w_t, (1,), generator=gen, device=x.device).item()
                is_copy = torch.rand(1, generator=gen, device=x.device).item() > 0.5
            else:
                top = torch.randint(0, H - h_t, (1,)).item()
                left = torch.randint(0, W - w_t, (1,)).item()
                is_copy = torch.rand(1).item() > 0.5

            masks[i, 0, top:top+h_t, left:left+w_t] = 1.0

            if is_copy:
                if gen is not None:
                    src_top = torch.randint(0, H - h_t, (1,), generator=gen, device=x.device).item()
                    src_left = torch.randint(0, W - w_t, (1,), generator=gen, device=x.device).item()
                else:
                    src_top = torch.randint(0, H - h_t, (1,)).item()
                    src_left = torch.randint(0, W - w_t, (1,)).item()
                tampered_x[i, :, top:top+h_t, left:left+w_t] = x[i, :, src_top:src_top+h_t, src_left:src_left+w_t]
            else:
                if gen is not None:
                    noise = torch.randn(C, h_t, w_t, generator=gen, device=x.device)
                else:
                    noise = torch.randn(C, h_t, w_t, device=x.device)
                tampered_x[i, :, top:top+h_t, left:left+w_t] = noise * 0.5

        return tampered_x, F.interpolate(masks, size=(56, 56), mode='area')

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()

        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):

        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)

        return x.flatten(1)

class ForensicIntegrityAnalyzer(nn.Module):
    def __init__(self, num_groups=8, num_res_blocks=6, fp_dim=256):
        super().__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 64), nn.GELU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 128), nn.GELU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 256), nn.GELU())
        self.processor = nn.Sequential(*[ResBlockGroupNorm(256, 256) for _ in range(num_res_blocks)])
        
        self.fpn_proj3 = nn.Conv2d(256, 128, 1)
        self.fpn_proj2 = nn.Conv2d(128, 128, 1)
        self.fpn_proj1 = nn.Conv2d(64,  128, 1)

        
        self.integrity_head = nn.Sequential(
            DecoderResBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        self.detector_head = nn.Sequential(GeMPooling(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4))
        
        self.latent_mean = nn.Conv2d(256, 256, 3, 1, 1)
        self.latent_logvar = nn.Conv2d(256, 256, 3, 1, 1)
        
        self.num_ids = fp_dim
        self.arc_margin = 0.25
        self.arc_scale = 30.0

        self.identity_centers = nn.Parameter(torch.randn(self.num_ids, fp_dim))

        nn.init.xavier_uniform_(self.identity_centers)

        self.identity_pool = GeMPooling()
        
        self.feat_dropout = nn.Dropout2d(p=0.03)
        self.embed_dropout = nn.Dropout(p=0.2)

        self.identity_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 256),
         
            nn.GELU(),
         
            nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256)
        )

        self.identity_proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, fp_dim)
        )

    def compute_arcface_logits(self, embeddings, labels):

        centers = F.normalize(self.identity_centers, dim=1)
        cosine = torch.matmul(embeddings, centers.T)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.arc_margin)
        one_hot = F.one_hot(labels, num_classes=self.num_ids).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits *= self.arc_scale

        return logits
    
    def forward(self, x):

        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        deep_features = self.processor(s3)

        robust_features = self.feat_dropout(deep_features)

        p3 = self.fpn_proj3(robust_features)
        # Change these:
        p2 = self.fpn_proj2(s2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.fpn_proj1(s1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        p1 = F.avg_pool2d(p1, 2)

        integrity_map = self.integrity_head(p1)

        global_logit = self.detector_head(robust_features)
        z_mean       = self.latent_mean(robust_features)
        z_logvar     = self.latent_logvar(robust_features)
        
        id_feat = self.identity_head(z_mean)
        id_feat = self.feat_dropout(id_feat) 

        pooled = self.identity_pool(id_feat)
        pooled = self.embed_dropout(pooled) 

        pred_fp = self.identity_proj(pooled)
        pred_fp = F.normalize(pred_fp, dim=1)
        
        return integrity_map, global_logit, z_mean, z_logvar, pred_fp

class ForensicInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))        
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature


        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel_edges(self, x):

        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    def forward(self, pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor, current_epoch=0, latent_weight=1.0, id_weight=1.0, max_epochs=80):
        
        loss_spatial_bce = self.bce(pred_int, gt_int)
        
        # Dice Loss for sharp tamper boundaries
        pred_sigmoid = torch.sigmoid(pred_int)
        
        intersection = (pred_sigmoid * gt_int).sum(dim=(2,3))
        union = pred_sigmoid.sum(dim=(2,3)) + gt_int.sum(dim=(2,3))
        loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        pred_edge = self._sobel_edges(pred_sigmoid)
        gt_edge = self._sobel_edges(gt_int)
        loss_edge = F.l1_loss(pred_edge, gt_edge)


        loss_spatial = loss_spatial_bce + loss_dice.mean() + (0.1 * loss_edge)
        loss_global  = self.ce(pred_glob, gt_glob)
        
        # [FIX 3] Normalize prediction to match target magnitude
        pred_latent = F.normalize(z_mean.flatten(1), dim=1).view_as(z_mean)
        loss_latent_recon = F.mse_loss(pred_latent, gt_latent)
        
        loss_kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        loss_energy = 0.001 * torch.mean(z_logvar.pow(2))
        kl_weight = min(0.02, (current_epoch / 40.0) * 0.02)
        
        fft_pred = torch.fft.rfft2(z_mean.float(), norm='ortho')
        fft_gt   = torch.fft.rfft2(gt_latent.float(), norm='ortho')
        loss_fft = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_gt))
        
        loss_latent = loss_latent_recon + (kl_weight * loss_kl) + loss_energy + (0.1 * loss_fft)

        features = F.normalize(pred_fp, dim=1)
        batch_size = features.shape[0]
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = identity_labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        
        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - max_sim.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss_infonce = -mean_log_prob_pos.mean()


        phase_3_start = int(0.75 * max_epochs)         
        if current_epoch < phase_3_start:
            lambda_infonce = 0.0
        else:
            lambda_infonce = 0.01

        arc_logits = extractor.compute_arcface_logits(features, identity_labels)
        loss_arc = F.cross_entropy(arc_logits, identity_labels)

        # Combine ArcFace with the throttled InfoNCE
        loss_identity = (loss_arc + lambda_infonce * loss_infonce) * id_weight
                
        total_loss = loss_spatial + loss_global + (loss_latent * latent_weight) + loss_identity
        
        return total_loss, {
            'loss_ext_total': total_loss.item(), 
            'loss_spatial': loss_spatial.item(),
            'loss_global': loss_global.item(), 
            'loss_latent': loss_latent.item(),
            'loss_fft' : loss_fft.item(),
            'loss_identity': loss_identity.item(),
            'loss_infonce': loss_infonce.item()
        }


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"    [ckpt] saved → {path}")

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location='cpu')
    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f"    [ckpt] resumed from epoch {ckpt['epoch']} → {path}")
    return ckpt['epoch'], ckpt['best_val_loss']

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ddp(rank, world_size, config):
    print(f"Rank {rank} initializing on GPU {torch.cuda.get_device_name(rank)}")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5)
    )

    batch_size     = config['batch_size']
    max_iterations = config['max_iterations']
    img_size       = config['img_size']
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    train_tf = SegmentTransform(img_size, p=0.5)
    val_tf   = SegmentTransform(img_size, p=0.0)

    train_ds, val_ds = split_dataset(
        config['images_dir'], config['mask_dir'],
        train_tf, val_tf,
        max_samples=config.get('max_samples'),
        split_size=config.get('split_size', 0.85),
        cache_ram=config.get('cache_ram', True),
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    nw = config['num_workers']
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, sampler=val_sampler,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)

    latent_channels = config.get('latent_channels', 256)
    latent_h = img_size[0] // 8
    latent_w = img_size[1] // 8
    
    ae = DualAutoencoder(latent_channels=latent_channels).to(rank)
    ae_weights = config.get('ae_weights')
    if ae_weights and os.path.isfile(ae_weights):
        state = torch.load(ae_weights, map_location='cpu')
        ae.load_state_dict(state.get('model', state), strict=False)
        if rank == 0: print(f"  [AE] loaded → {ae_weights}")
    
    ae = DDP(ae, device_ids=[rank])
    ae.eval()
    for p in ae.parameters(): p.requires_grad = False

    injector = SSWatermarkInjector(latent_channels, latent_h, latent_w).to(rank)
    
    inj_path = config.get('injector_weights')
    
    if inj_path and os.path.isfile(inj_path):
        injector.load_state_dict(torch.load(inj_path, map_location='cpu'))
        if rank == 0: print(f"  [injector] loaded → {inj_path}")
    injector.eval()
    for p in injector.parameters(): p.requires_grad = False

    fp_dim = config.get("latent_channels", 256)

    extractor = ForensicIntegrityAnalyzer(num_groups=8, num_res_blocks=6, fp_dim=fp_dim).to(rank)
    extractor = DDP(extractor, device_ids=[rank])

    if rank == 0:
        p = sum(v.numel() for v in extractor.parameters()) / 1e6
        print(f"  [extractor] ScratchCNN | params: {p:.2f}M  (all trainable)")

    semantic_masker = AdaptiveSemanticMasker().to(rank)
    attack_layer = AttackSimulationLayer(p_apply=config.get('attack_p', 0.5), deterministic=False).to(rank)
    val_attack_layer = AttackSimulationLayer(p_apply=config.get('attack_p', 0.5), deterministic=True).to(rank)
    tamper_layer = AdvancedTamperLayer().to(rank)
    criterion = ForensicInfoNCELoss(temperature=0.1).to(rank)

    optimizer = optim.AdamW(extractor.module.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_iterations'], eta_min=config['lr'] * 0.01)
    scaler       = GradScaler(device='cuda')


    start_epoch = 0
    best_val = float('inf')
    patience_ctr = 0

    if config.get('resume') and os.path.isfile(config['resume']):
        start_epoch, _ = load_checkpoint(config['resume'], extractor, optimizer, scheduler, scaler)
        
        # ═════════════════════════════════════════════════════════════════
        # [FIX] WIPE CORRUPTED HISTORY
        # ═════════════════════════════════════════════════════════════════
        best_val = float('inf')

    ckpt_dir       = config.get('checkpoint_dir', '/kaggle/working/checkpoints_ext_sc')
    best_model_dir = config.get('best_model_dir',  '/kaggle/working/best_model_ext_sc')
    accum_steps    = config.get('accum_steps', 1)
    grad_clip      = config.get('grad_clip', 1.0)
    patience       = config.get('patience', 10)
    min_delta      = config.get('min_delta', 1e-4)
    save_every     = config.get('save_every', 5)

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

    history = {
        'train_total': [], 'val_total': [], 'lr': [],
        'breakdown': {
            'loss_spatial': {'train': [], 'val': []},
            'loss_global':  {'train': [], 'val': []},
            'loss_latent':  {'train': [], 'val': []},
            'loss_identity': {'train': [], 'val': []},
        }
    }

    ema_decay = 0.999
    ema_extractor = ForensicIntegrityAnalyzer(num_groups=8, num_res_blocks=6, fp_dim=fp_dim).to(rank)
    
    for p in ema_extractor.parameters(): p.requires_grad = False 
    
    ema_extractor.load_state_dict(extractor.module.state_dict())
    ema_extractor.eval()

    for itr in range(start_epoch, max_iterations):
        start_time = time.time()
        train_sampler.set_epoch(itr)
        
        extractor.train(); attack_layer.train(); tamper_layer.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0; train_bd = defaultdict(float); train_n = 0

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(rank, non_blocking=True), masks.to(rank, non_blocking=True)
            B = images.size(0)

            # The Goldilocks Alpha Schedule

            if itr < 20:
                a_obj, a_bg = 0.15, 0.06
            elif itr < 40:
                a_obj, a_bg = 0.18, 0.08
            else:
                    a_obj = 0.225
                    a_bg  = 0.0925
            is_acc = (step + 1) % accum_steps != 0 and (step + 1) != len(train_loader)
            sync_ctx = extractor.no_sync() if is_acc else nullcontext()

            with sync_ctx:
                with autocast(device_type='cuda'):
                    soft_masks = TF.gaussian_blur(masks, kernel_size=[15,15], sigma=[5.0,5.0])
                    objs, bgs = images * soft_masks, images * (1 - soft_masks)
                    mask_obj, mask_bg = semantic_masker(objs), semantic_masker(bgs)

                    zo, sko = ae.module.enc_obj(objs)
                    zb, skb = ae.module.enc_bg(bgs)

                    idx_base = torch.randint(
                        0,
                        len(injector.base_w1_pool),
                        (B,),
                        device=rank
                    )

                    num_ids = 256
                    identity_ids = idx_base.clone()

                    base_w1 = injector.base_w1_pool[idx_base]
                    base_w2 = injector.base_w2_pool[idx_base]

                    wm_obj_z = injector.inject(zo, base_w1, a_obj, mask_obj)
                    wm_bg_z  = injector.inject(zb, base_w2, a_bg, mask_bg)

                    mask_latent = F.interpolate(soft_masks, size=(28,28), mode='nearest')
                    z_composite = wm_obj_z * mask_latent + wm_bg_z * (1 - mask_latent)
                    sk_composite = ae.module.blend_skips(sko, skb, soft_masks)
                    wm_composite = ae.module.shared_decoder(z_composite, sk_composite)

                    mode_weights = torch.tensor([0.3, 0.2, 0.2, 0.1, 0.2], device=rank)
                    attack_mode = torch.multinomial(mode_weights, 1).item()

                    gt_int  = torch.zeros(B, 1, 56, 56, device=rank)
                    gt_glob = torch.zeros(B, dtype=torch.long, device=rank)

                    current_w1, current_w2 = base_w1, base_w2

                    if attack_mode == 0:
                        
                        wm_attacked = attack_layer(wm_composite, itr, max_epochs=max_iterations)
                        gt_glob[:] = 0.0

                    elif attack_mode == 1:
                        
                        base_attacked = attack_layer(wm_composite, itr, max_epochs=max_iterations)
                        wm_attacked, gt_int = tamper_layer(base_attacked)
                        gt_glob[:] = 1
                    
                    elif attack_mode == 2:

                        idx_fake1 = torch.randint(0, len(injector.fake_w1_pool),(B,),device=rank)
                        idx_fake2 = torch.randint(0,len(injector.fake_w2_pool), (B,), device=rank)

                        identity_ids = idx_fake1.clone()

                        wrong_w1 = injector.fake_w1_pool[idx_fake1]
                        wrong_w2 = injector.fake_w2_pool[idx_fake2]
                        
                        
                        wrong_z_obj = injector.inject(zo, wrong_w1, a_obj, mask_obj)
                        wrong_z_bg  = injector.inject(zb, wrong_w2, a_bg, mask_bg)
                        z_fake = wrong_z_obj * mask_latent + wrong_z_bg * (1 - mask_latent)
                        fake_composite = ae.module.shared_decoder(z_fake, sk_composite)
                        wm_attacked = attack_layer(fake_composite, itr, max_epochs=max_iterations)
                        
                        gt_glob[:] = 2
                        gt_int = torch.zeros(B, 1, 56, 56, device=rank)
                        

                    elif attack_mode == 3:
                        wm_attacked = wm_composite
                        gt_glob[:] = 0

                    elif attack_mode == 4:
                        wm_attacked = images
                        gt_glob[:] = 3

                    else:
                        raise ValueError(f"Unknown attack_mode: {attack_mode}")

                    with torch.no_grad():
                        atk_z_obj, _ = ae.module.enc_obj(wm_attacked * soft_masks)
                        atk_z_bg, _  = ae.module.enc_bg(wm_attacked * (1 - soft_masks))
                        clean_obj, _ = ae.module.enc_obj(images * soft_masks)
                        clean_bg,  _ = ae.module.enc_bg(images * (1 - soft_masks))

                    # [FIX 1] Clean Structural Distillation (No fake projections!)
                    norm_atk_obj = F.normalize(atk_z_obj, dim=1)
                    norm_cln_obj = F.normalize(clean_obj, dim=1)
                    shift_obj = norm_atk_obj - norm_cln_obj
                    
                    norm_atk_bg = F.normalize(atk_z_bg, dim=1)
                    norm_cln_bg = F.normalize(clean_bg, dim=1)
                    shift_bg = norm_atk_bg - norm_cln_bg

                    # Combine shifts using the semantic mask
                    shift_total = shift_obj * mask_latent + shift_bg * (1 - mask_latent)
                    
                    # Target is the pure normalized direction of the watermark shift
                    # [FIX 4] Global Spatial Normalization for stable MSE
                    shift_flat = shift_total.flatten(1)
                    gt_latent = F.normalize(shift_flat, dim=1).view_as(shift_total)
                    
                    
                    pred_int, pred_glob, z_mean, z_logvar, pred_fp = extractor(wm_attacked)
                    
                    # ---------------------------------------------------------
                    # [FINAL FIX] Latent & Identity Curriculum Weights
                    # ---------------------------------------------------------
                    # 1. Latent Weight: Delay so identity/spatial learn first
                    # ════════════════════════════════════════════════════════════
                    # [PHASE CONTROLLER] LOSS WEIGHT CURRICULUM
                    # ════════════════════════════════════════════════════════════
                    
                    if itr < 40:
                        # PHASE 1: Spatial & Global ONLY
                        id_weight = 0.0
                        latent_weight = 0.0
                    
                    elif itr < 60:
                        id_weight = 1.0
                        latent_weight = 0.05
                    else:
                        id_weight = 1.0
                        latent_weight = 0.05
                        
                    # Always zero out these weights if attack mode is 4 (no watermark)
                    if attack_mode == 4:
                        id_weight = 0.0
                        latent_weight = 0.0
                    # ════════════════════════════════════════════════════════════


                    loss, bd = criterion(
                        pred_int, gt_int, pred_glob, gt_glob, z_mean,
                        z_logvar, gt_latent, pred_fp, identity_ids,
                        extractor.module, current_epoch=itr, 
                        latent_weight=latent_weight, id_weight=id_weight,
                        max_epochs=max_iterations  # <-- ADD THIS LINE
                    )

                    loss = loss / accum_steps
                
            if not torch.isfinite(loss):
                print(f"[Rank {rank}] WARNING: Non-finite loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()

            if 40 <= itr < 60:
                # We wipe the gradients AFTER DDP has synchronized them, 
                # so AdamW completely skips updating these layers.
                for name, p in extractor.module.named_parameters():
                    if 'layer1' in name or 'layer2' in name or 'layer3' in name:
                        p.grad = None

            if not is_acc:
                scaler.unscale_(optimizer)
                

                clip_grad_norm_(extractor.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # ═════════════════════════════════════════════════════════════════
                # [FIX] UPDATE EMA WEIGHTS
                # ═════════════════════════════════════════════════════════════════
                with torch.no_grad():
                    for p_ema, p_model in zip(ema_extractor.parameters(), extractor.module.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)
                # ═════════════════════════════════════════════════════════════════

            bs = images.size(0)
            train_loss += loss.item() * accum_steps * bs
            train_n += bs
            for k in bd: train_bd[k] += bd[k] * bs
        
        extractor.eval(); attack_layer.eval(); tamper_layer.eval()
        val_loss = 0.0; val_bd = defaultdict(float); val_n = 0

        with torch.inference_mode():
            for step, (images, masks) in enumerate(val_loader):
                images, masks = images.to(rank, non_blocking=True), masks.to(rank, non_blocking=True)
                B = images.size(0)

                with autocast(device_type='cuda'):
                    soft_masks = TF.gaussian_blur(masks, kernel_size=[15,15], sigma=[5.0,5.0])
                    objs, bgs = images * soft_masks, images * (1 - soft_masks)
                    mask_obj, mask_bg = semantic_masker(objs), semantic_masker(bgs)
                    zo, sko = ae.module.enc_obj(objs)
                    zb, skb = ae.module.enc_bg(bgs)

                    if itr < 20:
                        a_obj, a_bg = 0.15, 0.06
                    
                    elif itr < 40:
                        a_obj, a_bg = 0.18, 0.08
                    
                    else:
                        a_obj = 0.225
                        a_bg  = 0.0925

                    idx_val = torch.arange(step * B,(step + 1) * B,device=rank) % len(injector.base_w1_pool)

                    identity_ids = idx_val.clone()

                    base_w1 = injector.base_w1_pool[idx_val]
                    base_w2 = injector.base_w2_pool[idx_val]

                    wm_obj_z = injector.inject(zo, base_w1, a_obj, mask_obj)
                    wm_bg_z  = injector.inject(zb, base_w2, a_bg, mask_bg)

                    mask_latent = F.interpolate(soft_masks, size=(28,28), mode='bilinear', align_corners=False)
                    z_composite = wm_obj_z * mask_latent + wm_bg_z * (1 - mask_latent)
                    sk_composite = ae.module.blend_skips(sko, skb, soft_masks)
                    wm_composite = ae.module.shared_decoder(z_composite, sk_composite)

                    attack_mode = step % 5
                    gt_int  = torch.zeros(B, 1, 56, 56, device=rank)
                    gt_glob = torch.zeros(B, dtype=torch.long, device=rank)

                    current_w1, current_w2 = base_w1, base_w2

                    if attack_mode == 0:
                        wm_attacked = val_attack_layer(wm_composite, itr)
                        gt_glob[:] = 0

                    elif attack_mode == 1:
                        
                        
                        wm_attacked, gt_int = tamper_layer(val_attack_layer(wm_composite, itr), seed=step)
                        gt_glob[:] = 1
                    
                    elif attack_mode == 2:
                        
                        idx_fake1 = torch.arange(step * B,(step + 1) * B,device=rank) % len(injector.fake_w1_pool)
                        idx_fake2 = torch.arange(step * B,(step + 1) * B,device=rank) % len(injector.fake_w2_pool)
                        
                        identity_ids = idx_fake1.clone()
                        
                        wrong_w1 = injector.fake_w1_pool[idx_fake1]
                        wrong_w2 = injector.fake_w2_pool[idx_fake2]
                        
                        wrong_z_obj = injector.inject(zo, wrong_w1, a_obj, mask_obj)
                        wrong_z_bg  = injector.inject(zb, wrong_w2, a_bg, mask_bg)
                        z_fake = wrong_z_obj * mask_latent + wrong_z_bg * (1 - mask_latent)

                        fake_composite = ae.module.shared_decoder(z_fake, sk_composite)
                        wm_attacked = val_attack_layer(fake_composite, itr)
                        gt_glob[:] = 2
                        gt_int = torch.zeros(B, 1, 56, 56, device=rank)

                    elif attack_mode == 3:

                        wm_attacked = wm_composite
                        gt_glob[:] = 0
                        
                    elif attack_mode == 4:
                        wm_attacked = images
                        gt_glob[:] = 3
                    
                    else:
                        raise ValueError(f"Unknown attack_mode: {attack_mode}")

                    with torch.no_grad():
                        atk_z_obj, _ = ae.module.enc_obj(wm_attacked * soft_masks)
                        atk_z_bg, _  = ae.module.enc_bg(wm_attacked * (1 - soft_masks))
                        clean_obj, _ = ae.module.enc_obj(images * soft_masks)
                        clean_bg,  _ = ae.module.enc_bg(images * (1 - soft_masks))

                    # [FIX 1] Clean Structural Distillation (No fake projections!)
                    norm_atk_obj = F.normalize(atk_z_obj, dim=1)
                    norm_cln_obj = F.normalize(clean_obj, dim=1)
                    shift_obj = norm_atk_obj - norm_cln_obj
                    
                    norm_atk_bg = F.normalize(atk_z_bg, dim=1)
                    norm_cln_bg = F.normalize(clean_bg, dim=1)
                    shift_bg = norm_atk_bg - norm_cln_bg

                    # Combine shifts using the semantic mask
                    shift_total = shift_obj * mask_latent + shift_bg * (1 - mask_latent)
                    
                    # Target is the pure normalized direction of the watermark shift
                    # [FIX 4] Global Spatial Normalization for stable MSE
                    shift_flat = shift_total.flatten(1)
                    gt_latent = F.normalize(shift_flat, dim=1).view_as(shift_total)
                    
                    pred_int, pred_glob, z_mean, z_logvar, pred_fp = ema_extractor(wm_attacked)

                    # ════════════════════════════════════════════════════════════
                    # [PHASE CONTROLLER] LOSS WEIGHT CURRICULUM
                    # ════════════════════════════════════════════════════════════
                    
                    if itr < 40:
                        # PHASE 1: Spatial & Global ONLY
                        id_weight = 0.0
                        latent_weight = 0.0
                    elif itr < 60:
                        # PHASE 2: Add ArcFace Metric Geometry
                        id_weight = 1.0
                        latent_weight = 0.0
                    else:
                        # PHASE 3: Add Weak Latent Consistency
                        id_weight = 1.0
                        latent_weight = 0.05
                        
                    # Always zero out these weights if attack mode is 4 (no watermark)
                    if attack_mode == 4:
                        id_weight = 0.0
                        latent_weight = 0.0

                    # ════════════════════════════════════════════════════════════


                    val_step_loss, bd = criterion(
                        pred_int, gt_int, pred_glob, gt_glob, 
                        z_mean, z_logvar, gt_latent, pred_fp,
                        identity_ids, ema_extractor, current_epoch=itr,
                        latent_weight=latent_weight, id_weight=id_weight,
                        max_epochs=max_iterations  # <-- ADD THIS LINE
                    )

                bs = images.size(0)
                val_loss += val_step_loss.item() * bs
                val_n += bs
                for k in bd: val_bd[k] += bd[k] * bs

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        keys = ['loss_spatial', 'loss_global', 'loss_latent', 'loss_identity']
        metrics = torch.tensor(
            [train_loss] + [train_bd[k] for k in keys] + [float(train_n),
             val_loss]   + [val_bd[k]   for k in keys] + [float(val_n)],
            dtype=torch.float64, device=rank
        )
        dist.all_reduce(metrics)
        m = metrics.tolist()

        n_keys = len(keys)
        t_n = m[1+n_keys]
        v_n = m[3+2*n_keys]
        g_t = {k: m[1+i]/t_n for i,k in enumerate(keys)}
        g_t['total'] = m[0]/t_n
        g_v = {k: m[3+n_keys+i]/v_n for i,k in enumerate(keys)}
        g_v['total'] = m[2+n_keys]/v_n

        
        if rank == 0:

            print("Validation finished")
            elapsed = time.time() - start_time
            print(f"Epoch {itr+1:3d}/{max_iterations} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            print(f"  Train | total={g_t['total']:.5f}  spatial={g_t['loss_spatial']:.5f}  global={g_t['loss_global']:.5f}")
            print(f"  Val   | total={g_v['total']:.5f}  spatial={g_v['loss_spatial']:.5f}  global={g_v['loss_global']:.5f}")
            print('-' * 70)

            history['train_total'].append(g_t['total'])
            history['val_total'].append(g_v['total'])
            history['lr'].append(current_lr)
            for k in keys:
                history['breakdown'][k]['train'].append(g_t[k])
                history['breakdown'][k]['val'].append(g_v[k])

            phase_3_start = int(0.75 * max_iterations)
            if (itr) == 40 or (itr) == phase_3_start:
                print(f"    [!] Curriculum Shift Detected at Epoch {itr}. Resetting best_val.")
                best_val = float('inf')
                patience_ctr = 0

            if g_v['total'] < best_val - min_delta:
                
                best_val     = g_v['total']
                patience_ctr = 0

                print("Saving checkpoint...")
                
                save_checkpoint({
                    "epoch": itr+1, "model": extractor.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val, "config": config,
                }, os.path.join(ckpt_dir, "best_model.pth"))
                
                torch.save(extractor.module.state_dict(), os.path.join(best_model_dir, "best_weights.pth"))
                
                torch.save(ema_extractor.state_dict(), os.path.join(best_model_dir, "best_ema_weights.pth"))
                
                torch.save({
                    "epoch": itr+1, "best_val_loss": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "config":    config,
                }, os.path.join(best_model_dir, "best_optimizer_state.pth"))

                print("Checkpoint save finished")
            
            else:
            
                patience_ctr += 1

            if (itr + 1) % save_every == 0:
                save_checkpoint({
                    "epoch": itr+1, "model": extractor.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val, "config": config,
                }, os.path.join(ckpt_dir, f"epoch_{itr+1:03d}.pth"))

        stop_t = torch.tensor(patience_ctr, dtype=torch.int32, device=rank)
        dist.broadcast(stop_t, src=0)
        if stop_t.item() >= patience:
            if rank == 0: print(f"\n[!] Early stopping at epoch {itr+1}")
            break

    if rank == 0:
        history['best_val_loss'] = best_val
        hist_path = os.path.join(ckpt_dir, 'history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        save_checkpoint({
            "epoch": max_iterations, "model": extractor.module.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_val_loss": best_val, "config": config,
        }, os.path.join(ckpt_dir, "final_checkpoint.pth"))

    dist.destroy_process_group()

def get_config(stage_key):
    parser = argparse.ArgumentParser(description=f"Parser for {stage_key}")
    parser.add_argument('--config', type=str, required=True, help='Path to master config.json')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        master_json = json.load(f)

    cfg = {}
    cfg.update(master_json.get('paths', {}))
    cfg.update(master_json.get('shared', {}))
    cfg.update(master_json.get(stage_key, {}))

    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.lr is not None: cfg['lr'] = args.lr
    if args.resume is not None: cfg['resume'] = args.resume

    return cfg

if __name__ == '__main__':

    config = get_config(stage_key='stage3_extractor')
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")

    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)


    # ── Paths ───────────────────────────────────────────────────────────────

    # config.setdefault('images_dir',     '/kaggle/working/clean_pet_data/images')
    # config.setdefault('mask_dir',       '/kaggle/working/clean_pet_data/masks')
    # config.setdefault('checkpoint_dir', '/kaggle/working/checkpoints_e2e')
    # config.setdefault('best_model_dir', '/kaggle/working/best_model_e2e')

    # # Load your best AE weights from Stage 2b so it doesn't start from scratch!
    # config.setdefault('ae_weights',       '/kaggle/input/models/mrheavenly/stage-2b/pytorch/default/1/best_weights_wm.pth')
    # config.setdefault('injector_weights', '/kaggle/input/models/mrheavenly/stage-2b/pytorch/default/1/injector.pth')

    # # ── Data & Watermark ────────────────────────────────────────────────────
    # config.setdefault('img_size',    224)
    # config.setdefault('max_samples', None)
    # config.setdefault('split_size',  0.85)
    # config.setdefault('cache_ram',   False)

    # config.setdefault('latent_channels', 256)
    # # Note: 'alpha' is now overridden by the Differential Alphas (0.5 and 3.5) in the training loop!
    # config.setdefault('alpha',           0.15) 
    # config.setdefault('w1_seed',         'object')
    # config.setdefault('w2_seed',         'background')

    # # ── Attack Parameters ───────────────────────────────────────────────────
    # config.setdefault('attack_p', 0.8) # 80% chance to attack
    # config.setdefault('attack_warmup_epochs', 5) # 5 epochs of peace before the gauntlet

    # # ── End-to-End Training Params ──────────────────────────────────────────
    # config.setdefault('batch_size',      4)    # CRITICAL: Dropped to 4 for the Heavy Decoder!
    # config.setdefault('max_iterations',  80)   
    # config.setdefault('lr',              1e-4) # 1e-4 for stable joint training
    # config.setdefault('weight_decay',    1e-4)
    # config.setdefault('grad_clip',       1.0)
    # config.setdefault('accum_steps',     4)    # Effective batch size = 16 (4 * 4)
    # config.setdefault('patience',        100)   
    # config.setdefault('min_delta',       1e-4)
    # config.setdefault('resume',          None)
    # config.setdefault('save_every',      5)

    # # ── New End-to-End Loss Weights ─────────────────────────────────────────
    # # Note: These are now hardcoded in the loop as 5.0 and 5.0, but keeping here for reference
    # config.setdefault('lambda_l1',   5.0) 
    # config.setdefault('lambda_ext',  5.0) 

    # config['num_workers'] = 4

    # save_config(config, utils.CONFIG_PATH)