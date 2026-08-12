import os
import json
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from datetime import datetime
from torch.utils.data import DataLoader
import time

from torchvision.utils import save_image
from torchvision.io import read_image

import hashlib
from collections import OrderedDict
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision.transforms.functional as TF
import torch.nn as nn

class SkipDropout(nn.Module):
    def __init__(self, p=0.5, force_drop=False):
        super().__init__()
        self.p = p
        self.force_drop = force_drop
    def forward(self, x):
        if self.force_drop:
            return torch.zeros_like(x)
        if self.training:    
            if torch.rand(1, device=x.device).item() < self.p:
                return torch.zeros_like(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            norm_layer(out_ch),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            norm_layer(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.proj(x) + self.conv_path(x))

class ResNetAE(nn.Module):
    def __init__(self, latent_channels=256, num_groups=8, skip_proj_config=None):
        super().__init__()
        proj = skip_proj_config or {'s3': 64, 's2': 32, 's1': 16, 's0': 16}
        self.stem  = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.GroupNorm(num_groups,64), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1, bias=False), nn.GroupNorm(num_groups,64), nn.ReLU(inplace=True), ResBlock(64, 64))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.GroupNorm(num_groups,128), nn.ReLU(inplace=True), ResBlock(128, 128))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.GroupNorm(num_groups,256), nn.ReLU(inplace=True), ResBlock(256, 256))
        self.skip_proj_s3 = nn.Conv2d(256, proj['s3'], 1, bias=False)
        self.skip_proj_s2 = nn.Conv2d(128, proj['s2'], 1, bias=False)
        self.skip_proj_s1 = nn.Conv2d(64, proj['s1'], 1, bias=False)
        decoder_norm = lambda ch: nn.GroupNorm(num_groups, ch)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, 1, bias=False),
            nn.GroupNorm(num_groups, latent_channels), nn.ReLU(inplace=True),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
        )
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, 1, bias=False)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, 1, bias=False)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels + proj['s3'], 256, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 256), nn.ReLU(inplace=True),   
            ResBlock(256, 256, norm_layer=decoder_norm),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256 + proj['s2'], 128, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 128), nn.ReLU(inplace=True),
            ResBlock(128, 128, norm_layer=decoder_norm),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + proj['s1'], 64, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            ResBlock(64, 64, norm_layer=decoder_norm),
        )
        self.output = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())
        self.watermark_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 3, 1, 1)
        )
        self.s0_drop = SkipDropout(p=0.7)
        self.s1_drop = SkipDropout(p=0.5)
        self.s2_drop = SkipDropout(p=0.3)
        self.s3_drop = SkipDropout(p=0.2)
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    def extract_watermark(self, rec):
        return self.watermark_extractor(rec)
    def forward(self, x, force_skip_drop=False, z_override=None):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        feat = self.bottleneck(s3)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        z  = self.reparameterize(mu, logvar) if z_override is None else z_override
        if force_skip_drop:
            s0, s1, s2, s3 = torch.zeros_like(s0), torch.zeros_like(s1), torch.zeros_like(s2), torch.zeros_like(s3)
        else:
            s0 = self.s0_drop(s0)
            s1 = self.s1_drop(s1)
            s2 = self.s2_drop(s2)
            s3 = self.s3_drop(s3)
        s3_proj = self.skip_proj_s3(s3)
        s2_proj = self.skip_proj_s2(s2)
        s1_proj = self.skip_proj_s1(s1)
        x  = self.up1(torch.cat([z,  s3_proj], dim=1))
        x  = self.up2(torch.cat([x,  s2_proj], dim=1))
        x  = self.up3(torch.cat([x,  s1_proj], dim=1))
        out = self.output(x)
        return out, z, mu, logvar

class DualAutoencoder(nn.Module):

    def __init__(self, latent_channels=256, skip_proj_config=None):
        super().__init__()
        self.ae_obj = ResNetAE(latent_channels=latent_channels, skip_proj_config=skip_proj_config)
        self.ae_bg  = ResNetAE(latent_channels=latent_channels, skip_proj_config=skip_proj_config)
    def forward(self, objs, bgs, wm_std=0.0, compute_aux=True, compute_wm=True, force_skip_drop=False, z_override_obj=None, z_override_bg=None):

        rec_obj, z_obj, mu_obj, logvar_obj = self.ae_obj(objs, force_skip_drop, z_override=z_override_obj)
        rec_bg,  z_bg,  mu_bg,  logvar_bg  = self.ae_bg(bgs, force_skip_drop, z_override=z_override_bg)
        out = [rec_obj, rec_bg, z_obj, z_bg, mu_obj, logvar_obj, mu_bg, logvar_bg]

        if compute_aux:
            rec_obj_ns, _, _, _ = self.ae_obj(objs, force_skip_drop=True)
            rec_bg_ns,  _, _, _ = self.ae_bg(bgs,  force_skip_drop=True)
            out.extend([rec_obj_ns, rec_bg_ns])

        if compute_wm and wm_std > 0:

            wm_obj = torch.randn_like(z_obj) * wm_std
            wm_bg  = torch.randn_like(z_bg)  * wm_std
            rec_obj_wm, _, _, _ = self.ae_obj(objs, force_skip_drop, z_override=z_obj + wm_obj)
            rec_bg_wm,  _, _, _ = self.ae_bg(bgs,  force_skip_drop, z_override=z_bg + wm_bg)
            pred_wm_obj = self.ae_obj.extract_watermark(rec_obj_wm)
            pred_wm_bg  = self.ae_bg.extract_watermark(rec_bg_wm)
            out.extend([rec_obj_wm, rec_bg_wm, pred_wm_obj, pred_wm_bg, wm_obj, wm_bg])

        return tuple(out)


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

def generate_watermark_pool(latent_channels, latent_h, latent_w, prefix, num=256):
    N = latent_channels * latent_h * latent_w
    r, mu, T_thresh = 3.99, 0.99, 1.0
    
    xs = torch.zeros(num, dtype=torch.float32)
    ys = torch.zeros(num, dtype=torch.float32)
    
    for i in range(num):
        secret_key = f"{prefix}_{i}"
        hash_hex = hashlib.sha512(secret_key.encode('utf-8')).hexdigest()
        x0 = int(hash_hex[:64], 16) / (16**64)
        y0 = int(hash_hex[64:], 16) / (16**64)
        if x0 in (0, 0.5, 1): x0 = 0.12345
        if y0 in (0, 0.5, 1): y0 = 0.67890
        xs[i] = x0
        ys[i] = y0
        
    watermarks = torch.zeros(num, N, dtype=torch.float32)
    
    for i in range(N):
        xs = r * xs * (1.0 - xs)
        ys = mu * torch.sin(math.pi * ys)
        watermarks[:, i] = torch.where((xs + ys) > T_thresh, 1.0, -1.0)
        
    watermarks = watermarks.view(num, latent_channels, latent_h, latent_w)
    watermarks = watermarks / (watermarks.std(dim=(1,2,3), keepdim=True) + 1e-6)
    return torch.tanh(watermarks)

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
        self.register_buffer('base_w1_pool', generate_watermark_pool(latent_channels, latent_h, latent_w, "base_obj"))
        self.register_buffer('base_w2_pool', generate_watermark_pool(latent_channels, latent_h, latent_w, "base_bg"))
        self.register_buffer('fake_w1_pool', generate_watermark_pool(latent_channels, latent_h, latent_w, "fake_obj"))
        self.register_buffer('fake_w2_pool', generate_watermark_pool(latent_channels, latent_h, latent_w, "fake_bg"))
        self.project = nn.Identity()

    def inject(self, z, w, alpha, semantic_mask):
        w = w / (w.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        w = torch.tanh(w)
        w = w * 0.28
        if semantic_mask is not None:
            mask = (semantic_mask * 0.75) + 0.25
            w = w * mask
        delta = alpha * w
        delta = torch.clamp(delta, min=-0.25, max=0.25) 
        return z + delta

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device, vgg_weights_path=None):
        super().__init__()
        self.offline = False
        try:
            vgg = vgg16(weights=None)
            if vgg_weights_path and os.path.isfile(vgg_weights_path):
                vgg.load_state_dict(torch.load(vgg_weights_path, map_location='cpu'))
                print(f"  [VGG] loaded offline weights → {vgg_weights_path}")
            else:
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            
            vgg = vgg.features.to(device).eval()
            self.slice1 = nn.Sequential(*list(vgg.children())[:9])    
            self.slice2 = nn.Sequential(*list(vgg.children())[9:16])  
            for p in self.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"\n[!] Network Error: Could not download VGG16 weights for LPIPS ({e}). Skipping perceptual metrics.\n")
            self.offline = True
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.to(device)

    def preprocess(self, x):
        return ((x + 1.0) / 2.0 - self.mean) / self.std  

    def forward(self, x):
        if self.offline:
            return None, None
        x  = self.preprocess(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        return f1, f2

def compute_psnr(rec, target, data_range=2.0):
    mse = F.mse_loss(rec, target)
    return 20 * torch.log10(torch.tensor(data_range, device=rec.device)) - 10 * torch.log10(mse + 1e-8)

def compute_ssim(rec, target, data_range=2.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    channels = rec.shape[1]
    kernel = torch.ones(channels, 1, 3, 3, device=rec.device) / 9.0
    mu_rec = F.conv2d(rec, kernel, padding=1, groups=channels)
    mu_tgt = F.conv2d(target, kernel, padding=1, groups=channels)
    sigma_rec = F.conv2d(rec * rec, kernel, padding=1, groups=channels) - mu_rec ** 2
    sigma_tgt = F.conv2d(target * target, kernel, padding=1, groups=channels) - mu_tgt ** 2
    sigma_rt  = F.conv2d(rec * target, kernel, padding=1, groups=channels) - mu_rec * mu_tgt
    ssim_map = ((2 * mu_rec * mu_tgt + C1) * (2 * sigma_rt + C2)) / ((mu_rec ** 2 + mu_tgt ** 2 + C1) * (sigma_rec + sigma_tgt + C2))
    return ssim_map.mean()


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class Experiment:
    def __init__(self, name, results_dir):
        self.name = name
        self.results_dir = results_dir
        self.metrics = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def run(self, **kwargs):
        raise NotImplementedError

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Reconstruction Fidelity
# ══════════════════════════════════════════════════════════════════════════════

class ReconstructionExperiment(Experiment):
    def run(self, clean_image, wm_image, vgg_extractor):
        psnr_val = compute_psnr(wm_image, clean_image).item()
        ssim_val = compute_ssim(wm_image, clean_image).item()
        l1_val   = F.l1_loss(wm_image, clean_image).item()
        mse_val  = F.mse_loss(wm_image, clean_image).item()
        
        f1_c, f2_c = vgg_extractor(clean_image)
        if f1_c is not None:
            f1_w, f2_w = vgg_extractor(wm_image)
            lpips_val  = (F.mse_loss(f1_c, f1_w) + F.mse_loss(f2_c, f2_w)).item()
        else:
            lpips_val = 0.0
        
        self.metrics = {
            'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val,
            'L1': l1_val, 'MSE': mse_val
        }
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Latent Perturbation & Distribution
# ══════════════════════════════════════════════════════════════════════════════

class LatentAnalysisExperiment(Experiment):
    def run(self, z_clean, z_wm):
        delta_z = z_wm - z_clean
        
        energy_clean = torch.sum(z_clean**2).item()
        energy_wm    = torch.sum(z_wm**2).item()
        energy_ratio = energy_wm / (energy_clean + 1e-8)
        
        l1_norm = torch.norm(delta_z, p=1).item()
        l2_norm = torch.norm(delta_z, p=2).item()
        max_mag = torch.max(torch.abs(delta_z)).item()
        
        mean_clean = torch.mean(z_clean).item()
        var_clean  = torch.var(z_clean).item()
        mean_wm    = torch.mean(z_wm).item()
        var_wm     = torch.var(z_wm).item()
        
        # PCA Visualization
        B, C, H, W = z_clean.shape
        z_c_flat = z_clean.view(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu().numpy()
        z_w_flat = z_wm.view(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu().numpy()
        
        pca = PCA(n_components=2)
        idx = np.random.choice(z_c_flat.shape[0], min(2000, z_c_flat.shape[0]), replace=False)
        pts_c = pca.fit_transform(z_c_flat[idx])
        pts_w = pca.transform(z_w_flat[idx])
        
        plt.figure(figsize=(8,6))
        plt.scatter(pts_c[:,0], pts_c[:,1], alpha=0.5, label='Clean', s=10)
        plt.scatter(pts_w[:,0], pts_w[:,1], alpha=0.5, label='Watermarked', s=10)
        plt.legend()
        plt.title('PCA of Latent Space (Clean vs Watermarked)')
        plt.savefig(os.path.join(self.results_dir, '../plots/pca_tsne.png'))
        plt.close()
        
        self.metrics = {
            'Energy Ratio': energy_ratio, 'L1 Norm': l1_norm, 'L2 Norm': l2_norm,
            'Max Mag': max_mag, 'Mean Clean': mean_clean, 'Var Clean': var_clean,
            'Mean WM': mean_wm, 'Var WM': var_wm
        }
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Recovery & Consistency (BER)
# ══════════════════════════════════════════════════════════════════════════════

class RecoveryExperiment(Experiment):
    def compute_ber(self, pred_wm, gt_wm):
        bits_pred = (pred_wm > 0).float()
        bits_gt   = (gt_wm > 0).float()
        
        incorrect = (bits_pred != bits_gt).float().sum().item()
        total = bits_gt.numel()
        
        false_positives = ((bits_pred == 1) & (bits_gt == 0)).float().sum().item()
        false_negatives = ((bits_pred == 0) & (bits_gt == 1)).float().sum().item()
        true_negatives  = (bits_gt == 0).float().sum().item()
        true_positives  = (bits_gt == 1).float().sum().item()
        
        fpr = false_positives / max(true_negatives, 1)
        fnr = false_negatives / max(true_positives, 1)
        
        ber = incorrect / total
        accuracy = 1.0 - ber
        
        cos_sim = F.cosine_similarity(pred_wm.flatten(), gt_wm.flatten(), dim=0).item()
        
        p_mean = pred_wm.mean()
        g_mean = gt_wm.mean()
        corr = ((pred_wm - p_mean) * (gt_wm - g_mean)).sum() / \
               torch.sqrt(((pred_wm - p_mean)**2).sum() * ((gt_wm - g_mean)**2).sum() + 1e-8)
               
        return ber, accuracy, fpr, fnr, cos_sim, corr.item()

    def run(self, ae, rec_wm_image, payload, mask):
        with torch.no_grad():
            pred_wm = ae.ae_obj.extract_watermark(rec_wm_image)
            
        ber, acc, fpr, fnr, cos, corr = self.compute_ber(pred_wm, payload)
        
        self.metrics = {
            'BER': ber, 'Bit Accuracy': acc, 'FPR': fpr, 'FNR': fnr,
            'Cosine Sim': cos, 'Correlation': corr
        }
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Alpha Sweep
# ══════════════════════════════════════════════════════════════════════════════

class AlphaSweepExperiment(Experiment):
    def run(self, ae, clean_image, bg_image, blend_mask, z_clean, payload, mask, vgg_extractor, alphas=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]):
        results = []
        recovery_exp = RecoveryExperiment("AlphaRecovery", self.results_dir)
        recon_exp = ReconstructionExperiment("AlphaRecon", self.results_dir)
        
        for alpha in alphas:
            with torch.no_grad():
                w = payload / (payload.std(dim=(1,2,3), keepdim=True) + 1e-6)
                w = torch.tanh(w) * 0.28
                w = w * ((mask * 0.75) + 0.25)
                delta = torch.clamp(alpha * w, min=-0.25, max=0.25)
                z_wm = z_clean + delta
                
                rec_wm_img, _, _, _ = ae.ae_obj(clean_image, z_override=z_wm)
                rec_clean_bg, _, _, _ = ae.ae_bg(bg_image, z_override=None)
                final_img = rec_wm_img * blend_mask + rec_clean_bg * (1 - blend_mask)
                
                rec_clean_obj, _, _, _ = ae.ae_obj(clean_image, z_override=z_clean)
                clean_full = rec_clean_obj * blend_mask + rec_clean_bg * (1 - blend_mask)

                r_metrics = recon_exp.run(clean_full, final_img, vgg_extractor)
                pred_wm = ae.ae_obj.extract_watermark(final_img)
                ber, acc, _, _, _, corr = recovery_exp.compute_ber(pred_wm, payload)
                
                energy = torch.sum(z_wm**2).item()
                
                results.append({
                    'Alpha': alpha, 'BER': ber, 'Accuracy': acc, 'PSNR': r_metrics['PSNR'], 
                    'SSIM': r_metrics['SSIM'], 'LPIPS': r_metrics['LPIPS'], 'Energy': energy, 'Corr': corr
                })
        
        import csv
        with open(os.path.join(self.results_dir, '../metrics/alpha.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        alphas_plot = [r['Alpha'] for r in results]
        psnr_plot = [r['PSNR'] for r in results]
        ber_plot = [r['BER'] for r in results]
        
        plt.figure()
        plt.plot(alphas_plot, psnr_plot, marker='o')
        plt.xlabel('Alpha')
        plt.ylabel('PSNR')
        plt.title('Alpha vs PSNR')
        plt.savefig(os.path.join(self.results_dir, '../plots/alpha_psnr.png'))
        plt.close()
        
        plt.figure()
        plt.plot(alphas_plot, ber_plot, marker='x', color='red')
        plt.xlabel('Alpha')
        plt.ylabel('BER')
        plt.title('Alpha vs BER')
        plt.savefig(os.path.join(self.results_dir, '../plots/alpha_ber.png'))
        plt.close()
        
        self.metrics = {'sweeps': len(alphas)}
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: Cycle Stability
# ══════════════════════════════════════════════════════════════════════════════

class CycleExperiment(Experiment):
    def run(self, ae, clean_image, bg_image, blend_mask, z_wm, payload):
        cycles = [1, 3, 5, 10]
        results = {}
        recovery_exp = RecoveryExperiment("CycleRecovery", self.results_dir)
        
        with torch.no_grad():
            curr_z = z_wm.clone()
            
            for i in range(1, max(cycles) + 1):
                rec_img, _, _, _ = ae.ae_obj(clean_image, z_override=curr_z)
                rec_bg, _, _, _ = ae.ae_bg(bg_image, z_override=None)
                final_img = rec_img * blend_mask + rec_bg * (1 - blend_mask)
                
                _, next_z, _, _ = ae.ae_obj(final_img * blend_mask)
                curr_z = next_z
                
                if i in cycles:
                    pred_wm = ae.ae_obj.extract_watermark(final_img)
                    ber, acc, _, _, _, _ = recovery_exp.compute_ber(pred_wm, payload)
                    results[f'Cycle_{i}_BER'] = ber
                    results[f'Cycle_{i}_Acc'] = acc

        plt.figure()
        plt.plot(cycles, [results[f'Cycle_{i}_BER'] for i in cycles], marker='o', color='purple')
        plt.xlabel('Cycle Count')
        plt.ylabel('BER')
        plt.title('Cycle Stability (Drift)')
        plt.savefig(os.path.join(self.results_dir, '../plots/cycle_stability.png'))
        plt.close()
        
        self.metrics = results
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 6: Noise Robustness
# ══════════════════════════════════════════════════════════════════════════════

class NoiseRobustnessExperiment(Experiment):
    def run(self, ae, clean_image, bg_image, blend_mask, z_wm, payload, rec_wm_img):
        recovery_exp = RecoveryExperiment("NoiseRecovery", self.results_dir)
        results = {}
        
        with torch.no_grad():
            for sigma in [1/255.0, 2/255.0, 5/255.0]:
                noise = torch.randn_like(rec_wm_img) * sigma
                noisy_img = torch.clamp(rec_wm_img + noise, -1, 1)
                pred_wm = ae.ae_obj.extract_watermark(noisy_img)
                ber, _, _, _, _, _ = recovery_exp.compute_ber(pred_wm, payload)
                results[f'ImageNoise_sig{sigma*255:.0f}_BER'] = ber
                
            for scale in [1e-4, 1e-3, 1e-2]:
                mag = torch.mean(torch.abs(z_wm))
                noise = torch.randn_like(z_wm) * (mag * scale)
                z_noisy = z_wm + noise
                dec_noisy, _, _, _ = ae.ae_obj(clean_image, z_override=z_noisy)
                dec_bg, _, _, _ = ae.ae_bg(bg_image, z_override=None)
                comp = dec_noisy * blend_mask + dec_bg * (1-blend_mask)
                pred_wm = ae.ae_obj.extract_watermark(comp)
                ber, _, _, _, _, _ = recovery_exp.compute_ber(pred_wm, payload)
                results[f'LatentNoise_scale{scale}_BER'] = ber
                
            z_fp16 = z_wm.half().float()
            dec_fp16, _, _, _ = ae.ae_obj(clean_image, z_override=z_fp16)
            dec_bg, _, _, _ = ae.ae_bg(bg_image, z_override=None)
            comp = dec_fp16 * blend_mask + dec_bg * (1-blend_mask)
            pred_wm = ae.ae_obj.extract_watermark(comp)
            ber, _, _, _, _, _ = recovery_exp.compute_ber(pred_wm, payload)
            results['FP16_Truncation_BER'] = ber

        self.metrics = results
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 7: Frequency Analysis
# ══════════════════════════════════════════════════════════════════════════════

class FrequencyExperiment(Experiment):
    def run(self, z_clean, z_wm):
        delta = z_wm - z_clean
        
        fft_c = torch.fft.fftshift(torch.fft.fft2(z_clean.float()))
        fft_w = torch.fft.fftshift(torch.fft.fft2(z_wm.float()))
        fft_d = torch.fft.fftshift(torch.fft.fft2(delta.float()))
        
        mag_c = torch.log(torch.abs(fft_c) + 1e-8).mean(dim=(0,1)).cpu().numpy()
        mag_w = torch.log(torch.abs(fft_w) + 1e-8).mean(dim=(0,1)).cpu().numpy()
        mag_d = torch.log(torch.abs(fft_d) + 1e-8).mean(dim=(0,1)).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        axes[0].imshow(mag_c, cmap='viridis'); axes[0].set_title('FFT Clean')
        axes[1].imshow(mag_w, cmap='viridis'); axes[1].set_title('FFT Watermarked')
        axes[2].imshow(mag_d, cmap='viridis'); axes[2].set_title('FFT Delta')
        plt.savefig(os.path.join(self.results_dir, '../plots/fft_analysis.png'))
        plt.close()
        
        self.metrics = {'FFT_Generated': True}
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 8: Channel Ablation
# ══════════════════════════════════════════════════════════════════════════════

class ChannelExperiment(Experiment):
    def run(self, ae, clean_image, bg_image, blend_mask, z_clean, mask, payload, alpha=0.225):
        groups = 8
        channels = z_clean.size(1)
        chunk_size = channels // groups
        results = {}
        recovery_exp = RecoveryExperiment("ChannelRecovery", self.results_dir)
        
        w = payload / (payload.std(dim=(1,2,3), keepdim=True) + 1e-6)
        w = torch.tanh(w) * 0.28
        w = w * ((mask * 0.75) + 0.25)
        
        for g in range(groups):
            with torch.no_grad():
                z_ablated = z_clean.clone()
                start = g * chunk_size
                end = (g + 1) * chunk_size
                delta = torch.clamp(alpha * w[:, start:end], min=-0.25, max=0.25)
                z_ablated[:, start:end] += delta
                
                dec, _, _, _ = ae.ae_obj(clean_image, z_override=z_ablated)
                dec_bg, _, _, _ = ae.ae_bg(bg_image, z_override=None)
                comp = dec * blend_mask + dec_bg * (1 - blend_mask)
                
                pred_wm = ae.ae_obj.extract_watermark(comp)
                ber, acc, _, _, _, _ = recovery_exp.compute_ber(pred_wm, payload)
                results[f'Group_{g}_({start}-{end-1})_BER'] = ber

        self.metrics = results
        return self.metrics

# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def get_config(config_path, stage_key='stage3_extractor'):
    with open(config_path, 'r', encoding='utf-8') as f:
        master_json = json.load(f)
    cfg = {}
    cfg.update(master_json.get('paths', {}))
    cfg.update(master_json.get('shared', {}))
    cfg.update(master_json.get(stage_key, {}))
    return cfg

class ExperimentRunner:
    def __init__(self, config_path):
        self.config = get_config(config_path, stage_key='stage3_extractor')
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            
        self.base_dir = '/kaggle/working/phase0_results'
        os.makedirs(os.path.join(self.base_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'watermark'), exist_ok=True)
        
        self.ae = DualAutoencoder(
            latent_channels=self.config.get('latent_channels', 256),
            skip_proj_config=self.config.get('skip_proj')
        ).to(self.device).eval()
        
        if 'ae_weights' in self.config and os.path.exists(self.config['ae_weights']):
            state = torch.load(self.config['ae_weights'], map_location='cpu')
            if 'model' in state: state = state['model']
            self.ae.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            
        self.injector = SSWatermarkInjector(256, 28, 28).to(self.device).eval()
        if 'injector_weights' in self.config and self.config['injector_weights'] and os.path.exists(self.config['injector_weights']):
            state = torch.load(self.config['injector_weights'], map_location='cpu')
            if 'model' in state: state = state['model']
            self.injector.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            print(f"  [Injector] loaded weights from → {self.config['injector_weights']}")
        self.masker = AdaptiveSemanticMasker().to(self.device).eval()
        self.vgg = VGGFeatureExtractor(self.device, vgg_weights_path=self.config.get('vgg_weights_path'))
        
        self.timing = {}

    def save_images(self, clean, wm, diff, rec):

        save_image(clean * 0.5 + 0.5, os.path.join(self.base_dir, 'images/clean.png'))
        save_image(wm * 0.5 + 0.5, os.path.join(self.base_dir, 'images/watermarked.png'))
        diff_img = torch.abs(wm - clean) * 5.0
        save_image(diff_img, os.path.join(self.base_dir, 'images/difference.png'))
        save_image(rec * 0.5 + 0.5, os.path.join(self.base_dir, 'images/reconstruction.png'))

    def run_pipeline(self, image_path, mask_path):

        t0 = time.time()
        
        img = TF.resize(read_image(image_path).float() / 255.0, [224, 224])
        mask = TF.resize(read_image(mask_path).float() / 255.0, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(img, [0.5]*3, [0.5]*3).unsqueeze(0).to(self.device)
        mask = (mask > 0.5).float().unsqueeze(0).to(self.device)
        if mask.shape[1] > 1: mask = mask[:, 0:1]
        
        t1 = time.time()
        soft_masks = TF.gaussian_blur(mask, [15, 15], [5.0, 5.0])
        objs, bgs = img * soft_masks, img * (1 - soft_masks)
        mask_obj = self.masker(objs)
        
        with torch.no_grad():
            _, z_obj, _, _ = self.ae.ae_obj(objs)
            _, z_bg, _, _ = self.ae.ae_bg(bgs)
        self.timing['Encoder'] = time.time() - t1
        
        t2 = time.time()
        payload = self.injector.base_w1_pool[0:1].clone()
        alpha = 0.225
        z_wm = self.injector.inject(z_obj, payload, alpha, mask_obj)
        self.timing['Injection'] = time.time() - t2
        
        t3 = time.time()
        with torch.no_grad():
            rec_wm_obj, _, _, _ = self.ae.ae_obj(objs, z_override=z_wm)
            rec_clean_bg, _, _, _ = self.ae.ae_bg(bgs, z_override=z_bg)
            wm_image = rec_wm_obj * soft_masks + rec_clean_bg * (1 - soft_masks)
            
            rec_cl_obj, _, _, _ = self.ae.ae_obj(objs, z_override=z_obj)
            clean_image = rec_cl_obj * soft_masks + rec_clean_bg * (1 - soft_masks)
        self.timing['Decoder'] = time.time() - t3
        
        torch.save(payload, os.path.join(self.base_dir, 'watermark/watermark_payload.pt'))
        torch.save(z_wm, os.path.join(self.base_dir, 'watermark/watermarked_latent.pt'))
        torch.save(self.injector.state_dict(), os.path.join(self.base_dir, 'watermark/injector.pth'))
        with open(os.path.join(self.base_dir, 'watermark/injection_params.json'), 'w') as f:
            json.dump({
                'alpha': alpha, 'seed': self.seed,
                'embedding_method': 'tanh additive with semantic gain control',
                'recovery_method': 'Stage 2 VAE extract_watermark(image) > 0'
            }, f, indent=2)
            
        self.save_images(clean_image, wm_image, wm_image - clean_image, clean_image)
        
        t4 = time.time()
        with torch.no_grad():
            self.ae.ae_obj.extract_watermark(wm_image)
        self.timing['Recovery'] = time.time() - t4
        
        re = ReconstructionExperiment("Recon", os.path.join(self.base_dir, 'metrics'))
        m_re = re.run(clean_image.clone().detach(), wm_image.clone().detach(), self.vgg)
        
        le = LatentAnalysisExperiment("Latent", os.path.join(self.base_dir, 'metrics'))
        m_le = le.run(z_obj.clone().detach(), z_wm.clone().detach())
        
        rc = RecoveryExperiment("Recovery", os.path.join(self.base_dir, 'metrics'))
        m_rc = rc.run(self.ae, wm_image.clone().detach(), payload.clone().detach(), soft_masks.clone().detach())
        
        ae_exp = AlphaSweepExperiment("Alpha", os.path.join(self.base_dir, 'metrics'))
        m_ae = ae_exp.run(self.ae, objs.clone().detach(), bgs.clone().detach(), soft_masks.clone().detach(), z_obj.clone().detach(), payload.clone().detach(), mask_obj.clone().detach(), self.vgg)
        
        ce = CycleExperiment("Cycle", os.path.join(self.base_dir, 'metrics'))
        m_ce = ce.run(self.ae, objs.clone().detach(), bgs.clone().detach(), soft_masks.clone().detach(), z_wm.clone().detach(), payload.clone().detach())
        
        ne = NoiseRobustnessExperiment("Noise", os.path.join(self.base_dir, 'metrics'))
        m_ne = ne.run(self.ae, objs.clone().detach(), bgs.clone().detach(), soft_masks.clone().detach(), z_wm.clone().detach(), payload.clone().detach(), wm_image.clone().detach())
        
        fe = FrequencyExperiment("Freq", os.path.join(self.base_dir, 'metrics'))
        m_fe = fe.run(z_obj.clone().detach(), z_wm.clone().detach())
        
        ch = ChannelExperiment("Channel", os.path.join(self.base_dir, 'metrics'))
        m_ch = ch.run(self.ae, objs.clone().detach(), bgs.clone().detach(), soft_masks.clone().detach(), z_obj.clone().detach(), mask_obj.clone().detach(), payload.clone().detach())
        
        self.timing['Total'] = time.time() - t0
        
        with open(os.path.join(self.base_dir, 'report.txt'), 'w') as f:
            f.write("=========================================\n")
            f.write(" PHASE 0: LATENT VERIFICATION REPORT\n")
            f.write("=========================================\n\n")
            
            f.write("--- TIMING ---\n")
            for k, v in self.timing.items(): f.write(f"{k}: {v:.4f}s\n")
            
            f.write("\n--- RECONSTRUCTION FIDELITY ---\n")
            for k, v in m_re.items(): f.write(f"{k}: {v:.4f}\n")
                
            f.write("\n--- WATERMARK RECOVERY ---\n")
            for k, v in m_rc.items(): f.write(f"{k}: {v:.4f}\n")
            f.write("Recovery Method: Passed watermarked image through Stage 2 'extract_watermark' CNN. Predicted bits where pred > 0.\n")
                
            f.write("\n--- LATENT ANALYSIS ---\n")
            for k, v in m_le.items(): f.write(f"{k}: {v:.4f}\n")
                
            f.write("\n--- NOISE ROBUSTNESS ---\n")
            for k, v in m_ne.items(): f.write(f"{k}: {v:.4f}\n")
                
            f.write("\n--- CHANNEL ABLATION (BER) ---\n")
            for k, v in m_ch.items(): f.write(f"{k}: {v:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--image', type=str, required=False)
    parser.add_argument('--mask', type=str, required=False)
    args = parser.parse_args()
    
    config = get_config(args.config, stage_key='stage3_extractor')
        
    img_path = args.image
    mask_path = args.mask
    
    if not img_path or not mask_path:
        print("Image or mask not provided. Fetching from dataset...")
        dataset = OxfordPetDataset(config['images_dir'], config['mask_dir'])
        if not img_path:
            img_path = dataset.img_list[0]
            print(f"Auto-selected image: {img_path}")
        if not mask_path:
            mask_path = dataset.mask_list[0]
            print(f"Auto-selected mask: {mask_path}")
            
    runner = ExperimentRunner(args.config)
    runner.run_pipeline(img_path, mask_path)
    print(f"Phase 0 Verification Complete. Results saved to {runner.base_dir}")
