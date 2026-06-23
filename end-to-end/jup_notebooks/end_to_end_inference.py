"""
ADSA Inference Engine  —  All Stages
======================================
Self-contained inference for the full Adversarial Dual-Stream
Autoencoder (ADSA) watermarking pipeline.

Supported operations:
    embed   — inject invisible spread-spectrum watermark into an image
    verify  — extract watermark maps and score them against known patterns
    segment — run segmentation only and return/save the binary mask

All architectures are defined inline so this file has no imports from
any training script.

Pipeline stages used at inference time:
    Stage 1  SegUNet            → binary foreground mask
    Stage 2b DualAutoencoder    → watermark injection via SSInj
    Stage 3/4 HeavyWatermarkDecoder → watermark extraction

Usage (CLI):
    python inference.py --config inference_config.json --mode embed   --input img.jpg --output wm_img.png
    python inference.py --config inference_config.json --mode verify  --input wm_img.png
    python inference.py --config inference_config.json --mode segment --input img.jpg  --output mask.png
    python inference.py --config inference_config.json --mode embed   --input_dir images/ --output_dir wm_images/

Usage (API):
    from inference import ADSAPipeline
    pipeline = ADSAPipeline.from_config("inference_config.json")
    result   = pipeline.embed("cat.jpg", "cat_wm.png")
    scores   = pipeline.verify("cat_wm.png")
"""

import os
import json
import math
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — SEGMENTATION  (SegUNet)
# ══════════════════════════════════════════════════════════════════════════════    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False), nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
    def forward(self, x, skip): return self.conv(torch.cat([self.up(x), skip], dim=1))

class SegUNet(nn.Module):
    """
    Outputs raw logits [B, 1, H, W].
    Apply sigmoid + threshold at inference.
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
        self.head       = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x); s2 = self.enc2(s1); s3 = self.enc3(s2); s4 = self.enc4(s3)
        z  = self.bottleneck(s4)
        x  = self.dec4(z, s4); x = self.dec3(x, s3); x = self.dec2(x, s2); x = self.dec1(x, s1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2b — DUAL AUTOENCODER  (DAE)
#
#  Key names match Stage 4 saved weights (ae_o.enc.*, ae_b.enc.*).
#  Stage 2b weights (ae_o.ae_obj.stem.*, ...) differ — use strict=False
#  and they map via the partial overlap that still works.
# ══════════════════════════════════════════════════════════════════════════════

class RB(nn.Module):
    def __init__(self, c, oc):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(c,  oc, 3, 1, 1, bias=False), nn.InstanceNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.InstanceNorm2d(oc),
        )
        self.p = nn.Sequential(nn.Conv2d(c, oc, 1, bias=False), nn.InstanceNorm2d(oc)) if c != oc else nn.Identity()
    def forward(self, x): return F.relu(self.p(x) + self.c(x))

class ResAE(nn.Module):
    def __init__(self, lat=256):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3,   64,  3, 1, 1, bias=False), nn.InstanceNorm2d(64),  nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64,  64,  4, 2, 1, bias=False), nn.InstanceNorm2d(64),  nn.ReLU(True), RB(64,  64)),
            nn.Sequential(nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True), RB(128, 128)),
            nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True), RB(256, 256)),
        ])
        self.bot = nn.Sequential(
            nn.Conv2d(256, lat, 1, bias=False), nn.InstanceNorm2d(lat), nn.ReLU(True),
            RB(lat, lat), RB(lat, lat), RB(lat, lat),
        )
        self.dec = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(lat+256, 256, 3, 1, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True), RB(256, 256)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256+128, 128, 3, 1, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True), RB(128, 128)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128+64,   64, 3, 1, 1, bias=False), nn.InstanceNorm2d(64),  nn.ReLU(True), RB(64,  64)),
        ])
        self.out = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())

    def encode(self, x):
        skips, curr = [], x
        for b in self.enc: curr = b(curr); skips.append(curr)
        return self.bot(curr), skips

    def decode(self, z, skips):
        curr = z
        for i, b in enumerate(self.dec): curr = b(torch.cat([curr, skips[-(i+1)]], dim=1))
        return self.out(torch.cat([curr, skips[0]], dim=1))

class DAE(nn.Module):
    """DualAutoencoder — one stream for foreground, one for background."""
    def __init__(self, lat=256):
        super().__init__()
        self.ae_o = ResAE(lat)
        self.ae_b = ResAE(lat)

    def forward_watermarked(self, o, b, inj, ao, ab, emo=None, emb=None):
        zo, sko = self.ae_o.encode(o)
        zb, skb = self.ae_b.encode(b)
        zwo, zwb = inj(zo, zb, ao, ab, emo, emb)
        return self.ae_o.decode(zwo, sko), self.ae_b.decode(zwb, skb)


# ══════════════════════════════════════════════════════════════════════════════
#  WATERMARK PRIMITIVES  (SSInj, EdgeMask, watermark generator)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, secret_key):
    hash_hex = hashlib.sha512(secret_key.encode('utf-8')).hexdigest()
    x0 = int(hash_hex[:64], 16) / (16**64)
    y0 = int(hash_hex[64:], 16) / (16**64)
    if x0 in (0, 0.5, 1): x0 = 0.12345
    if y0 in (0, 0.5, 1): y0 = 0.67890

    N = latent_channels * latent_h * latent_w
    r, mu, T_thresh = 3.99, 0.99, 1.0
    x, y = x0, y0
    wf = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        x = r * x * (1.0 - x)
        y = mu * math.sin(math.pi * y)
        wf[i] = 1.0 if (x + y) > T_thresh else -1.0
    return wf.view(1, latent_channels, latent_h, latent_w)

class SSInj(nn.Module):
    """Spread-spectrum watermark injector. Buffers are deterministic from seeds."""
    def __init__(self, c, h, w, w1_seed="object", w2_seed="background"):
        super().__init__()
        self.register_buffer('w1', _generate_hybrid_chaotic_watermark(c, h, w, w1_seed))
        self.register_buffer('w2', _generate_hybrid_chaotic_watermark(c, h, w, w2_seed))

    def forward(self, zo, zb, ao, ab, emo=None, emb=None):
        so = ao * self.w1.expand_as(zo)
        sb = ab * self.w2.expand_as(zb)
        if emo is not None: so = so * (emo + 0.05).clamp(max=1.0)
        if emb is not None: sb = sb * (emb + 0.05).clamp(max=1.0)
        return zo + so, zb + sb

class EdgeMask(nn.Module):
    """Sobel edge map downsampled to latent resolution (28×28)."""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.f = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        self.f.weight.data[0, 0] = sx
        self.f.weight.data[1, 0] = sy
        self.f.weight.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        e    = self.f(gray)
        mag  = torch.sqrt(e[:,0:1]**2 + e[:,1:2]**2 + 1e-6)
        return F.adaptive_max_pool2d((mag > 0.5).float(), (28, 28))


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3/4 — WATERMARK EXTRACTOR  (HeavyWatermarkDecoder)
# ══════════════════════════════════════════════════════════════════════════════

class DecoderResBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.GroupNorm(num_groups, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.GroupNorm(num_groups, channels),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))

class HeavyWatermarkDecoder(nn.Module):
    """
    Input  : (img [B,3,224,224], mask [B,1,224,224])
    Output : watermark_maps [B,2,28,28], deep_features [B,256,28,28]
    """
    def __init__(self, num_groups=8, num_res_blocks=6):
        super().__init__()
        self.layer1    = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 64), nn.GELU())
        self.layer2    = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 128), nn.GELU())
        self.layer3    = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.GroupNorm(num_groups, 256), nn.GELU())
        self.processor = nn.Sequential(*[DecoderResBlock(256, num_groups) for _ in range(num_res_blocks)])
        self.out_conv  = nn.Conv2d(256, 2, 3, 1, 1)

    def forward(self, x, mask):
        x_in          = torch.cat([x, mask], dim=1)
        x1            = self.layer1(x_in)
        x2            = self.layer2(x1)
        features      = self.layer3(x2)
        deep_features = self.processor(features)
        return self.out_conv(deep_features), deep_features


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_state(sd):
    """Strip 'module.' prefix left over from DDP checkpoints."""
    return OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())

def _load_weights(model, path, key=None, strict=True):
    raw = torch.load(path, map_location='cpu')
    sd  = raw.get(key, raw) if key else raw
    if not isinstance(sd, dict):
        raise ValueError(f"Expected a state dict at key '{key}' in {path}")
    model.load_state_dict(_clean_state(sd), strict=strict)
    print(f"  [loaded] {model.__class__.__name__} ← {path}")

def _pil_to_tensor(path, img_size):
    """Load an image from disk → normalised [1,3,H,W] tensor in [-1,1]."""
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    t   = TF.to_tensor(img)          # [3, H, W] in [0, 1]
    t   = TF.normalize(t, [0.5]*3, [0.5]*3)  # → [-1, 1]
    return t.unsqueeze(0)            # [1, 3, H, W]

def _tensor_to_pil(t):
    """Convert [1,3,H,W] tensor in [-1,1] back to a PIL image."""
    t = t.squeeze(0).clamp(-1, 1)   # [3, H, W]
    t = (t + 1.0) / 2.0             # [0, 1]
    return TF.to_pil_image(t.cpu())

def _psnr(original, reconstructed):
    """PSNR between two [1,3,H,W] tensors in [-1,1]."""
    mse = F.mse_loss(reconstructed, original).item()
    return 10.0 * math.log10(4.0 / (mse + 1e-12))  # max value is 2 in [-1,1] space

def _cosine_score(pred, target):
    return F.cosine_similarity(pred.flatten(1), target.flatten(1)).mean().item()

def _normalized_corr(a, b):
    a = a.flatten()
    b = b.flatten()

    a = a - a.mean()
    b = b - b.mean()

    return (a*b).sum() / (
        torch.sqrt((a*a).sum()) *
        torch.sqrt((b*b).sum()) + 1e-8
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ADSAPipeline:
    """
    Main inference interface for the ADSA watermarking pipeline.

    Parameters
    ----------
    config : dict
        Must contain:
            seg_weights   — path to SegUNet best_weights.pth
            ae_weights    — path to Stage 4 ae_best_weights.pth
            ext_weights   — path to Stage 4 ext_best_weights.pth
        Optional (with defaults shown):
            img_size        : 224
            latent_channels : 256
            latent_h        : 28   (img_size // 8)
            latent_w        : 28   (img_size // 8)
            w1_seed         : "object"
            w2_seed         : "background"
            alpha_obj       : 0.5
            alpha_bg        : 3.5
            seg_threshold   : 0.5
            soft_mask_k     : 15      (gaussian blur kernel for soft mask)
            soft_mask_sigma : 5.0
            device          : "cuda" if available, else "cpu"
    """

    def __init__(self, config: dict):
        req_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if 'cuda' in req_device and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(req_device)

        self.img_size       = config.get('img_size',        224)
        self.lat_ch         = config.get('latent_channels', 256)
        self.lat_h          = config.get('latent_h', self.img_size // 8)
        self.lat_w          = config.get('latent_w', self.img_size // 8)
        self.w1_seed        = config.get('w1_seed',    'object')
        self.w2_seed        = config.get('w2_seed',    'background')
        self.alpha_obj      = config.get('alpha_obj',  0.5)
        self.alpha_bg       = config.get('alpha_bg',   3.5)
        self.seg_thresh     = config.get('seg_threshold', 0.5)
        self.soft_k         = config.get('soft_mask_k',     15)
        self.soft_sigma     = config.get('soft_mask_sigma', 5.0)

        # ── Segmentor ─────────────────────────────────────────────────────────
        self.seg = SegUNet().to(self.device)
        _load_weights(self.seg, config['seg_weights'], key='model', strict=True)
        self.seg.eval()

        # ── Dual Autoencoder ──────────────────────────────────────────────────
        self.ae = DAE(lat=self.lat_ch).to(self.device)
        raw = torch.load(config['ae_weights'], map_location='cpu')
        # Stage 4 checkpoint saves bare state dict; Stage 2b wraps it under 'model'
        ae_sd = raw.get('model', raw)
        self.ae.load_state_dict(_clean_state(ae_sd), strict=False)
        print(f"  [loaded] DAE ← {config['ae_weights']}")
        self.ae.eval()

        # ── Watermark injector (deterministic, no weights to load) ────────────
        self.inj = SSInj(self.lat_ch, self.lat_h, self.lat_w,
                         self.w1_seed, self.w2_seed).to(self.device)
        self.inj.eval()

        # ── Edge masker ───────────────────────────────────────────────────────
        self.edge = EdgeMask().to(self.device)
        self.edge.eval()

        # ── Extractor ─────────────────────────────────────────────────────────
        self.ext = HeavyWatermarkDecoder(num_res_blocks=6).to(self.device)
        raw = torch.load(config['ext_weights'], map_location='cpu')
        ext_sd = raw.get('model', raw)
        self.ext.load_state_dict(_clean_state(ext_sd), strict=True)
        print(f"  [loaded] HeavyWatermarkDecoder ← {config['ext_weights']}")
        self.ext.eval()

        # ── Precompute target watermark maps (channel-mean, 28×28) ────────────
        # These are compared against extractor output during verify()
        with torch.no_grad():
            self._tw1 = self.inj.w1.clone()   # [1,256,28,28]
            self._tw2 = self.inj.w2.clone()

        print(f"  [pipeline] ready on {self.device}")

    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> 'ADSAPipeline':
        """Build a pipeline from a JSON config file (same format as training)."""
        with open(config_path, 'r') as f:
            master = json.load(f)
        cfg = {}
        cfg.update(master.get('paths',  {}))
        cfg.update(master.get('shared', {}))
        cfg.update(master.get('inference', {}))
        return cls(cfg)

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _preprocess(self, img_path: str):
        """Returns normalised tensor [1,3,H,W] on self.device."""
        return _pil_to_tensor(img_path, self.img_size).to(self.device)

    @torch.no_grad()
    def _get_soft_mask(self, mask_hard: torch.Tensor) -> torch.Tensor:
        """Gaussian-blur a binary mask to produce a soft compositing weight."""
        return TF.gaussian_blur(mask_hard,
                                [self.soft_k, self.soft_k],
                                [self.soft_sigma, self.soft_sigma])

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def segment(self, img_path: str,
                save_path: Optional[str] = None) -> torch.Tensor:
        """
        Run the segmentor on a single image.

        Returns
        -------
        mask : [1, 1, H, W] hard binary mask on self.device  (1 = foreground)

        Optionally saves the mask as a greyscale PNG if save_path is given.
        """
        img  = self._preprocess(img_path)
        logit = self.seg(img)                         # [1, 1, H, W] logits
        mask  = (torch.sigmoid(logit) > self.seg_thresh).float()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            mask_pil = TF.to_pil_image(mask.squeeze(0).cpu())
            mask_pil.save(save_path)
            print(f"  [segment] mask saved → {save_path}")

        return mask

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, img_path: str, output_path: str) -> dict:
        """
        Watermark an image and save the result.

        Returns
        -------
        dict with keys:
            psnr         : float  — quality metric (higher is better, >38 dB is imperceptible)
            corr_obj     : float  — self-check: obj watermark correlation in embedded image
            corr_bg      : float  — self-check: bg  watermark correlation in embedded image
            output_path  : str
        """
        img     = self._preprocess(img_path)                # [1, 3, H, W]
        mask    = self.segment(img_path)                    # [1, 1, H, W] hard
        soft_m  = self._get_soft_mask(mask)                 # [1, 1, H, W] soft

        objs = img * mask
        bgs  = img * (1.0 - mask)

        em_o = self.edge(objs)
        em_b = self.edge(bgs)

        # Inject watermark through the dual AE
        wm_o, wm_b = self.ae.forward_watermarked(
            objs,
            bgs,
            self.inj,
            self.alpha_obj,
            self.alpha_bg,
            em_o,
            em_b,
        )

        # Composite using soft mask so the seam is invisible
        wm_img = wm_o * soft_m + wm_b * (1.0 - soft_m)    # [1, 3, H, W]

        # Self-check: run extractor immediately to confirm signal survived
        pwm, _ = self.ext(wm_img, soft_m)                  # [1, 2, 28, 28]
        
        print("\n[PWM DEBUG]")
        print("shape :", pwm.shape)

        print("obj mean :", pwm[:,0:1].mean().item())
        print("obj std  :", pwm[:,0:1].std().item())

        print("bg mean  :", pwm[:,1:2].mean().item())
        print("bg std   :", pwm[:,1:2].std().item())

        print("obj min/max :", pwm[:,0:1].min().item(),
                            pwm[:,0:1].max().item())

        print("bg min/max  :", pwm[:,1:2].min().item(),
                            pwm[:,1:2].max().item())
        
        m28    = F.interpolate(soft_m, size=(28, 28), mode='area')
        tw1 = self._tw1.mean(1, keepdim=True)
        tw2 = self._tw2.mean(1, keepdim=True)

        co = _normalized_corr(
            pwm[:,0:1],
            tw1
        ).item()

        cb = _normalized_corr(
            pwm[:,1:2],
            tw2
        ).item()

        psnr = _psnr(img, wm_img)

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _tensor_to_pil(wm_img).save(output_path)

        result = {
            'psnr':        psnr,
            'corr_obj':    co,
            'corr_bg':     cb,
            'output_path': output_path,
        }
        print(f"  [embed] PSNR={psnr:.2f} dB  corr_obj={co:+.4f}  corr_bg={cb:+.4f}  → {output_path}")
        return result

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def verify(self, img_path: str,
               mask: Optional[torch.Tensor] = None) -> dict:
        """
        Extract and score the watermark in an image.

        Parameters
        ----------
        img_path : path to the (possibly watermarked) image
        mask     : optional pre-computed hard binary mask [1,1,H,W].
                   If None the segmentor runs automatically.

        Returns
        -------
        dict with keys:
            corr_obj  : cosine similarity for the foreground watermark  (-1 … 1)
            corr_bg   : cosine similarity for the background watermark  (-1 … 1)
            verdict   : "WATERMARKED"  if both correlations > threshold
                        "SUSPECT"      if one correlation > threshold
                        "CLEAN"        otherwise
            threshold : float — the decision boundary used
        """
        img    = self._preprocess(img_path)
        if mask is None:
            mask = self.segment(img_path)
        soft_m = self._get_soft_mask(mask)

        pwm, _ = self.ext(img, soft_m)   # [1, 2, 28, 28]

        tw1 = self._tw1.mean(1, keepdim=True)
        tw2 = self._tw2.mean(1, keepdim=True)

        co = _normalized_corr(
            pwm[:,0:1],
            tw1
        ).item()

        cb = _normalized_corr(
            pwm[:,1:2],
            tw2
).item()

        # Heuristic threshold — tune this on your validation set
        thresh  = 0.3
        n_above = (co > thresh) + (cb > thresh)
        if   n_above == 2: verdict = "WATERMARKED"
        elif n_above == 1: verdict = "SUSPECT"
        else:              verdict = "CLEAN"

        result = {
            'corr_obj':  co,
            'corr_bg':   cb,
            'verdict':   verdict,
            'threshold': thresh,
        }
        print(f"  [verify] corr_obj={co:+.4f}  corr_bg={cb:+.4f}  → {verdict}")
        return result

    # ──────────────────────────────────────────────────────────────────────────

    def embed_dir(self, input_dir: str, output_dir: str,
                  extensions=('.jpg', '.jpeg', '.png')) -> list:
        """Batch-embed an entire directory. Returns list of per-image result dicts."""
        paths   = [p for p in sorted(Path(input_dir).iterdir()) if p.suffix.lower() in extensions]
        results = []
        for p in paths:
            out = os.path.join(output_dir, p.name)
            results.append(self.embed(str(p), out))
        return results

    def verify_dir(self, input_dir: str,
                   extensions=('.jpg', '.jpeg', '.png')) -> list:
        """Batch-verify an entire directory. Returns list of per-image result dicts."""
        paths   = [p for p in sorted(Path(input_dir).iterdir()) if p.suffix.lower() in extensions]
        results = []
        for p in paths:
            r = self.verify(str(p))
            r['path'] = str(p)
            results.append(r)
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="ADSA Inference Engine")
    p.add_argument('--config',     required=True, help='Path to inference_config.json')
    p.add_argument('--mode',       required=True, choices=['embed', 'verify', 'segment'],
                   help='Operation to run')
    p.add_argument('--input',      default=None,  help='Single input image path')
    p.add_argument('--output',     default=None,  help='Output path for embed / segment')
    p.add_argument('--input_dir',  default=None,  help='Batch: directory of input images')
    p.add_argument('--output_dir', default=None,  help='Batch: directory for output images (embed)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    pipeline = ADSAPipeline.from_config(args.config)

#     {
#   "paths": {
#             "seg_weights":  "/kaggle/input/models/mrheavenly/stage-1/pytorch/default/1/best_weights.pth",
#             "ae_weights":   "/kaggle/input/models/mrheavenly/stage-4/pytorch/default/1/ae_best_weights.pth",
#             "ext_weights":  "/kaggle/input/models/mrheavenly/stage-4/pytorch/default/1/ext_best_weights.pth",
#             "output_dir":   "/kaggle/working/inference_output"
#         },

#     "shared": {
#             "img_size":        224,
#             "latent_channels": 256,
#             "w1_seed":         "object",
#             "w2_seed":         "background",
#             "device":          "cuda"
#         },

        # "inference": {
        #     "seg_weights": "/kaggle/working/best_model_unet/best_weights.pth",

        #     "ae_weights": "/kaggle/working/best_model_gan/ae_best_weights.pth",

        #     "ext_weights": "/kaggle/working/best_model_gan/ext_best_weights.pth",

        #     "device": "cuda:0",

        #     "alpha_obj": 0.1,
        #     "alpha_bg": 0.2,

        #     "seg_threshold": 0.5,

        #     "soft_mask_k": 15,
        #     "soft_mask_sigma": 5.0
        # }
#     }

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.input_dir:
        if args.mode == 'embed':
            if not args.output_dir:
                raise ValueError("--output_dir required for batch embed")
            results = pipeline.embed_dir(args.input_dir, args.output_dir)
            print(f"\n[done] {len(results)} images watermarked → {args.output_dir}")

        elif args.mode == 'verify':
            results = pipeline.verify_dir(args.input_dir)
            watermarked = sum(r['verdict'] == 'WATERMARKED' for r in results)
            print(f"\n[done] {len(results)} images verified — {watermarked} WATERMARKED")

        else:
            raise ValueError("--mode segment does not support --input_dir")

    # ── Single image mode ─────────────────────────────────────────────────────
    elif args.input:
        if args.mode == 'embed':
            if not args.output:
                raise ValueError("--output required for embed")
            pipeline.embed(args.input, args.output)

        elif args.mode == 'verify':
            pipeline.verify(args.input)

        elif args.mode == 'segment':
            out = args.output or 'mask.png'
            pipeline.segment(args.input, save_path=out)

    else:
        raise ValueError("Provide either --input or --input_dir")