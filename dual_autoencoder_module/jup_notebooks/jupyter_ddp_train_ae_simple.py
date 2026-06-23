import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms as T
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
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


# ══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.

    Trimaps use pixel values (NOT 0–255 range):
      1 = Foreground (pet)
      2 = Background
      3 = Not classified / border

    We binarise: foreground=1.0, everything else=0.0
    Images are standard JPEGs, masks are PNGs.
    """
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
        # torchvision read_image only supports jpeg/png/webp/gif — skip avif/heic
        SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        img_files  = sorted(f for f in os.listdir(self.img_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)
        mask_files = sorted(f for f in os.listdir(self.mask_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)

        # Oxford images and masks share the same base filename
        # e.g. Abyssinian_1.jpg  ↔  Abyssinian_1.png
        img_dict  = {os.path.splitext(f)[0]: f for f in img_files}
        mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}

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
            # PIL fallback for any format read_image can't handle (avif, heic, etc.)
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

        # img: uint8 [C, H, W] → float [0, 1]
        # Some Oxford images are RGBA (4ch) — strip alpha to keep 3ch
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)   # grayscale → RGB
        img = img.float().div(255.0)

        # Trimap: raw pixel values are 1/2/3 (NOT scaled by 255)
        # Foreground = 1 → 1.0,  Background/border = 2,3 → 0.0
        mask = (mask == 1).float()   # [1, H, W] binary

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask


class VOCDataset(Dataset):
    """Pascal VOC 2012 with pre-generated binary masks (kept for compatibility)."""
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
        img_dict  = {os.path.splitext(f)[0]: f
                     for f in sorted(os.listdir(self.img_dir))}
        mask_dict = {os.path.splitext(f)[0].replace('_mask', ''): f
                     for f in sorted(os.listdir(self.mask_dir))}
        for key in sorted(set(img_dict) & set(mask_dict)):
            self.img_list.append(os.path.join(self.img_dir,  img_dict[key]))
            self.mask_list.append(os.path.join(self.mask_dir, mask_dict[key]))

    def _load(self, idx):
        if self.cache_ram and idx in self.cache:
            return self.cache[idx]
        img  = read_image(self.img_list[idx])
        mask = read_image(self.mask_list[idx])
        if self.cache_ram:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[idx] = (img, mask)
        return img, mask

    def __len__(self): return self.len

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        img  = img.float().div(255.0)
        mask = (mask.float().div(255.0) > 0.5).float()
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
                  dataset_cls=OxfordPetDataset,
                  max_samples=None, split_size=0.85, cache_ram=False, max_cache_size=None):
    # max_samples handled inside dataset_cls.__init__ — no double-sampling
    dataset = dataset_cls(img_dir, mask_dir,
                          transforms=None,
                          max_samples=max_samples,
                          cache_ram=cache_ram,
                          max_cache_size=max_cache_size)

    n_train = int(split_size * len(dataset))
    n_val   = len(dataset) - n_train
    train_sub, val_sub = random_split(dataset, [n_train, n_val])

    return TransformedDataset(train_sub, train_tf), TransformedDataset(val_sub, val_tf)


def benchmark_num_workers(batch_size, img_size, candidates=[0, 2, 4, 8]):
    tf      = T.Compose([T.Resize(img_size), T.ToTensor()])
    dataset = datasets.FakeData(transform=tf)
    results = {}
    for nw in candidates:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=nw, pin_memory=True)
        t0 = time.time()
        for i, (x, _) in enumerate(loader):
            x.cuda()
            if i >= 50: break
        results[nw] = time.time() - t0
    best = min(results, key=results.get)
    print(f"  [workers] {results} → best={best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  LOSSES  — L1 + Perceptual ONLY
# ══════════════════════════════════════════════════════════════════════════════

class MaskedL1Loss(nn.Module):
    def forward(self, rec, target, mask):
        return (mask * (rec - target).abs()).sum() / (mask.sum() + 1e-8)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        # relu2_2 (shallow texture) + relu3_3 (mid-level structure)
        self.slice1 = nn.Sequential(*list(vgg.children())[:9])    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def preprocess(self, x):
        return ((x + 1.0) / 2.0 - self.mean) / self.std  # [-1,1] → VGG normalised

    def forward(self, x):
        x  = self.preprocess(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        return f1, f2


class SimpleCombinedLoss(nn.Module):
    """
    Two losses only:
      total = lambda_l1 * MaskedL1  +  lambda_perceptual * VGG(relu2_2, relu3_3)

    SSIM removed: redundant alongside L1, adds AMP instability at boundaries.
    """
    def __init__(self, device, lambda_l1=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1         = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.vgg               = VGGFeatureExtractor(device)
        self.l1                = MaskedL1Loss()
        self.mse               = nn.MSELoss()

    def forward(self, rec, target, mask):
        mask_exp = mask.expand_as(rec)   # [B,1,H,W] → [B,3,H,W]

        # ── L1 ────────────────────────────────────────────────────────────────
        loss_l1 = self.l1(rec, target, mask_exp)

        # ── Perceptual ────────────────────────────────────────────────────────
        # img_size=224 → already VGG-ready, no interpolation needed
        m_rec    = rec    * mask_exp
        m_target = target * mask_exp

        feats_rec = self.vgg(m_rec)
        with torch.inference_mode():           # FIX 6: was .detach() — still ran grad graph
            feats_tgt = self.vgg(m_target)

        loss_perc = sum(self.mse(fr, ft) for fr, ft in zip(feats_rec, feats_tgt))

        total = self.lambda_l1 * loss_l1 + self.lambda_perceptual * loss_perc

        return total, {'l1': loss_l1.item(), 'perceptual': loss_perc.item()}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL — ResNet Autoencoder with Skip Connections
#
#  ResNet backbone (strided conv down, bilinear up) with UNet-style skip
#  connections from each encoder level to the matching decoder level.
#  Skips give the decoder access to fine spatial detail it would otherwise
#  have to reconstruct from the compressed bottleneck alone — important for
#  high-quality reconstruction at 224×224.
#
#  Shape trace at 224×224, latent_channels=128:
#    Input      [B,   3, 224, 224]
#    stem  s0   [B,  64, 224, 224]
#    down1 s1   [B,  64, 112, 112]
#    down2 s2   [B, 128,  56,  56]
#    down3 s3   [B, 256,  28,  28]
#    bottleneck [B, 128,  28,  28]  ← latent z
#                                     (cat with s3 → 128+256=384)
#    up1        [B, 256,  56,  56]   (cat with s2 → 256+128=384)
#    up2        [B, 128, 112, 112]   (cat with s1 → 128+64=192)
#    up3        [B,  64, 224, 224]   (cat with s0 → 64+64=128)
#    output     [B,   3, 224, 224]
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.proj(x) + self.conv_path(x))


class ResNetAE(nn.Module):
    """ResNet encoder-bottleneck-decoder with skip connections."""
    def __init__(self, latent_channels=128):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem  = nn.Sequential(                       # s0: 224×224, ch=64
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(                       # s1: 112×112, ch=64
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64, 64),
        )
        self.down2 = nn.Sequential(                       # s2:  56×56, ch=128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128, 128),
        )
        self.down3 = nn.Sequential(                       # s3:  28×28, ch=256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256, 256),
        )

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, 1, bias=False),
            nn.InstanceNorm2d(latent_channels), nn.ReLU(inplace=True),
            ResBlock(latent_channels, latent_channels),
            ResBlock(latent_channels, latent_channels),
            ResBlock(latent_channels, latent_channels),
        )

        # ── Decoder with skip connections ─────────────────────────────────────
        # Each up block receives: upsample(prev) cat skip → conv → ResBlock
        # in_ch = prev_out + skip_ch
        self.up1 = nn.Sequential(                         # cat(z,s3):  128+256=384
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels + 256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256, 256),
        )
        self.up2 = nn.Sequential(                         # cat(up1,s2): 256+128=384
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256 + 128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128, 128),
        )
        self.up3 = nn.Sequential(                         # cat(up2,s1): 128+64=192
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + 64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64, 64),
        )
        self.output = nn.Sequential(                      # cat(up3,s0):  64+64=128
            nn.Conv2d(64 + 64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        s0 = self.stem(x)           # 224×224  ch=64
        s1 = self.down1(s0)         # 112×112  ch=64
        s2 = self.down2(s1)         #  56×56   ch=128
        s3 = self.down3(s2)         #  28×28   ch=256

        z  = self.bottleneck(s3)    #  28×28   ch=latent  ← watermark lives here

        x  = self.up1(torch.cat([z,  s3], dim=1))   # 128+256 → 56×56
        x  = self.up2(torch.cat([x,  s2], dim=1))   # 256+128 → 112×112
        x  = self.up3(torch.cat([x,  s1], dim=1))   # 128+64  → 224×224
        return self.output(torch.cat([x, s0], dim=1)), z  # 64+64 → output


class DualAutoencoder(nn.Module):
    """One AE for object stream, one for background stream. Single DDP module."""
    def __init__(self, latent_channels=128):
        super().__init__()
        self.ae_obj = ResNetAE(latent_channels=latent_channels)
        self.ae_bg  = ResNetAE(latent_channels=latent_channels)

    def forward(self, objs, bgs):
        rec_obj, z_obj = self.ae_obj(objs)
        rec_bg,  z_bg  = self.ae_bg(bgs)
        return rec_obj, rec_bg, z_obj, z_bg


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILS
# ══════════════════════════════════════════════════════════════════════════════

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
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # ── Hyperparams ───────────────────────────────────────────────────────────
    batch_size     = config['batch_size']
    max_iterations = config['max_iterations']
    img_size       = config['img_size']
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_tf = SegmentTransform(img_size, p=0.5)
    val_tf   = SegmentTransform(img_size, p=0.0)

    train_ds, val_ds = split_dataset(
        config['images_dir'], config['mask_dir'],
        train_tf, val_tf,
        dataset_cls=OxfordPetDataset,
        max_samples=config.get('max_samples'),
        split_size=config.get('split_size', 0.85),
        cache_ram=config.get('cache_ram', True),
        max_cache_size=config.get('max_cache_ram', 7500),
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DualAutoencoder(latent_channels=config.get('latent_channels', 128)).to(rank)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        p = sum(v.numel() for v in model.parameters()) / 1e6
        print(f"  [model] ResNetAE (with skips) | params: {p:.2f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = SimpleCombinedLoss(
        device=rank,
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_perceptual=config.get('lambda_perceptual', 0.1),
    ).to(rank)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iterations, eta_min=config['lr'] * 0.01)

    scaler       = GradScaler(device='cuda')
    start_epoch  = 0
    best_val     = float('inf')
    patience_ctr = 0

    if config.get('resume') and os.path.isfile(config['resume']):
        start_epoch, best_val = load_checkpoint(
            config['resume'], model, optimizer, scheduler, scaler)

    # ── Dirs ──────────────────────────────────────────────────────────────────
    ckpt_dir       = config.get('checkpoint_dir', '/kaggle/working/checkpoints')
    best_model_dir = config.get('best_model_dir',  '/kaggle/working/best_model')
    accum_steps    = config.get('accum_steps', 1)
    grad_clip      = config.get('grad_clip', 1.0)
    patience       = config.get('patience', 10)
    min_delta      = config.get('min_delta', 1e-4)
    save_every     = config.get('save_every', 5)

    if rank == 0:
        os.makedirs(ckpt_dir,       exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        print(f"  [dirs] checkpoints → {ckpt_dir}")
        print(f"  [dirs] best model  → {best_model_dir}")

    history = {
        'train_total': [], 'val_total': [], 'lr': [],
        'breakdown': {
            'l1':         {'train': [], 'val': []},
            'perceptual': {'train': [], 'val': []},
        }
    }

    if rank == 0:
        print(f"  [train] {len(train_ds)} samples | [val] {len(val_ds)} samples")
        print(f"  Starting training for {max_iterations} epochs\n")

    # ══════════════════════════════════════════════════════════════════════════
    for itr in range(start_epoch, max_iterations):

        start_time = time.time()
        train_sampler.set_epoch(itr)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        train_bd   = defaultdict(float)
        train_n    = 0

        for step, (images, masks) in enumerate(train_loader):
            images = images.cuda(rank, non_blocking=True)
            masks  = masks.cuda(rank,  non_blocking=True)

            # FIX 5: use masked streams as targets (consistent with val)
            objs = images * masks
            bgs  = images * (1.0 - masks)

            is_acc   = (step + 1) % accum_steps != 0 and (step + 1) != len(train_loader)
            sync_ctx = model.no_sync() if is_acc else nullcontext()

            with sync_ctx:
                with autocast(device_type='cuda'):
                    rec_obj, rec_bg, _, _ = model(objs, bgs)
                    loss_obj, bd_obj = criterion(rec_obj, objs, masks)
                    loss_bg,  bd_bg  = criterion(rec_bg,  bgs,  1.0 - masks)
                    loss = (loss_obj + loss_bg) / accum_steps

                # capture BEFORE backward — graph freed after
                raw = (loss_obj + loss_bg).item()
                scaler.scale(loss).backward()

            if not is_acc:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                # FIX 4: removed torch.cuda.synchronize() — added ~30% wall time per epoch
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = images.size(0)
            train_loss += raw * bs
            train_n    += bs
            for k in bd_obj:
                train_bd[k] += (bd_obj[k] + bd_bg[k]) * bs

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_bd   = defaultdict(float)
        val_n    = 0

        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.cuda(rank, non_blocking=True)
                masks  = masks.cuda(rank,  non_blocking=True)
                objs   = images * masks
                bgs    = images * (1.0 - masks)

                with autocast(device_type='cuda'):
                    rec_obj, rec_bg, _, _ = model(objs, bgs)
                    loss_obj, bd_obj = criterion(rec_obj, objs, masks)
                    loss_bg,  bd_bg  = criterion(rec_bg,  bgs,  1.0 - masks)

                bs = images.size(0)
                val_loss += (loss_obj + loss_bg).item() * bs
                val_n    += bs
                for k in bd_obj:
                    val_bd[k] += (bd_obj[k] + bd_bg[k]) * bs

        # capture LR BEFORE step → shows LR used this epoch
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ── All-reduce across GPUs ─────────────────────────────────────────────
        metrics = torch.tensor([
            train_loss, train_bd['l1'], train_bd['perceptual'], float(train_n),
            val_loss,   val_bd['l1'],   val_bd['perceptual'],   float(val_n),
        ], dtype=torch.float64, device=rank)
        dist.all_reduce(metrics)

        (t_loss, t_l1, t_perc, t_n,
         v_loss, v_l1, v_perc, v_n) = metrics.tolist()

        g_t_total = t_loss / t_n
        g_t_l1    = t_l1   / t_n
        g_t_perc  = t_perc / t_n
        g_v_total = v_loss / v_n
        g_v_l1    = v_l1   / v_n
        g_v_perc  = v_perc / v_n

        # ── Logging & Checkpointing (rank 0 only) ─────────────────────────────
        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {itr+1:3d}/{max_iterations} | LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")
            print(f"  Train | total={g_t_total:.5f}  l1={g_t_l1:.5f}  "
                  f"perceptual={g_t_perc:.5f}")
            print(f"  Val   | total={g_v_total:.5f}  l1={g_v_l1:.5f}  "
                  f"perceptual={g_v_perc:.5f}")
            print('-' * 70)

            history['train_total'].append(g_t_total)
            history['val_total'].append(g_v_total)
            history['lr'].append(current_lr)
            history['breakdown']['l1']['train'].append(g_t_l1)
            history['breakdown']['l1']['val'].append(g_v_l1)
            history['breakdown']['perceptual']['train'].append(g_t_perc)
            history['breakdown']['perceptual']['val'].append(g_v_perc)

            if g_v_total < best_val - min_delta:
                best_val     = g_v_total
                patience_ctr = 0

                # Full checkpoint (for resuming)
                save_checkpoint({
                    "epoch": itr+1, "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val, "config": config,
                }, os.path.join(ckpt_dir, "best_model.pth"))

                # Separate files in best_model_dir — directly Kaggle-downloadable
                torch.save(model.module.state_dict(),
                           os.path.join(best_model_dir, "best_weights.pth"))
                torch.save({
                    "epoch": itr+1, "best_val_loss": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "config":    config,
                }, os.path.join(best_model_dir, "best_optimizer_state.pth"))

                print(f"    [best] weights        → {best_model_dir}/best_weights.pth")
                print(f"    [best] optimizer state → {best_model_dir}/best_optimizer_state.pth")

            else:
                patience_ctr += 1
                print(f"  [early stop] no improvement {patience_ctr}/{patience}")

            if (itr + 1) % save_every == 0:
                save_checkpoint({
                    "epoch": itr+1, "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val, "config": config,
                }, os.path.join(ckpt_dir, f"epoch_{itr+1:03d}.pth"))

        # ── Early stopping broadcast ───────────────────────────────────────────
        stop_t = torch.tensor(patience_ctr, dtype=torch.int32, device=rank)
        dist.broadcast(stop_t, src=0)
        if stop_t.item() >= patience:
            if rank == 0:
                print(f"\n[!] Early stopping triggered at epoch {itr+1}")
            break

    # ── End of training ───────────────────────────────────────────────────────
    if rank == 0:
        history['best_val_loss'] = best_val
        hist_path = os.path.join(ckpt_dir, 'history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"History saved → {hist_path}")

        save_checkpoint({
            "epoch": max_iterations, "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val, "config": config,
        }, os.path.join(ckpt_dir, "final_checkpoint.pth"))

    dist.destroy_process_group()

def get_config(stage_key):
    """
    Reads the master JSON and applies command-line overrides.
    """
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

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    config = get_config(stage_key='stage2_autoencoder')
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")
    
    print(f"Loaded Stage 2 Config from master JSON.")
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)


    # config.json values take priority — setdefault won't overwrite existing keys
    # # ── Paths ─────────────────────────────────────────────────────────────────
    # config.setdefault('images_dir',     '/kaggle/input/oxford-iiit-pet/images')
    # config.setdefault('mask_dir',       '/kaggle/input/oxford-iiit-pet/annotations/trimaps')
    # config.setdefault('checkpoint_dir', '/kaggle/working/checkpoints')
    # config.setdefault('best_model_dir', '/kaggle/working/best_model')

    # # ── Data ──────────────────────────────────────────────────────────────────
    # config.setdefault('img_size',       224)   # 224 = VGG native, no interpolation needed
    # config.setdefault('max_samples',    None)  # Oxford has ~7k — use all
    # config.setdefault('split_size',     0.85)
    # config.setdefault('cache_ram',      True)

    # # ── Model ─────────────────────────────────────────────────────────────────
    # config.setdefault('latent_channels', 128)

    # # ── Training ──────────────────────────────────────────────────────────────
    # config.setdefault('batch_size',      16)   # 224px is heavier than 128px — 16 safe on T4
    # config.setdefault('max_iterations',  60)
    # config.setdefault('lr',              1e-3)
    # config.setdefault('weight_decay',    1e-4)
    # config.setdefault('grad_clip',       1.0)
    # config.setdefault('accum_steps',     2)    # effective batch = 32
    # config.setdefault('patience',        10)
    # config.setdefault('min_delta',       1e-4)
    # config.setdefault('resume',          None)
    # config.setdefault('save_every',      5)

    # # ── Losses ────────────────────────────────────────────────────────────────
    # config.setdefault('lambda_l1',          1.0)
    # config.setdefault('lambda_perceptual',  0.1)

    # config['num_workers'] = benchmark_num_workers(
    #     config['batch_size'], config['img_size'])

    # save_config(config, utils.CONFIG_PATH)