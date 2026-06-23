"""
UNet Segmentation Training  —  Stage 1
========================================
Trains a UNet to predict binary foreground masks from images.
Output masks are saved to disk and used by the DualAutoencoder (Stage 2)
to split images into object and background streams.

Pipeline position:
    [This file] → masks on disk → train_ae_clean.py (DualAutoencoder)

Usage:
    1. Run this to train the segmentor.
    2. Run inference separately to generate and save masks for all images.
    3. Feed saved masks into train_ae_clean.py.

Loss: BCE (0.4) + Dice (0.4) + KL Divergence (0.2)
    - BCE  : per-pixel binary cross-entropy
    - Dice : overlap-based, handles class imbalance (background >> foreground)
    - KL   : distribution-level penalty — pushes predicted probabilities to be
             well-calibrated, not just locally correct
"""

import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
import torch.multiprocessing as mp
from contextlib import nullcontext
from torchvision.io import read_image
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split

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
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.
    Trimaps: 1=Foreground, 2=Background, 3=Border → binarised to 0/1.
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
        SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        img_files  = sorted(f for f in os.listdir(self.img_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)
        mask_files = sorted(f for f in os.listdir(self.mask_dir)
                            if not f.startswith('.')
                            and os.path.splitext(f)[1].lower() in SUPPORTED)
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
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img  = img.float().div(255.0)
        mask = (mask == 1).float()    # [1, H, W] binary
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
    dataset = OxfordPetDataset(img_dir, mask_dir,
                               transforms=None,
                               max_samples=max_samples,
                               cache_ram=cache_ram)
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
#  LOSS — BCE + Dice + KL
# ══════════════════════════════════════════════════════════════════════════════

class BCEDiceKLLoss(nn.Module):
    """
    BCE (0.4) + Dice (0.4) + KL Divergence (0.2)

    BCE  — standard per-pixel binary cross-entropy on logits
    Dice — overlap loss, robust to foreground/background class imbalance
    KL   — distribution-level loss; penalises miscalibrated confidence.
           A model that is wrong AND overconfident gets hit harder than one
           that is wrong but uncertain. Improves mask boundary quality.
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.4, kl_weight=0.2, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.kl_weight   = kl_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()
        # 'mean' divides by B×H×W — 'batchmean' only divides by B,
        # which causes KL to explode by a factor of H×W (~50k at 224px)
        self.kl          = nn.KLDivLoss(reduction='mean')

    def forward(self, logits, targets):
        """
        logits  : [B, 1, H, W]  raw unactivated predictions
        targets : [B, 1, H, W]  binary float {0.0, 1.0}
        """
        # ── BCE ───────────────────────────────────────────────────────────────
        loss_bce = self.bce(logits, targets)

        # ── Dice ──────────────────────────────────────────────────────────────
        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice      = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss_dice = 1.0 - dice.mean()

        # ── KL Divergence ─────────────────────────────────────────────────────
        # KLDivLoss(input, target): input must be log-probs, target must be probs
        # log_sigmoid(logits) = log(P(foreground)) directly
        log_probs = F.logsigmoid(logits)
        loss_kl   = self.kl(log_probs, targets.clamp(0.0, 1.0))

        total = (self.bce_weight  * loss_bce
               + self.dice_weight * loss_dice
               + self.kl_weight   * loss_kl)

        return total, {
            'bce':  loss_bce.item(),
            'dice': loss_dice.item(),
            'kl':   loss_kl.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL — Segmentation UNet
#
#  Single UNet: image in → binary mask logit out [B, 1, H, W]
#  No reconstruction head. No dual stream. Pure segmentation.
#
#  Shape trace at 224×224:
#    Input      [B,   3, 224, 224]
#    enc1  s1   [B,  64, 224, 224]
#    enc2  s2   [B, 128, 112, 112]
#    enc3  s3   [B, 256,  56,  56]
#    enc4  s4   [B, 512,  28,  28]
#    bottle     [B, 512,  14,  14]
#    dec4       [B, 256,  28,  28]  cat(up, s4)
#    dec3       [B, 128,  56,  56]  cat(up, s3)
#    dec2       [B,  64, 112, 112]  cat(up, s2)
#    dec1       [B,  64, 224, 224]  cat(up, s1)
#    output     [B,   1, 224, 224]  logit (no sigmoid — applied in loss)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def iou_score(logits, targets, threshold=0.5):
    """Mean IoU over batch. For logging only — not used in loss."""
    probs  = torch.sigmoid(logits) > threshold
    inter  = (probs & targets.bool()).float().sum(dim=(1, 2, 3))
    union  = (probs | targets.bool()).float().sum(dim=(1, 2, 3))
    iou    = (inter + 1e-8) / (union + 1e-8)
    return iou.mean().item()


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

    model = SegUNet().to(rank)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        p = sum(v.numel() for v in model.parameters()) / 1e6
        print(f"  [model] SegUNet | params: {p:.2f}M")

    criterion = BCEDiceKLLoss(
        bce_weight=config.get('bce_weight',  0.4),
        dice_weight=config.get('dice_weight', 0.4),
        kl_weight=config.get('kl_weight',    0.2),
    ).to(rank)

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

    ckpt_dir       = config.get('checkpoint_dir', '/kaggle/working/checkpoints_unet')
    best_model_dir = config.get('best_model_dir',  '/kaggle/working/best_model_unet')
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
        'train_total': [], 'val_total': [], 'train_iou': [], 'val_iou': [], 'lr': [],
        'breakdown': {
            'bce':  {'train': [], 'val': []},
            'dice': {'train': [], 'val': []},
            'kl':   {'train': [], 'val': []},
        }
    }

    if rank == 0:
        print(f"  [train] {len(train_ds)} samples | [val] {len(val_ds)} samples")
        print(f"  Starting training for {max_iterations} epochs\n")

    for itr in range(start_epoch, max_iterations):
        start_time = time.time()
        train_sampler.set_epoch(itr)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        train_iou  = 0.0
        train_bd   = defaultdict(float)
        train_n    = 0

        for step, (images, masks) in enumerate(train_loader):
            images = images.cuda(rank, non_blocking=True)
            masks  = masks.cuda(rank,  non_blocking=True)

            is_acc   = (step + 1) % accum_steps != 0 and (step + 1) != len(train_loader)
            sync_ctx = model.no_sync() if is_acc else nullcontext()

            with sync_ctx:
                with autocast(device_type='cuda'):
                    logits = model(images)              # [B, 1, H, W]
                    loss, bd = criterion(logits, masks)
                    loss = loss / accum_steps

                raw = loss.item() * accum_steps
                scaler.scale(loss).backward()

            if not is_acc:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = images.size(0)
            train_loss += raw * bs
            train_iou  += iou_score(logits.detach(), masks) * bs
            train_n    += bs
            for k in bd:
                train_bd[k] += bd[k] * bs

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        val_bd   = defaultdict(float)
        val_n    = 0

        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.cuda(rank, non_blocking=True)
                masks  = masks.cuda(rank,  non_blocking=True)

                with autocast(device_type='cuda'):
                    logits = model(images)
                    val_step_loss, bd = criterion(logits, masks)

                bs = images.size(0)
                val_loss += val_step_loss.item() * bs
                val_iou  += iou_score(logits, masks) * bs
                val_n    += bs
                for k in bd:
                    val_bd[k] += bd[k] * bs

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ── All-reduce ────────────────────────────────────────────────────────
        metrics = torch.tensor([
            train_loss, train_iou, train_bd['bce'], train_bd['dice'], train_bd['kl'], float(train_n),
            val_loss,   val_iou,   val_bd['bce'],   val_bd['dice'],   val_bd['kl'],   float(val_n),
        ], dtype=torch.float64, device=rank)
        dist.all_reduce(metrics)

        (t_loss, t_iou, t_bce, t_dice, t_kl, t_n,
         v_loss, v_iou, v_bce, v_dice, v_kl, v_n) = metrics.tolist()

        g_t_total = t_loss / t_n
        g_t_iou   = t_iou  / t_n
        g_t_bce   = t_bce  / t_n
        g_t_dice  = t_dice / t_n
        g_t_kl    = t_kl   / t_n
        g_v_total = v_loss / v_n
        g_v_iou   = v_iou  / v_n
        g_v_bce   = v_bce  / v_n
        g_v_dice  = v_dice / v_n
        g_v_kl    = v_kl   / v_n

        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {itr+1:3d}/{max_iterations} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            print(f"  Train | loss={g_t_total:.5f}  IoU={g_t_iou:.4f}  "
                  f"bce={g_t_bce:.5f}  dice={g_t_dice:.5f}  kl={g_t_kl:.5f}")
            print(f"  Val   | loss={g_v_total:.5f}  IoU={g_v_iou:.4f}  "
                  f"bce={g_v_bce:.5f}  dice={g_v_dice:.5f}  kl={g_v_kl:.5f}")
            print('-' * 70)

            history['train_total'].append(g_t_total)
            history['val_total'].append(g_v_total)
            history['train_iou'].append(g_t_iou)
            history['val_iou'].append(g_v_iou)
            history['lr'].append(current_lr)
            history['breakdown']['bce']['train'].append(g_t_bce)
            history['breakdown']['bce']['val'].append(g_v_bce)
            history['breakdown']['dice']['train'].append(g_t_dice)
            history['breakdown']['dice']['val'].append(g_v_dice)
            history['breakdown']['kl']['train'].append(g_t_kl)
            history['breakdown']['kl']['val'].append(g_v_kl)

            if g_v_total < best_val - min_delta:
                best_val     = g_v_total
                patience_ctr = 0
                save_checkpoint({
                    "epoch": itr+1, "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val, "config": config,
                }, os.path.join(ckpt_dir, "best_model.pth"))
                torch.save(model.module.state_dict(),
                           os.path.join(best_model_dir, "best_weights.pth"))
                torch.save({
                    "epoch": itr+1, "best_val_loss": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "config":    config,
                }, os.path.join(best_model_dir, "best_optimizer_state.pth"))
                print(f"    [best] val_loss={best_val:.5f}  val_IoU={g_v_iou:.4f}")
                print(f"    [best] weights → {best_model_dir}/best_weights.pth")
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

        stop_t = torch.tensor(patience_ctr, dtype=torch.int32, device=rank)
        dist.broadcast(stop_t, src=0)
        if stop_t.item() >= patience:
            if rank == 0:
                print(f"\n[!] Early stopping triggered at epoch {itr+1}")
            break

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


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
#  Run after training to generate and save masks for all images.
#  These saved masks are the input to train_ae_clean.py (Stage 2).
# ══════════════════════════════════════════════════════════════════════════════

def generate_masks(weights_path, img_dir, out_mask_dir, img_size=224, threshold=0.5, device='cuda'):
    """
    Load trained SegUNet and run inference on every image in img_dir.
    Saves binary masks as PNGs to out_mask_dir.

    Usage:
        generate_masks(
            weights_path = '/kaggle/working/best_model_unet/best_weights.pth',
            img_dir      = '/kaggle/input/oxford-iiit-pet/images',
            out_mask_dir = '/kaggle/working/generated_masks',
        )
    """
    from torchvision.utils import save_image

    os.makedirs(out_mask_dir, exist_ok=True)

    model = SegUNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model = model.to(device).eval()

    SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    img_files = sorted(f for f in os.listdir(img_dir)
                       if not f.startswith('.')
                       and os.path.splitext(f)[1].lower() in SUPPORTED)

    tf_resize = T.Resize([img_size, img_size])
    tf_norm   = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    print(f"  Generating masks for {len(img_files)} images → {out_mask_dir}")

    with torch.inference_mode():
        for fname in img_files:
            try:
                img = read_image(os.path.join(img_dir, fname))
            except RuntimeError:
                from PIL import Image as PILImage
                img = TF.pil_to_tensor(PILImage.open(os.path.join(img_dir, fname)).convert('RGB'))

            if img.shape[0] == 4: img = img[:3]
            if img.shape[0] == 1: img = img.repeat(3, 1, 1)

            img_t = tf_norm(tf_resize(img.float().div(255.0))).unsqueeze(0).to(device)
            logit = model(img_t)                          # [1, 1, H, W]
            mask  = (torch.sigmoid(logit) > threshold).float()

            base = os.path.splitext(fname)[0]
            save_image(mask, os.path.join(out_mask_dir, f"{base}_mask.png"))

    print("  Done.")

def get_config(stage_key):
    """
    Reads the master JSON and applies command-line overrides.
    'stage_key' tells the function which section of the JSON to load (e.g., 'stage1_unet').
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

    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.resume is not None:
        cfg['resume'] = args.resume

    return cfg

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config = get_config(stage_key='stage1_unet')

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")

    print(f"Loaded Stage 1 Config from master JSON.")
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)
    

    # config.setdefault('images_dir',     '/kaggle/input/oxford-iiit-pet/images')
    # config.setdefault('mask_dir',       '/kaggle/input/oxford-iiit-pet/annotations/trimaps')
    # config.setdefault('checkpoint_dir', '/kaggle/working/checkpoints_unet')
    # config.setdefault('best_model_dir', '/kaggle/working/best_model_unet')

    # config.setdefault('img_size',    224)
    # config.setdefault('max_samples', None)
    # config.setdefault('split_size',  0.85)
    # config.setdefault('cache_ram',   True)

    # config.setdefault('batch_size',      16)
    # config.setdefault('max_iterations',  50)
    # config.setdefault('lr',              1e-3)
    # config.setdefault('weight_decay',    1e-4)
    # config.setdefault('grad_clip',       1.0)
    # config.setdefault('accum_steps',     2)    # effective batch = 32
    # config.setdefault('patience',        10)
    # config.setdefault('min_delta',       1e-4)
    # config.setdefault('resume',          None)
    # config.setdefault('save_every',      5)

    # config.setdefault('bce_weight',  0.4)
    # config.setdefault('dice_weight', 0.4)
    # config.setdefault('kl_weight',   0.2)

    # config['num_workers'] = benchmark_num_workers(
    #     config['batch_size'], config['img_size'])

    # save_config(config, utils.CONFIG_PATH)


    # ── After training: generate masks for the full dataset ───────────────────
    # Uncomment and run in a separate cell after training completes:
    #
    # generate_masks(
    #     weights_path = '/kaggle/working/best_model_unet/best_weights.pth',
    #     img_dir      = '/kaggle/input/oxford-iiit-pet/images',
    #     out_mask_dir = '/kaggle/working/generated_masks',
    # )