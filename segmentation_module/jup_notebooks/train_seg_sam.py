"""
SAM Segmentation Training  —  Stage 1
========================================
Fine-tunes MobileSAM's mask decoder to predict binary foreground masks.
Output masks are saved to disk and used by the DualAutoencoder (Stage 2).

Pipeline position:
    [This file] → masks on disk → train_ae_clean.py (DualAutoencoder)

This file is structurally identical to train_seg_unet.py. Every class and
function that is the same keeps the exact same name and signature so the
codebase stays consistent and explainable.

WHAT CHANGED vs train_seg_unet.py  (and why):
──────────────────────────────────────────────
1. _load() — mask loading
   OLD: mask = read_image(...)
   NEW: mask = TF.pil_to_tensor(PILImage.open(...).convert('L'))
   WHY: Oxford trimaps are palette-mode PNGs. read_image() silently converts
        them to RGB, turning raw index 1 into the RGB triple (1,1,1). After
        any resize those tiny values smear to non-integers so (mask==1)
        matches nothing → all-zero GT everywhere → all-black masks.
        PIL .convert('L') preserves the raw uint8 palette indices {1,2,3}.

2. __getitem__() — image dtype
   OLD: img = img.float().div(255.0)   → [0,1] float for UNet
   NEW: img stays uint8                → SAM's sam.preprocess() normalises
   WHY: SAM's image encoder was pretrained expecting its own normalisation
        (pixel_mean/pixel_std applied after padding to 1024). Applying
        .div(255) here and then SAM's preprocessing double-normalises and
        breaks embeddings.

3. SegmentTransform — no TF.normalize
   OLD: img = TF.normalize(img, mean, std)
   NEW: removed
   WHY: Same reason as above — SAM handles normalisation internally.

4. BCEDiceKLLoss → BCEDiceLoss  (KL removed)
   WHY: nn.KLDivLoss requires log-probabilities as input. Applying
        F.logsigmoid(logits) only gives log P(foreground), not a valid
        probability distribution. Also SAM outputs 256×256 low-res logits
        that must be resized to img_size before any pixel-level comparison
        — KL on mismatched resolution is wrong. BCE + Dice is correct.

5. iou_score — upsample before threshold
   WHY: SAM's decoder outputs 256×256. Must interpolate back to img_size
        before computing IoU against the img_size GT mask.

6. Model: SegUNet → MobileSAM (mask decoder only, encoder frozen)
7. Forward: model(images) → sam_forward(model.module, images, masks, H, rank)
   WHY: SAM needs explicit preprocessing, prompt derivation, and per-sample
        encode + decode. Wrapped in sam_forward() to keep the train loop
        as clean as the UNet version.

EVERYTHING ELSE IS IDENTICAL to train_seg_unet.py:
  Utils, load_config, save_config, OxfordPetDataset (structure),
  TransformedDataset, split_dataset, benchmark_num_workers,
  save_checkpoint, load_checkpoint, train_ddp loop structure,
  all_reduce layout, history dict format, checkpoint saves,
  early-stop broadcast, generate_masks signature, entry point.

Install:
    !pip install mobile-sam -q
"""

import os
import json
import time
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from PIL import Image as PILImage
from mobile_sam import sam_model_registry


SAM_INPUT_SIZE = 1024   # SAM encoder always works at 1024×1024
SAM_LOW_RES    = 256    # SAM mask decoder always outputs 256×256 logits


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — identical to train_seg_unet.py
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

@dataclass
class Utils:
    CONFIG_PATH: str = "/kaggle/working/config.json"
    def __post_init__(self):
        self.CONFIG = load_config(self.CONFIG_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS  — identical to train_seg_unet.py EXCEPT TF.normalize removed
#
#  TF.normalize is removed because SAM handles its own normalisation inside
#  sam.preprocess() during the forward pass. Adding it here would
#  double-normalise and corrupt the image encoder embeddings.
# ══════════════════════════════════════════════════════════════════════════════

class SegmentTransform:
    def __init__(self, img_size, p=0.5):
        self.image_size = [img_size, img_size] if isinstance(img_size, int) else img_size
        self.p          = p

    def __call__(self, img, mask):
        img  = TF.resize(img,  self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        if torch.rand(1) < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # NOTE: no TF.normalize here — SAM normalises inside sam.preprocess()
        return img, mask


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  — identical to train_seg_unet.py EXCEPT two lines in _load()
#             and one line in __getitem__()
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
        # identical to train_seg_unet.py
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

        # Image loading — identical to train_seg_unet.py
        try:
            img = read_image(self.img_list[idx])
        except RuntimeError:
            img = TF.pil_to_tensor(PILImage.open(self.img_list[idx]).convert('RGB'))

        # CHANGED: mask must always use PIL .convert('L')
        # read_image() converts palette-mode PNGs to RGB, turning raw trimap
        # index 1 into the RGB triple (1,1,1). After resize those tiny values
        # smear to non-integers so (mask==1) matches nothing → all-black masks.
        # PIL .convert('L') preserves raw uint8 palette indices {1=fg,2=bg,3=border}.
        mask = TF.pil_to_tensor(PILImage.open(self.mask_list[idx]).convert('L'))

        if self.cache_ram:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[idx] = (img, mask)
        return img, mask

    def __len__(self): return self.len

    def __getitem__(self, idx):
        img, mask = self._load(idx)

        # Channel handling — identical to train_seg_unet.py
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # CHANGED: img stays uint8 [0,255] — SAM normalises inside sam.preprocess()
        # train_seg_unet.py did img.float().div(255.0) here; SAM cannot use that.

        mask = (mask == 1).float()    # [1, H, W] binary — identical to train_seg_unet.py

        if self.transforms:
            img, mask = self.transforms(img, mask)
        return img, mask


class TransformedDataset(Dataset):
    # identical to train_seg_unet.py
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
    # identical to train_seg_unet.py
    dataset = OxfordPetDataset(img_dir, mask_dir,
                               transforms=None,
                               max_samples=max_samples,
                               cache_ram=cache_ram)
    n_train = int(split_size * len(dataset))
    n_val   = len(dataset) - n_train
    train_sub, val_sub = random_split(dataset, [n_train, n_val])
    return TransformedDataset(train_sub, train_tf), TransformedDataset(val_sub, val_tf)


def benchmark_num_workers(batch_size, img_size, candidates=[0, 2, 4, 8]):
    # identical to train_seg_unet.py
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
#  LOSS — BCE + Dice  (KL removed vs train_seg_unet.py)
#
#  KL removed because:
#    nn.KLDivLoss requires log-probabilities as input. F.logsigmoid(logits)
#    only gives log P(foreground), not a proper probability distribution.
#    Additionally SAM outputs 256×256 low-res logits — computing KL against
#    an img_size GT without resizing is wrong, and resizing then KL is noisy.
#    BCE + Dice is correct and sufficient for binary segmentation.
# ══════════════════════════════════════════════════════════════════════════════

class BCEDiceLoss(nn.Module):
    """
    BCE (bce_weight) + Dice (dice_weight).

    logits  : [B, 1, 256, 256]  SAM low-res decoder output (before sigmoid)
    targets : [B, 1, H,   W  ]  binary float {0.0, 1.0}

    targets are resized to 256×256 before loss (nearest-neighbour) to match
    SAM's low-res output. Upsampling back to img_size happens only for IoU.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Resize GT to match SAM's 256×256 low-res output
        if logits.shape != targets.shape:
            targets = F.interpolate(targets, size=logits.shape[-2:], mode='nearest')

        loss_bce  = self.bce(logits, targets)

        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice      = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss_dice = 1.0 - dice.mean()

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total, {'bce': loss_bce.item(), 'dice': loss_dice.item()}


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC — iou_score  (same purpose as train_seg_unet.py, upsample added)
# ══════════════════════════════════════════════════════════════════════════════

def iou_score(logits, targets, img_size, threshold=0.5):
    """
    Mean IoU over batch. For logging only — not used in loss.

    CHANGED vs train_seg_unet.py: logits are SAM's 256×256 low-res output,
    must be upsampled to img_size before comparing against img_size targets.
    """
    pred  = F.interpolate(logits.float(), size=(img_size, img_size),
                          mode='bilinear', align_corners=False)
    pred  = (torch.sigmoid(pred) > threshold)
    inter = (pred & targets.bool()).float().sum(dim=(1, 2, 3))
    union = (pred | targets.bool()).float().sum(dim=(1, 2, 3))
    return ((inter + 1e-8) / (union + 1e-8)).mean().item()


# ══════════════════════════════════════════════════════════════════════════════
#  SAM HELPERS  (SAM-specific — not in train_seg_unet.py)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_batch(images: torch.Tensor, sam) -> torch.Tensor:
    """
    images : [B, 3, H, W]  uint8  [0, 255]
    returns: [B, 3, 1024, 1024]  float  normalised + padded to 1024

    Calls sam.preprocess() per image — the exact same function SamPredictor
    uses internally. This normalises with pixel_mean/pixel_std then pads
    (not squashes) to 1024×1024.
    """
    return torch.stack([sam.preprocess(img.float()) for img in images])


def _get_prompts(masks: torch.Tensor, img_size: int, device: str):
    """
    Derive centroid point + tight bounding box from GT masks.
    All coordinates are scaled from img_size → SAM's 1024 internal space.

    masks : [B, 1, H, W]  binary float
    Returns points [B,1,2], labels [B,1], boxes [B,4]  — all in 1024 space
    """
    B     = masks.shape[0]
    scale = SAM_INPUT_SIZE / img_size

    points = torch.zeros(B, 1, 2, dtype=torch.float32, device=device)
    labels = torch.ones( B, 1,    dtype=torch.int,     device=device)
    boxes  = torch.zeros(B, 4,    dtype=torch.float32, device=device)

    for b in range(B):
        m = masks[b, 0]
        if m.sum() > 0:
            ys, xs          = torch.where(m > 0.5)
            points[b, 0, 0] = xs.float().mean() * scale
            points[b, 0, 1] = ys.float().mean() * scale
            boxes[b, 0]     = xs.min().float()  * scale
            boxes[b, 1]     = ys.min().float()  * scale
            boxes[b, 2]     = xs.max().float()  * scale
            boxes[b, 3]     = ys.max().float()  * scale
        else:
            half            = SAM_INPUT_SIZE / 2.0
            points[b, 0]    = torch.tensor([half, half])
            boxes[b]        = torch.tensor([0., 0.,
                                            float(SAM_INPUT_SIZE),
                                            float(SAM_INPUT_SIZE)])
    return points, labels, boxes


def sam_forward(sam, images, masks, img_size, device):
    """
    SAM-specific forward pass. Replaces the single model(images) call in
    train_seg_unet.py with the three-stage SAM pipeline:
      1. sam.preprocess()   — normalise + pad images to 1024×1024
      2. image_encoder      — frozen, wrapped in no_grad
      3. prompt_encoder     — frozen, per-sample centroid + bbox prompts
      4. mask_decoder       — trained, outputs 256×256 low-res logits

    images : [B, 3, img_size, img_size]  uint8
    masks  : [B, 1, img_size, img_size]  float binary  (for prompt derivation)
    returns: [B, 1, 256, 256]  raw logits
    """
    imgs_1024 = _preprocess_batch(images, sam)        # [B, 3, 1024, 1024]

    with torch.no_grad():                              # encoder is frozen
        emb = sam.image_encoder(imgs_1024)             # [B, C, 64, 64]

    points, labels, boxes = _get_prompts(masks, img_size, device)

    all_logits = []
    for i in range(images.shape[0]):
        sparse_e, dense_e = sam.prompt_encoder(
            points=(points[i:i+1], labels[i:i+1]),
            boxes=boxes[i:i+1],
            masks=None,
        )
        logit_i, _ = sam.mask_decoder(
            image_embeddings         = emb[i:i+1],
            image_pe                 = sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_e,
            dense_prompt_embeddings  = dense_e,
            multimask_output         = False,
        )
        all_logits.append(logit_i)    # [1, 1, 256, 256]

    return torch.cat(all_logits, dim=0)   # [B, 1, 256, 256]


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILS  — identical to train_seg_unet.py
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
#  TRAINING  — identical structure to train_seg_unet.py
#  DDP setup, accumulation, all_reduce, history, checkpointing, early-stop
#  broadcast are all the same. Only model init and forward call differ.
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
    H = img_size[0]

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

    # ── Model: MobileSAM, decoder only trained ────────────────────────────────
    sam = sam_model_registry[config.get('sam_model_type', 'vit_t')](
              checkpoint=config['sam_checkpoint'])
    sam = sam.to(rank)
    for p in sam.image_encoder.parameters():  p.requires_grad = False
    for p in sam.prompt_encoder.parameters(): p.requires_grad = False
    for p in sam.mask_decoder.parameters():   p.requires_grad = True

    model = DDP(sam, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_train = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in sam.parameters())
        print(f"  [model] MobileSAM | trainable: {n_train/1e6:.2f}M "
              f"/ total: {n_total/1e6:.2f}M  (mask decoder only)")

    criterion = BCEDiceLoss(
        bce_weight  = config.get('bce_weight',  0.5),
        dice_weight = config.get('dice_weight', 0.5),
    ).to(rank)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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

    ckpt_dir       = config.get('checkpoint_dir', '/kaggle/working/checkpoints_sam')
    best_model_dir = config.get('best_model_dir',  '/kaggle/working/best_model_sam')
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

    # History — same nested structure as train_seg_unet.py (kl removed)
    history = {
        'train_total': [], 'val_total': [], 'train_iou': [], 'val_iou': [], 'lr': [],
        'breakdown': {
            'bce':  {'train': [], 'val': []},
            'dice': {'train': [], 'val': []},
        }
    }

    if rank == 0:
        print(f"  [train] {len(train_ds)} samples | [val] {len(val_ds)} samples")
        print(f"  Starting training for {max_iterations} epochs\n")

    for itr in range(start_epoch, max_iterations):
        start_time = time.time()
        train_sampler.set_epoch(itr)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        model.module.image_encoder.eval()    # frozen — must stay in eval mode
        model.module.prompt_encoder.eval()   # frozen — must stay in eval mode
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        train_iou  = 0.0
        train_bd   = defaultdict(float)
        train_n    = 0

        for step, (images, masks) in enumerate(train_loader):
            images = images.cuda(rank, non_blocking=True)   # uint8 [B,3,H,W]
            masks  = masks.cuda(rank,  non_blocking=True)   # float [B,1,H,W]

            is_acc   = (step + 1) % accum_steps != 0 and (step + 1) != len(train_loader)
            sync_ctx = model.no_sync() if is_acc else nullcontext()

            with sync_ctx:
                with autocast(device_type='cuda'):
                    logits   = sam_forward(model.module, images, masks, H, rank)
                    loss, bd = criterion(logits, masks)
                    loss     = loss / accum_steps

                raw = loss.item() * accum_steps
                scaler.scale(loss).backward()

            if not is_acc:
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs          = images.size(0)
            train_loss += raw * bs
            train_iou  += iou_score(logits.detach(), masks, H) * bs
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
                    logits        = sam_forward(model.module, images, masks, H, rank)
                    val_step_loss, bd = criterion(logits, masks)

                bs        = images.size(0)
                val_loss += val_step_loss.item() * bs
                val_iou  += iou_score(logits, masks, H) * bs
                val_n    += bs
                for k in bd:
                    val_bd[k] += bd[k] * bs

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ── All-reduce — identical structure to train_seg_unet.py ─────────────
        metrics = torch.tensor([
            train_loss, train_iou, train_bd['bce'], train_bd['dice'], float(train_n),
            val_loss,   val_iou,   val_bd['bce'],   val_bd['dice'],   float(val_n),
        ], dtype=torch.float64, device=rank)
        dist.all_reduce(metrics)

        (t_loss, t_iou, t_bce, t_dice, t_n,
         v_loss, v_iou, v_bce, v_dice, v_n) = metrics.tolist()

        g_t_total = t_loss / t_n
        g_t_iou   = t_iou  / t_n
        g_t_bce   = t_bce  / t_n
        g_t_dice  = t_dice / t_n
        g_v_total = v_loss / v_n
        g_v_iou   = v_iou  / v_n
        g_v_bce   = v_bce  / v_n
        g_v_dice  = v_dice / v_n

        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {itr+1:3d}/{max_iterations} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            print(f"  Train | loss={g_t_total:.5f}  IoU={g_t_iou:.4f}  "
                  f"bce={g_t_bce:.5f}  dice={g_t_dice:.5f}")
            print(f"  Val   | loss={g_v_total:.5f}  IoU={g_v_iou:.4f}  "
                  f"bce={g_v_bce:.5f}  dice={g_v_dice:.5f}")
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
#  INFERENCE HELPER  — same signature as train_seg_unet.py's generate_masks()
#  Run after training to generate and save masks for all images.
#  These saved masks are the input to train_ae_clean.py (Stage 2).
# ══════════════════════════════════════════════════════════════════════════════

def generate_masks(weights_path, img_dir, out_mask_dir,
                   img_size=224, threshold=0.5, device='cuda',
                   model_type='vit_t'):
    """
    Load fine-tuned SAM and run inference on every image in img_dir.
    Saves binary masks as PNGs to out_mask_dir.

    At inference there is no GT mask, so we use a blind prompt:
      - Centre point of the image in 1024 space
      - Full-image bounding box
    multimask_output=True → pick the mask with the highest iou_pred score.

    Usage:
        generate_masks(
            weights_path = '/kaggle/working/best_model_sam/best_weights.pth',
            img_dir      = '/kaggle/input/oxford-iiit-pet/images',
            out_mask_dir = '/kaggle/working/generated_masks',
        )
    """
    from torchvision.utils import save_image

    os.makedirs(out_mask_dir, exist_ok=True)

    sam = sam_model_registry[model_type]()
    sam.load_state_dict(torch.load(weights_path, map_location='cpu'))
    sam = sam.to(device).eval()
    for p in sam.parameters(): p.requires_grad = False

    SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    img_files = sorted(f for f in os.listdir(img_dir)
                       if not f.startswith('.')
                       and os.path.splitext(f)[1].lower() in SUPPORTED)

    half   = float(SAM_INPUT_SIZE) / 2.0
    points = torch.tensor([[[half, half]]], dtype=torch.float32, device=device)
    labels = torch.tensor([[1]],            dtype=torch.int,     device=device)
    boxes  = torch.tensor([[0., 0., float(SAM_INPUT_SIZE), float(SAM_INPUT_SIZE)]],
                           dtype=torch.float32, device=device)

    print(f"  Generating masks for {len(img_files)} images → {out_mask_dir}")

    with torch.inference_mode():
        for fname in img_files:
            try:
                img = read_image(os.path.join(img_dir, fname))
            except RuntimeError:
                img = TF.pil_to_tensor(PILImage.open(os.path.join(img_dir, fname)).convert('RGB'))

            if img.shape[0] == 4: img = img[:3]
            if img.shape[0] == 1: img = img.repeat(3, 1, 1)

            img_t     = TF.resize(img, [img_size, img_size]).unsqueeze(0).to(device)
            imgs_1024 = _preprocess_batch(img_t, sam)
            emb       = sam.image_encoder(imgs_1024)

            sparse_e, dense_e = sam.prompt_encoder(
                points=(points, labels), boxes=boxes, masks=None)

            logits, iou_preds = sam.mask_decoder(
                image_embeddings         = emb,
                image_pe                 = sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_e,
                dense_prompt_embeddings  = dense_e,
                multimask_output         = True,   # pick best at inference
            )
            best_i = iou_preds[0].argmax().item()
            logit  = logits[0, best_i:best_i+1].unsqueeze(0)   # [1,1,256,256]
            logit  = F.interpolate(logit, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False)
            mask   = (torch.sigmoid(logit) > threshold).float()

            base = os.path.splitext(fname)[0]
            save_image(mask, os.path.join(out_mask_dir, f"{base}_mask.png"))

    print("  Done.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT  — identical pattern to train_seg_unet.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    utils  = Utils()
    config = utils.CONFIG

    # config.setdefault('images_dir',     '/kaggle/input/oxford-iiit-pet/images')
    # config.setdefault('mask_dir',       '/kaggle/input/oxford-iiit-pet/annotations/trimaps')
    # config.setdefault('checkpoint_dir', '/kaggle/working/checkpoints_sam')
    # config.setdefault('best_model_dir', '/kaggle/working/best_model_sam')

    # config.setdefault('sam_model_type', 'vit_t')
    # config.setdefault('sam_checkpoint', '/kaggle/input/models/dibyenducontroversy/mobile-sam/pytorch/default/1/mobile_sam.pt')

    # config.setdefault('img_size',    224)
    # config.setdefault('max_samples', None)
    # config.setdefault('split_size',  0.85)
    # config.setdefault('cache_ram',   True)

    # config.setdefault('batch_size',      8)
    # config.setdefault('max_iterations',  40)
    # config.setdefault('lr',              5e-5)
    # config.setdefault('weight_decay',    1e-4)
    # config.setdefault('grad_clip',       1.0)
    # config.setdefault('accum_steps',     4)
    # config.setdefault('patience',        10)
    # config.setdefault('min_delta',       1e-4)
    # config.setdefault('resume',          None)
    # config.setdefault('save_every',      5)

    # config.setdefault('bce_weight',  0.5)
    # config.setdefault('dice_weight', 0.5)

    # config['num_workers'] = benchmark_num_workers(
    #     config['batch_size'], config['img_size'])

    # save_config(config, utils.CONFIG_PATH)

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")

    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)

    # ── After training: generate masks for the full dataset ───────────────────
    # Uncomment and run in a separate cell after training completes:
    #
    # generate_masks(
    #     weights_path = '/kaggle/working/best_model_sam/best_weights.pth',
    #     img_dir      = '/kaggle/input/oxford-iiit-pet/images',
    #     out_mask_dir = '/kaggle/working/generated_masks',
    # )