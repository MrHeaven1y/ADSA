"""
Stage 4: GAN Adversarial Training (Complete with all features)
AE learns to hide watermark. D learns to detect it. E learns to extract despite D.
"""

import os, json, time, datetime, torch, hashlib, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import hashlib
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from collections import OrderedDict, defaultdict
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. FIXED IMPORT: Import the exact Extractor architecture from Stage 3
import train_extractor_scratch as ex

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS & SEEDING
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42, rank=0):
    """Ensure deterministic training per rank."""
    seed += rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class SegTf:
    def __init__(self, sz, p=0.5): self.sz, self.p = [sz, sz], p
    def __call__(self, img, mask):
        img = TF.resize(img, self.sz); mask = TF.resize(mask, self.sz, interpolation=TF.InterpolationMode.NEAREST)
        if torch.rand(1) < self.p: img, mask = TF.hflip(img), TF.hflip(mask)
        return TF.normalize(img, [0.5]*3, [0.5]*3), mask

class PetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, tf=None):
        self.img_list, self.mask_list, self.tf = [], [], tf
        sup = {'.jpg', '.jpeg', '.png'}
        i_dict = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in sup}
        m_dict = {os.path.splitext(f)[0].replace('_mask', ''): f for f in os.listdir(mask_dir) if os.path.splitext(f)[1].lower() in sup}
        for k in sorted(set(i_dict) & set(m_dict)):
            self.img_list.append(os.path.join(img_dir, i_dict[k]))
            self.mask_list.append(os.path.join(mask_dir, m_dict[k]))
    def __len__(self): return len(self.img_list)
    def __getitem__(self, idx):
        img, mask = read_image(self.img_list[idx]), read_image(self.mask_list[idx])
        if img.shape[0] == 4: img = img[:3]
        elif img.shape[0] == 1: img = img.repeat(3, 1, 1)
        if mask.shape[0] > 1: mask = mask[0:1]
        img, mask = img.float() / 255.0, (mask.float() > 127.0).float()
        if self.tf: img, mask = self.tf(img, mask)
        return img, mask

class TD(Dataset):
    def __init__(self, sub, tf): self.sub, self.tf = sub, tf
    def __len__(self): return len(self.sub)
    def __getitem__(self, i): img, mask = self.sub[i]; return self.tf(img, mask)

def split_ds(cfg):
    ds = PetDataset(cfg['images_dir'], cfg['mask_dir'])
    nt = int(0.85 * len(ds))
    train, val = random_split(ds, [nt, len(ds) - nt], generator=torch.Generator().manual_seed(42))
    return TD(train, SegTf(cfg['img_size'], 0.5)), TD(val, SegTf(cfg['img_size'], 0.0))

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class PatchD(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        seq = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        for n in range(1, n_layers):
            nf_prev, nf = min(2**(n-1), 8), min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_prev, ndf*nf, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = min(2**(n_layers-1), 8), min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf*nf_prev, ndf*nf, 4, 1, 1, bias=False), nn.InstanceNorm2d(ndf*nf), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf*nf, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)
    def forward(self, x): return self.model(x)

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
        
    return watermark_flat.view(1, latent_channels, latent_h, latent_w)

class SSInj(nn.Module):
    def __init__(self, c, h, w, w1s="object", w2s="background"):
        super().__init__()
        self.register_buffer('w1', generate_hybrid_chaotic_watermark(c, h, w, w1s))
        self.register_buffer('w2', generate_hybrid_chaotic_watermark(c, h, w, w2s))
    def forward(self, zo, zb, ao, ab, emo=None, emb=None):
        so, sb = ao * self.w1.expand_as(zo), ab * self.w2.expand_as(zb)
        if emo is not None: so = so * (emo + 0.05).clamp(max=1.0)
        if emb is not None: sb = sb * (emb + 0.05).clamp(max=1.0)
        return zo + so, zb + sb

class RB(nn.Module):
    def __init__(self, c, oc):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(c, oc, 3, 1, 1, bias=False), nn.InstanceNorm2d(oc), nn.ReLU(True),
                               nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.InstanceNorm2d(oc))
        self.p = nn.Sequential(nn.Conv2d(c, oc, 1, bias=False), nn.InstanceNorm2d(oc)) if c != oc else nn.Identity()
    def forward(self, x): return F.relu(self.p(x) + self.c(x))

class ResAE(nn.Module):
    def __init__(self, lat=256):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True), RB(64, 64)),
            nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True), RB(128, 128)),
            nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True), RB(256, 256))
        ])
        self.bot = nn.Sequential(nn.Conv2d(256, lat, 1, bias=False), nn.InstanceNorm2d(lat), nn.ReLU(True),
                                 RB(lat, lat), RB(lat, lat), RB(lat, lat))
        self.dec = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(lat+256, 256, 3, 1, 1, bias=False), nn.InstanceNorm2d(256), nn.ReLU(True), RB(256, 256)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256+128, 128, 3, 1, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(True), RB(128, 128)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128+64, 64, 3, 1, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(True), RB(64, 64))
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
    def __init__(self, lat=256):
        super().__init__()
        self.ae_o, self.ae_b = ResAE(lat), ResAE(lat)
    def forward_watermarked(self, o, b, inj, ao, ab, emo, emb):
        zo, sko = self.ae_o.encode(o); zb, skb = self.ae_b.encode(b)
        zwo, zwb = inj(zo, zb, ao, ab, emo, emb)
        return self.ae_o.decode(zwo, sko), self.ae_b.decode(zwb, skb)

class EdgeMask(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.f = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        self.f.weight.data[0, 0], self.f.weight.data[1, 0] = sx, sy
        self.f.weight.requires_grad = False
    @torch.no_grad()
    def forward(self, x):
        gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        e = self.f(gray)
        sm = torch.sqrt(e[:,0:1]**2 + e[:,1:2]**2 + 1e-6)
        return F.adaptive_max_pool2d((sm > 0.5).float(), (28, 28))

class AttackLayer(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x, ep=None, wp=10): 
        if ep is not None and ep < wp: return x
        if torch.rand(1).item() < self.p:
            sigma = torch.empty(1).uniform_(0.01, 0.05).item()
            x = (x + torch.randn_like(x) * sigma).clamp(-1, 1)
        return x

class SpatialForensicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_wm_maps, target_wm_maps, pred_wm_logit, pred_clean_maps, pred_clean_logit, mask_28x28):
        
        # 1. Binary Detection Loss (Real vs Fake)
        loss_detect_wm = self.bce(pred_wm_logit, torch.ones_like(pred_wm_logit))
        loss_detect_cl = self.bce(pred_clean_logit, torch.zeros_like(pred_clean_logit))
        loss_detect = loss_detect_wm + loss_detect_cl

        # 2. Full Reconstruction Loss (No mean-squashing!)
        pred_obj, pred_bg = pred_wm_maps[:, 0:256], pred_wm_maps[:, 256:512]
        targ_obj, targ_bg = target_wm_maps[:, 0:256], target_wm_maps[:, 256:512]
        
        # Normalize to prevent amplitude collapse
        pred_obj = F.layer_norm(pred_obj, pred_obj.shape[1:])
        pred_bg = F.layer_norm(pred_bg, pred_bg.shape[1:])

        loss_rec = self.l1(pred_wm_maps, target_wm_maps)

        # 3. Correlation on FULL 256-Channel Tensors
        cos_obj = F.cosine_similarity(pred_obj.flatten(1), targ_obj.flatten(1)).mean()
        cos_bg = F.cosine_similarity(pred_bg.flatten(1), targ_bg.flatten(1)).mean()
        loss_corr = (1.0 - cos_obj) + (1.0 - cos_bg)

        # 4. Clean Penalty & Leakage
        loss_clean_maps = self.l1(pred_clean_maps, torch.zeros_like(pred_clean_maps))
        
        bg_mask_28x28 = 1.0 - mask_28x28
        leak_obj = (pred_obj.abs() * bg_mask_28x28).mean()
        leak_bg = (pred_bg.abs() * mask_28x28).mean()
        loss_leak = leak_obj + leak_bg

        # Combined Dual-Objective Loss
        total_loss = (10.0 * loss_detect) + (1.0 * loss_rec) + (2.0 * loss_corr) + (5.0 * loss_clean_maps) + (1.0 * loss_leak)

        return total_loss, {
            'loss_det': loss_detect.item(),
            'loss_rec': loss_rec.item(),
            'corr_obj': cos_obj.item(),
            'corr_bg': cos_bg.item(),
        }
# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def save_ckpt(state, path):
    torch.save(state, path)
    print(f"    [ckpt] Saved → {path}")

def load_ckpt(path, ae, ext, disc, opt_g, opt_d, sched_g, sched_d, scaler):
    ckpt = torch.load(path, map_location='cpu')
    
    (ae.module if isinstance(ae, DDP) else ae).load_state_dict(ckpt['ae'])
    (ext.module if isinstance(ext, DDP) else ext).load_state_dict(ckpt['ext'])
    (disc.module if isinstance(disc, DDP) else disc).load_state_dict(ckpt['disc'])
    
    opt_g.load_state_dict(ckpt['opt_g'])
    opt_d.load_state_dict(ckpt['opt_d'])
    if 'sched_g' in ckpt and sched_g is not None: sched_g.load_state_dict(ckpt['sched_g'])
    if 'sched_d' in ckpt and sched_d is not None: sched_d.load_state_dict(ckpt['sched_d'])
    if 'scaler' in ckpt: scaler.load_state_dict(ckpt['scaler'])
    
    print(f"    [ckpt] Resumed from epoch {ckpt.get('epoch', 0)} → {path}")
    return ckpt.get('epoch', 0), ckpt.get('best_metric', float('inf')), ckpt.get('global_step', 0)

def clean_state(s): return OrderedDict((k.replace('module.', ''), v) for k, v in s.items())

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_gan_ddp(rank, ws, cfg):
    set_seed(42, rank)
    print(f"[Rank {rank}] Initializing on {torch.cuda.get_device_name(rank)}")
    
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', '12356'
    dist.init_process_group('nccl', init_method='env://', world_size=ws, rank=rank)
    torch.cuda.set_device(rank)

    # Loaders
    tr_ds, vl_ds = split_ds(cfg)
    ts = DistributedSampler(tr_ds, num_replicas=ws, rank=rank, shuffle=True)
    vs = DistributedSampler(vl_ds, num_replicas=ws, rank=rank, shuffle=False)
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    tr_loader = DataLoader(tr_ds, sampler=ts, **loader_args)
    vl_loader = DataLoader(vl_ds, sampler=vs, **loader_args)

    # Models
    ae = DAE().to(rank)
    ae.load_state_dict(clean_state(torch.load(cfg['ae_weights'], map_location='cpu').get('model', torch.load(cfg['ae_weights'], map_location='cpu'))), strict=False)
    ae = DDP(ae, device_ids=[rank])

    # 2. FIXED EXTRACTOR INIT: Load the correct exact architecture
    ext = ex.HeavyWatermarkDecoder(num_res_blocks=6).to(rank)
    ext_st = torch.load(cfg['ext_weights'], map_location='cpu')
    ext.load_state_dict(clean_state(ext_st.get('model', ext_st)), strict=True)
    ext = DDP(ext, device_ids=[rank])

    disc = PatchD().to(rank)
    disc = DDP(disc, device_ids=[rank])

    # Frozen elements
    inj = SSInj(256, 28, 28).to(rank)
    inj.eval()
    edge = EdgeMask().to(rank)
    edge.eval()
    atk = AttackLayer(cfg['attack_p']).to(rank)

    # Optimizers & Loss
    opt_g = optim.AdamW(list(ae.parameters()) + list(ext.parameters()), lr=cfg['lr_g'], weight_decay=cfg['weight_decay'])
    opt_d = optim.AdamW(disc.parameters(), lr=cfg['lr_d'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler('cuda')
    crit_ext = SpatialForensicLoss().to(rank)
    crit_gan = nn.BCEWithLogitsLoss().to(rank)
    
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=cfg['max_iterations'])
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg['max_iterations'])

    if rank == 0:
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
        os.makedirs(cfg['best_model_dir'], exist_ok=True)
        print(f"  [train] {len(tr_ds)} samples | [val] {len(vl_ds)} samples")
        print(f"  Starting Stage 4 GAN training for {cfg['max_iterations']} epochs\n")

    history = {'train_total': [], 'val_total': [], 'lr_g': [], 'lr_d': [], 'breakdown': defaultdict(lambda: {'train': [], 'val': []})}
    
    acc = cfg['accum_steps']
    best_metric = float('inf')
    patience_ctr, global_step = 0, 0
    patience, min_delta, save_every = cfg.get('patience', 15), cfg.get('min_delta', 1e-4), cfg.get('save_every', 5)
    start_epoch = 0

    if cfg.get('resume') and os.path.isfile(cfg['resume']):
        start_epoch, best_metric, global_step = load_ckpt(cfg['resume'], ae, ext, disc, opt_g, opt_d, sched_g, sched_d, scaler)

    total_start_time = time.time()

    try:
        for itr in range(start_epoch, cfg['max_iterations']):
            epoch_start_time = time.time()
            ts.set_epoch(itr)
            ae.train(); ext.train(); disc.train(); atk.train()
            opt_g.zero_grad(set_to_none=True); opt_d.zero_grad(set_to_none=True)

            tr_loss_d, tr_loss_g, tr_bd, tr_n = 0.0, 0.0, defaultdict(float), 0

            for step, (img, mask) in enumerate(tr_loader):
                img, mask = img.to(rank, non_blocking=True), mask.to(rank, non_blocking=True)
                objs, bgs = img * mask, img * (1.0 - mask)
                B = img.size(0)

                # 1. NEW: Generate RANDOM targets for the GAN training loop
                random_w1 = torch.sign(torch.randn(B, 256, 28, 28, device=rank))
                random_w2 = torch.sign(torch.randn(B, 256, 28, 28, device=rank))
                random_w1[random_w1 == 0] = 1.0
                random_w2[random_w2 == 0] = 1.0
                target_wm = torch.cat([random_w1, random_w2], dim=1)
                
                is_acc = (step + 1) % acc != 0 and (step + 1) != len(tr_loader)
                sync_ctx = ae.no_sync() if is_acc else nullcontext()
                
                with sync_ctx:
                    with autocast('cuda'):
                        soft_m = TF.gaussian_blur(mask, [15,15], [5.0,5.0])
                        em_o, em_b = edge(objs), edge(bgs)
                        
                        # 2. NEW: Manual dynamic injection to replace the static 'inj'
                        zo, sko = ae.module.ae_o.encode(objs)
                        zb, skb = ae.module.ae_b.encode(bgs)
                        
                        zwo = zo + (0.5 * random_w1 * (em_o + 0.05).clamp(max=1.0))
                        zwb = zb + (3.5 * random_w2 * (em_b + 0.05).clamp(max=1.0))
                        
                        wm_o, wm_b = ae.module.ae_o.decode(zwo, sko), ae.module.ae_b.decode(zwb, skb)
                        wm_img = (wm_o * soft_m) + (wm_b * (1.0 - soft_m))
                        
                        with torch.no_grad():
                            cl_o, cl_b = ae.module.forward_watermarked(objs, bgs, inj, 0.0, 0.0, None, None)
                            cl_img = (cl_o * soft_m) + (cl_b * (1.0 - soft_m))

                        # D Step
                        pred_real = disc(cl_img)
                        pred_fake = disc(wm_img.detach())
                        loss_d = ((crit_gan(pred_real, torch.ones_like(pred_real)) + crit_gan(pred_fake, torch.zeros_like(pred_fake))) * 0.5) / acc

                scaler.scale(loss_d).backward()

                sync_ctx = ae.no_sync() if is_acc else nullcontext()
                with sync_ctx:
                    with autocast('cuda'):
                        # G Step
                        pred_fake_g = disc(wm_img)
                        loss_g_gan = crit_gan(pred_fake_g, torch.ones_like(pred_fake_g))
                        loss_g_l1 = F.l1_loss(wm_img, img)
                        
                        atk_wm = atk(wm_img, itr, cfg['attack_warmup_epochs'])
                        atk_cl = atk(cl_img, itr, cfg['attack_warmup_epochs'])
                        
                        pwm_maps, pwm_logit, _ = ext(atk_wm, soft_m)
                        pcl_maps, pcl_logit, _ = ext(atk_cl, soft_m)
                        
                        m28 = F.interpolate(soft_m, size=(28,28), mode='area')
                        
                        # Use target_wm which was already correctly generated at the top of the loop
                        loss_g_ext, bd = crit_ext(
                            pwm_maps, target_wm, pwm_logit,
                            pcl_maps, pcl_logit, m28
                        )
                        loss_g = ((loss_g_l1 * cfg['lambda_l1']) + (loss_g_ext * cfg['lambda_ext']) + (loss_g_gan * cfg['lambda_gan'])) / acc

                scaler.scale(loss_g).backward()

                if not is_acc:
                    scaler.unscale_(opt_d)
                    scaler.unscale_(opt_g)
                    clip_grad_norm_(disc.parameters(), 1.0)
                    clip_grad_norm_(list(ae.parameters()) + list(ext.parameters()), 1.0)
                    scaler.step(opt_d)
                    scaler.step(opt_g)
                    scaler.update()
                    opt_d.zero_grad(set_to_none=True)
                    opt_g.zero_grad(set_to_none=True)
                    global_step += 1

                bs = img.size(0)
                tr_loss_d += loss_d.item() * acc * bs
                tr_loss_g += loss_g.item() * acc * bs
                tr_bd['loss_d'] += loss_d.item() * acc * bs
                tr_bd['loss_g_gan'] += loss_g_gan.item() * bs
                tr_bd['loss_g_l1'] += loss_g_l1.item() * bs
                tr_bd['loss_g_ext'] += loss_g_ext.item() * bs
                tr_bd['corr_o'] += bd['corr_obj'] * bs
                tr_bd['corr_b'] += bd['corr_bg'] * bs
                tr_n += bs

            # ── Validation ────────────────────────────────────────────────────────
            ae.eval(); ext.eval(); disc.eval(); atk.eval()
            vl_loss_d, vl_loss_g, vl_bd, vl_n = 0.0, 0.0, defaultdict(float), 0

            with torch.inference_mode():
                for img, mask in vl_loader:
                    img, mask = img.to(rank, non_blocking=True), mask.to(rank, non_blocking=True)
                    objs, bgs = img * mask, img * (1.0 - mask)
                    B = img.size(0)

                    # 1. Random targets for validation
                    random_w1 = torch.sign(torch.randn(B, 256, 28, 28, device=rank))
                    random_w2 = torch.sign(torch.randn(B, 256, 28, 28, device=rank))
                    random_w1[random_w1 == 0] = 1.0
                    random_w2[random_w2 == 0] = 1.0
                    target_wm = torch.cat([random_w1, random_w2], dim=1)

                    with autocast('cuda'):
                        soft_m = TF.gaussian_blur(mask, [15,15], [5.0,5.0])
                        em_o, em_b = edge(objs), edge(bgs)
                        
                        zo, sko = ae.module.ae_o.encode(objs)
                        zb, skb = ae.module.ae_b.encode(bgs)
                        
                        zwo = zo + (0.5 * random_w1 * (em_o + 0.05).clamp(max=1.0))
                        zwb = zb + (3.5 * random_w2 * (em_b + 0.05).clamp(max=1.0))
                        
                        wm_o, wm_b = ae.module.ae_o.decode(zwo, sko), ae.module.ae_b.decode(zwb, skb)
                        wm_img = (wm_o * soft_m) + (wm_b * (1.0 - soft_m))
                        
                        cl_o, cl_b = ae.module.forward_watermarked(objs, bgs, inj, 0.0, 0.0, None, None)
                        cl_img = (cl_o * soft_m) + (cl_b * (1.0 - soft_m))

                        pred_real = disc(cl_img)
                        pred_fake = disc(wm_img)
                        loss_d_val = ((crit_gan(pred_real, torch.ones_like(pred_real)) + crit_gan(pred_fake, torch.zeros_like(pred_fake))) * 0.5)

                        pred_fake_g = disc(wm_img)
                        loss_g_gan_val = crit_gan(pred_fake_g, torch.ones_like(pred_fake_g))
                        loss_g_l1_val = F.l1_loss(wm_img, img)
                        
                        atk_wm = atk(wm_img, itr, cfg['attack_warmup_epochs'])
                        atk_cl = atk(cl_img, itr, cfg['attack_warmup_epochs'])
                        
                        pwm_maps, pwm_logit, _ = ext(atk_wm, soft_m)
                        pcl_maps, pcl_logit, _ = ext(atk_cl, soft_m)
                        
                        m28 = F.interpolate(soft_m, size=(28,28), mode='area')
                        
                        loss_g_ext_val, bd_val = crit_ext(
                            pwm_maps, target_wm, pwm_logit,
                            pcl_maps, pcl_logit, m28
                        )
                        
                        loss_g_val = (loss_g_l1_val * cfg['lambda_l1']) + (loss_g_ext_val * cfg['lambda_ext']) + (loss_g_gan_val * cfg['lambda_gan'])

                    bs = img.size(0)
                    vl_loss_d += loss_d_val.item() * bs
                    vl_loss_g += loss_g_val.item() * bs
                    vl_bd['loss_d'] += loss_d_val.item() * bs
                    vl_bd['loss_g_gan'] += loss_g_gan_val.item() * bs
                    vl_bd['loss_g_l1'] += loss_g_l1_val.item() * bs
                    vl_bd['loss_g_ext'] += loss_g_ext_val.item() * bs
                    vl_bd['corr_o'] += bd_val['corr_obj'] * bs
                    vl_bd['corr_b'] += bd_val['corr_bg'] * bs
                    vl_n += bs

            current_lr_g = opt_g.param_groups[0]['lr']
            sched_g.step()
            sched_d.step()

            # ── All-reduce metrics ─────────────────────────────────────────────────
            metrics = torch.tensor([tr_loss_d, tr_loss_g, float(tr_n), vl_loss_d, vl_loss_g, float(vl_n)], dtype=torch.float64, device=rank)
            dist.all_reduce(metrics)
            m = metrics.tolist()
            t_d, t_g, t_n, v_d, v_g, v_n = m[0]/m[2], m[1]/m[2], m[2], m[3]/m[5], m[4]/m[5], m[5]

            bd_keys = ['loss_d', 'loss_g_gan', 'loss_g_l1', 'loss_g_ext', 'corr_o', 'corr_b']
            bd_m = torch.tensor([tr_bd[k] for k in bd_keys] + [vl_bd[k] for k in bd_keys], dtype=torch.float64, device=rank)
            dist.all_reduce(bd_m)
            bd_m = bd_m.tolist()
            tr_bd_final = {k: bd_m[i] / t_n for i, k in enumerate(bd_keys)}
            vl_bd_final = {k: bd_m[len(bd_keys)+i] / v_n for i, k in enumerate(bd_keys)}

            # ── Logging & Saving ───────────────────────────────────────────────────
            if rank == 0:
                epoch_time = str(datetime.timedelta(seconds=int(time.time() - epoch_start_time)))
                total_time = str(datetime.timedelta(seconds=int(time.time() - total_start_time)))
                
                print(f"Epoch {itr+1:3d}/{cfg['max_iterations']} | LR_G: {current_lr_g:.6f} | Time: {epoch_time} (Total: {total_time})")
                print(f"  Train | Loss_D: {tr_bd_final['loss_d']:.4f} | Loss_G: {t_g:.4f} | Corr_O: {tr_bd_final['corr_o']:.3f} | Corr_B: {tr_bd_final['corr_b']:.3f}")
                print(f"  Val   | Loss_D: {vl_bd_final['loss_d']:.4f} | Loss_G: {v_g:.4f} | Corr_O: {vl_bd_final['corr_o']:.3f} | Corr_B: {vl_bd_final['corr_b']:.3f}")
                print('-' * 90)

                history['train_total'].append(t_g); history['val_total'].append(v_g)
                history['lr_g'].append(current_lr_g); history['lr_d'].append(opt_d.param_groups[0]['lr'])
                for k in bd_keys:
                    history['breakdown'][k]['train'].append(tr_bd_final[k])
                    history['breakdown'][k]['val'].append(vl_bd_final[k])

                val_metric = v_g + vl_bd_final['loss_d']
                
                ckpt_payload = {
                    "epoch": itr+1, "global_step": global_step,
                    "ae": ae.module.state_dict(), "ext": ext.module.state_dict(), "disc": disc.module.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                    "scaler": scaler.state_dict(), "best_metric": best_metric,
                }
                
                if val_metric < best_metric - min_delta:
                    best_metric = val_metric
                    patience_ctr = 0
                    save_ckpt(ckpt_payload, os.path.join(cfg['checkpoint_dir'], "best_model.pth"))
                    torch.save(ae.module.state_dict(), os.path.join(cfg['best_model_dir'], 'ae_best_weights.pth'))
                    torch.save(ext.module.state_dict(), os.path.join(cfg['best_model_dir'], 'ext_best_weights.pth'))
                    torch.save(disc.module.state_dict(), os.path.join(cfg['best_model_dir'], 'disc_best_weights.pth'))
                else:
                    patience_ctr += 1
                    print(f"  [early stop] no improvement {patience_ctr}/{patience}")

                if (itr + 1) % save_every == 0:
                    save_ckpt(ckpt_payload, os.path.join(cfg['checkpoint_dir'], f"epoch_{itr+1:03d}.pth"))

            stop_t = torch.tensor(patience_ctr, dtype=torch.int32, device=rank)
            dist.broadcast(stop_t, src=0)
            if stop_t.item() >= patience:
                if rank == 0: print(f"\n[!] Early stopping at epoch {itr+1}")
                break

    except KeyboardInterrupt:
        if rank == 0:
            print("\n[!] KeyboardInterrupt detected. Saving emergency checkpoint...")
            emergency_ckpt = {
                "epoch": itr, "global_step": global_step,
                "ae": ae.module.state_dict(), "ext": ext.module.state_dict(), "disc": disc.module.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                "scaler": scaler.state_dict(), "best_metric": best_metric,
            }
            save_ckpt(emergency_ckpt, os.path.join(cfg['checkpoint_dir'], "interrupt_checkpoint.pth"))
            print("Emergency save complete. Exiting gracefully.")
        dist.destroy_process_group()
        return

    if rank == 0:
        history['best_metric'] = best_metric
        with open(os.path.join(cfg['checkpoint_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining Complete. Total Time: {str(datetime.timedelta(seconds=int(time.time() - total_start_time)))}")

    dist.destroy_process_group()

def get_config(stage_key):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr_g', type=float, default=None)
    parser.add_argument('--lr_d', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f: master_json = json.load(f)
    cfg = {**master_json.get('paths', {}), **master_json.get('shared', {}), **master_json.get(stage_key, {})}

    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.lr_g is not None: cfg['lr_g'] = args.lr_g
    if args.lr_d is not None: cfg['lr_d'] = args.lr_d
    if args.resume is not None: cfg['resume'] = args.resume
    return cfg

if __name__ == '__main__':
    cfg = get_config(stage_key='stage4_gan')

    # "stage4_gan": {
    #     "images_dir": "/kaggle/working/clean_pet_data/images",
    #     "mask_dir": "/kaggle/working/clean_pet_data/masks",
    #     "checkpoint_dir": "/kaggle/working/checkpoints_gan",
    #     "best_model_dir": "/kaggle/working/best_model_gan",
    #     "ae_weights": "/kaggle/input/models/mrheavenly/stage-2b/pytorch/default/1/best_weights_wm.pth",
    #     "ext_weights": "/kaggle/working/best_model_e2e/best_weights.pth",
    #     "batch_size": 4,
    #     "max_iterations": 60,
    #     "lr_g": 0.0001,
    #     "lr_d": 0.00005,
    #     "weight_decay": 0.0001,
    #     "accum_steps": 4,
    #     "lambda_l1": 10.0,
    #     "lambda_ext": 5.0,
    #     "lambda_gan": 1.0,
    #     "attack_p": 0.8,
    #     "attack_warmup_epochs": 2,
    #     "patience": 15,
    #     "min_delta": 0.0001,
    #     "save_every": 5
    # }

    ws = torch.cuda.device_count()
    if ws < 1: raise RuntimeError("No GPUs found.")
    print("Loaded Stage 4 Config from master JSON.")
    mp.spawn(train_gan_ddp, args=(ws, cfg), nprocs=ws, join=True)