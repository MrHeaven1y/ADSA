"""
DDP training loop for the UNet mask generator.

Features:
- Mixed precision (AMP) + gradient scaling
- Gradient accumulation for higher effective batch sizes on limited VRAM
- Cosine annealing LR schedule
- All-reduce synchronisation across GPUs for accurate metric logging

Run via torchrun or mp.spawn.
"""

import os
import time
import json
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from .segment_models import SegUNet
from collections import defaultdict
from .segment_metrics import iou_score
from torch.utils.data import DataLoader
from .segment_loss import BCEDiceKLLoss
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from .segment_transforms import SegmentTransform
from .datasets import split_dataset, benchmark_num_workers
from .utils import save_checkpoint, load_checkpoint, Utils
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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


if __name__ == "__main__":
    
    utils = Utils()
    config = utils.CONFIG
    
    config.setdefault("images_dir", "/kaggle/input/oxford-iiit-pet/images")
    config.setdefault("mask_dir", "/kaggle/input/oxford-iiit-pet/annotations/trimaps")
    config.setdefault("checkpoint_dir", "/kaggle/working/checkpoints_unet")
    config.setdefault("best_model_dir", "/kaggle/working/best_model_unet")
    
    config.setdefault("img_size", 224)
    config.setdefault("max_samples", None)
    config.setdefault("split_size", 0.85)
    config.setdefault("cache_ram", True)

    config.setdefault("batch_size", 16)
    config.setdefault("max_iterations", 60)
    config.setdefault("lr", 0.001)
    config.setdefault("weight_decay", 0.0001)
    config.setdefault("grad_clip", 1.0)
    config.setdefault("accum_steps", 2)
    config.setdefault("patience", 10)
    config.setdefault("min_delta", 0.0001)
    config.setdefault("resume", None)
    config.setdefault("save_every", 5)
    config.setdefault("bce_weight", 0.4),
    config.setdefault("dice_weight", 0.4),
    config.setdefault("kl_weight", 0.2),
    

    config["num_workers"] = benchmark_num_workers(
        config['batch_size'],
        config['img_size'],
        candidates=[0, 2, 4, 8]
    )
    