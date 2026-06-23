import torch
import time
from collections import defaultdict
import os
import json
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from .datasets import split_dataset
from .ae_transforms import SegmentTransform
from .ae_models import DualAutoencoder
from .ae_loss import SimpleCombinedLoss
from .utils import save_checkpoint, load_checkpoint, Utils

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


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    utils  = Utils()
    config = utils.CONFIG

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
    # config.setdefault('latent_channels', 256)

    # # ── Training ──────────────────────────────────────────────────────────────
    # config.setdefault('batch_size',      16)   # 224px is heavier than 256px — 16 safe on T4
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

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")

    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)