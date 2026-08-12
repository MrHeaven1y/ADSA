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
import torchvision.models as models
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
def run_forensic_diagnostics(epoch, config, ae_module, injector, ema_extractor,
                              semantic_masker, attack_layer_det, tamper_layer, device):
    """
    Runs the 4 forensic tests mid-training and saves results to /kaggle/working/diagnostics/epoch_XXX/
    """
    diag_dir = f"/kaggle/working/diagnostics/epoch_{epoch:03d}"
    os.makedirs(diag_dir, exist_ok=True)

    log_path = os.path.join(diag_dir, "diagnostics_log.txt")
    log_file = open(log_path, 'w')
    _orig_stdout = sys.stdout

    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, text):
            for f in self.files: f.write(text); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    sys.stdout = _Tee(_orig_stdout, log_file)

    def _to_numpy(t):
        t = t[0].detach().cpu().permute(1, 2, 0)
        return ((t * 0.5 + 0.5) * 255).clamp(0, 255).byte().numpy()

    def _compute_dice(pred, gt, eps=1e-6):
        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        return ((2 * inter + eps) / (union + eps)).item()

    status_map = {0: "Authentic", 1: "Tampered", 2: "Fake Identity", 3: "No Watermark"}

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    identity_threshold = config.get('identity_threshold', 0.55)
    image_name = config.get('test_image', 'Abyssinian_100')
    img_dir = config.get('images_dir', '/kaggle/working/clean_pet_data/images')
    mask_dir = config.get('mask_dir', '/kaggle/working/clean_pet_data/masks')
    max_iters = config.get('max_iterations', 80)
    test_attack_epoch = max(max_iters, 50)

    ae_module.eval(); injector.eval(); ema_extractor.eval()
    semantic_masker.eval(); attack_layer_det.eval(); tamper_layer.eval()

    try:
        print("=" * 70)
        print(f"  FORENSIC DIAGNOSTICS — EPOCH {epoch}")
        print(f"  Test image: {image_name} | Threshold: {identity_threshold}")
        print("=" * 70)

        # 1. Load Image + Mask
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            c = os.path.join(img_dir, f"{image_name}{ext}")
            if os.path.isfile(c): img_path = c; break
        if img_path is None:
            orig_img_dir = "/kaggle/input/oxford-iiit-pet/images"
            for ext in ['.jpg', '.jpeg', '.png']:
                c = os.path.join(orig_img_dir, f"{image_name}{ext}")
                if os.path.isfile(c): img_path = c; break

        mask_path = None
        for pat in [f"{image_name}_mask.png", f"{image_name}.png", f"{image_name}_mask.jpg"]:
            c = os.path.join(mask_dir, pat)
            if os.path.isfile(c): mask_path = c; break
        if mask_path is None:
            orig_mask_dir = "/kaggle/input/oxford-iiit-pet/annotations/trimaps"
            for pat in [f"{image_name}.png", f"{image_name}_mask.png"]:
                c = os.path.join(orig_mask_dir, pat)
                if os.path.isfile(c): mask_path = c; break

        img_raw = read_image(img_path)
        mask_raw = read_image(mask_path)
        if img_raw.shape[0] == 4: img_raw = img_raw[:3]
        elif img_raw.shape[0] == 1: img_raw = img_raw.repeat(3, 1, 1)
        if mask_raw.shape[0] > 1: mask_raw = mask_raw[0:1]
        mask_raw = (mask_raw > 127).float()
        img_raw = img_raw.float() / 255.0

        img = TF.resize(img_raw, [224, 224])
        mask = TF.resize(mask_raw, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(img, [0.5]*3, [0.5]*3)
        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        # 2. Encode
        with torch.inference_mode():
            objs, bgs = img * mask, img * (1 - mask)
            soft_masks = TF.gaussian_blur(mask, [15, 15], [5.0, 5.0])
            mask_obj, mask_bg = semantic_masker(objs), semantic_masker(bgs)
            zo, sko = ae_module.enc_obj(objs)
            zb, skb = ae_module.enc_bg(bgs)
            mask_latent = F.interpolate(soft_masks, (28, 28), mode='nearest')

        # 3. Authentic Watermark
        wm_idx = 0
        with torch.inference_mode():
            base_w1 = injector.base_w1_pool[wm_idx:wm_idx+1].to(device)
            base_w2 = injector.base_w2_pool[wm_idx:wm_idx+1].to(device)
            wm_obj_z = injector.inject(zo, base_w1, alpha=0.225, semantic_mask=mask_obj)
            wm_bg_z = injector.inject(zb, base_w2, alpha=0.0925, semantic_mask=mask_bg)
            z_auth = wm_obj_z * mask_latent + wm_bg_z * (1 - mask_latent)
            sk_composite = ae_module.blend_skips(sko, skb, soft_masks)
            wm_image = ae_module.shared_decoder(z_auth, sk_composite)
            z_clean = zo * mask_latent + zb * (1 - mask_latent)
            clean_image = ae_module.shared_decoder(z_clean, sk_composite)

        # 4. Build Samples
        samples = []
        with torch.inference_mode():
            authentic_attacked = attack_layer_det(wm_image, current_epoch=test_attack_epoch, max_epochs=max(80, max_iters))
            samples.append({'name': 'Authentic Attack', 'image': authentic_attacked, 'gt_class': 0, 'gt_id': wm_idx, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})
            
            tampered_img, tamper_gt = tamper_layer(authentic_attacked, seed=42)
            samples.append({'name': 'Tampered', 'image': tampered_img, 'gt_class': 1, 'gt_id': wm_idx, 'gt_mask': tamper_gt})
            
            fake_idx = 17
            fake_w1 = injector.fake_w1_pool[fake_idx:fake_idx+1].to(device)
            fake_w2 = injector.fake_w2_pool[fake_idx:fake_idx+1].to(device)
            fake_obj_z = injector.inject(zo, fake_w1, alpha=0.225, semantic_mask=mask_obj)
            fake_bg_z = injector.inject(zb, fake_w2, alpha=0.0925, semantic_mask=mask_bg)
            z_fake = fake_obj_z * mask_latent + fake_bg_z * (1 - mask_latent)
            fake_image = ae_module.shared_decoder(z_fake, sk_composite)
            fake_image = attack_layer_det(fake_image, current_epoch=test_attack_epoch, max_epochs=max(80, max_iters))
            samples.append({'name': 'Fake Identity', 'image': fake_image, 'gt_class': 2, 'gt_id': fake_idx, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})
            
            samples.append({'name': 'No Watermark', 'image': clean_image, 'gt_class': 3, 'gt_id': -1, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})

        # TEST 1
        print("\nTEST 1: FORENSIC EVALUATION")
        with torch.inference_mode():
            for sample in samples:
                pred_int, pred_glob, _, _, pred_fp = ema_extractor(sample['image'])
                pred_class = torch.argmax(pred_glob, dim=1).item()
                dice = _compute_dice(torch.sigmoid(pred_int), sample['gt_mask'])
                if sample['gt_id'] != -1:
                    target_center = F.normalize(ema_extractor.identity_centers[sample['gt_id']], dim=0)
                    cos_sim = F.cosine_similarity(pred_fp, target_center.unsqueeze(0)).item()
                else:
                    cos_sim = F.cosine_similarity(pred_fp, ema_extractor.identity_centers[0:1]).item()
                print(f"  {sample['name']:<18} | GT: {status_map[sample['gt_class']]} | Pred: {status_map[pred_class]} | Dice: {dice:.4f} | Sim: {cos_sim:.4f}")

        # TEST 2
        print("\nTEST 2: VISUALIZATION + IDENTITY MATCHING")
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4 * len(samples)))
            if len(samples) == 1: axes = [axes]
            
            with torch.inference_mode():
                identity_bank = F.normalize(ema_extractor.identity_centers, dim=1)
                for row, sample in enumerate(samples):
                    pred_int, pred_glob, _, _, pred_fp = ema_extractor(sample['image'])
                    pred_fp = F.normalize(pred_fp, dim=1)
                    sims = torch.matmul(pred_fp, identity_bank.T)
                    max_sim, predicted_id = sims.max(dim=1)
                    max_sim_val = max_sim.item()
                    predicted_id_val = predicted_id.item()
                    
                    if max_sim_val < identity_threshold: 
                        predicted_id_val = -1
                    
                    dice = _compute_dice(torch.sigmoid(pred_int), sample['gt_mask'])
                    
                    axes[row][0].imshow(_to_numpy(img)); axes[row][0].set_title("Original")
                    axes[row][1].imshow(_to_numpy(wm_image)); axes[row][1].set_title("Authentic WM")
                    axes[row][2].imshow(_to_numpy(sample['image'])); axes[row][2].set_title(sample['name'])
                    im = axes[row][3].imshow(torch.sigmoid(pred_int)[0,0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                    # FIX: Removed .item() from predicted_id_val
                    axes[row][3].set_title(f"GT_ID:{sample['gt_id']} | PRED_ID:{predicted_id_val}\nCONF:{max_sim_val:.3f} | DICE:{dice:.3f}")
                    plt.colorbar(im, ax=axes[row][3], fraction=0.046, pad=0.04)
                    for c in range(4): axes[row][c].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(diag_dir, "forensic_evaluation_results.png"), bbox_inches='tight')
            plt.close(fig)
            print("  Saved visualization.")
        except Exception as e:
            print(f"  Visualization failed: {e}")

        # TEST 3
        print("\nTEST 3: SIMILARITY STATS")
        with torch.inference_mode():
            _, _, _, _, pred_fp = ema_extractor(authentic_attacked)
            pred_fp = F.normalize(pred_fp, dim=1)
            identity_bank = F.normalize(ema_extractor.identity_centers, dim=1)
            sims = torch.matmul(pred_fp, identity_bank.T)
            print(f"  Pred Norm: {pred_fp.norm(dim=1).mean().item():.4f}")
            print(f"  Sim Stats -> Min: {sims.min().item():.4f} | Max: {sims.max().item():.4f} | Mean: {sims.mean().item():.4f}")
            topk_vals, topk_idx = torch.topk(sims[0], k=10)
            print("  Top-10 Matches:")
            for r in range(10):
                # FIX: Safe casting to int just in case
                idx_val = topk_idx[r].item() if torch.is_tensor(topk_idx[r]) else int(topk_idx[r])
                sim_val = topk_vals[r].item() if torch.is_tensor(topk_vals[r]) else float(topk_vals[r])
                print(f"    {r+1:02d} | ID={idx_val:3d} | SIM={sim_val:.4f}")

        # TEST 4
        print("\nTEST 4: LATENT SPACE SANITY CHECK")
        with torch.inference_mode():
            z_clean_obj, _ = ae_module.enc_obj(objs)
            z_clean_bg, _ = ae_module.enc_bg(bgs)
            z_wm_obj = injector.inject(z_clean_obj, base_w1, alpha=0.225, semantic_mask=mask_obj)
            z_wm_bg = injector.inject(z_clean_bg, base_w2, alpha=0.0925, semantic_mask=mask_bg)
            
            diff_obj = z_wm_obj - z_clean_obj
            diff_bg = z_wm_bg - z_clean_bg
            obj_ratio = diff_obj.pow(2).mean().item() / (z_clean_obj.pow(2).mean().item() + 1e-8)
            bg_ratio = diff_bg.pow(2).mean().item() / (z_clean_bg.pow(2).mean().item() + 1e-8)
            print(f"  OBJ Energy Ratio: {obj_ratio:.8f} | BG Energy Ratio: {bg_ratio:.8f}")
            
            z_auth_t4 = z_wm_obj * mask_latent + z_wm_bg * (1 - mask_latent)
            wm_image_t4 = ae_module.shared_decoder(z_auth_t4, sk_composite)
            
            pred_int_t4, pred_glob_t4, _, _, pred_fp_t4 = ema_extractor(wm_image_t4)
            pred_fp_t4 = F.normalize(pred_fp_t4, dim=1)
            sims_t4 = torch.matmul(pred_fp_t4, identity_bank.T)
            conf_t4, pred_id_t4 = sims_t4.max(dim=1)
            print(f"  Pred Status: {status_map[pred_glob_t4.argmax(1).item()]} | GT ID: {wm_idx} | Rec ID: {pred_id_t4.item()} | Conf: {conf_t4.item():.4f}")

        print("\n" + "=" * 70)
        print(f"  DIAGNOSTICS COMPLETE — Results in {diag_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Diagnostic failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        sys.stdout = _orig_stdout
        log_file.close()
        torch.cuda.empty_cache()
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

def get_mode_weights(epoch, max_epochs):
    """Progressive curriculum - start easy, add harder modes later"""
    if epoch < max_epochs * 0.25:
        # Early: mostly clean attacked, some tampered, no fake
        return torch.tensor([0.50, 0.30, 0.00, 0.10, 0.10])
    elif epoch < max_epochs * 0.50:
        # Mid: introduce fake watermarks
        return torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10])
    elif epoch < max_epochs * 0.75:
        # Late: more identity confusion
        return torch.tensor([0.20, 0.20, 0.30, 0.15, 0.15])
    else:
        # Final: balanced with emphasis on hard cases
        return torch.tensor([0.15, 0.20, 0.30, 0.20, 0.15])

def generate_hybrid_chaotic_watermark(latent_channels, latent_h, latent_w, secret_key, cache_dir="/kaggle/working/wm_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    
    # Hash the parameters to create a unique filename
    params_str = f"{latent_channels}_{latent_h}_{latent_w}_{secret_key}"
    file_hash = hashlib.md5(params_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{file_hash}.pt")
    
    # 1. Try to load, but catch corrupted files!
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path)
        except Exception:
            os.remove(cache_path)
            
    # 2. Compute the watermark
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
    
    # 3. Save to cache ATOMICALLY using PID to prevent DDP race conditions
    temp_path = cache_path + f".tmp_{os.getpid()}"
    torch.save(w, temp_path)
    try:
        os.replace(temp_path, cache_path)
    except OSError:
        # If another rank beat us to it, just delete our temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
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
            nn.Conv2d(256, latent_channels, 1, bias=False), nn.GroupNorm(num_groups, latent_channels), nn.ReLU(inplace=True),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
            ResBlock(latent_channels, latent_channels, norm_layer=decoder_norm),
        )
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, 1, bias=False)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, 1, bias=False)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(latent_channels + proj['s3'], 256, 3, 1, 1, bias=False), nn.GroupNorm(num_groups, 256), nn.ReLU(inplace=True), ResBlock(256, 256, norm_layer=decoder_norm))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256 + proj['s2'], 128, 3, 1, 1, bias=False), nn.GroupNorm(num_groups, 128), nn.ReLU(inplace=True), ResBlock(128, 128, norm_layer=decoder_norm))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128 + proj['s1'], 64, 3, 1, 1, bias=False), nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True), ResBlock(64, 64, norm_layer=decoder_norm))
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
            s0, s1, s2, s3 = self.s0_drop(s0), self.s1_drop(s1), self.s2_drop(s2), self.s3_drop(s3)
        s3_proj = self.skip_proj_s3(s3)
        s2_proj = self.skip_proj_s2(s2)
        s1_proj = self.skip_proj_s1(s1)
        x  = self.up1(torch.cat([z,  s3_proj], dim=1))
        x  = self.up2(torch.cat([x,  s2_proj], dim=1))
        x  = self.up3(torch.cat([x,  s1_proj], dim=1))
        out = self.output(x)
        return out, z, mu, logvar

class DualAutoencoder(nn.Module):
    def __init__(self, latent_channels=128, skip_proj_config=None):
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
            rec_obj_wm, _, _, _ = self.ae_obj(objs, z_override=z_obj + wm_obj)
            rec_bg_wm,  _, _, _ = self.ae_bg(bgs,  z_override=z_bg + wm_bg)
            pred_wm_obj = self.ae_obj.extract_watermark(rec_obj_wm)
            pred_wm_bg  = self.ae_bg.extract_watermark(rec_bg_wm)
            out.extend([rec_obj_wm, rec_bg_wm, pred_wm_obj, pred_wm_bg, wm_obj, wm_bg])
        return tuple(out)

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
    def __init__(self, fp_dim=256, pretrained=True, resnet_weights_path=None):
        
        super().__init__()

        resnet = models.resnet34(weights=None)
        
        if pretrained:
            if resnet_weights_path and os.path.isfile(resnet_weights_path):
                resnet.load_state_dict(torch.load(resnet_weights_path, map_location='cpu'))

            else:
                # Fallback if no path is provided
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                resnet = models.resnet34(weights=weights)
        
        # 2. THE FORENSIC FIX: Replace MaxPool with AvgPool
        # AvgPool preserves the high-frequency watermark distribution, 
        # while keeping the spatial math exactly the same (56x56 output).
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1) 
        ) 
        
        self.layer1 = resnet.layer1 # Output: 64 ch, 56x56
        self.layer2 = resnet.layer2 # Output: 128 ch, 28x28
        self.layer3 = resnet.layer3 # Output: 256 ch, 14x14  <-- We pull Latents from here!
        self.layer4 = resnet.layer4 # Output: 512 ch, 7x7    <-- Deepest semantics
        
        # 3. Custom FPN Projections (Wired for ResNet-34 dimensions)
        self.fpn_proj3 = nn.Conv2d(256, 128, 1)
        self.fpn_proj2 = nn.Conv2d(128, 128, 1)
        self.fpn_proj1 = nn.Conv2d(64,  128, 1)

        self.integrity_head = nn.Sequential(
            DecoderResBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        # 4. Heads updated for new routing
        # Global detector still looks at the deepest layer (512 ch)
        self.detector_head = nn.Sequential(
            GeMPooling(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128), 
            nn.GELU(), 
            nn.Linear(128, 4)
        )
        
        # nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.latent_mean = nn.Conv2d(128, 256, 3, 1, 1)
        self.latent_logvar = nn.Conv2d(128, 256, 3, 1, 1)
        
        self.num_ids = fp_dim
        self.arc_margin = 0.2
        self.arc_scale = 32.0

        init = torch.linalg.qr(torch.randn(fp_dim, self.num_ids))[0].T
        self.identity_centers = nn.Parameter(init)

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
        x_stem = self.stem(x)
        s1 = self.layer1(x_stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3) 

        robust_features_deep = self.feat_dropout(s4)
        robust_features_mid  = self.feat_dropout(s2)

        # Feature Pyramid Network (FPN) 
        p3 = self.fpn_proj3(s3)
        p2 = self.fpn_proj2(s2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.fpn_proj1(s1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        integrity_map = self.integrity_head(p1)

        # Global logit from deepest features (s4)
        global_logit = self.detector_head(robust_features_deep)
        
        # Latents from middle features (s3) - 14x14!
        z_mean       = self.latent_mean(robust_features_mid)
        z_logvar     = self.latent_logvar(robust_features_mid)
        
        # Identity pooling from z_mean
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
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))        
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.temperature = temperature


        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel_edges(self, x):

        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    def forward(self, pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor, current_epoch=0, latent_weight=1.0, id_weight=1.0, collapse_weight=0.1, max_epochs=80):
        
        loss_spatial_bce = self.bce(pred_int, gt_int)
        
        # Dice Loss for sharp tamper boundaries
        pred_sigmoid = torch.sigmoid(pred_int)
        
        intersection = (pred_sigmoid * gt_int).sum(dim=(2,3))
        union = pred_sigmoid.sum(dim=(2,3)) + gt_int.sum(dim=(2,3))
        
        has_temper = (gt_int.sum(dim=[2,3]) > 0).float()
        loss_dice_raw = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

        loss_dice = (loss_dice_raw * has_temper).sum() / (has_temper.sum() + 1e-6)

        pred_edge = self._sobel_edges(pred_sigmoid)
        gt_edge = self._sobel_edges(gt_int)
        loss_edge = F.l1_loss(pred_edge, gt_edge)


        loss_spatial = loss_spatial_bce + loss_dice + (0.1 * loss_edge)
        loss_global  = self.ce(pred_glob, gt_glob)
        
        pred_latent = z_mean
        
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


        phase_2_start = int(0.50 * max_epochs) # e.g., 40 if max_epochs=80
        phase_3_start = int(0.75 * max_epochs) # e.g., 60 if max_epochs=80
        
        if current_epoch < phase_2_start:
            lambda_infonce = 0.0
        elif current_epoch < phase_3_start:
            lambda_infonce = 0.02
        else:
            lambda_infonce = 0.05

        arc_logits = extractor.compute_arcface_logits(features, identity_labels)
        loss_arc = F.cross_entropy(arc_logits, identity_labels)

        # anti-collapse regularizer 
        embed_std = features.std(dim=0).mean()
        loss_collapse = -embed_std

        # Combine ArcFace with the throttled InfoNCE
        loss_identity = (loss_arc + lambda_infonce * loss_infonce) * id_weight

        total_loss = loss_spatial + loss_global + (loss_latent * latent_weight) + loss_identity + (collapse_weight * loss_collapse)
        
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
    
    return ckpt['epoch'], ckpt['best_val_loss'], ckpt

# --------------------------
#  TRAINING

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
        timeout=timedelta(minutes=2)
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
    
    ae = DualAutoencoder(
        latent_channels=latent_channels,
        skip_proj_config=config.get('skip_proj')
    ).to(rank)
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
    resnet_weights_path = config.get("resnet_weights_path", None)

    extractor = ForensicIntegrityAnalyzer(
        fp_dim=fp_dim,
        pretrained=True,
        resnet_weights_path=resnet_weights_path
    ).to(rank)
    
    extractor = DDP(extractor, device_ids=[rank])

    if rank == 0:
        p = sum(v.numel() for v in extractor.parameters()) / 1e6
        print(f"  [extractor] ScratchCNN | params: {p:.2f}M  (all trainable)")

    semantic_masker = AdaptiveSemanticMasker().to(rank)
    attack_layer = AttackSimulationLayer(p_apply=config.get('attack_p', 0.5), deterministic=False).to(rank)
    val_attack_layer = AttackSimulationLayer(p_apply=config.get('attack_p', 0.5), deterministic=True).to(rank)
    tamper_layer = AdvancedTamperLayer().to(rank)
    criterion = ForensicInfoNCELoss(temperature=0.1).to(rank)

    backbone_params = []
    head_params = []
    
    for name, param in extractor.module.named_parameters():
        if 'stem' in name or 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    # Backbone learns 10x slower than the config learning rate
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1}, 
        {'params': head_params, 'lr': config['lr']}            
    ], weight_decay=config['weight_decay'])


    # safenet for the plateaus
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,T_0=config.get('restart_interval', 20), T_mult=1, eta_min=config['lr'] * 0.01
    )
    scaler = GradScaler(device='cuda')


    start_epoch = 0
    best_val = float('inf')
    patience_ctr = 0

    if config.get('resume') and os.path.isfile(config['resume']):
        start_epoch, prev_best, ckpt = load_checkpoint(config['resume'], extractor, optimizer, scheduler, scaler)
        best_val = prev_best
        patience_ctr = 0 
        if 'ema_state' in ckpt:
            ema_extractor.load_state_dict(ckpt['ema_state'])

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

    ema_decay = 0.995
    ema_extractor = ForensicIntegrityAnalyzer(
        fp_dim=fp_dim,
        pretrained=True,
        resnet_weights_path=resnet_weights_path
    ).to(rank)
        
    for p in ema_extractor.parameters(): p.requires_grad = False 

    for itr in range(start_epoch, max_iterations):
        start_time = time.time()
        train_sampler.set_epoch(itr)
        
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0; train_bd = defaultdict(float); train_n = 0
        
        extractor.train(); attack_layer.train(); tamper_layer.train()
        def set_bn_eval(m):
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        extractor.apply(set_bn_eval)

        for step, (images, masks) in enumerate(train_loader):
            
            images, masks = images.to(rank, non_blocking=True), masks.to(rank, non_blocking=True)
            B = images.size(0)

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

                    with torch.no_grad():
                        _, zo, _, _ = ae.module.ae_obj(objs)
                        _, zb, _, _ = ae.module.ae_bg(bgs)

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
                    wm_obj_img, _, _, _ = ae.module.ae_obj(objs, z_override=wm_obj_z)
                    wm_bg_img, _, _, _  = ae.module.ae_bg(bgs, z_override=wm_bg_z)
                    wm_composite = wm_obj_img * soft_masks + wm_bg_img * (1 - soft_masks)

                    # ═══════════════════════════════════════════════════════════════════════
                    # MODE SELECTION - Per-sample for better gradient diversity
                    # ═══════════════════════════════════════════════════════════════════════
                    mode_weights = get_mode_weights(itr, max_iterations).to(rank)

                    # Option A: Per-batch mode (simple, your original approach)
                    attack_mode = torch.multinomial(mode_weights, 1).item()

                    # Option B: Per-sample mode (better diversity - uncomment if desired)
                    # attack_modes = torch.multinomial(mode_weights.repeat(B, 1), 1).squeeze(1)

                    gt_int  = torch.zeros(B, 1, 56, 56, device=rank)
                    gt_glob = torch.zeros(B, dtype=torch.long, device=rank)
                    identity_ids = idx_base.clone()  # Default: use base watermark IDs

                    if attack_mode == 0:
                        # Attacked watermark, no tampering
                        wm_attacked = attack_layer(wm_composite, itr, max_epochs=max_iterations)
                        gt_glob[:] = 0

                    elif attack_mode == 1:
                        # Attacked + Tampered
                        base_attacked = attack_layer(wm_composite, itr, max_epochs=max_iterations)
                        wm_attacked, gt_int = tamper_layer(base_attacked)
                        gt_glob[:] = 1
                        
                    elif attack_mode == 2:
                        # Wrong/Fake watermark (identity confusion)
                        idx_fake1 = torch.randint(0, len(injector.fake_w1_pool), (B,), device=rank)
                        idx_fake2 = torch.randint(0, len(injector.fake_w2_pool), (B,), device=rank)
                        
                        identity_ids = idx_fake1.clone()  # Use fake IDs for identity loss
                        
                        wrong_w1 = injector.fake_w1_pool[idx_fake1]
                        wrong_w2 = injector.fake_w2_pool[idx_fake2]
                        
                        wrong_z_obj = injector.inject(zo, wrong_w1, a_obj, mask_obj)
                        wrong_z_bg  = injector.inject(zb, wrong_w2, a_bg, mask_bg)
                        fake_obj_img, _, _, _ = ae.module.ae_obj(objs, z_override=wrong_z_obj)
                        fake_bg_img, _, _, _  = ae.module.ae_bg(bgs, z_override=wrong_z_bg)
                        fake_composite = fake_obj_img * soft_masks + fake_bg_img * (1 - soft_masks)
                        wm_attacked = attack_layer(fake_composite, itr, max_epochs=max_iterations)
                        gt_glob[:] = 2
                        # gt_int stays zeros (no spatial tampering in this mode)

                    elif attack_mode == 3:
                        # Clean watermark (no attack)
                        wm_attacked = wm_composite
                        gt_glob[:] = 0

                    elif attack_mode == 4:
                        # No watermark at all (clean image)
                        wm_attacked = images
                        gt_glob[:] = 3

                    else:
                        raise ValueError(f"Unknown attack_mode: {attack_mode}")

                    with torch.no_grad():
                        _, atk_z_obj, _, _ = ae.module.ae_obj(wm_attacked * soft_masks)
                        _, atk_z_bg, _, _  = ae.module.ae_bg(wm_attacked * (1 - soft_masks))
                        _, clean_obj, _, _ = ae.module.ae_obj(images * soft_masks)
                        _, clean_bg, _, _  = ae.module.ae_bg(images * (1 - soft_masks))

                    # [FIX 1] Clean Structural Distillation (No fake projections!)
                    norm_atk_obj = F.normalize(atk_z_obj, dim=1)
                    norm_cln_obj = F.normalize(clean_obj, dim=1)
                    shift_obj = norm_atk_obj - norm_cln_obj
                    
                    norm_atk_bg = F.normalize(atk_z_bg, dim=1)
                    norm_cln_bg = F.normalize(clean_bg, dim=1)
                    shift_bg = norm_atk_bg - norm_cln_bg

                    # Combine shifts using the semantic mask
                    shift_total = shift_obj * mask_latent + shift_bg * (1 - mask_latent)
                    
                    shift_flat = shift_total.flatten(1)
                    gt_latent = F.normalize(shift_flat, dim=1).view_as(shift_total)
                    
                    
                    if rank == 0 and step == 0:
                        print(f"  [DBG] mode={attack_mode} | gt_glob dist={torch.bincount(gt_glob, minlength=4)}")
                        print(f"  [DBG] pred_glob dist={torch.bincount(pred_glob.argmax(1), minlength=4)}")
                        print(f"  [DBG] shift mag={shift_total.abs().mean():.6f}")


                    if rank == 0 and step == 0:
                        print(f"  [DBG] mode={attack_mode} | shift mag={shift_total.abs().mean():.6f}")
                    
                    pred_int, pred_glob, z_mean, z_logvar, pred_fp = extractor(wm_attacked)
                    
                    phase_2_start = int(0.50 * max_iterations)
                    ramp_length = config.get('phase_2_ramp_length', 10)

                    # Base weights from config
                    base_id_weight = config.get('id_weight_max', 1.0)
                    base_latent_weight = config['latent_weight']
                    collapse_weight = config.get('collapse_weight', 0.1)

                    if attack_mode == 4:
                        # No watermark case - disable identity and latent losses
                        id_weight = 0.0
                        latent_weight = 0.0
                    elif itr < phase_2_start:
                        # Phase 1: Focus on spatial/global, light latent
                        id_weight = 0.0
                        latent_weight = base_latent_weight  # Reduced early on
                    else:
                        # Phase 2: Ramp up identity loss
                        progress = min(1.0, (itr - phase_2_start) / ramp_length)
                        id_weight = progress * base_id_weight
                        latent_weight = base_latent_weight


                    loss, bd = criterion(
                        pred_int, gt_int, pred_glob, gt_glob, z_mean,
                        z_logvar, gt_latent, pred_fp, identity_ids,
                        extractor.module, current_epoch=itr, 
                        latent_weight=latent_weight, id_weight=id_weight,
                        collapse_weight=collapse_weight,
                        max_epochs=max_iterations
                    )

                    loss = loss / accum_steps
                
            if not torch.isfinite(loss):
                print(f"[Rank {rank}] WARNING: Non-finite loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()

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
                    with torch.no_grad():
                        _, zo, _, _ = ae.module.ae_obj(objs)
                        _, zb, _, _ = ae.module.ae_bg(bgs)

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

                    mask_latent = F.interpolate(soft_masks, size=(28,28), mode='nearest')
                    wm_obj_img, _, _, _ = ae.module.ae_obj(objs, z_override=wm_obj_z)
                    wm_bg_img, _, _, _  = ae.module.ae_bg(bgs, z_override=wm_bg_z)
                    wm_composite = wm_obj_img * soft_masks + wm_bg_img * (1 - soft_masks)

                    mode_weights = get_mode_weights(itr, max_iterations).to(rank)
                    # Make it deterministic per-step but follow the same distribution
                    attack_mode = torch.multinomial(
                        mode_weights, 1, generator=torch.Generator(device=rank).manual_seed(step)
                    ).item()
                    gt_int  = torch.zeros(B, 1, 56, 56, device=rank)
                    gt_glob = torch.zeros(B, dtype=torch.long, device=rank)

                    current_w1, current_w2 = base_w1, base_w2

                    if attack_mode == 0:
                        wm_attacked = val_attack_layer(wm_composite, itr, max_epochs=max_iterations)
                        gt_glob[:] = 0

                    elif attack_mode == 1:
                        
                        
                        wm_attacked, gt_int = tamper_layer(val_attack_layer(wm_composite, itr, max_epochs=max_iterations), seed=step)
                        gt_glob[:] = 1
                    
                    elif attack_mode == 2:
                        
                        idx_fake1 = torch.arange(step * B,(step + 1) * B,device=rank) % len(injector.fake_w1_pool)
                        idx_fake2 = torch.arange(step * B,(step + 1) * B,device=rank) % len(injector.fake_w2_pool)
                        
                        identity_ids = idx_fake1.clone()
                        
                        wrong_w1 = injector.fake_w1_pool[idx_fake1]
                        wrong_w2 = injector.fake_w2_pool[idx_fake2]
                        
                        wrong_z_obj = injector.inject(zo, wrong_w1, a_obj, mask_obj)
                        wrong_z_bg  = injector.inject(zb, wrong_w2, a_bg, mask_bg)
                        fake_obj_img, _, _, _ = ae.module.ae_obj(objs, z_override=wrong_z_obj)
                        fake_bg_img, _, _, _  = ae.module.ae_bg(bgs, z_override=wrong_z_bg)
                        fake_composite = fake_obj_img * soft_masks + fake_bg_img * (1 - soft_masks)
                        wm_attacked = val_attack_layer(fake_composite, itr, max_epochs=max_iterations)
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
                        _, atk_z_obj, _, _ = ae.module.ae_obj(wm_attacked * soft_masks)
                        _, atk_z_bg, _, _  = ae.module.ae_bg(wm_attacked * (1 - soft_masks))
                        _, clean_obj, _, _ = ae.module.ae_obj(images * soft_masks)
                        _, clean_bg, _, _  = ae.module.ae_bg(images * (1 - soft_masks))

                    norm_atk_obj = F.normalize(atk_z_obj, dim=1)
                    norm_cln_obj = F.normalize(clean_obj, dim=1)
                    shift_obj = norm_atk_obj - norm_cln_obj
                    
                    norm_atk_bg = F.normalize(atk_z_bg, dim=1)
                    norm_cln_bg = F.normalize(clean_bg, dim=1)
                    shift_bg = norm_atk_bg - norm_cln_bg

                    shift_total = shift_obj * mask_latent + shift_bg * (1 - mask_latent)

                    shift_flat = shift_total.flatten(1)
                    gt_latent = F.normalize(shift_flat, dim=1).view_as(shift_total)

                    if rank == 0 and step == 0:
                        print(f"  [DBG] mode={attack_mode} | gt_glob dist={torch.bincount(gt_glob, minlength=4)}")
                        print(f"  [DBG] pred_glob dist={torch.bincount(pred_glob.argmax(1), minlength=4)}")
                        print(f"  [DBG] shift mag={shift_total.abs().mean():.6f}")

                    if rank == 0 and step == 0:
                        print(f"  [VAL DBG] mode={attack_mode} | shift mag={shift_total.abs().mean():.6f}")

                    eval_extractor = ema_extractor if itr >= 5 else extractor.module
                    pred_int, pred_glob, z_mean, z_logvar, pred_fp = eval_extractor(wm_attacked)

                    
                    phase_2_start = int(0.50 * max_iterations)
                    ramp_length = config.get('phase_2_ramp_length', 10)
                    
                    # Before passing to criterion:
                    if attack_mode == 4:  # No watermark case
                        id_weight = 0.0
                        latent_weight = 0.0
                    elif itr < phase_2_start:
                        id_weight = 0.0
                        latent_weight = config['latent_weight']
                    else:
                        progress = min(1.0, (itr - phase_2_start) / ramp_length)
                        id_weight = progress * config.get('id_weight_max', 1.0)
                        latent_weight = config['latent_weight']



                    val_step_loss, bd = criterion(
                        pred_int, gt_int, pred_glob, gt_glob, 
                        z_mean, z_logvar, gt_latent, pred_fp,
                        identity_ids, ema_extractor, current_epoch=itr,
                        latent_weight=latent_weight, id_weight=id_weight,
                        collapse_weight=collapse_weight,
                        max_epochs=max_iterations 
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

            phase_2_start = int(0.50 * max_iterations)
            phase_3_start = int(0.75 * max_iterations)
            
            # Dynamically reset whenever we cross into Phase 2 or Phase 3
            # Optional: save a phase-transition checkpoint as a backup, but don't reset best_val
            if (itr) == phase_2_start or (itr) == phase_3_start:
                # Save a backup at the transition, but don't reset best_val
                save_checkpoint({
                    "epoch": itr+1,
                    "model": extractor.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val,          # unchanged
                    "config": config,
                    "ema_state": ema_extractor.state_dict(),
                }, os.path.join(ckpt_dir, f"phase_transition_{itr+1:03d}.pth"))
                
                patience_ctr = 0   # optional, reset early-stopping counter

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
                    "ema_state": ema_extractor.state_dict(),
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
                    "ema_state": ema_extractor.state_dict()
                }, os.path.join(ckpt_dir, f"epoch_{itr+1:03d}.pth"))

        # ═════════════════════════════════════════════════════════════════
        #  EMBEDDED DIAGNOSTICS TRIGGER
        # ═════════════════════════════════════════════════════════════════
        _test_epochs = set(config.get('test_epochs', []))
        _should_test = (itr + 1) in _test_epochs

        if rank == 0 and _should_test:
            print(f"\n{'='*70}\n  RUNNING FORENSIC DIAGNOSTICS AT EPOCH {itr+1}\n{'='*70}\n")
            run_forensic_diagnostics(
                epoch=itr + 1,
                config=config,
                ae_module=ae.module,
                injector=injector,
                ema_extractor=ema_extractor,
                semantic_masker=semantic_masker,
                attack_layer_det=val_attack_layer,
                tamper_layer=tamper_layer,
                device=rank
            )
            torch.cuda.empty_cache()

        if _should_test:
            dist.barrier()
        # ═════════════════════════════════════════════════════════════════

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
            "ema_state": ema_extractor.state_dict(),
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

    # save_config(config, utils.CONFIG_PATHdef main():
    print("Running Phase 0 dry-run verification...")
    # 1. Instantiate VAE
    ae = DualAutoencoder(latent_channels=256)
    print("VAE DualAutoencoder instantiated.")
    
    # 2. Instantiate Injector and save
    injector = SSWatermarkInjector(latent_channels=256, latent_h=28, latent_w=28)
    injector.eval()
    torch.save(injector.state_dict(), '/kaggle/working/injector_weights.pth')
    print("Saved injector_weights.pth")
    
    # 3. Instantiate Extractor
    extractor = ForensicIntegrityAnalyzer(fp_dim=256, pretrained=False)
    print("ForensicIntegrityAnalyzer instantiated.")
    
    print("Phase 0 verification complete!")

if __name__ == "__main__":
    main()
