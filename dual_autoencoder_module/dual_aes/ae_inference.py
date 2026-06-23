"""
Offline inference pipeline for Stage-2 DualAutoencoder.

Two entry points:
  infer_single() — one image at a time, useful for debugging
  infer_batch()  — full folder, uses DataLoader for throughput

Both save rec_obj and rec_bg PNGs to out_dir, and optionally dump
latent tensors as .pt files for downstream stages.
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from .ae_models import DualAutoencoder
from .ae_transforms import SegmentTransform


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device, latent_channels=256):
    model = DualAutoencoder(latent_channels=latent_channels).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    # checkpoint may be a full training state or just weights
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  [inference] loaded → {checkpoint_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def denormalize(tensor):
    """[-1, 1] → [0, 1], clamped."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def save_outputs(out_dir, stem, rec_obj, rec_bg, z_obj, z_bg, save_latents):
    os.makedirs(out_dir, exist_ok=True)
    # FIX: was saving rec_obj twice — rec_bg never written
    save_image(denormalize(rec_obj), os.path.join(out_dir, f"{stem}_rec_obj.png"))
    save_image(denormalize(rec_bg),  os.path.join(out_dir, f"{stem}_rec_bg.png"))
    if save_latents:
        torch.save(
            {'z_obj': z_obj.cpu(), 'z_bg': z_bg.cpu()},
            os.path.join(out_dir, f"{stem}_latents.pt")
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (used by infer_batch)
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    """
    Pairs images with their masks for batch inference.
    If mask_dir is None, uses all-ones mask (no masking).

    mask filenames are expected to match image stems after stripping '_mask':
      e.g. Abyssinian_1.jpg  ↔  Abyssinian_1_mask.png
    or exact stem match if no '_mask' suffix present.
    """
    def __init__(self, img_dir, img_size, mask_dir=None):
        self.transform = SegmentTransform(img_size, p=0.0)
        self.img_size  = img_size
        self.mask_dir  = mask_dir
        self.items     = []

        img_files = sorted(
            f for f in os.listdir(img_dir)
            if not f.startswith('.')
        )

        if mask_dir:
            mask_dict = {
                os.path.splitext(f)[0].replace('_mask', ''): os.path.join(mask_dir, f)
                for f in sorted(os.listdir(mask_dir))
                if not f.startswith('.')
            }
            for f in img_files:
                stem = os.path.splitext(f)[0]
                if stem in mask_dict:
                    self.items.append((
                        os.path.join(img_dir, f),
                        mask_dict[stem],
                        stem
                    ))
        else:
            for f in img_files:
                self.items.append((
                    os.path.join(img_dir, f),
                    None,
                    os.path.splitext(f)[0]
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.items[idx]

        img = read_image(img_path).float().div(255.0)

        if img.shape[0] == 4: img = img[:3]
        if img.shape[0] == 1: img = img.repeat(3, 1, 1)

        if mask_path is not None:
            mask = read_image(mask_path).float().div(255.0)
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones(1, img.shape[1], img.shape[2])

        img, mask = self.transform(img, mask)
        # FIX: was returning (img, mask) — infer_batch unpacks 3 values including stem
        return img, mask, stem


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE IMAGE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_single(model, img_path, out_dir,
                 img_size=224, device='cuda',
                 mask_path=None, save_latents=True):
    """
    Run inference on one image. Useful for quick checks during development.

    mask_path=None → uses all-ones mask (reconstructs full image in both streams).
    """
    transform = SegmentTransform(img_size, p=0.0)

    img = read_image(img_path).float().div(255.0)
    if img.shape[0] == 4: img = img[:3]
    if img.shape[0] == 1: img = img.repeat(3, 1, 1)

    if mask_path is not None:
        mask = read_image(mask_path).float().div(255.0)
        mask = (mask > 0.5).float()
    else:
        mask = torch.ones(1, img.shape[1], img.shape[2])

    img, mask = transform(img, mask)
    img  = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    objs = img * mask
    bgs  = img * (1.0 - mask)

    with autocast(device_type=device):
        rec_obj, rec_bg, z_obj, z_bg = model(objs, bgs)

    stem = Path(img_path).stem
    save_outputs(out_dir, stem, rec_obj, rec_bg, z_obj, z_bg, save_latents)
    print(f"  [inference] saved outputs for '{stem}' → {out_dir}")

    return {
        'rec_obj': rec_obj.cpu(),
        'rec_bg':  rec_bg.cpu(),
        'z_obj':   z_obj.cpu(),
        'z_bg':    z_bg.cpu(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_batch(model, img_dir, out_dir,
                img_size=224, batch_size=8, num_workers=4,
                device='cuda', mask_dir=None, save_latents=True):
    """
    Run inference on every image in img_dir. Saves mask-decomposed
    reconstructions and optionally latent .pt files per image.

    Returns a list of dicts with stem + latent tensors for all images.
    """
    dataset = InferenceDataset(img_dir, img_size, mask_dir=mask_dir)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == 'cuda')
    )

    os.makedirs(out_dir, exist_ok=True)
    all_latents = []

    for imgs, masks, stems in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        objs = imgs * masks
        bgs  = imgs * (1.0 - masks)

        with autocast(device_type=device):
            rec_obj, rec_bg, z_obj, z_bg = model(objs, bgs)

        for i, stem in enumerate(stems):
            save_outputs(
                out_dir, stem,
                rec_obj[i:i+1], rec_bg[i:i+1],
                z_obj[i:i+1],   z_bg[i:i+1],
                save_latents
            )
            all_latents.append({
                'stem':  stem,
                'z_obj': z_obj[i].cpu(),
                'z_bg':  z_bg[i].cpu(),
            })

    print(f"  [inference] done. {len(dataset)} images → {out_dir}")
    return all_latents


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT = '/kaggle/working/best_model/best_weights.pth'
    IMG_SIZE   = 224
    OUT_DIR    = '/kaggle/working/inference_outputs'

    model = load_model(CHECKPOINT, DEVICE)

    # ── single image ──────────────────────────────────────────────────────────
    # infer_single(
    #     model,
    #     img_path  = '/path/to/image.jpg',
    #     mask_path = '/path/to/mask.png',   # None to skip masking
    #     out_dir   = OUT_DIR,
    #     img_size  = IMG_SIZE,
    #     device    = DEVICE,
    # )

    # ── batch folder ──────────────────────────────────────────────────────────
    # infer_batch(
    #     model,
    #     img_dir    = '/kaggle/input/oxford-iiit-pet/images',
    #     mask_dir   = '/kaggle/working/generated_masks',
    #     out_dir    = OUT_DIR,
    #     img_size   = IMG_SIZE,
    #     batch_size = 8,
    #     num_workers= 4,
    #     device     = DEVICE,
    # )