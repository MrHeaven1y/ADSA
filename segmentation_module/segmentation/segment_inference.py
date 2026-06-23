"""
Offline inference pipeline for Stage-1.

Loads frozen SegUNet weights to generate and dump binary masks to disk.
These saved PNGs act as the static dataset inputs for the downstream dual-stream autoencoder.
"""
import os
import torch
from .segment_models import SegUNet
from torchvision.io import read_image
from torchvision import transforms as T
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from mobile_sam import sam_model_registry
from .segment_models import SegUNet
from .segment_sam import _preprocess_batch, SAM_INPUT_SIZE


def generate_masks_unet(weights_path, img_dir, out_mask_dir, img_size=224, threshold=0.5, device='cuda'):
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


def generate_masks_sam(weights_path, img_dir, out_mask_dir,
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
