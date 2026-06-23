import torch


SAM_INPUT_SIZE = 1024   # SAM encoder always works at 1024×1024
SAM_LOW_RES    = 256  

def _preprocess_batch(images: torch.Tensor, sam) -> torch.Tensor:
    """
    images : [B, 3, H, W]  uint8  [0, 255]
    returns: [B, 3, 1024, 1024]  float  normalised + padded to 1024

    Calls sam.preprocess() per image — the exact same function SamPredictor
    uses internally. This normalises with pixel_mean/pixel_std then pads
    (not squashes) to (1024 X 1024).
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
      1. sam.preprocess()   — normalise + pad images to (1024 x 1024)
      2. image_encoder      — frozen, wrapped in no_grad
      3. prompt_encoder     — frozen, per-sample centroid + bbox prompts
      4. mask_decoder       — trained, outputs (256 X 256) low-res logits

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
