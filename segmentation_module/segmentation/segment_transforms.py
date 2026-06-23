"""
Deterministic spatial transforms for image-mask pairs.
Uses nearest-neighbor interpolation for masks to prevent generating floating-point
artifacts (e.g., 0.5) on class boundaries.

normalize=False for the SAM path — sam.preprocess() handles normalisation
internally. Running TF.normalize here too would double-normalise and corrupt
the image encoder embeddings.
"""

import torch
import torchvision.transforms.functional as TF


class SegmentTransform:
    def __init__(self, img_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 p=0.5, normalize=True):
        self.image_size = [img_size, img_size] if isinstance(img_size, int) else img_size
        self.mean      = mean
        self.std       = std
        self.p         = p
        self.normalize = normalize

    def __call__(self, img, mask):
        img  = TF.resize(img,  self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        if torch.rand(1) < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if self.normalize:
            img = TF.normalize(img, self.mean, self.std)
        return img, mask