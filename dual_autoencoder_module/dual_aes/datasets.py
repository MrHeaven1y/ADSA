"""
Data loaders and caching logic for the Oxford-IIIT Pet dataset.

Converts trimaps to binary masks (Foreground vs. Background/Border).
Includes aggressive RAM caching (`cache_ram=True`) to bypass disk I/O bottlenecks 
during epoch iterations. Use `benchmark_num_workers` to tune DataLoader prefetches.
"""

import os
import time
import torch
from collections import OrderedDict
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader, random_split


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.
    Trimaps: 1=Foreground, 2=Background, 3=Border → binarised to 0/1.
    """
    def __init__(self, img_dir, mask_dir, transforms=None,
                 max_samples=None, cache_ram=True, max_cache_size=7500,
                 sam_mode=False):
        
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
        self.sam_mode = sam_mode

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
        except RuntimeError:
            from PIL import Image as PILImage
            
            img  = TF.pil_to_tensor(PILImage.open(self.img_list[idx]).convert('RGB'))
            
        if self.sam_mode:
            
            from PIL import Image as PILImage
            
            mask = TF.pil_to_tensor(PILImage.open(self.mask_list[idx]).convert('L'))
        else:
            try:
                mask = read_image(self.mask_list[idx])
            except RuntimeError:
                from PIL import Image as PILImage
                mask = TF.pil_to_tensor(PILImage.open(self.mask_list[idx]).convert('L'))

        return img, mask

    def __len__(self): return self.len

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        if not self.sam_mode:
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
                  max_samples=None, split_size=0.85, cache_ram=True,
                  sam_mode=False):
    
    dataset = OxfordPetDataset(img_dir, mask_dir,
                               transforms=None,
                               max_samples=max_samples,
                               cache_ram=cache_ram,
                               sam_mode=sam_mode)
    
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

