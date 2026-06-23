"""
Global I/O and checkpointing utilities.
Handles unwrapping DDP modules during state_dict loading to prevent key mismatches 
('module.conv1' vs 'conv1').
"""
import json
import torch
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP

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
