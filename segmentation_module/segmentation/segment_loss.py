"""
Composite loss function for binary segmentation.
Balances pixel-level accuracy (BCE), foreground/background overlap (Dice), 
and confidence calibration (KL Divergence).

Note: KLDivLoss uses 'mean' reduction. 'batchmean' scales with HxW, 
which blows up gradients (~50k factor at 224px).
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceKLLoss(nn.Module):
    """
    BCE (0.4) + Dice (0.4) + KL Divergence (0.2)

    BCE  — standard per-pixel binary cross-entropy on logits
    Dice — overlap loss, robust to foreground/background class imbalance
    KL   — distribution-level loss; penalises miscalibrated confidence.
           A model that is wrong AND overconfident gets hit harder than one
           that is wrong but uncertain. Improves mask boundary quality.
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.4, kl_weight=0.2, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.kl_weight   = kl_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()
        # 'mean' divides by B×H×W — 'batchmean' only divides by B,
        # which causes KL to explode by a factor of H×W (~50k at 224px)
        self.kl          = nn.KLDivLoss(reduction='mean') # Keeping reduction as mean because batchmean cause loss goes up higher due to low  batch size

    def forward(self, logits, targets):
        """
        logits  : [B, 1, H, W]  raw unactivated predictions
        targets : [B, 1, H, W]  binary float {0.0, 1.0}
        """
        # ── BCE ───────────────────────────────────────────────────────────────
        loss_bce = self.bce(logits, targets)

        # ── Dice ──────────────────────────────────────────────────────────────
        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice      = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss_dice = 1.0 - dice.mean()

        # ── KL Divergence ─────────────────────────────────────────────────────
        # KLDivLoss(input, target): input must be log-probs, target must be probs
        # log_sigmoid(logits) = log(P(foreground)) directly
        log_probs = F.logsigmoid(logits)
        loss_kl   = self.kl(log_probs, targets.clamp(0.0, 1.0))

        total = (self.bce_weight  * loss_bce
               + self.dice_weight * loss_dice
               + self.kl_weight   * loss_kl)

        return total, {
            'bce':  loss_bce.item(),
            'dice': loss_dice.item(),
            'kl':   loss_kl.item(),
        }


class BCEDiceLoss(nn.Module):
    """
    BCE (bce_weight) + Dice (dice_weight).

    logits  : [B, 1, 256, 256]  SAM low-res decoder output (before sigmoid)
    targets : [B, 1, H,   W  ]  binary float {0.0, 1.0}

    targets are resized to (256 x 256) before loss (nearest-neighbour) to match
    SAM's low-res output. Upsampling back to img_size happens only for IoU.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Resize GT to match SAM's 256×256 low-res output
        if logits.shape != targets.shape:
            targets = F.interpolate(targets, size=logits.shape[-2:], mode='nearest')

        loss_bce  = self.bce(logits, targets)

        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice      = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss_dice = 1.0 - dice.mean()

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total, {'bce': loss_bce.item(), 'dice': loss_dice.item()}

