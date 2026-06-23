import torch
import torch.nn.functional as F

def iou_score(logits, targets, img_size=None, threshold=0.5):
    """
    Mean IoU over batch. For logging only - not used in loss.
    img_size: pass for SAM — logits are (256 x 256), need upsampling to match GT.
              leave None for UNet — logits are already at img_size.
    """

    pred = logits.float()
    
    if img_size is not None:
    
        pred = F.interpolate(pred, size=(img_size, img_size),
                             mode='bilinear', align_corners=False)
    
    pred = torch.sigmoid(pred) > threshold

    inter = (pred & targets.bool()).float().sum(dim=(1, 2, 3))
    union = (pred | targets.bool()).float().sum(dim=(1, 2, 3))

    return ((inter + 1e-8) / (union + 1e-8)).mean().item()

