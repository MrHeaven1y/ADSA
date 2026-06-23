import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torchvision.models import vgg16, VGG16_Weights


class VGGFeatureExtractor(nn.Module):
  
  def __init__(self, device):
    super().__init__()
    self.device = device

    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(self.device).eval()

    
    self.slice1 = nn.Sequential(*list(vgg.children())[:4])
    self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
    self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
    self.slice4 = nn.Sequential(*list(vgg.children())[16:23])

    for p in self.parameters():
      p.requires_grad = False

    self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
    self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

  def preprocess(self, x):
    x = (x + 1.0) / 2.0 # [-1, 1] -> [0, 1]

    return (x - self.mean) / self.std

  def forward(self, x):
    x = self.preprocess(x)

    f1 = self.slice1(x)
    f2 = self.slice2(f1)
    f3 = self.slice3(f2)
    f4 = self.slice4(f3)

    return [f1, f2, f3, f4]
  
class PerceptualLoss(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.mse = nn.MSELoss()


  def forward(self, feats_rec, feats_target):
    
    loss = sum(
      self.mse(fr, ft)
      for fr, ft in zip(feats_rec, feats_target)
    )

    
    return loss


class StyleLoss(nn.Module):
  def __init__(self):
    super().__init__() 

    self.mse = nn.MSELoss()

  @staticmethod
  def gram_matrix(feat):
  
    B, C, H, W = feat.shape
    f = feat.view(B, C, H*W).to(torch.float32)
    gram = torch.bmm(f, f.transpose(1, 2))

    return gram / (C * H * W) 
  
  def forward(self, feats_rec, feats_target):

    with torch.amp.autocast('cuda', enabled=False):
      loss = 0.0
      for fr, ft in zip(feats_rec, feats_target):
        
        fr32 = fr.to(torch.float32)
        
        ft32 = ft.to(torch.float32)
        
        loss += self.mse(self.gram_matrix(fr32), self.gram_matrix(ft32))

    return loss

class MaskedL1Loss(nn.Module):
  
  def forward(self, rec, target, mask_expanded):
    
    loss = (mask_expanded * (rec - target).abs()).sum() / (mask_expanded.sum() + 1e-8)
    
    return loss

class SSIMLoss(nn.Module):
  def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):    
    super().__init__()

    self.window_size = window_size
    self.C1 = C1
    self.C2 = C2

    kernel_1d = self._gaussian_kernel(window_size, sigma)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)

    self.register_buffer('window', kernel_2d.unsqueeze(0).unsqueeze(0))

  @staticmethod
  def _gaussian_kernel(size, sigma):
  
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma ** 2))

    return g / g.sum()
  
  def _ssim_map(self, x, y):
    
    pad = self.window_size // 2
    w = self.window

    mu_x = F.conv2d(x, w, padding=pad, groups=1)
    mu_y = F.conv2d(y, w, padding=pad, groups=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x*x, w, padding=pad, groups=1) - mu_x2
    sig_y2 = F.conv2d(y*y, w, padding=pad, groups=1) - mu_y2
    sig_xy = F.conv2d(x*y, w, padding=pad, groups=1) - mu_xy

    num = (2 *  mu_xy + self.C1) * (2 * sig_xy + self.C2)
    den = (mu_x2 + mu_y2 + self.C1) * (sig_x2 + sig_y2 + self.C2)

    return num / den
  
  def forward(self, rec, target, mask_expanded):
    
    rec    = rec.float()      # ← force fp32 regardless of AMP
    target = target.float()
    mask_expanded   = mask_expanded.float()
    
    ssim_val = torch.zeros(1, device=rec.device)
    for c in range(rec.shape[1]):
      ssim_map = self._ssim_map(
        rec[:, c:c+1, :, :],
        target[:, c:c+1, :, :]
      )

      m = mask_expanded[:, c:c+1, :, :]

      ssim_val = ssim_val + (m * ssim_map).sum() / (m.sum() + 1e-8)

    ssim_val = ssim_val / rec.shape[1]

    return 1.0 - ssim_val

class EdgeLoss(nn.Module):
  def __init__(self):
    super().__init__()

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    self.register_buffer('sobel_x', sobel_x)
    self.register_buffer('sobel_y', sobel_y)

  def _gradient_magnitude(self, img):
    
    B, C, H, W = img.shape
    x = img.view(B*C, 1, H, W)

    gx = F.conv2d(x, self.sobel_x, padding=1)
    gy = F.conv2d(x, self.sobel_y, padding=1)

    mag = (gx**2 + gy**2 + 1e-8).sqrt()

    return mag.view(B, C, H, W)
  
  def forward(self, rec, target, mask_expanded):
    

    grad_rec = self._gradient_magnitude(rec)
    grad_target = self._gradient_magnitude(target)

    diff = (grad_rec - grad_target).abs()
    loss = (mask_expanded * diff).sum() / (mask_expanded.sum() + 1e-8)
    return loss

class MaskedCombinedLoss(nn.Module):
  def __init__(self, 
               device, 
               lambda_l1=1.0, 
               lambda_ssim=0.2, 
               lambda_perceptual=0.05,
               lambda_edge=0.05
  ):

    super().__init__()


    self.lambda_l1 = lambda_l1
    self.lambda_ssim = lambda_ssim
    self.lambda_perceptual = lambda_perceptual
    # self.lambda_style = lambda_style
    self.lambda_edge = lambda_edge

    self.vgg = VGGFeatureExtractor(device)

    self.l1 = MaskedL1Loss()
    self.ssim = SSIMLoss()
    self.perceptual = PerceptualLoss()
    # self.style = StyleLoss()
    self.edge = EdgeLoss()

  def forward(self, rec, target, mask):
    
    mask_expanded = mask.expand_as(rec)

    loss_l1 = self.l1(rec, target, mask_expanded)
    loss_ssim = self.ssim(rec, target, mask_expanded)
    loss_edge = self.edge(rec, target, mask_expanded)

    masked_rec = rec * mask_expanded
    masked_target = target * mask_expanded

    masked_rec = F.interpolate(masked_rec, (224,224), mode="bilinear", align_corners=False)
    masked_target = F.interpolate(masked_target, (224,224), mode="bilinear", align_corners=False)

    feats_rec = self.vgg(masked_rec)

    with torch.inference_mode():
      feats_target = self.vgg(masked_target)

    loss_perc = self.perceptual(feats_rec, feats_target)
    # loss_style = self.style(feats_rec.float(), feats_target.float())


    total = (
      self.lambda_l1 * loss_l1 +
      self.lambda_ssim * loss_ssim +
      self.lambda_perceptual * loss_perc +
      # self.lambda_style * loss_style +
      self.lambda_edge * loss_edge 
    )

    breakdown = {
      'l1': loss_l1.item(),
      'ssim': loss_ssim.item(),
      'perceptual': loss_perc.item(),
      # 'style': loss_style.item(),
      'edge': loss_edge.item()

    }

    return total, breakdown
  

class SimpleCombinedLoss(nn.Module):
    """
    Two losses only:
      total = lambda_l1 * MaskedL1  +  lambda_perceptual * VGG(relu2_2, relu3_3)

    SSIM removed: redundant alongside L1, adds AMP instability at boundaries.
    """
    def __init__(self, device, lambda_l1=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_l1         = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.vgg               = VGGFeatureExtractor(device)
        self.l1                = MaskedL1Loss()
        self.mse               = nn.MSELoss()

    def forward(self, rec, target, mask):
        mask_exp = mask.expand_as(rec)   # [B,1,H,W] → [B,3,H,W]

        # ── L1 ────────────────────────────────────────────────────────────────
        loss_l1 = self.l1(rec, target, mask_exp)

        # ── Perceptual ────────────────────────────────────────────────────────
        # img_size=224 → already VGG-ready, no interpolation needed
        m_rec    = rec    * mask_exp
        m_target = target * mask_exp

        feats_rec = self.vgg(m_rec)
        with torch.inference_mode():           # FIX 6: was .detach() — still ran grad graph
            feats_tgt = self.vgg(m_target)

        loss_perc = sum(self.mse(fr, ft) for fr, ft in zip(feats_rec, feats_tgt))

        total = self.lambda_l1 * loss_l1 + self.lambda_perceptual * loss_perc

        return total, {'l1': loss_l1.item(), 'perceptual': loss_perc.item()}
