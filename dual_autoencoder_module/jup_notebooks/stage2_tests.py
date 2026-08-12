# stage2_tests.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_psnr(rec, target, data_range=2.0):
    mse = F.mse_loss(rec, target)
    return (20 * torch.log10(torch.tensor(data_range, device=rec.device)) - 10 * torch.log10(mse + 1e-8)).item()

def to_numpy(t):
    if t.min() < 0:
        t = t * 0.5 + 0.5
    return (t[0].cpu().permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()

def run_tests(model, epoch, device, config):
    """Runs visual reconstruction, watermark sensitivity, and skip dropout tests."""
    import stage2_ae_training as ae_module 
    
    print(f"\n{'='*70}\n  RUNNING STAGE 2 DIAGNOSTICS AT EPOCH {epoch}\n{'='*70}\n")
    
    ae = model.module if hasattr(model, 'module') else model
    ae.eval()
    
    img_dir = config['images_dir']
    mask_dir = config['mask_dir']
    img_size = config.get('img_size', 224)
    
    print(f"  [Diag] Loading dataset from config: {img_dir}")
    dataset = ae_module.OxfordPetDataset(
        img_dir, mask_dir, 
        transforms=ae_module.SegmentTransform(img_size, p=0.0),
        cache_ram=False
    )
    
    if len(dataset) == 0:
        print("  [Diag] Dataset is empty! Check paths in config. Skipping test.")
        return
        
    img, mask = dataset[0]
    img, mask = img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
    print("  [Diag] Successfully loaded test image.")

    objs, bgs = img * mask, img * (1.0 - mask)

    with torch.inference_mode():
        # 1. Full reconstruction WITH skips (also capture latent variables)
        rec_obj, rec_bg, z_obj, z_bg, mu_obj, logvar_obj, mu_bg, logvar_bg = ae(
            objs, bgs, compute_aux=False, compute_wm=False
        )
        full_rec = rec_obj * mask + rec_bg * (1.0 - mask)
        psnr = compute_psnr(full_rec, img)
        print(f"  [Diag] PSNR with Skips: {psnr:.2f} dB")

        # 2. WATERMARK SENSITIVITY TEST – inject random noise into latent mu
        torch.manual_seed(42)
        fake_wm = torch.randn_like(mu_obj) * 0.25          # [1, 256, 28, 28]
        wm_injected = mu_obj + fake_wm

        # Decode through the full decoder (with skips)
        s0 = ae.ae_obj.stem(objs)
        s1 = ae.ae_obj.down1(s0)
        s2 = ae.ae_obj.down2(s1)
        s3 = ae.ae_obj.down3(s2)
        s3_proj = ae.ae_obj.skip_proj_s3(s3)
        s2_proj = ae.ae_obj.skip_proj_s2(s2)
        s1_proj = ae.ae_obj.skip_proj_s1(s1)
        x = ae.ae_obj.up1(torch.cat([wm_injected, s3_proj], dim=1))
        x = ae.ae_obj.up2(torch.cat([x, s2_proj], dim=1))
        x = ae.ae_obj.up3(torch.cat([x, s1_proj], dim=1))
        rec_injected_obj = ae.ae_obj.output(x)

        wm_diff = (rec_injected_obj - rec_obj).abs().mean().item()
        print(f"  [Diag] Watermark Distortion: {wm_diff:.6f} (Should be > 0.001)")

        # 3. SKIP DROPOUT TEST – use force_skip_drop=True (works in inference)
        rec_obj_dropped, rec_bg_dropped, *_ = ae(objs, bgs, force_skip_drop=True)
        full_rec_dropped = rec_obj_dropped * mask + rec_bg_dropped * (1.0 - mask)
        psnr_dropped = compute_psnr(full_rec_dropped, img)
        print(f"  [Diag] PSNR with Skips Dropped: {psnr_dropped:.2f} dB")

        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(to_numpy(img)); axes[0].set_title("Original")
        axes[1].imshow(to_numpy(full_rec)); axes[1].set_title(f"Reconstruction\nPSNR: {psnr:.2f} dB")
        axes[2].imshow(to_numpy(rec_injected_obj)); axes[2].set_title(f"Watermarked Latent\nDistort: {wm_diff:.6f}")
        axes[3].imshow(to_numpy(full_rec_dropped)); axes[3].set_title(f"Skips Dropped\nPSNR: {psnr_dropped:.2f} dB")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        
        save_dir = "/kaggle/working/stage2_diags"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"stage2_diag_epoch_{epoch:03d}.png")
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [Diag] Visualization saved → {save_path}\n{'='*70}\n")

if __name__ == '__main__':
    import json
    print("Running standalone test...")
    
    config_path = '/kaggle/working/master_config.json'
    with open(config_path, 'r') as f:
        master_json = json.load(f)
    config = master_json['stage2_autoencoder']
    config['images_dir'] = master_json['paths']['images_dir']
    config['mask_dir'] = master_json['paths']['mask_dir']
    
    import stage2_ae_training as ae_module
    ae = ae_module.DualAutoencoder(latent_channels=config.get('latent_channels', 256)).to(device)
    weights_path = '/kaggle/working/best_model_ae/best_weights.pth'
    
    if os.path.exists(weights_path):
        ae.load_state_dict(torch.load(weights_path, map_location=device))
        run_tests(ae, 0, device, config)
    else:
        print(f"Could not find weights at {weights_path}")