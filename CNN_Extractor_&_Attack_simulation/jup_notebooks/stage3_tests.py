# stage3_test.py
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image

def run_forensic_diagnostics(epoch, config, ae_module, injector, ema_extractor,
                              semantic_masker, attack_layer_det, tamper_layer, device):
    """
    Runs the 4 forensic tests mid-training and saves results to /kaggle/working/diagnostics/epoch_XXX/
    """
    diag_dir = f"/kaggle/working/diagnostics/epoch_{epoch:03d}"
    os.makedirs(diag_dir, exist_ok=True)

    log_path = os.path.join(diag_dir, "diagnostics_log.txt")
    log_file = open(log_path, 'w')
    _orig_stdout = sys.stdout

    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, text):
            for f in self.files: f.write(text); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    sys.stdout = _Tee(_orig_stdout, log_file)

    def _to_numpy(t):
        t = t[0].detach().cpu().permute(1, 2, 0)
        return ((t * 0.5 + 0.5) * 255).clamp(0, 255).byte().numpy()

    def _compute_dice(pred, gt, eps=1e-6):
        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        return ((2 * inter + eps) / (union + eps)).item()

    status_map = {0: "Authentic", 1: "Tampered", 2: "Fake Identity", 3: "No Watermark"}

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    identity_threshold = config.get('identity_threshold', 0.55)
    image_name = config.get('test_image', 'Abyssinian_100')
    img_dir = config.get('images_dir', '/kaggle/working/clean_pet_data/images')
    mask_dir = config.get('mask_dir', '/kaggle/working/clean_pet_data/masks')
    max_iters = config.get('max_iterations', 80)
    test_attack_epoch = max(max_iters, 50)

    ae_module.eval(); injector.eval(); ema_extractor.eval()
    semantic_masker.eval(); attack_layer_det.eval(); tamper_layer.eval()

    try:
        print("=" * 70)
        print(f"  FORENSIC DIAGNOSTICS — EPOCH {epoch}")
        print(f"  Test image: {image_name} | Threshold: {identity_threshold}")
        print("=" * 70)

        # 1. Load Image + Mask
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            c = os.path.join(img_dir, f"{image_name}{ext}")
            if os.path.isfile(c): img_path = c; break
        if img_path is None:
            orig_img_dir = "/kaggle/input/oxford-iiit-pet/images"
            for ext in ['.jpg', '.jpeg', '.png']:
                c = os.path.join(orig_img_dir, f"{image_name}{ext}")
                if os.path.isfile(c): img_path = c; break

        mask_path = None
        for pat in [f"{image_name}_mask.png", f"{image_name}.png", f"{image_name}_mask.jpg"]:
            c = os.path.join(mask_dir, pat)
            if os.path.isfile(c): mask_path = c; break
        if mask_path is None:
            orig_mask_dir = "/kaggle/input/oxford-iiit-pet/annotations/trimaps"
            for pat in [f"{image_name}.png", f"{image_name}_mask.png"]:
                c = os.path.join(orig_mask_dir, pat)
                if os.path.isfile(c): mask_path = c; break

        img_raw = read_image(img_path)
        mask_raw = read_image(mask_path)
        if img_raw.shape[0] == 4: img_raw = img_raw[:3]
        elif img_raw.shape[0] == 1: img_raw = img_raw.repeat(3, 1, 1)
        if mask_raw.shape[0] > 1: mask_raw = mask_raw[0:1]
        mask_raw = (mask_raw > 127).float()
        img_raw = img_raw.float() / 255.0

        img = TF.resize(img_raw, [224, 224])
        mask = TF.resize(mask_raw, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(img, [0.5]*3, [0.5]*3)
        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        # 2. Encode
        with torch.inference_mode():
            objs, bgs = img * mask, img * (1 - mask)
            soft_masks = TF.gaussian_blur(mask, [15, 15], [5.0, 5.0])
            mask_obj, mask_bg = semantic_masker(objs), semantic_masker(bgs)
            zo, sko = ae_module.enc_obj(objs)
            zb, skb = ae_module.enc_bg(bgs)
            mask_latent = F.interpolate(soft_masks, (28, 28), mode='nearest')

        # 3. Authentic Watermark
        wm_idx = 0
        with torch.inference_mode():
            base_w1 = injector.base_w1_pool[wm_idx:wm_idx+1].to(device)
            base_w2 = injector.base_w2_pool[wm_idx:wm_idx+1].to(device)
            wm_obj_z = injector.inject(zo, base_w1, alpha=0.225, semantic_mask=mask_obj)
            wm_bg_z = injector.inject(zb, base_w2, alpha=0.0925, semantic_mask=mask_bg)
            z_auth = wm_obj_z * mask_latent + wm_bg_z * (1 - mask_latent)
            sk_composite = ae_module.blend_skips(sko, skb, soft_masks)
            wm_image = ae_module.shared_decoder(z_auth, sk_composite)
            z_clean = zo * mask_latent + zb * (1 - mask_latent)
            clean_image = ae_module.shared_decoder(z_clean, sk_composite)

        # 4. Build Samples
        samples = []
        with torch.inference_mode():
            authentic_attacked = attack_layer_det(wm_image, current_epoch=test_attack_epoch, max_epochs=max(80, max_iters))
            samples.append({'name': 'Authentic Attack', 'image': authentic_attacked, 'gt_class': 0, 'gt_id': wm_idx, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})
            
            tampered_img, tamper_gt = tamper_layer(authentic_attacked, seed=42)
            samples.append({'name': 'Tampered', 'image': tampered_img, 'gt_class': 1, 'gt_id': wm_idx, 'gt_mask': tamper_gt})
            
            fake_idx = 17
            fake_w1 = injector.fake_w1_pool[fake_idx:fake_idx+1].to(device)
            fake_w2 = injector.fake_w2_pool[fake_idx:fake_idx+1].to(device)
            fake_obj_z = injector.inject(zo, fake_w1, alpha=0.225, semantic_mask=mask_obj)
            fake_bg_z = injector.inject(zb, fake_w2, alpha=0.0925, semantic_mask=mask_bg)
            z_fake = fake_obj_z * mask_latent + fake_bg_z * (1 - mask_latent)
            fake_image = ae_module.shared_decoder(z_fake, sk_composite)
            fake_image = attack_layer_det(fake_image, current_epoch=test_attack_epoch, max_epochs=max(80, max_iters))
            samples.append({'name': 'Fake Identity', 'image': fake_image, 'gt_class': 2, 'gt_id': fake_idx, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})
            
            samples.append({'name': 'No Watermark', 'image': clean_image, 'gt_class': 3, 'gt_id': -1, 'gt_mask': torch.zeros(1, 1, 56, 56, device=device)})

        # TEST 1
        print("\nTEST 1: FORENSIC EVALUATION")
        with torch.inference_mode():
            for sample in samples:
                pred_int, pred_glob, _, _, pred_fp = ema_extractor(sample['image'])
                pred_class = torch.argmax(pred_glob, dim=1).item()
                dice = _compute_dice(torch.sigmoid(pred_int), sample['gt_mask'])
                if sample['gt_id'] != -1:
                    target_center = F.normalize(ema_extractor.identity_centers[sample['gt_id']], dim=0)
                    cos_sim = F.cosine_similarity(pred_fp, target_center.unsqueeze(0)).item()
                else:
                    cos_sim = F.cosine_similarity(pred_fp, ema_extractor.identity_centers[0:1]).item()
                print(f"  {sample['name']:<18} | GT: {status_map[sample['gt_class']]} | Pred: {status_map[pred_class]} | Dice: {dice:.4f} | Sim: {cos_sim:.4f}")

        # TEST 2
        print("\nTEST 2: VISUALIZATION + IDENTITY MATCHING")
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4 * len(samples)))
            if len(samples) == 1: axes = [axes]
            
            with torch.inference_mode():
                identity_bank = F.normalize(ema_extractor.identity_centers, dim=1)
                for row, sample in enumerate(samples):
                    pred_int, pred_glob, _, _, pred_fp = ema_extractor(sample['image'])
                    pred_fp = F.normalize(pred_fp, dim=1)
                    sims = torch.matmul(pred_fp, identity_bank.T)
                    max_sim, predicted_id = sims.max(dim=1)
                    max_sim_val = max_sim.item()
                    predicted_id_val = predicted_id.item()
                    
                    if max_sim_val < identity_threshold: 
                        predicted_id_val = -1
                    
                    dice = _compute_dice(torch.sigmoid(pred_int), sample['gt_mask'])
                    
                    axes[row][0].imshow(_to_numpy(img)); axes[row][0].set_title("Original")
                    axes[row][1].imshow(_to_numpy(wm_image)); axes[row][1].set_title("Authentic WM")
                    axes[row][2].imshow(_to_numpy(sample['image'])); axes[row][2].set_title(sample['name'])
                    im = axes[row][3].imshow(torch.sigmoid(pred_int)[0,0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                    # FIX: Removed .item() from predicted_id_val
                    axes[row][3].set_title(f"GT_ID:{sample['gt_id']} | PRED_ID:{predicted_id_val}\nCONF:{max_sim_val:.3f} | DICE:{dice:.3f}")
                    plt.colorbar(im, ax=axes[row][3], fraction=0.046, pad=0.04)
                    for c in range(4): axes[row][c].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(diag_dir, "forensic_evaluation_results.png"), bbox_inches='tight')
            plt.close(fig)
            print("  Saved visualization.")
        except Exception as e:
            print(f"  Visualization failed: {e}")

        # TEST 3
        print("\nTEST 3: SIMILARITY STATS")
        with torch.inference_mode():
            _, _, _, _, pred_fp = ema_extractor(authentic_attacked)
            pred_fp = F.normalize(pred_fp, dim=1)
            identity_bank = F.normalize(ema_extractor.identity_centers, dim=1)
            sims = torch.matmul(pred_fp, identity_bank.T)
            print(f"  Pred Norm: {pred_fp.norm(dim=1).mean().item():.4f}")
            print(f"  Sim Stats -> Min: {sims.min().item():.4f} | Max: {sims.max().item():.4f} | Mean: {sims.mean().item():.4f}")
            topk_vals, topk_idx = torch.topk(sims[0], k=10)
            print("  Top-10 Matches:")
            for r in range(10):
                # FIX: Safe casting to int just in case
                idx_val = topk_idx[r].item() if torch.is_tensor(topk_idx[r]) else int(topk_idx[r])
                sim_val = topk_vals[r].item() if torch.is_tensor(topk_vals[r]) else float(topk_vals[r])
                print(f"    {r+1:02d} | ID={idx_val:3d} | SIM={sim_val:.4f}")

        # TEST 4
        print("\nTEST 4: LATENT SPACE SANITY CHECK")
        with torch.inference_mode():
            z_clean_obj, _ = ae_module.enc_obj(objs)
            z_clean_bg, _ = ae_module.enc_bg(bgs)
            z_wm_obj = injector.inject(z_clean_obj, base_w1, alpha=0.225, semantic_mask=mask_obj)
            z_wm_bg = injector.inject(z_clean_bg, base_w2, alpha=0.0925, semantic_mask=mask_bg)
            
            diff_obj = z_wm_obj - z_clean_obj
            diff_bg = z_wm_bg - z_clean_bg
            obj_ratio = diff_obj.pow(2).mean().item() / (z_clean_obj.pow(2).mean().item() + 1e-8)
            bg_ratio = diff_bg.pow(2).mean().item() / (z_clean_bg.pow(2).mean().item() + 1e-8)
            print(f"  OBJ Energy Ratio: {obj_ratio:.8f} | BG Energy Ratio: {bg_ratio:.8f}")
            
            z_auth_t4 = z_wm_obj * mask_latent + z_wm_bg * (1 - mask_latent)
            wm_image_t4 = ae_module.shared_decoder(z_auth_t4, sk_composite)
            
            pred_int_t4, pred_glob_t4, _, _, pred_fp_t4 = ema_extractor(wm_image_t4)
            pred_fp_t4 = F.normalize(pred_fp_t4, dim=1)
            sims_t4 = torch.matmul(pred_fp_t4, identity_bank.T)
            conf_t4, pred_id_t4 = sims_t4.max(dim=1)
            print(f"  Pred Status: {status_map[pred_glob_t4.argmax(1).item()]} | GT ID: {wm_idx} | Rec ID: {pred_id_t4.item()} | Conf: {conf_t4.item():.4f}")

        print("\n" + "=" * 70)
        print(f"  DIAGNOSTICS COMPLETE — Results in {diag_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Diagnostic failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        sys.stdout = _orig_stdout
        log_file.close()
        torch.cuda.empty_cache()