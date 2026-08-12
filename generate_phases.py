import os
import re

base_file = r'c:\Workspace\Projects\DeepLearning\ADSA--simplified\CNN_Extractor_&_Attack_simulation\jup_notebooks\train_extractor_pretrained_synced.py'
out_dir = r'c:\Workspace\Projects\DeepLearning\ADSA--simplified\CNN_Extractor_&_Attack_simulation\jup_notebooks'

with open(base_file, 'r', encoding='utf-8') as f:
    base_content = f.read()

# VAE is already synced in train_extractor_pretrained_synced.py
base_with_vae = base_content

# Now we generate the 5 files by manipulating base_with_vae

def write_file(name, content):
    path = os.path.join(out_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created {name}")

# --- Phase 0: Verify ---
phase0 = base_with_vae.replace(
    'if __name__ == "__main__":\n    import multiprocessing\n    multiprocessing.set_start_method("spawn", force=True)\n    parser = argparse.ArgumentParser()',
    '''if __name__ == "__main__":
    print("--- Phase 0: Verify ---")
    
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()'''
)

# For Phase 0, we just want to instantiate the injector and save it, and instantiate the model to verify.
# Let's replace the training loop in __main__ for phase 0.
main_start = phase0.find('def main():')
phase0_main = '''def main():
    print("Running Phase 0 dry-run verification...")
    # 1. Instantiate VAE
    ae = DualAutoencoder(latent_channels=256)
    print("VAE DualAutoencoder instantiated.")
    
    # 2. Instantiate Injector and save
    injector = SSWatermarkInjector(latent_channels=256, latent_h=28, latent_w=28)
    injector.eval()
    torch.save(injector.state_dict(), '/kaggle/working/injector_weights.pth')
    print("Saved injector_weights.pth")
    
    # 3. Instantiate Extractor
    extractor = ForensicIntegrityAnalyzer(fp_dim=256, pretrained=False)
    print("ForensicIntegrityAnalyzer instantiated.")
    
    print("Phase 0 verification complete!")

if __name__ == "__main__":
    main()
'''
phase0 = phase0[:main_start] + phase0_main
write_file('stage3_phase0_verify.py', phase0)


# Function to generate phase 1 to 4
def gen_phase(phase_num, active_heads, active_losses):
    content = base_with_vae
    
    # We inject the unfreezing logic right after model creation in train_ddp
    model_creation = "extractor = DDP(extractor, device_ids=[rank], find_unused_parameters=True)"
    
    unfreeze_logic = f"""
    # PHASE {phase_num} Unfreezing logic
    print("Setting requires_grad for Phase {phase_num}...")
    for name, param in extractor.named_parameters():
        param.requires_grad = False
        
    for name, param in extractor.named_parameters():
        for head in {active_heads}:
            if head in name:
                param.requires_grad = True
                
    extractor = DDP(extractor, device_ids=[rank], find_unused_parameters=True)
    """
    content = content.replace(model_creation, unfreeze_logic)
    
    # Update loss weight parameters in train_ddp call to ForensicInfoNCELoss
    # The call looks like: total_loss, losses = criterion(pred_int, gt_int, pred_glob, gt_glob, ...)
    # Actually, we can just replace the loss arguments when calling criterion.
    # In train_extractor_pretrained.py, the criterion is called in the loop.
    # Let's just do a regex or exact replace for the loss weights in the forward pass of ForensicInfoNCELoss
    
    # Or simply add a block at the start of the epoch loop to override weights in config dict
    override_logic = f"""
        # PHASE {phase_num} LOSS WEIGHTS
        loss_weights = {active_losses}
        # Assuming criterion uses these. We'll modify the loop where total_loss is computed.
    """
    
    # To be safe, let's inject into the criterion call.
    # Base code: loss_ext, losses_ext = criterion(pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor.module, itr, config['latent_weight'], id_weight_current, config['collapse_weight'], max_iterations)
    
    old_call = "loss_ext, losses_ext = criterion(pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor.module, itr, config['latent_weight'], id_weight_current, config['collapse_weight'], max_iterations)"
    
    new_call = f"""
            # PHASE {phase_num} overrides
            lw = {active_losses}
            loss_ext, losses_ext = criterion(pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor.module, itr, lw.get('latent', 0.0), lw.get('id', 0.0), lw.get('collapse', 0.0), max_iterations)
            
            # Manually apply spatial and global weights since criterion doesn't take them as args directly in the base script,
            # Wait, the base script's ForensicInfoNCELoss hardcodes spatial and global addition!
            # Let's override total_loss inside the criterion.
    """
    
    # Better approach: Modify ForensicInfoNCELoss forward signature and return.
    loss_def_old = "def forward(self, pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor, current_epoch=0, latent_weight=1.0, id_weight=1.0, collapse_weight=0.1, max_epochs=80):"
    loss_def_new = f"def forward(self, pred_int, gt_int, pred_glob, gt_glob, z_mean, z_logvar, gt_latent, pred_fp, identity_labels, extractor, current_epoch=0, latent_weight=1.0, id_weight=1.0, collapse_weight=0.1, max_epochs=80, spatial_weight={active_losses.get('spatial', 1.0)}, global_weight={active_losses.get('global', 1.0)}):"
    
    content = content.replace(loss_def_old, loss_def_new)
    
    loss_sum_old = "total_loss = loss_spatial + loss_global + (loss_latent * latent_weight) + loss_identity + (collapse_weight * loss_collapse)"
    loss_sum_new = "total_loss = (loss_spatial * spatial_weight) + (loss_global * global_weight) + (loss_latent * latent_weight) + loss_identity + (collapse_weight * loss_collapse)"
    content = content.replace(loss_sum_old, loss_sum_new)
    
    # Fix the criterion call in training loop to override weights
    crit_args_old = "latent_weight=latent_weight, id_weight=id_weight,"
    crit_args_new = f"latent_weight={active_losses.get('latent', 0.0)}, id_weight={active_losses.get('id', 0.0)},"
    content = content.replace(crit_args_old, crit_args_new)
    
    crit_args2_old = "collapse_weight=collapse_weight,"
    crit_args2_new = f"collapse_weight={active_losses.get('collapse', 0.0)},"
    content = content.replace(crit_args2_old, crit_args2_new)
    
    # Load injector weights logic
    inj_load_new = """
    # ALWAYS load injector weights from phase0
    injector.load_state_dict(torch.load('/kaggle/working/injector_weights.pth', map_location='cpu'))
    injector.eval()
    """
    content = re.sub(r"inj_path = config\.get\('injector_weights'\).*?map_location='cpu'\)\)", inj_load_new, content, flags=re.DOTALL)
    
    write_file(f'stage3_train_phase{phase_num}.py', content)

# Phase 1: Tamper Localisation
gen_phase(1, 
          active_heads="['integrity_head', 'fpn', 'stem', 'layer1', 'layer2', 'layer3', 'layer4']", 
          active_losses={'spatial': 1.0, 'global': 0.0, 'latent': 0.0, 'id': 0.0, 'collapse': 0.0})

# Phase 2: Global Attack Classification
gen_phase(2, 
          active_heads="['detector_head', 'integrity_head', 'fpn', 'stem', 'layer1', 'layer2', 'layer3', 'layer4']", 
          active_losses={'spatial': 0.1, 'global': 1.0, 'latent': 0.0, 'id': 0.0, 'collapse': 0.0})

# Phase 3: Latent Shift Reconstruction
gen_phase(3, 
          active_heads="['latent_mean', 'latent_logvar', 'detector_head', 'integrity_head', 'fpn', 'stem', 'layer1', 'layer2', 'layer3', 'layer4']", 
          active_losses={'spatial': 0.01, 'global': 0.01, 'latent': 1.0, 'id': 0.0, 'collapse': 0.0})

# Phase 4: Identity Verification
gen_phase(4, 
          active_heads="['identity_head', 'identity_proj', 'identity_centers', 'latent_mean', 'latent_logvar', 'detector_head', 'integrity_head', 'fpn', 'stem', 'layer1', 'layer2', 'layer3', 'layer4']", 
          active_losses={'spatial': 0.01, 'global': 0.01, 'latent': 0.1, 'id': 1.0, 'collapse': 0.1})

