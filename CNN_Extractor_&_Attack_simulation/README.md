# Stage 3: CNN Extractor & Attack Simulation

This directory contains the codebase for Stage 3 of the ADSA pipeline (Watermark Extraction and Attack Simulation).

## Important Context for AI Agents & Developers

When modifying or analyzing the codebase, please be aware of the following architecture and generation flow:

### 1. Base Files (`*_synced.py`)
- **`train_extractor_pretrained_synced.py`** and **`train_extractor_scratch_synced.py`** are the **primary source files**.
- Do **NOT** use or modify `train_extractor_pretrained.py` or `train_extractor_scratch.py`. Those are outdated/legacy files.
- The `_synced.py` files already contain the fully synchronized Dual VAE implementation from Stage 2 (`dual_autoencoder_module`). 

### 2. Phase Generation (`generate_phases.py`)
- The script `generate_phases.py` (located in the project root) automates the generation of the curriculum training phases (Phase 0 through Phase 4).
- It uses `train_extractor_pretrained_synced.py` as the base file.
- It injects specific logic into the base file to generate `stage3_train_phase1.py`, `stage3_train_phase2.py`, `stage3_train_phase3.py`, and `stage3_train_phase4.py` inside the `jup_notebooks` folder.
- **Modifications**: If you need to make core structural changes to the Extractor or its training loop, make them in `train_extractor_pretrained_synced.py` and then run `generate_phases.py` to propagate the changes.
- `generate_phases.py` automatically configures which network heads are frozen/unfrozen and overrides the loss function weights (`spatial_weight`, `global_weight`, `latent_weight`, `id_weight`, `collapse_weight`) dynamically per phase.

### 3. Inline Tests (`inline_tests.py`)
- Diagnostic and testing logic for Stage 3 is stored in `jup_notebooks/stage3_tests.py`.
- After running `generate_phases.py`, you must run `inline_tests.py` (also in the project root).
- `inline_tests.py` extracts the `run_forensic_diagnostics` function from `stage3_tests.py` and directly injects it into all generated phase scripts (`stage3_phase0_verify.py` and `stage3_train_phaseX.py`).

### Summary of Workflow
If you edit the Extractor architecture or training loops:
1. Edit `train_extractor_pretrained_synced.py`
2. Run `python generate_phases.py` from the project root.
3. Run `python inline_tests.py` from the project root.
4. The generated `stage3_train_phaseX.py` scripts are now ready to be run or pushed.

### 4. Injector Weights & Watermark Cache (`wm_cache`)
- The Stage 3 pipeline relies on a pre-computed watermark pool stored in `/kaggle/working/injector_weights.pth`.
- This `.pth` file is generated automatically by running the **Phase 0 verification script** (`stage3_phase0_verify.py`).
- **Caching Mechanism**: During Phase 0, the script computes the chaotic tensors and caches them individually in the `/kaggle/working/wm_cache` directory. If this directory already exists with the cached `.pt` tensors, Phase 0 will skip the heavy mathematical computation, instantly load the `.pt` files, and compile them into `injector_weights.pth` in seconds.
- Ensure `"injector_weights": null` in your configuration when running Phase 0 to trigger this generation. Phases 1-4 are hardcoded to automatically load the resulting `injector_weights.pth`.
