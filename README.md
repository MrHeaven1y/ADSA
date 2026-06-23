# Adversarial Dual-Stream Autoencoder (ADSA) for Forensic Tamper Detection

A modular, PyTorch-based research pipeline for image forensics, fragile watermarking, and tamper localization. This project implements a 4-stage architecture that segments an image, hides invisible chaotic watermarks in dual latent spaces, extracts them under severe geometric and noise attacks, and ultimately trains an adversarial network to make the watermark undetectable to forensic tools.

> **Status:** Stage 1, 2, and 3 are fully functional and highly stable. Stage 4 (GAN) is currently under development.

---

## 🧠 Pipeline Architecture

The project is broken down into four modular stages. Each stage outputs weights or data that feed into the next.

### Stage 1: Semantic Segmentation
* **Goal:** Separate the foreground object from the background.
* **Architecture:** Fine-tuned MobileSAM (Mask Decoder only) or custom UNet.
* **Output:** Binary masks used to create dual-stream inputs for the Autoencoder.

### Stage 2: Dual-Stream Autoencoder & Watermark Injection
* **Goal:** Compress images into a latent space and inject chaotic watermarks.
* **Architecture:** A custom ResNet-based Autoencoder with UNet skip connections. It features separate encoders for the Object (`enc_obj`) and Background (`enc_bg`) streams, but a shared decoder.
* **Watermarking:** Utilizes a Logistic-Tent chaotic map to generate 1,024 unique, structured high-frequency watermarks. These are injected into the 28x28 latent space using variance normalization and semantic masking (stronger in textured areas, weaker in flat areas).

### Stage 3: Forensic Watermark Extractor (Current Focus)
* **Goal:** Given an attacked image, detect tampering, extract the latent watermark, and identify the specific watermark identity.
* **Architecture (`ForensicIntegrityAnalyzer`):** ResNet-34 backbone (re-engineered for forensic high-frequency retention) with three specialized heads:
  1. **Spatial Integrity Head:** An FPN-based decoder outputting a 56x56 tamper probability map.
  2. **Global Detector Head:** A GeMPooled classifier determining the image status (Authentic, Tampered, Fake Identity, No Watermark).
  3. **Identity Head:** An ArcFace-regularized metric learning head that maps the extracted latent to a specific 256-dimensional identity center.

### Stage 4: Adversarial Training (GAN) [WIP]
* **Goal:** Train a Discriminator to detect the watermark, forcing the Autoencoder to hide it better, while the Extractor learns to read through the Discriminator's noise.
* **Status:** Architectural wiring is in progress.

---

## ✨ Key Engineering Innovations & Fixes

During the development of Stage 3, several critical engineering challenges were solved to ensure research-grade stability:

1. **The MaxPool Trap:** Standard ResNet aggressively pools high frequencies in its stem, destroying the hidden watermark. We replaced the `MaxPool` layer with an `AvgPool` to preserve the mathematical footprint of the chaotic signal.
2. **Dimension Alignment:** The extractor's intermediate layers were specifically routed to output a `28x28` latent grid, aligning perfectly with the Autoencoder's latent space to prevent MSE loss smearing.
3. **Small-Batch BatchNorm Collapse:** With a batch size of 4, standard `BatchNorm` running statistics became highly erratic. A `set_bn_eval` hook was implemented to freeze ResNet's BN layers to ImageNet statistics during training, preventing validation loss explosions.
4. **Curriculum Learning:** Training is divided into phases:
   * *Phase 1 (Epochs 0-40):* Focus entirely on Spatial and Global detection.
   * *Phase 2 (Epochs 40-60):* Activate ArcFace Identity loss.
   * *Phase 3 (Epochs 60+):* Introduce weak latent consistency loss.
5. **Chaotic Watermark Caching:** Generating 1,024 chaotic maps via pure Python loops takes ~5 minutes. An atomic, PID-safe caching system (`/kaggle/working/wm_cache`) was implemented to ensure this computation only happens once per environment.

---

## 📊 Forensic Diagnostics

To verify model generalization mid-training, a custom diagnostic script (`stage3_tests.py`) is embedded directly into the training loop. It runs at specified epochs and tests the model against 4 distinct attack modes:

1. **Authentic Attack:** Watermarked image passes through JPEG, blur, and noise.
2. **Tampered:** Authentic image has a 1/3rd region spliced or noise-injected.
3. **Fake Identity:** Image is watermarked with an identity not belonging to the original distribution.
4. **No Watermark:** Clean image with no watermark injected.

The script outputs Dice scores for tamper localization, Cosine Similarities for identity verification, and saves visual heatmaps to `/kaggle/working/diagnostics/`.

---

## 🚀 Installation & Usage

### 1. Environment Setup
This project was developed on Kaggle using Tesla T4 GPUs. 
```bash
pip install torch torchvision mobile-sam matplotlib
```

### 2. Directory Structure
```text
├── config.json                 # Master configuration file
├── train_seg_sam.py            # Stage 1
├── train_extractor_pretrained  # Stage 3 (Pretrained ResNet)
├── train_extractor_scratch.py  # Stage 3 (From Scratch CNN)
├── stage3_tests.py             # Embedded diagnostic tools
└── wm_cache/                   # Auto-generated watermark cache
```

### 3. Configuration
All hyperparameters, paths, and learning rates are controlled via a master `config.json`. 
```json
{
  "stage3_extractor": {
    "images_dir": "/kaggle/working/clean_pet_data/images",
    "mask_dir": "/kaggle/working/clean_pet_data/masks",
    "test_epochs": [10, 25, 40, 55, 70, 80],
    "resume": null,
    "batch_size": 4,
    "max_iterations": 80,
    "lr": 1e-4,
    "accum_steps": 4
  }
}
```

### 4. Run Training
Start Stage 3 training using DDP (Distributed Data Parallel) across available GPUs:
```bash
python train_extractor_pretrained.py --config config.json
```

---

## 📈 Current Results (Stage 3)

The applied fixes have completely eliminated the overfitting collapse observed in early iterations. Current metrics at Epoch 40:

* **Tamper Localization (Dice):** `0.9541` (Perfectly isolates the tampered 1/3rd region)
* **Watermark Invisibility (Energy Ratio):** `0.00007671` (Virtually invisible to the human eye)
* **Generalization:** Train/Val loss tracks within 2% of each other, proving the model is learning forensic features, not memorizing noise.

---

## 🛣️ Roadmap

- [x] Fix Stage 3 train/val supervision leakage.
- [x] Stabilize ArcFace identity geometry.
- [ ] Complete Stage 4 GAN architecture wiring (Extractor <-> Discriminator).
- [ ] Test Stage 4 adversarial payload extraction robustness.
- [ ] Write final research paper summarizing the ADSA framework.

## 📄 License

This project is developed for academic research purposes.