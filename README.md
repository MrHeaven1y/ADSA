# ADSA — Adversarial Defense & Segmentation Architecture

ADSA is a modular deep learning pipeline focused on adversarial attack simulations, reconstruction-based defenses, and advanced image segmentation using Segment Anything Model (SAM) and U-Net. 

---

## 🚀 System Architecture

The pipeline is organized into distinct, specialized modules that handle everything from attack simulations to segmentation and refinement:

```
ADSA/
├── CNN_Extractor_&_Attack_simulation/  # Evaluates model robustness against adversarial noise
├── dual_autoencoder_module/            # Autoencoder-based reconstruction & anomaly detection
├── segmentation_module/                # SAM and U-Net pipelines for region of interest masking
├── GAN_Based_Refinement/               # Refines segmentation outputs using GAN discriminators
├── end-to-end/                         # Coordinates the execution of all modules
└── master_config.json                  # Global configuration for paths, thresholds, and parameters
```

---

## 🛠️ Modules Overview

### 1. CNN Extractor & Adversarial Attack Simulation
Simulates adversarial attacks (such as FGSM/PGD) on CNN feature extractors. Used to evaluate model robustness and test defense mechanisms under controlled noise perturbations.

### 2. Dual Autoencoder Module
A reconstruction-based defense and anomaly detection system. It trains twin autoencoders to learn nominal features and reconstruct input images, effectively filtering out adversarial perturbations and identifying anomalous regions.

### 3. Segmentation Module
Integrates advanced segmentation models:
* **U-Net**: Trained locally for custom semantic segmentation.
* **Segment Anything Model (SAM)**: Utilized for high-fidelity zero-shot instance masking of regions of interest.

### 4. GAN-Based Refinement
Features a refinement generator and a discriminator to polish segmentation masks. The discriminator ensures that generated boundaries and segmentations align perfectly with real-world leaf/object shapes, reducing noise.

### 5. End-to-End Orchestrator
Stitches the modules together. Controlled by `activity.py` and configured through `master_config.json`, this module feeds images through the adversarial defense filters, runs the segmentation engines, and outputs refined masks automatically.

---

## ⚙️ Configuration & Usage

The entire pipeline is dynamically configured via `master_config.json`. Adjust model paths, hyper-parameters, thresholds, and input/output folders here.

To execute the full end-to-end pipeline:
```bash
python activity.py
```

> [!NOTE]
> Ensure that all required models and weight artifacts (stored under each module's `artifacts` directory) are downloaded and correctly pointed to in `master_config.json` before running the pipeline.
