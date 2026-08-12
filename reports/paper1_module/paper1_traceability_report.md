# Scientific Traceability Report: Paper 1

**Target Paper:** *CAE-DCNN Architectures for Image Watermarking and Detection on Edge-IoT Networks*

This document provides a rigorous, section-by-section traceability mapping between the theoretical claims in Paper 1 and the actual codebase (`ADSA`). It focuses on the architectural, objective-level, and experimental deviations between the paper's original intent and the repository's current implementation.

---

### Implementation Status Categories
*   **Direct implementation:** Code fully follows the paper.
*   **Modified implementation:** Code differs from the paper, requiring scientific defense.
*   **Observational Correction / Experimental Fix:** Solutions introduced because the original paper's methods failed the core objective of real-world deepfake detection (e.g., they only detected pixel noise, or provided no diagnostic utility).
*   **Extended implementation:** Code adds new features beyond the paper's scope.
*   **Missing implementation:** Proposed feature is absent in the code.

---

## 1. Paradigm Shift: From Edge Authentication to Forensic Traceability

The `ADSA` repository fundamentally re-interprets the problem presented in Paper 1. 

| Dimension | Paper 1 (Edge Authentication) | ADSA Repository (Forensic Traceability) |
| :--- | :--- | :--- |
| **Primary Objective** | Lightweight, time-critical image authentication. | Heavyweight, adversarially robust forensic traceability (Deepfake Detection). |
| **Detection** | Binary classification (Watermarked / Not). | 4-Class Triage (Authentic, Tampered, Fake, No WM). |
| **Localization** | Not explicitly handled by the DCNN. | Pixel-dense integrity mapping (`integrity_head`). |
| **Identity** | Implicitly verified via binary detection. | Explicitly clustered via InfoNCE / ArcFace (`identity_head`). |
| **Attack Robustness** | Passive evaluation post-training (Standard JPEG). | Active curriculum via Differentiable Attack & Tamper Layers. |

---

## 2. Research Chronology: The Evolution of Stage 2 & 3

The current architecture emerged from an iterative cycle of engineering observations during the development of the current run:

1. **Paper Baseline:** Convolutional Cascaded Autoencoder (CCAE).
2. **Observation 1 (The Skip-Connection Bypass):** During early development, watermark extraction exhibited severe failure post-reconstruction. The decoder drew high-quality spatial data from the skip connections, exhibiting insufficient dependence on the deep latent bottleneck (`z`). Because the watermark was injected into `z`, the signal was bypassed.
3. **Modification 1 (Experimental Fix):** A 2-pass forward pass with a skip-drop auxiliary reconstruction objective was adopted to force bottleneck utilization.
4. **Observation 2 (Deterministic Suppression):** The deterministic latent representation was inadequate for retaining small watermark perturbations; extraction was exceedingly hard and redundant. Standard latent reconstruction MSE failed to explicitly preserve the watermark's frequency signature.
5. **Modification 2 (Observational Correction):** The VAE formulation was adopted alongside a watermark-aware FFT Magnitude loss to isolate and preserve latent variation.

---

## 3. Section-by-Section Traceability & Fidelity Analysis

### 3.1 Proposed Method: Autoencoder Architecture (CAE vs VAE)

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Uses a deterministic Convolutional Cascaded Autoencoder (CCAE). |
| **Current ADSA Method** | A probabilistic Variational Autoencoder (VAE) via `ResNetAE`. |
| **Status** | Observational Correction |
| **Engineering Observation** | The deterministic AE exhibited inadequate watermark preservation. The latent space failed to separate small, high-frequency perturbations from ambient noise. |
| **Adopted Rationale** | The VAE formulation was adopted to explicitly preserve embedding capacity. The continuous probabilistic distribution ensures the watermark survives smooth latent-space interpolations typical of modern deepfakes. |

### 3.2 Proposed Method: Spread Spectrum Embedding

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Embeds an i.i.d. Gaussian spread spectrum watermark uniformly using a static scalar $\alpha$. |
| **Current ADSA Method** | `AdaptiveSemanticMasker` dynamically scales a SHA-512 hybrid chaotic watermark using pooled Sobel edge magnitudes and local variance. |
| **Status** | Experimental Fix |
| **Engineering Observation** | A static scalar $\alpha$ is theoretically vulnerable to statistical estimation and introduces perceptual artifacts in smooth, low-variance regions. |
| **Adopted Rationale** | Adaptive masking was adopted as an experimental fix to modulate embedding strength via local variance, concentrating the signal in complex textures and exploiting human visual masking. |

### 3.3 Proposed Method: Watermark Detection (DCNN)

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Simple DCNN binary classifier (Watermarked vs Not). |
| **Current ADSA Method** | `ForensicIntegrityAnalyzer`: a multi-task network outputting an `integrity_map`, `global_logit` (4-class), latent shifts, and `pred_fp`. |
| **Status** | Observational Correction |
| **Engineering Observation** | A binary decision trained on pixel noise fails the core objective of deepfake detection. Deepfakes are structural manipulations; a binary "Yes/No" provides zero diagnostic utility for incident response and is blind to macroscopic tampering. |
| **Adopted Rationale** | The multi-task network was adopted as an observational correction to provide comprehensive forensic situational awareness, allowing the network to learn actual semantic integrity violations. |

---

## 4. Conclusion: Summary of Experimental Fixes

Because the baseline methods proposed in the original paper tend to fail the primary objective of real-world deepfake detection, the ADSA implementation introduced the following necessary experimental fixes:

1. **The VAE & FFT Shift:** Replaced the deterministic CAE and spatial MSE with a Variational Autoencoder and an Orthogonal FFT loss. This was an experimental fix to prevent the network from treating the watermark as negligible noise, ensuring it survives structural deepfake generation.
2. **The Multi-Task Network:** Replaced the simple binary classifier with a multi-task `ForensicIntegrityAnalyzer`. Previous binary methods fail the deepfake detection objective because they cannot localize tampering or identify the source.
3. **The Auxiliary Skip-Drop:** Introduced a 2-pass forward pass to solve the "lazy network" problem, ensuring the decoder actually relies on the watermark-bearing bottleneck.
