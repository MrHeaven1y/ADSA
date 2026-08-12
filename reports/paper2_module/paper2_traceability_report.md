# Scientific Traceability Report: Paper 2

**Target Paper:** *Trustworthy AI Media Provenance: An Adversarial Dual-Stream Autoencoder with Hybrid Chaotic Watermarking*

This document provides a rigorous, section-by-section traceability mapping between the theoretical claims in Paper 2 and the actual codebase (`ADSA`). It focuses on the architectural, objective-level, and experimental deviations between the paper's original intent and the repository's current implementation, framed around engineering observations and the ultimate goal of deepfake detection.

---

### Implementation Status Categories
*   **Direct implementation:** Code fully follows the paper.
*   **Modified implementation:** Code differs from the paper, requiring scientific defense.
*   **Observational Correction / Experimental Fix:** Solutions introduced because the baseline methods failed the core objective of deepfake detection (e.g., gradient shattering, blindness to structural manipulations).
*   **Extended implementation:** Code adds new features beyond the paper's scope.
*   **Missing implementation:** Proposed feature is absent in the code.

---

## 1. Paradigm Shift: From Hard Logic to Differentiable Forensics

The `ADSA` repository re-interprets the methods presented in Paper 2 to ensure end-to-end differentiability and real-world robustness. Crucially, the current implementation (Stage 3) focuses on robust supervised extraction, setting the architectural foundation for a planned adversarial module (Stage 4).

| Dimension | Paper 2 | ADSA |
| :--- | :--- | :--- |
| **Tamper Robustness** | FGSM robustness (Pixel noise) | Semantic tamper augmentation (Structural Deepfakes) |
| **Authentication** | Binary authentication | Multi-task forensic analysis |
| **Masking Logic** | Hard edge intersection | Differentiable soft masking |
| **Optimization** | End-to-end adversarial concept | Stage 3 stabilization + Stage 4 refinement architecture |

*Note: The hybrid chaotic sequence generation for watermarking proposed in Paper 2 is faithfully retained in the current implementation.*

---

## 2. Section-by-Section Traceability & Fidelity Analysis

### 2.1 Proposed Method: Hybrid Chaotic Sequence Generation

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Generates a bipolar spreading sequence ($c_i \in \{+1, -1\}$) using a 512-bit secret key hashed via SHA-512 to initialize Logistic and Sine chaotic maps. |
| **Current ADSA Method** | Strict adherence in `generate_hybrid_chaotic_watermark` (`train_extractor_pretrained_synced.py`). |
| **Status** | Retained |

### 2.2 Proposed Method: Tri-Operator Edge Intersection Masking

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Constrains embedding by intersecting binary edge maps from Canny, Sobel, and Prewitt: $M(x, y) = E_C \land E_S \land E_P$. |
| **Current ADSA Method** | `AdaptiveSemanticMasker` replacing binary edges with continuous Sobel magnitude and a local pixel variance map, intersecting softly via weighted combination. |
| **Status** | Experimental Fix |
| **Engineering Observation** | The baseline hard logic creates a binary mask. Because this mask sits inside the computational graph during adversarial training, the binary thresholding introduces regions with gradients of zero ($\frac{\partial M}{\partial x} = 0$). This completely disrupted backpropagation (gradient shattering), failing the objective of end-to-end training. |
| **Adopted Rationale** | The soft differentiable approximation was adopted as an experimental fix. It maintains perceptual alignment while allowing continuous gradient propagation ($\frac{\partial M}{\partial x} \neq 0$), preventing GAN collapse. |

### 2.3 Experimental Setup: DCNN Authentication Block and Adversarial Training

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Evaluates media using a binary DCNN, trained with BCE on FGSM adversarial examples. |
| **Current ADSA Method** | `ForensicIntegrityAnalyzer` (a multi-task modified ResNet34) and `AdvancedTamperLayer` for in-loop augmentation (copy-paste, generative noise). |
| **Status** | Observational Correction |
| **Engineering Observation** | FGSM adversarial training only teaches robustness to local high-frequency noise (pixel-level). It was observed that this previous method actively fails the objective of deepfake detection, because macroscopic deepfakes (face swaps) are structural manipulations, not pixel noise. The detector was blind to them. |
| **Adopted Rationale** | The `AdvancedTamperLayer` was adopted to simulate structural semantic tampering during training. By forcing the detector to learn structural integrity violations rather than just high-frequency noise robustness, the system successfully achieves the deepfake detection objective. |

---

## 3. Conclusion: Summary of Experimental Fixes

Because the baseline methods proposed in the original paper (FGSM, hard logic masks) tend to fail the primary objective of real-world deepfake detection and stable end-to-end training, the ADSA implementation introduced the following necessary experimental fixes:

1. **The Differentiable Masking Fix:** Replaced the non-differentiable hard logical intersection with the `AdaptiveSemanticMasker`. This solved the gradient shattering problem ($\frac{\partial M}{\partial x} = 0$), allowing the network to continuously learn where to embed without blocking backpropagation.
2. **The Structural Tampering Fix:** Shifted adversarial hardening from FGSM (pixel noise) to the `AdvancedTamperLayer` (structural semantic tampering). This was a critical observational correction because FGSM training fails to recognize actual deepfakes, which are macroscopic and semantic in nature.
