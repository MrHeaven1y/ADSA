# Code and Mathematical Evidence for ADSA Claims (Paper 2)

This document provides direct mathematical formulations, code evidence, and deep intuitive explanations for the architectural claims made in the Paper 2 traceability report. It maps the theoretical paradigm shifts directly to their implementation in the ADSA repository.

---

## 1. Differentiable Soft-Intersection Masking (Fixing Gradient Shattering)

**The Claim:** 
Because the baseline hard logical intersection shattered gradients and crashed end-to-end training, a continuous, differentiable soft approximation was adopted as an experimental fix to preserve continuous gradient flow.

**Deep Dive & Intuitive Explanation:**
*   **The Problem with Hard Binary Masks:** The original paper proposed a hard logical AND intersection ($M(x,y) = E_C \land E_S \land E_P$). While this filters out noise, the resulting mask is binary (0/1). In backpropagation, gradients must flow continuously through the masking operation. With a binary mask, the derivative is zero almost everywhere ($\frac{\partial M(x,y)}{\partial x} = 0$). This creates "dead zones"—the generator receives no usable gradient signals about where to embed, causing optimization to fail.
*   **The Soft Approximation:** The `AdaptiveSemanticMasker` uses continuous-valued features: Sobel magnitude (continuous edge strength) and local pixel variance (continuous texture measure). 
*   **Why Differentiability Matters:** The weighted combination is differentiable everywhere ($\frac{\partial M(x,y)}{\partial x} \neq 0$). During adversarial training, the generator receives usable gradient signals about where to embed without breaking optimization.
*   **Intuitive Analogy (Stencil vs. Mesh):** A binary mask is like a cardboard stencil with holes cut out—paint either goes through or doesn't, offering no nuance. If you try to adjust the spray angle, nothing changes. The adopted differentiable mask is like a mesh screen—paint flows through with varying intensity depending on mesh density, allowing the system to learn subtle, fine-grained adjustments.

**Code Evidence (`train_extractor_pretrained_synced.py`):**
```python
# Extracted from AdaptiveSemanticMasker
# Continuous Sobel magnitude instead of binary edges
sobel_mag = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2 + 1e-6)

# Local variance instead of Canny/Prewitt
local_mean = self.avg_pool(gray)
local_var = self.avg_pool((gray - local_mean)**2)

# Differentiable soft intersection (avoids derivative = 0)
intersect = (0.4 * sobel_mag + 0.6 * local_var).clamp(0, 1)
return F.adaptive_avg_pool2d(intersect, (28, 28))
```

---

## 2. Advanced Tamper Layer vs. FGSM (Failing Deepfake Detection)

**The Claim:** 
Previous methods relying on FGSM adversarial training actively fail the objective of deepfake detection. The `AdvancedTamperLayer` was adopted as an observational correction to simulate macroscopic, semantic deepfake tampering directly within the training loop.

**Deep Dive & Intuitive Explanation:**
*   **The Limitation of FGSM:** FGSM teaches the network robustness to local, high-frequency pixel noise. However, deepfakes (like face swaps) are macroscopic structural manipulations. A network trained exclusively on FGSM noise is blind to deepfakes because they do not resemble gradient noise.
*   **In-Loop Simulation (Data Augmentation):** By dynamically cropping and pasting regions, or injecting generative Gaussian noise during the forward pass, we create a continuous curriculum of structural tampering. This forces the network to learn semantic integrity violations.
*   **Intuitive Analogy (The Guard Dog):** Training only with FGSM is like training a guard dog to recognize a thief strictly by the specific brand of shoes they wear. If the thief wears different shoes (a structural deepfake), the dog fails to react. Training with in-loop semantic tampering is like teaching the dog to recognize the *act of breaking in*—it's profoundly more robust and actually solves the deepfake detection objective.

**Code Evidence (`train_extractor_pretrained_synced.py`):**
```python
# Extracted from AdvancedTamperLayer
is_copy = torch.rand(1).item() > 0.5
if is_copy:
    # Simulate object splicing / face swapping (Structural Tampering)
    tampered_x[i, :, top:top+h_t, left:left+w_t] = x[i, :, src_top:src_top+h_t, src_left:src_left+w_t]
else:
    # Simulate generative diffusion noise replacement
    noise = torch.randn(C, h_t, w_t, device=x.device)
    tampered_x[i, :, top:top+h_t, left:left+w_t] = noise * 0.5
```
