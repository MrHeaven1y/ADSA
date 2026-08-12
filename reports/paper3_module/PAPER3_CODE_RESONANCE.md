# Paper 3: Semantic & Structural Resonance with Code Implementation

**Target Paper:** *Adversarial Dual-Stream Autoencoders: A Semantic-Aware Robust Watermarking Framework for Deepfake Defence and Media Provenance*
**Objective:** To outline the philosophical, semantic, and structural differences between the theoretical framework proposed in Paper 3 and our actual Python implementation (`ADSA`), explicitly highlighting why baseline methods fail the objective of deepfake detection and necessitated experimental fixes.

---

## 1. The Core Philosophy: The Digital Witness

### Paper Context
The core conceptual idea of Paper 3 is the semantic-aware dual-stream approach. It intentionally assigns different forensic roles to different image regions: the **object watermark** is designed to be fragile (so it breaks when a face-swap occurs), while the **background watermark** is designed to be highly robust (to survive and act as an immutable "witness"). Therefore, a state of $BER_{obj} \uparrow$ and $BER_{bg} \downarrow$ provides evidence of localized semantic manipulation.

### Code Implementation Divergence
*   **Philosophy Retained:** The ADSA implementation strictly retains this philosophical approach. The structural separation of the object and background streams for differential robustness remains the fundamental pillar of the project.
*   **What Changed:** What ADSA modifies is not the *philosophy* of the digital witness, but the *decision mechanism* used to evaluate its testimony, derived via experimental fixes because the baseline mechanism failed the deepfake detection objective in practice.

## 2. Decision Architecture: Fixed Thresholds vs. Learned Classifier

### Paper Context
The paper proposes evaluating the digital witness using a fixed decision boundary: a deepfake is confirmed when $(BER_{obj}, BER_{bg})$ violate static hard-coded thresholds ($\tau_{high} = 0.35$ and $\tau_{low} = 0.02$).

### Code Implementation Divergence
*   **Architectural Shift (Experimental Fix):** Because the baseline method fails the objective of robust deepfake detection in dynamic environments, our codebase changes the decision architecture itself. Rather than extracting a discrete metric (BER) and passing it to a fixed threshold rule, the pipeline flows from $\text{deep semantic features} \rightarrow \text{learned detector} \rightarrow \text{forensic class}$. 
*   **Philosophical Reason:** The fixed BER rule is highly sensitive to shifts in BER caused by additional external degradation (e.g., heavy compression), which can artificially inflate the background BER and cause false negatives. The baseline simply breaks in the wild. By replacing the fixed decision boundary with a learned feature-space decision mechanism as an observational correction, the architecture allows the network to learn the boundaries of real-world tampering more flexibly.

## 3. Provenance Representation: Discrete Agreement vs. Continuous Identity Projection

### Paper Context
The paper verifies media provenance by computing the raw bit-error rate of the extracted watermark against the original secret key.

### Code Implementation Divergence
*   **Metric Shift (Experimental Fix):** We introduced an `identity_head` (inspired by ArcFace/InfoNCE) that projects the extracted features into a metric identity space, measuring provenance survival via Cosine Distance.
*   **Philosophical Reason:** The identity projection provides a continuous learned representation for provenance comparison rather than relying exclusively on discrete bit-error counts. This observational correction provides a smoother, gradient-friendly space for measuring structural identity preservation.

## 4. Attack Training: Attack Presence vs. Attack Scheduling

### Paper Context
The paper implies a training paradigm where the network optimizes against a differentiable attack layer modeling various transmission degradations continuously.

### Code Implementation Divergence
*   **Curriculum Shift (Observational Correction):** Because the original paper's baseline mechanism of immediate maximum attack severity actively fails the objective of training a robust generator, we shifted from *attack presence* to *attack scheduling*. We treat the attack layer as a progressive optimization problem.
*   **Philosophical Reason:** Exposing the generator to maximum attack severity immediately creates chaotic gradients that prevent the model from learning the fundamental embedding task (like teaching calculus to a child before addition). By scheduling the attacks ($\text{clean} \rightarrow \text{mild degradation} \rightarrow \text{strong degradation}$), the curriculum stabilizes early training while ensuring robust convergence.

## 5. Implementation Construct: Staged Training (Stage 3 $\rightarrow$ Stage 4)

### Paper Context
The paper implies a monolithic, end-to-end framework where embedding, attack resistance, and extraction are all learned concurrently.

### Code Implementation Divergence
*   **Construct Shift (Experimental Fix):** We decoupled the architecture into strict phases: Stage 3 (supervised extractor/detector training) and Stage 4 (GAN-based generative refinement).
*   **Philosophical Reason:** The paper does not provide a particular way to achieve stable convergence of all these competing objectives simultaneously. Through empirical testing, attempting to train the embedder, extractor, and a GAN discriminator from scratch simultaneously leads to catastrophic collapse, failing the entire objective of the project. Thus, the Stage 3 $\rightarrow$ Stage 4 pipeline was explicitly constructed as an experimental fix to achieve the paper's theoretical vision in reality.
