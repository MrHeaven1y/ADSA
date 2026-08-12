# Research Traceability Report: Paper 3

## Adversarial Dual-Stream Autoencoders: A Semantic-Aware Robust Watermarking Framework for Deepfake Defence and Media Provenance

This document maps the theoretical claims and architectural designs proposed in **Paper 3** to their practical implementation in the repository. It categorizes each engineering decision to justify deviations from the paper's original scope, providing a defensible framework for the implemented architecture.

---

### Implementation Status Categories
*   **Direct implementation:** Code fully follows the paper.
*   **Modified implementation:** Code differs from the paper, requiring scientific defense.
*   **Observational Correction / Experimental Fix:** Solutions introduced because the original paper lacked specific implementation mechanisms or failed the core objective of real-world deepfake detection (e.g., gradient collapse, threshold compounding).
*   **Extended implementation:** Code adds new features beyond the paper's scope.
*   **Missing implementation:** Proposed feature is absent in the code.

---

## Section 1: Goals and Contributions

**Paper Goal:** To propose a semantic-aware ADSA framework that utilizes explicit differential Bit Error Rates (BER) between the object and background streams to verify provenance and detect deepfakes. The core philosophy is the "Digital Witness": the object watermark is fragile and expected to be destroyed by semantic manipulation (like a face-swap), while the background watermark is robust and expected to survive. Thus, high $BER_{obj}$ combined with low $BER_{bg}$ serves as evidence of localized manipulation.

**Paper Contribution:**
1. Semantic-Aware Dual-Stream Watermarking Architecture.
2. Adversarial-trained differentiable attack layer.
3. Differential BER-based forensic decision rule (using hard thresholds $\tau_{low}, \tau_{high}$).
4. Background as an immutable "digital witness".

**Status:** Retained / Observational Corrections / Experimental Fixes
*   **Where implemented:** `CNN_Extractor_&_Attack_simulation/` (Stage 3), `GAN_Based_Refinement_Discriminator_Integration/` (Stage 4).
*   **Philosophy Status (Retained):** The fundamental dual-stream digital witness philosophy is strictly retained. The object and background still possess differential robustness.
*   **Deviation (The Stage 3 $\rightarrow$ Stage 4 Construct):** The paper does not provide a particular way to achieve stable convergence for all these systems simultaneously. As an experimental fix, our implementation explicitly decouples the training into two distinct phases: first stabilizing the extractor/detector in a highly supervised curriculum (**Stage 3**), and then utilizing those frozen weights for adversarial generative refinement (**Stage 4**).
*   **Deviation (Decision Architecture):** Because the baseline fixed BER thresholding rule actively fails the objective of real-world deepfake detection (due to varying noise floors breaking the math), we replaced it with a learned multi-class detection head and an InfoNCE-based Identity projection space as an observational correction.

---

## Section 2: Architecture & Attacks

### 2.1 Differentiable Attack Layer ($\eta$)

**Paper:** Models the transmission channel and malicious attacks via a differentiable layer simulating JPEG compression, Gaussian blur, and geometric transformations. The assumption is that the attack layer exposes the network to transmission degradations during training.

**Code Mapping:** `DifferentiableJPEG` and `AttackSimulationLayer` in `train_extractor_pretrained_synced.py`.

**Status:** Observational Correction (Attack presence $\rightarrow$ Attack scheduling)

**Scientific Justification:**
*   **Alignment:** The code meticulously implements differentiable JPEG (using DCT block approximations) and Gaussian blur convolutions. 
*   **Adopted Rationale:** The baseline method of simply applying strong attacks immediately causes gradient collapse, failing the objective of training a robust embedder. As an experimental fix, the implementation changes this from simple "attack presence" to "attack scheduling" (a progressive optimization problem). The curriculum ensures the network first learns to embed in a clean channel before facing mild degradation, and finally strong degradation.

---

## Section 3: Deepfake Detection Rule

### 3.1 Differential BER Thresholding

**Paper:** Proposes a hard-coded forensic decision rule:
*   Semantic deepfake (face swap): $BER_{obj} > \tau_{high} \land BER_{bg} \le \tau_{low}$ (where $\tau_{low} = 0.02, \tau_{high} = 0.35$).

**Code Mapping:** `ForensicIntegrityAnalyzer` detector heads.

**Status:** Experimental Fix (Change in Decision Architecture)

**Scientific Justification:**
*   **Engineering Observation (Failing Deepfake Detection):** The fixed BER rule actively fails the core objective of deepfake detection in real-world scenarios. Compounding degradations (e.g., severe social media compression) inflate $BER_{bg}$ beyond the rigid $\tau_{low}$ threshold, causing massive false negatives (tampered media looks "authentic" to the algorithm).
*   **Adopted Rationale:** Because the baseline failed to adapt to varying noise floors, we introduced an experimental fix that changed the **forensic decision architecture**. 
    *   *Paper:* $(BER_{obj}, BER_{bg}) \rightarrow \text{fixed thresholds} \rightarrow \text{deepfake decision}$
    *   *ADSA:* $\text{watermark representation} \rightarrow \text{deep semantic features} \rightarrow \text{learned detector} \rightarrow \text{forensic class}$
*   By appending a `detector_head` onto the deepest features, the architecture allows the network to learn the feature-space decision mechanism for boundaries between "Authentic", "Tampered", "Fake", and "No WM".

---

## 4. Conclusion: Summary of Experimental Fixes

Because the baseline methods proposed in the original paper outline an abstract theoretical framework but lack concrete engineering directives, they tend to fail the primary objective of real-world deepfake detection and stable convergence.

The ADSA implementation introduces critical **observational corrections**:
1.  **Staged Training Pipeline:** Splitting the architecture into a Stage 3 (Extractor Stabilization) and Stage 4 (GAN Refinement) construct to ensure convergence, because end-to-end simultaneous training leads to catastrophic collapse.
2.  **Attack Scheduling:** Shifting from "attack presence" to a progressive attack curriculum to prevent gradients from collapsing early in training.
3.  **Learned Decision Boundaries:** Replacing the brittle fixed decision boundary (BER thresholds) with a learned feature-space decision mechanism, because fixed thresholds fail deepfake detection under varying real-world noise floors.
