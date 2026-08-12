# Code and Mathematical Evidence for ADSA Claims (Paper 3)

This document provides direct mathematical formulations, code evidence, and deep intuitive explanations for the architectural claims made in the Paper 3 traceability report. It maps the theoretical paradigm shifts directly to their implementation in the ADSA repository, highlighting where experimental fixes were required because previous methods failed the core objective of deepfake detection.

---

## 1. The Digital Witness Philosophy (Differential Degradation)

**The Claim:** 
The fundamental conceptual contribution of Paper 3—the dual-stream "digital witness" philosophy where the object watermark is fragile and the background watermark is robust—is retained and explicitly implemented in the embedding architecture.

**Deep Dive & Intuitive Explanation:**
*   The architecture intentionally gives the two regions different forensic roles. The object (foreground) receives a low-strength embedding to preserve visual fidelity, rendering it fragile to semantic manipulations like face-swapping. The background receives a high-strength embedding, expected to survive transmission. The differential degradation between these two streams serves as the core evidence of localized manipulation.

**Code Evidence (`dual_autoencoder_module` equivalent):**
*(Note: Code for varying alpha/embedding strength between streams is implemented in the embedder module, configuring differential robustness for $M_{obj}$ vs $M_{bg}$.)*

---

## 2. Differentiable Attack Layer: Attack Scheduling Curriculum

**The Claim:** 
Because the original paper's baseline method of immediately applying strong attacks caused gradient collapse and failed the objective of training a robust generator, the attack layer was modified from simple "attack presence" to a progressive optimization problem ("attack scheduling") as an observational correction.

**Deep Dive & Intuitive Explanation:**
*   **The Problem with Immediate Adversity:** Presenting a generative model with extreme structural noise (like high-compression JPEG) on epoch 0 destroys nascent gradients. The generator cannot learn stable embedding patterns if the signal is instantly obliterated by the transmission channel.
*   **Intuitive Analogy:** Throwing strong attacks at the model early is like trying to teach a child calculus before they've learned addition—they just shut down and training stalls.
*   **The Experimental Fix (Curriculum):** We progressively schedule the probability and intensity of attacks based on the training epoch: $\text{clean} \rightarrow \text{mild degradation} \rightarrow \text{strong degradation}$. This construct is vital for stabilizing the Stage 3 training phase.

**Code Evidence (`train_extractor_pretrained_synced.py`):**
```python
# Extracted from AttackSimulationLayer
# Progressive attack scheduling (Experimental Fix for gradient collapse)
if current_epoch >= stage1 and (self.deterministic or torch.rand(1).item() < self.p_apply):
    q = 60 if self.deterministic else torch.randint(50, 91, (1,)).item()
    x = self.jpeg(x, quality=q)
    
if current_epoch >= stage2 and (self.deterministic or torch.rand(1).item() < self.p_apply):
    x = self._gaussian_blur(x)
```

---

## 3. Change in Decision Architecture: Learned Detector Head

**The Claim:** 
Because the paper's baseline of fixed BER thresholding fails the ultimate objective of real-world deepfake detection (by yielding false negatives under varying noise floors), an experimental fix was introduced: the forensic decision architecture was changed to a learned feature-space decision mechanism.

**Deep Dive & Intuitive Explanation:**
*   **The Failure of Fixed Thresholds:** A fixed BER rule is highly sensitive to shifts caused by additional real-world degradation (e.g., severe compression). If heavy JPEG compression inflates $BER_{bg}$ above $0.02$, the rigid mathematical condition fails, and a tampered deepfake is falsely classified as "authentic." This actively fails the deepfake detection objective.
*   **The Architectural Shift:** We changed the decision architecture itself. The flow shifted from $(BER_{obj}, BER_{bg}) \rightarrow \text{hard rule}$ to $\text{deep features} \rightarrow \text{learned classifier}$.
*   **The Rationale:** Appending a multi-layer perceptron (`detector_head`) onto the deepest 512-channel features allows the model to learn the multidimensional boundaries separating "Authentic", "Tampered", "Fake", and "No WM" from the feature space directly, adapting dynamically to varying noise distributions.

**Code Evidence (`ForensicIntegrityAnalyzer`):**
```python
# The learned feature-space decision mechanism
# mapping the 512-channel deep robust features directly to a 4-class decision
self.detector_head = nn.Sequential(
    GeMPooling(),
    nn.Dropout(p=0.1),
    nn.Linear(512, 128), 
    nn.GELU(), 
    nn.Linear(128, 4) # Authentic vs. Tampered vs. Fake vs. No WM
)
```
