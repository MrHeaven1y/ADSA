# Paper 3 Module Context & Documentation

This module contains the definitive traceability and evidence documentation linking **Paper 3** (*Adversarial Dual-Stream Autoencoders: A Semantic-Aware Robust Watermarking Framework for Deepfake Defence and Media Provenance*) to its realization in the ADSA codebase.

## 1. Context: The Paradigm Shift
Initially, the objective of Paper 3 was to provide a semantic-aware dual-stream watermarking architecture utilizing explicit differential Bit Error Rates (BER) between the object and background streams to verify provenance and detect deepfakes. However, during the integration with the broader ADSA pipeline, certain constraints necessitated significant shifts:
*   **The Fragility of Static BER Thresholds:** The proposed hard-coded forensic decision rule (e.g., $BER_{obj} > \tau_{high} \land BER_{bg} \le \tau_{low}$) proved brittle in the wild. Varying camera quality, social media compression, or adversarial noise can artificially inflate BER, causing false negatives in deepfake detection.
*   **Curriculum Collapse:** Simulating all attacks at maximum intensity from the start overwhelmed the model, necessitating a dynamically scaling curriculum.

Consequently, our implementation shifted from **Static Thresholding** to **Learned Feature Boundaries**, relying on a learned multi-task detection head over deep semantic features.

## 2. Module Contents
This directory packages the documents that scientifically map and defend these theoretical choices against rigorous academic scrutiny.

*   **[`paper3_traceability_report.md`](./paper3_traceability_report.md):** The core section-by-section audit of Paper 3 versus our implementation. It strictly maps what the paper proposed, what we implemented, why the original method failed in our context, and the logical validation of our modifications using the epistemic tagging framework.
*   **[`CODE_CLAIMS_EVIDENCE.md`](./CODE_CLAIMS_EVIDENCE.md):** The definitive "shield" document containing mathematical formulations and exact Python code snippets directly backing up our theoretical solutions (Differentiable Attack Layer Curriculum, Learned Multi-Class Detector Head).
*   **[`PAPER3_CODE_RESONANCE.md`](./PAPER3_CODE_RESONANCE.md):** Outlines the philosophical and semantic differences between the paper's framework and our implementation.
