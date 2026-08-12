# Paper 1 Module Context & Documentation

This module contains the definitive traceability and evidence documentation linking **Paper 1** (*CAE-DCNN Architectures for Image Watermarking and Detection on Edge-IoT Networks*) to its advanced realization in the ADSA codebase.

## 1. Context: The Paradigm Shift
Initially, the objective of Paper 1 was to provide lightweight, binary authentication on edge devices. However, during the implementation of the ADSA pipeline, several core theoretical and architectural limitations of the original paper were discovered:
*   **The Deterministic Problem:** Standard deterministic Autoencoders suppressed the subtle watermark perturbation, treating it as ambient noise.
*   **The Lazy Decoder Problem:** CCAE skip-connections allowed the network to bypass the watermark-bearing deep bottleneck entirely, erasing the signal.
*   **The Frequency Loss Problem:** Spatial losses (MSE/BCE) failed to capture the spectral footprint of the watermark.
*   **The Triage Limitation:** A simple binary detector could not answer *where* a deepfake occurred or *who* the watermark belonged to, making it useless for incident response.
*   **The Gradient Interference Problem:** Training a sophisticated forensic multi-task network simultaneously caused gradients from different tasks to destructively overlap.

Consequently, the ADSA pipeline shifted the objective from **Edge Authentication** to **Adversarially Robust Forensic Traceability**, resulting in deep, logically verifiable architectural deviations (VAE, 2-Pass Skip-Dropout, Watermark-Aware FFT Loss, Adaptive Masking, and Phased Curriculum).

## 2. Module Contents
This directory packages the documents that scientifically map and defend these theoretical choices against rigorous academic scrutiny.

*   **[`paper1_traceability_report.md`](./paper1_traceability_report.md):** The core section-by-section audit of Paper 1 versus our implementation. It strictly maps what the paper proposed, what we implemented, why the original method failed in our context, and the logical validation of our modifications.
*   **[`CODE_CLAIMS_EVIDENCE.md`](./CODE_CLAIMS_EVIDENCE.md):** The definitive "shield" document. It contains deep intuitive explanations (like the "whispering in a song" analogy), explicit mathematical formulations, and exact Python code snippets directly backing up our theoretical solutions (VAE, FFT Loss, Curriculum, etc.).
*   **[`PAPER1_CODE_RESONANCE.md`](./PAPER1_CODE_RESONANCE.md):** An older historical draft of the code resonance, kept here for archival tracking purposes.

*Reviewers auditing the ADSA reproduction of Paper 1 should begin with the Traceability Report and cross-reference the Code Claims Evidence to verify the mathematics of the architectural shifts.*
