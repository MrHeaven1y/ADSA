# Paper 2 Module Context & Documentation

This module contains the definitive traceability and evidence documentation linking **Paper 2** (*Trustworthy AI Media Provenance: An Adversarial Dual-Stream Autoencoder with Hybrid Chaotic Watermarking*) to its realization in the ADSA codebase.

## 1. Context: The Paradigm Shift
Initially, the objective of Paper 2 was to provide cryptographically secured media provenance utilizing a hybrid chaotic spread-spectrum sequence and tri-operator edge masking to dynamically control embedding strength. However, during the integration with the broader ADSA pipeline, several architectural and theoretical constraints necessitated significant shifts:
*   **The Non-Differentiability Problem:** The proposed logical AND intersection of binary edge maps (Canny, Sobel, Prewitt) shattered gradients, making end-to-end adversarial training impossible.
*   **The Semantic Vulnerability:** Training the authenticator solely against mathematical FGSM gradient perturbations created a model blind to real-world, macroscopic deepfake tampering (like face-swapping).
*   **The Triage Limitation:** As with Paper 1, a binary authenticator could not answer the critical forensic questions required for deepfake incident response (*where* did the tamper occur, and *who* does the provenance belong to).

Consequently, our implementation shifted the objective from **Cryptographic Authentication** to **Differentiable Forensic Traceability**, resulting in logical deviations including the differentiable Soft-Masking approximation and the in-loop Advanced Tamper Layer augmentation.

## 2. Module Contents
This directory packages the documents that scientifically map and defend these theoretical choices against rigorous academic scrutiny.

*   **[`paper2_traceability_report.md`](./paper2_traceability_report.md):** The core section-by-section audit of Paper 2 versus our implementation. It strictly maps what the paper proposed, what we implemented, why the original method failed in our context, and the logical validation of our modifications using the epistemic tagging framework.
*   **[`CODE_CLAIMS_EVIDENCE.md`](./CODE_CLAIMS_EVIDENCE.md):** The definitive "shield" document containing mathematical formulations and exact Python code snippets directly backing up our theoretical solutions (Chaotic Sequence Generation, Soft Semantic Masking, Advanced Tamper Layer).

*Reviewers auditing the ADSA reproduction of Paper 2 should begin with the Traceability Report and cross-reference the Code Claims Evidence to verify the mathematics of the architectural shifts.*
