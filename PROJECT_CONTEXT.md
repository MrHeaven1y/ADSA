# ADSA: Comprehensive Project Context & Architecture Guide

This document serves as a deep dive into the **Adversarial Dual-Stream Autoencoder (ADSA)** project. It explains the theoretical background, what the system accomplishes, how the underlying mechanics work, and how the various stages of the codebase synchronize to form a complete end-to-end deepfake defense pipeline.

---

## 1. The Core Philosophy: What are we doing and why?

### The Problem
Traditional digital watermarking treats an image as a single, uniform grid of pixels. When embedding a hidden signal (watermark) across an entire image, there is an unavoidable trade-off between **imperceptibility** (how invisible the watermark is) and **robustness** (how well it survives compression or malicious attacks). Furthermore, against localized semantic manipulations like **Deepfake face-swapping**, global watermarks often fail because the attacker completely replaces the primary subject while leaving the rest of the image intact, destroying the provenance chain.

### The ADSA Solution
This project implements a **Semantic-Aware Hypothesis**. We recognize that not all pixels in an image are equal:
*   **The Object (Region of Interest, e.g., a Face):** Requires strict high-fidelity preservation because human eyes are highly sensitive to facial artifacts. It necessitates a delicate, low-strength watermark.
*   **The Background (Context, e.g., foliage, buildings):** Contains high-variance textures that naturally hide noise. It can absorb a highly robust, redundant watermark without triggering perceptual alarms.

By splitting the image into two distinct streams, we embed different watermarks into the object and the background. 

### The "Digital Witness"
If a malicious actor swaps a face in the image, the delicate object watermark is destroyed. However, the robust background watermark survives perfectly. The surviving background acts as an immutable **"Digital Witness,"** proving the media's original provenance while simultaneously flagging the missing object watermark as evidence of targeted semantic tampering (a deepfake).

---

## 2. Project Stages & Codebase Synchronization

The repository is modularly structured into sequential training stages, governed by a central configuration file (`master_config.json`). Here is how the code implements the ADSA architecture:

### Stage 1: Semantic Partitioning
**Directory:** [`segmentation_module/`](file:///c:/Workspace/Projects/DeepLearning/ADSA--simplified/segmentation_module)
*   **What is happening:** The pipeline begins by analyzing the original cover image. Using a U-Net or SAM (Segment Anything Model) architecture, it generates precise binary masks: `M_obj` (the foreground subject) and `M_bg` (the environmental background).
*   **Sync:** These masks are saved and subsequently used to split the input image into isolated tensors for the next stage.

### Stage 2: Dual-Stream Autoencoder (The Embedder)
**Directory:** [`dual_autoencoder_module/`](file:///c:/Workspace/Projects/DeepLearning/ADSA--simplified/dual_autoencoder_module)
*   **What is happening:** The isolated object and background are passed into parallel Convolutional Cascaded Autoencoders (CCAE). 
    *   **Cryptographic Chaos:** A 512-bit secret key is hashed via SHA-512 to generate a hybrid chaotic spread-spectrum (SS) sequence. 
    *   **Latent Embedding:** This sequence is injected into the *latent feature bottlenecks* of the autoencoders. The background stream is heavily compressed but receives a high-strength watermark. The object stream retains a larger latent dimension to preserve fidelity but receives a low-strength watermark.
    *   **Reconstruction:** The decoders reconstruct the separated parts, which are then fused back together using a Gaussian soft-edge blending mask to prevent visible boundary seams.

### Stage 3: Adversarial Extractor & Attack Simulation (The Detector)
**Directory:** [`CNN_Extractor_&_Attack_simulation/`](file:///c:/Workspace/Projects/DeepLearning/ADSA--simplified/CNN_Extractor_&_Attack_simulation)
*   **What is happening:** Once the image is watermarked, we need a network capable of reliably extracting the watermark, even if the image is transmitted over the internet or maliciously attacked. 
*   **Threat Model Expansion:** While foundational theories (e.g., Paper 2) focused on pixel-level adversarial noise (like FGSM), our extractor is trained with an `AdvancedTamperLayer` that simulates structural semantic manipulations (like face swaps). This forces the network to learn macroscopic integrity violations rather than just high-frequency gradient noise. Note that combining FGSM with semantic tampering remains a potential future enhancement.
*   **The 5-Phase Curriculum:** To prevent "curriculum collapse" (where the model gets overwhelmed by complex tasks), the DCNN Extractor is trained across 5 distinct phases:
    1.  **Phase 0:** Verify network graphs and latent alignment.
    2.  **Phase 1:** Tamper Localisation (Detecting local pixel tampering using spatial losses).
    3.  **Phase 2:** Global Attack Classification (Identifying JPEG compression, Gaussian blur, etc.).
    4.  **Phase 3:** Latent Shift Reconstruction.
    5.  **Phase 4:** Identity Verification (Provenance clustering).
*   **Sync:** This DCNN learns to rely on robust statistical signatures—specifically **Kullback-Leibler Divergence (KLD)** and **Edge Entropy**—rather than fragile spatial patterns. Furthermore, instead of using hard-coded Bit Error Rate (BER) thresholds for deepfake detection (which are brittle in the wild, as noted in Paper 3 analysis), we append a multi-task `detector_head` and `identity_head` to learn multidimensional boundary conditions inherently.

### Stage 4: Generative Adversarial Refinement (Planned Architecture)
**Directory:** [`GAN_Based_Refinement_Discriminator_Integration/`](file:///c:/Workspace/Projects/DeepLearning/ADSA--simplified/GAN_Based_Refinement_Discriminator_Integration)
*   **Current Status:** *Architecturally prepared, but not yet implemented.* The choices in Stage 2 and 3 (such as replacing hard logical edge intersections with a differentiable soft-masking approximation) were made specifically to remove architectural barriers for this future integration.
*   **What is planned:** To maximize visual imperceptibility and adversarial security, the dual-stream autoencoder from Stage 2 will be situated as a **Generator** against a **GAN Discriminator**.
*   **Differentiable Attack Layer:** During this future GAN training loop, the watermarked images will be passed through an in-loop attack layer that simulates JPEG compression, cropping, and noise. The Generator will be penalized if the Discriminator can tell the image is watermarked, forcing it to hide the chaotic sequence deep within high-variance textures.

### End-to-End Inference
**Directory:** [`end-to-end/`](file:///c:/Workspace/Projects/DeepLearning/ADSA--simplified/end-to-end)
*   **What is happening:** The culmination of the project. This module loads the best saved `.pth` weights from the U-Net, the Autoencoder, the Extractor, and the GAN discriminator. It takes a raw input image, runs it through the complete segmented embedding process, applies a simulated attack, and attempts to extract the provenance data to verify the "Digital Witness" logic.

---

## 3. How the Pipeline is Orchestrated

*   **`master_config.json`**: This file is the "brain" of the project's hyperparameters. Rather than hardcoding learning rates, batch sizes, or latent dimensions across dozens of scripts, every module imports settings from this central JSON. It ensures that if we change the latent space dimension for the background in Stage 2, the Extractor in Stage 3 automatically knows the correct dimensions to expect.
*   **Data Flow**: The pipeline is largely file-based for modularity. Stage 1 saves masks, Stage 2 saves watermarked images and `injector_weights.pth`, which Stage 3 loads to train the forensic analyzer. 

## Summary
In essence, we are building an active, tamper-evident AI governance tool. Rather than waiting to passively detect a deepfake after it has been created, ADSA proactively "vaccinates" the original media with a cryptographically secure, semantically aware dual-watermark. 
