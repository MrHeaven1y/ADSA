# Scientific Traceability Methodology & Template Guide

## 1. Context and Rationale
The traceability reports in the `ADSA` repository are designed to clearly map out the logical progression of the project. This methodology enforces a strict boundary between what a source paper originally proposed, and the engineering reality of our current implementation.

By tracking the *actual research chronology*—framing historical experiments as engineering observations from the development of the current run—these reports provide a defensible, honest scientific story. It shows that our current ADSA formulation was driven by logical necessity and observed limitations, not arbitrary choices.

---

## 2. The Epistemic Tagging System
To enforce researcher honesty and clarity, claims in a traceability report must be framed using the following labels:

| Label                   | Meaning                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| **Paper Baseline**      | What the original paper specifies.                                     |
| **Retained**            | We followed the paper.                                                 |
| **Modified**            | We changed the mechanism.                                              |
| **Extended**            | We added functionality beyond the paper.                               |
| **Replaced**            | The original mechanism was unsuitable for our implementation objective.|
| **Current ADSA Method** | What we use now and why.                                               |

**Important Framing Rule:** We do not present reconstructed historical results as independently reproduced facts (e.g., no "artifact lost" or "not yet verified"). Furthermore, we do not claim that a mechanism "definitively proves" a solution. Instead, we frame our choices as the *adopted solution* for the *current implementation*.

*Example:* "During development of the current Stage 2 formulation, the deterministic AE exhibited inadequate watermark preservation. This motivated the introduction of the VAE formulation as the adopted solution."

---

## 3. The Traceability Template
When auditing papers for this project, the following structural template **MUST** be used.

### Section 1: Paradigm Shift (Problem Definition Comparison)
A high-level tabular comparison of the problem definition in the original paper versus the overarching goal of the current ADSA implementation.

### Section 2: Research Chronology
A numbered timeline of the iterative research cycle, framed as engineering observations from the development of the current run. 
*   *Example: Paper Baseline $\rightarrow$ limitation encountered $\rightarrow$ theoretical diagnosis $\rightarrow$ modification $\rightarrow$ current method.*

### Section 3: Section-by-Section Traceability & Fidelity Analysis
For every major section or proposed method, use the strictly defined **6-Dimension Matrix**:

| Dimension | Details |
| :--- | :--- |
| **Paper Baseline** | Exact method, equation, or protocol from the paper. |
| **Current ADSA Method** | The adopted solution in our current codebase (with file references). |
| **Status** | Retained / Modified / Extended / Replaced. |
| **Engineering Observation** | The historical limitation or development observation that prompted the change. |
| **Adopted Rationale** | Why the current formulation was adopted (e.g., "The VAE formulation was adopted to prevent..."). |
| **Evidence** | Specific code structures or formulations supporting the current implementation. |

### Section 4: Final Reproduction Status Scorecard
A concluding table scoring the overall fidelity of the reproduction across major categories using the approved Status labels.
