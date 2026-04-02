# Literature Review Guide

## Our positioning
We combine **conformal prediction** (statistical guarantees) + **k-space data consistency** (physics-based hallucination detection) for MRI reconstruction. No existing paper does both.

---

## Papers you MUST read (priority order)

### 1. Conformal prediction (our core method)
- **Angelopoulos & Bates (2023)** — "A Gentle Introduction to Conformal Prediction" — THE tutorial. Read Sections 1-3 for the theory we use. Explains split conformal, exchangeability, finite-sample correction.
- **Angelopoulos et al. (2022)** — "Image-to-Image Regression with Distribution-Free UQ" — Extends CP to pixel-level image tasks. This is the closest prior work to ours. Key difference: they don't apply it to MRI or combine with physics.

### 2. CP in medical imaging (our closest competitors)
- **Kutiel et al. (2023)** — "Conformal Prediction Masks" — Uses CP to create binary reliable/unreliable region masks. Different from our pixel-wise intervals.
- **Wen et al. (2024)** — "Task-Driven UQ in Inverse Problems via CP" — Applies CP to downstream tasks from MRI, not pixel-level reconstruction quality.
- **Fischer et al. (2025)** — "CUTE-MRI" — Uses conformalized intervals for adaptive MRI acquisition (deciding what to scan next). Different goal from ours.

### 3. UQ in MRI reconstruction (what we're improving on)
- **Gal & Ghahramani (2016)** — MC Dropout. Read to understand why it lacks coverage guarantees.
- **Kustner et al. (2024)** — Deep ensembles on fastMRI. Shows uncertainty correlates with error but no formal guarantees. Good comparison point.
- **Edupuganti et al. (2021)** — VAE-based UQ for MRI. Bayesian approach, no coverage guarantees either.

### 4. Hallucination detection (our second contribution)
- **Bhadra et al. (2021)** — "On Hallucinations in Tomographic Reconstruction" — Formalises hallucinations via null-space analysis. Key paper. Our k-space consistency is complementary: we detect range-space inconsistency, they analyse null-space.
- **Tivnan et al. (2024)** — "Hallucination Index" — Spectral analysis approach. Different methodology from ours.

### 5. MRI reconstruction baselines (context)
- **Zbontar et al. (2018)** — fastMRI dataset paper. Read the benchmark section.
- **Ronneberger et al. (2015)** — U-Net. Skim for architecture details.
- **Hammernik et al. (2018)** — Variational networks. Understand data consistency layers (related to our k-space consistency but used *inside* the network, not as post-hoc analysis).

---

## The gap we fill (state this clearly in the paper)

| Method | Coverage guarantee? | Physics-informed? | Post-hoc? |
|--------|-------------------|--------------------|-----------|
| MC Dropout | No | No | Yes |
| Deep ensembles | No | No | No (need multiple models) |
| VAE-based | No | No | No (need special architecture) |
| Angelopoulos (2022) | Yes | No | Yes |
| Wen (2024) | Yes | No | Yes (but task-level, not pixel) |
| **Ours** | **Yes** | **Yes** | **Yes** |

---

## Papers to search for (possible additions)

Look on Google Scholar / arXiv for:
- "conformal prediction MRI" — check if anyone did pixel-level CP on MRI recon since 2024
- "data consistency hallucination MRI" — check if anyone used k-space residual as standalone metric
- "calibration uncertainty MRI reconstruction" — papers showing MC Dropout is miscalibrated

---

## How to strengthen the Related Work section

1. **Don't just list papers** — group them by theme and explain the progression
2. **State what each lacks** — "X does A but not B; Y does B but not C; we do A+B+C"
3. **Be specific about our novelty** — pixel-level CP on MRI reconstruction + k-space consistency as post-hoc hallucination detector + showing these are complementary
4. **Acknowledge limitations of our comparison** — we compare against MC Dropout (simple baseline), not ensembles or VAEs (more expensive but potentially better calibrated)
