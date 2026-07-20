# Paper 3 — MoPE (Beyond Sinusoids)

**Beyond Sinusoids: A Morlet Wavelet Framework for Transformer Positional
Encoding**

Introduces Morlet Positional Encoding (MoPE): each embedding dimension learns
its own centre frequency and locality bandwidth from data. The main theoretical
result is a unification — sinusoidal PE and the RoPE correlation kernel both
emerge as limiting cases of MoPE when locality is switched off (σ → ∞); MoPE's
phase recovers the RoPE rotation angle exactly, while its amplitude adds a
learned Gaussian locality kernel that standard encodings lack.

- **arXiv:** [2606.01258](https://arxiv.org/abs/2606.01258)
- **Status:** published

## Code

Paper 3 shares its experimental campaign with Paper 2: the MoPE models
(PE-MORLET, EGA-MORLET) are trained by Paper 2's `gpt_phase4_memfix.py`, and
this paper's figures are produced from that trained EGA-MORLET checkpoint
(`exp2_egam.pt`).

- `morlet_pe_figures.py` — figure generation (CPU, ~2 min, no training). Loads
  `exp2_egam.pt` and produces the uncertainty-plane figure, the learned
  frequency (ω) and bandwidth (σ) spectra, and a numerical parameter dump.
- **Training:** see [`../paper2_ega_mope/gpt_phase4_memfix.py`](../paper2_ega_mope/),
  which trains the EGA-MORLET model whose checkpoint this script analyses.

## Run

```bash
# 1. Train EGA-MORLET (produces exp2_egam.pt) — from Paper 2:
#    python ../paper2_ega_mope/gpt_phase4_memfix.py
# 2. Generate this paper's figures from that checkpoint:
python morlet_pe_figures.py
```

## Citation

```bibtex
@misc{zeris2025mope,
  author = {Athanasios Zeris},
  title  = {Beyond Sinusoids: A Morlet Wavelet Framework for
            Transformer Positional Encoding},
  note   = {arXiv:2606.01258},
  year   = {2025}
}
```
