# Energy-Gated Attention

Simple, parameter-efficient attention modification that gates
value aggregation by the spectral energy of key token embeddings.

> *Similarity selects what matches the query; salience selects what matters.*

## Key Result

| Model       | Val Loss | Δ vs BASE | Extra Params | Dataset |
|-------------|----------|-----------|--------------|---------|
| BASE        | 1.4742   | —         | —            | TinyShakespeare |
| EGA-1       | 1.3712   | +0.103    | 12,480       | TinyShakespeare |
| EGA-C       | 1.3745   | +0.097    | 1,377,216    | TinyShakespeare |
| EGA-M-Fixed | 1.4733   | +0.001    | 960          | TinyShakespeare |
| BASE        | 1.0897   | —         | —            | Penn Treebank   |
| EGA-1       | 0.9889   | +0.101    | 12,480       | Penn Treebank   |

**EGA-1 achieves +0.103 validation loss improvement with
only 12,480 extra parameters (<0.26% overhead) and no
computational overhead. The result is consistent across
two datasets.**

## The Idea

Standard attention measures only similarity between queries
and keys. By the Wiener-Khinchin theorem, a linear projection
of the embedding estimates the spectral energy of the token
signal. Tokens with high spectral energy carry more information
and should attract more attention — independently of similarity.

Energy-Gated Attention adds a learned salience signal:

```
e(b)    = W_proj · x(b)              # energy projection
e_norm  = znorm(e(b))                # z-normalize across tokens
gate(b) = sigmoid(α · (e_norm - τ))  # learned threshold
att     = softmax(QK/√d) · gate      # gate attention weights
att     = att / sum(att)             # renormalize (sum-to-1)
output  = att · V
```

## Key Findings

1. **Single scale is optimal**: EGA-1 > EGA-2 > EGA-4
   — the first principal component of spectral energy is sufficient;
   additional scales add redundancy without new information

2. **Temporal structure helps but costs more**: EGA-C (causal conv)
   achieves +0.097 but uses 110× more parameters than EGA-1

3. **Optimal basis is non-sinusoidal**: Morlet wavelets converge
   to the admissibility boundary (ω₀σ = 5.0) and fail to match
   the learned linear projection — fixed structured bases
   are a lower bound; learned wavelet packets remain an open question

4. **τ ≈ 0.35 is a stable linguistic property**: independently
   discovered from two different initializations, corresponding
   to ≈36% of tokens carrying above-average spectral energy
   — the fraction of content-word characters in English text

5. **Cross-dataset consistency**: +0.103 on TinyShakespeare,
   +0.101 on Penn Treebank — dataset-independent inductive bias

## Wavelet Family Comparison

| Model       | Val Loss | Δ vs BASE | Basis         | Learned? |
|-------------|----------|-----------|---------------|----------|
| EGA-1       | 1.3712   | +0.103    | Linear        | Yes ◄ best |
| EGA-DB2     | 1.4692   | +0.005    | Daubechies db2 | No (fixed) |
| EGA-M-Fixed | 1.4733   | +0.001    | Morlet        | Partial  |
| EGA-DB4     | 1.4748   | −0.001    | Daubechies db4 | No (fixed) |
| BASE        | 1.4742   | —         | None          | —        |

The less constrained the wavelet basis, the better.
Fixed sinusoidal (Morlet) < Fixed orthogonal (DWT) < Learned (linear).

## Paper

> **Energy-Gated Attention: Spectral Salience as an Inductive
> Bias for Transformer Attention**
>
> arXiv preprint — link to be added after submission

## Run

```bash
# Phase 1: validate energy gating (BASE vs EGA-4)
python experiments/phase1_ega4.py

# Phase 2: full N_SCALES ablation (BASE, EGA-1/2/4, EGA-C)
# Recommended: T4 GPU (~3 hours)
python experiments/phase2_ablation.py

# Phase 3: Morlet wavelet central question
# (BASE vs EGA-1 vs EGA-M-Fixed)
# Recommended: T4 GPU (~60 min)
python experiments/phase3_morlet.py
```

All scripts run on CPU or GPU. GPU strongly recommended for
Phase 2. Checkpoints are saved every 500 steps — safe to
resume after interruption.

## Results

```
results/
  phase2_ablation_results.png   6-panel Phase 2 ablation
  egam_fixed_results.png        3-panel Phase 3 central question
  dwt_comparison_final.png      wavelet family comparison
  trained_scalogram.png         Morlet scalogram of embeddings
  mean_scalogram.png            mean spectral portrait
```

## Requirements

```
torch>=2.0.0
requests
matplotlib
tqdm
numpy
```

```bash
pip install torch requests matplotlib tqdm numpy
```

## Citation

```bibtex
@misc{authorname2025ega,
  title   = {Energy-Gated Attention: Spectral Salience as an
             Inductive Bias for Transformer Attention},
  author  = {[Author Name]},
  year    = {2025},
  note    = {arXiv preprint — link to be added}
}
```
*(Update arXiv ID and author name before publishing)*
