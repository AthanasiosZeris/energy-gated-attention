# Paper 5b — FourierQK: Filter Shape and Admissibility

**FourierQK: Filter Shape, Admissibility, and the Leakage–Coverage Law**

Companion analysis to Paper 5a. Given that frequency-collapse attention works,
*which* filter properties are necessary? A controlled ablation tests five
hypotheses — DC suppression, Nyquist suppression, bandwidth, centre frequency,
and multi-scale coverage — and establishes the paper's headline methodological
result: bilateral-FFT leakage scales monotonically with spectral coverage, made
measurable by a shuffled-gap diagnostic (gap > +4 clean, gap < +2 leaky).
Admissible (zero-mean) filters provide partial leakage protection. FourierQK is
therefore characterised as effective in bidirectional (encoder-style, e.g. BERT)
settings, with causal generation deferred to the companion MorletQK work.

- **Status:** submitted (arXiv link to be added)

## Code

- `train_spectral_compression_fixed.py` — self-contained spectral-compression /
  filter-shape ablation (own data loader, GPT, training loop, and the
  shuffled-gap leakage diagnostic). Tests the paper's Table 1 variants: DC-QK,
  Nyquist-QK, GaussNarrow (σ=0.5), GaussWide (σ=16), MexHat-K4 (admissible), with
  an Init4 (σ=2 learned) FourierQK reference. This is the corrected version
  (single 1/√hs scaling, real-only scoring for the DC/Nyquist null test); the
  earlier buggy draft is superseded and not included.

## Run

```bash
python train_spectral_compression_fixed.py
```

Downloads TinyShakespeare on first run; GPU recommended.

## Citation

```bibtex
@misc{zeris2025filtershape,
  author = {Athanasios Zeris},
  title  = {FourierQK: Filter Shape, Admissibility, and the
            Leakage--Coverage Law},
  note   = {arXiv preprint --- link to be added},
  year   = {2025}
}
```
