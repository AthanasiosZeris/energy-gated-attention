# Paper 5a — FourierQK

**FourierQK: Spectral Preprocessing of Query–Key Projections Improves Transformer
Attention**

Applies a learnable spectral mask to the (bilateral) FFT of the query and key
streams before the attention inner product — a bandpass-filtered inner product
at a learned frequency. Character-level language modelling.

- **arXiv:** [2607.07478](https://arxiv.org/abs/2607.07478)
- **Status:** published

## Code

Three self-contained experiment scripts (each builds its own GPT/data/training
scaffold). All are the corrected final versions — the earlier buggy drafts are
superseded and not included; the fixes are documented in each file header.

- `train_freq_ablation_fixed.py` — fixed-frequency ablation (main experiment).
  Corrects five bugs vs the original draft: single attention scaling, documented
  bilateral-FFT leakage regime, weight-decay excluded from `log_freq` (so any
  frequency migration reflects the loss, not the optimiser), and consistent
  frequency/period logging. → `freq_ablation_fixed_results.json`
- `train_phase_randomised_v2.py` — phase-coherence decomposition via eval-time
  Theiler surrogates (a trained model is evaluated under phase-randomisation and
  amplitude-only surrogates; phase noise is applied at eval time only, never
  during training).
- `train_morlet_scales_v3.py` — causal Morlet scale sweep with early stopping and
  a shuffled-gap leakage diagnostic (confirms the v2 U-shaped curve was
  overfitting, not leakage). → `morlet_scale_sweep_v3.json`

## Run

```bash
python train_freq_ablation_fixed.py
python train_phase_randomised_v2.py
python train_morlet_scales_v3.py
```

Each downloads TinyShakespeare on first run; GPU recommended.

## Citation

```bibtex
@misc{zeris2025fourierqk,
  author = {Athanasios Zeris},
  title  = {FourierQK: Spectral Preprocessing of Query--Key Projections
            Improves Transformer Attention},
  note   = {arXiv:2607.07478},
  year   = {2025}
}
```
