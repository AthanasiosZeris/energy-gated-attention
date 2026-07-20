# Paper 2 — EGA + MoPE

**Energy-Gated Attention and Wavelet Positional Encoding: Complementary
Inductive Biases for Transformer Attention**

Combines the energy gate of Paper 1 (EGA) with a learned Morlet wavelet
positional encoding (MoPE). The central finding is *superadditivity*: EGA alone
gives +0.092 validation-loss improvement on TinyShakespeare and MoPE alone is
−0.032, but together they reach +0.119 — more than the sum of parts —
demonstrating that spectral salience and scale-selective locality are
complementary inductive biases. Five supporting experiments (convolution
attention, scale-initialised heads, multi-quantity gating, spectral cascade)
test the spectral-filtering interpretation of attention.

- **arXiv:** [2605.26355](https://arxiv.org/abs/2605.26355)
- **Status:** published

## Code

The paper's five experiments run from two self-contained scripts (each rebuilds
the full GPT/data/training scaffold):

- `gpt_phase4_memfix.py` — **run first.** Experiment 1 (BASE-DOT, CONV-L4,
  CONV-L8, EGA-1) and Experiment 2 (PE-SINCOS, PE-ROPE, PE-MORLET,
  EGA-MORLET). CONV-L4 replaces the original CONV-L16, which exceeded T4 VRAM
  (see paper Appendix B); this is the configuration reported in the paper.
- `gpt_phase4_exp345.py` — **run second.** Experiment 3 (SCALE-INIT),
  Experiment 4 (MQ-E / MQ-EP / MQ-EF multi-quantity attention), and Experiment 5
  (spectral cascade interpretability, no training). Loads the `exp1_base.pt`,
  `exp1_ega1.pt`, and `exp2_egam.pt` checkpoints written by the first script.

## Run

```bash
# Colab T4. Experiments 1-2 (~2-3 h), then 3-5 (~2.5 h).
python gpt_phase4_memfix.py      # writes checkpoints for exp1/exp2
python gpt_phase4_exp345.py      # consumes those checkpoints, runs exp3/4/5
```

Checkpoints are saved per model — both scripts are safe to restart after
interruption.

## Citation

```bibtex
@misc{zeris2025egamope,
  author = {Athanasios Zeris},
  title  = {Energy-Gated Attention and Wavelet Positional Encoding:
            Complementary Inductive Biases for Transformer Attention},
  note   = {arXiv:2605.26355},
  year   = {2025}
}
```
