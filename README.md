# Spectral Methods in Transformer Attention

A research series applying signal-processing and fluid-mechanics tools —
spectral energy, wavelets, Proper Orthogonal Decomposition, and the
Hilbert–Huang transform — to the query/key representations of transformer
attention. The unifying view treats attention as a data-dependent, adaptive
filtering operation, and asks which spectral inductive biases improve language
modelling and how they reshape the learned representations.

Each paper lives in its own subfolder with its code, results, and a dedicated
README. This repository was originally released as the Energy-Gated Attention
(Paper 1) code; the URL `github.com/AthanasiosZeris/energy-gated-attention` is
kept unchanged so that the links printed in Papers 1–4 continue to resolve.

## Papers

| # | Name | Contribution | arXiv | Code |
|---|------|--------------|-------|------|
| 1 | **EGA** | Spectral energy gating as attention salience | [2605.21842](https://arxiv.org/abs/2605.21842) | [`paper1_ega/`](paper1_ega/) |
| 2 | **EGA + MoPE** | Complementary biases: salience and time-frequency locality | [2605.26355](https://arxiv.org/abs/2605.26355) | [`paper2_ega_mope/`](paper2_ega_mope/) |
| 3 | **MoPE** | Morlet-wavelet unification of positional encodings | [2606.01258](https://arxiv.org/abs/2606.01258) | [`paper3_mope/`](paper3_mope/) |
| 4 | **POD** | Multiscale covariance decomposition via scale-selective POD | [2606.06573](https://arxiv.org/abs/2606.06573) | [`paper4_pod/`](paper4_pod/) |
| 5a | **FourierQK** | Bilateral spectral preprocessing of Q/K projections | [2607.07478](https://arxiv.org/abs/2607.07478) | [`paper5a_fourierqk/`](paper5a_fourierqk/) |
| 5b | **FourierQK: Filter Shape & Admissibility** | The leakage–coverage law | *submitted* | [`paper5b_filtershape/`](paper5b_filtershape/) |

<!-- Later series entries (5c MorletQK, 5d HilbertQK, …) get their own
     subfolder here as they are released. A revised version of an existing
     paper updates that paper's folder in place — git history and tags record
     prior versions; no duplicate "_v2" folders. -->

## The through-line

The series builds one idea across architectural roles. Paper 1 gates attention
by the spectral energy of key tokens; Paper 3 recasts positional encoding as a
Morlet wavelet family; Paper 2 shows those two biases are complementary; Paper 4
applies POD — the optimal linear decomposition — to the attention field itself.
Papers 5a–5b move the spectral operation directly into the query/key
computation (FourierQK), and study how the filter's shape and spectral coverage
govern the performance gain.

## Requirements

```bash
pip install torch requests matplotlib tqdm numpy
```

Individual papers may add dependencies (e.g. `EMD-signal`, `PyWavelets`); see
each subfolder's README. Scripts run on CPU or GPU; a T4-class GPU is
recommended for the training experiments. Long runs checkpoint periodically and
are safe to resume after interruption.

## Citation

Please cite the specific paper you use; BibTeX entries are in each subfolder's
README. For the series as a whole:

```bibtex
@misc{zeris_spectral_attention,
  author = {Athanasios Zeris},
  title  = {Spectral Methods in Transformer Attention},
  note   = {github.com/AthanasiosZeris/energy-gated-attention},
  year   = {2025}
}
```

## License

See [LICENSE](LICENSE).
