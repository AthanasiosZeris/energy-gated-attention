# Paper 4 — POD (Multiscale POD of Attention Fields)

**Multiscale POD of Transformer Attention Fields: Scale-Selective Analysis via
Morlet Scalogram**

Applies Proper Orthogonal Decomposition — the optimal linear dimensionality
reduction — to the transformer attention field, treating each attention matrix as
a two-dimensional snapshot analogous to a turbulent velocity field. A Morlet
scalogram along the lag diagonal makes the decomposition scale-selective,
extracting energetically ordered coherent structures at each scale.

- **arXiv:** [2606.06573](https://arxiv.org/abs/2606.06573)
- **Status:** published (originally submitted physics.flu-dyn, recategorised cs.LG)

## Code

Paper 4 is an analysis of trained models, not a new architecture: it decomposes
the attention fields of the BASE and EGA-1 checkpoints from Paper 1.

- `pod_analysis.py` — self-contained analysis implementing Algorithm 1 from the
  paper. Loads a BASE and an EGA-1 checkpoint, collects attention snapshots,
  computes the Morlet scalogram and scale-selective POD, and produces every
  figure and table: `attn_scalogram.png` (Fig 1), `pod_modes.png` (Fig 2),
  `coherency_map.png` (Fig 3), `reynolds_table.txt` (Table 2), `heads_table.txt`
  (Table 3).

## Run

```bash
python pod_analysis.py --base path/to/base.pt --ega path/to/ega1.pt
# optional: --data <text file>  --n_snapshots 500  --outdir <dir>
# --inspect prints a checkpoint's structure to help diagnose loading
```

The `base.pt` and `ega1.pt` checkpoints are the standard GPT and EGA-1 models
from Paper 1 (see [`../paper1_ega/`](../paper1_ega/)).

## Citation

```bibtex
@misc{zeris2025pod,
  author = {Athanasios Zeris},
  title  = {Multiscale POD of Transformer Attention Fields:
            Scale-Selective Analysis via Morlet Scalogram},
  note   = {arXiv:2606.06573},
  year   = {2025}
}
```
