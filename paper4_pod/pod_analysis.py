"""
Scale-Selective POD Analysis of Transformer Attention Fields
Implements Algorithm 1 from pod_paper.tex appendix.

Usage:
    python pod_analysis.py --base path/to/base.pt --ega path/to/ega1.pt

Produces:
    attn_scalogram.png   → Figure 1 (Experiment 1)
    pod_modes.png        → Figure 2 (Experiment 2)
                            lag profiles for layers 2,4,6
                            (layers with non-degenerate POD structure)
    coherency_map.png    → Figure 3 (Experiment 5)
    reynolds_table.txt   → Table 2 values
    heads_table.txt      → Table 3 values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from scipy.stats import linregress
import argparse
import os
import sys

# ── publication style ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── GPT model definition (must match training code) ─────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.register_buffer('bias', torch.tril(
            torch.ones(config['block_size'], config['block_size']))
            .view(1, 1, config['block_size'], config['block_size']))
        self.attn_weights = None  # store for analysis

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.attn_weights = att.detach()  # (B, H, T, T)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.act    = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            h   = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f= nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def get_attention_weights(self):
        """Return list of (B, H, T, T) attention tensors, one per layer."""
        return [block.attn.attn_weights for block in self.transformer.h]


# ── Morlet wavelet transform ─────────────────────────────────────
def morlet_wavelet(scale, omega0=5.0, N=256):
    """Complex Morlet wavelet at given scale."""
    t = np.arange(-N//2, N//2) / scale
    psi = (np.pi**(-0.25)) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)
    return psi / (scale ** 0.5)


def wavelet_transform_1d(signal_1d, scales, omega0=5.0):
    """CWT of a 1D signal at given scales. Returns (n_scales, N) complex array."""
    N = len(signal_1d)
    sig_fft = np.fft.fft(signal_1d, n=2*N)
    freqs = np.fft.fftfreq(2*N)
    W = np.zeros((len(scales), N), dtype=complex)
    for i, scale in enumerate(scales):
        psi_fft = np.sqrt(2 * np.pi * scale) * np.exp(
            -(2 * np.pi * freqs * scale - omega0)**2 / 2) * (freqs > 0)
        conv = np.fft.ifft(sig_fft * np.conj(psi_fft))[:N]
        W[i] = conv
    return W


def compute_attention_scalogram(attn_matrix, scales, omega0=5.0):
    """
    Compute scalogram of attention field along lag diagonal.
    attn_matrix: (T, T) numpy array
    Returns: (n_scales, T) scalogram energy
    """
    T = attn_matrix.shape[0]
    scalogram = np.zeros((len(scales), T))
    for b in range(T):
        # extract diagonal at position b (lag = j - b for j > b)
        diag = np.array([attn_matrix[b, min(b+s, T-1)]
                         for s in range(T)])
        W = wavelet_transform_1d(diag, scales, omega0)
        scalogram += np.abs(W)**2
    return scalogram / T


# ── Snapshot collection ──────────────────────────────────────────
@torch.no_grad()
def collect_snapshots(model, data_tokens, n_snapshots=500, block_size=256,
                      head_avg=True):
    """
    Collect attention field snapshots.
    Returns: list of length n_layers, each (n_snapshots, T, T) numpy arrays
             (averaged over heads if head_avg=True)
    """
    model.eval()
    n_layers = model.config['n_layer']
    T = block_size
    snapshots = [[] for _ in range(n_layers)]

    total_tokens = len(data_tokens)
    print(f"  Collecting {n_snapshots} snapshots (T={T})...")

    for s in range(n_snapshots):
        # random starting position
        start = np.random.randint(0, max(1, total_tokens - T))
        chunk = data_tokens[start:start+T]
        if len(chunk) < T:
            chunk = data_tokens[:T]
        idx = torch.tensor(chunk[:T], dtype=torch.long).unsqueeze(0)
        _ = model(idx)
        attn_weights = model.get_attention_weights()

        for l in range(n_layers):
            w = attn_weights[l]  # (1, H, T, T)
            if head_avg:
                w = w[0].mean(0).cpu().numpy()   # (T, T)
            else:
                w = w[0].cpu().numpy()           # (H, T, T) — not used here
            snapshots[l].append(w)

        if (s+1) % 100 == 0:
            print(f"    {s+1}/{n_snapshots} snapshots done")

    return [np.stack(snaps) for snaps in snapshots]  # each: (N, T, T)


# ── Algorithm 1: Scale-Selective POD ────────────────────────────
def scale_selective_pod(snapshots_per_layer, scales, n_modes=3,
                        omega0=5.0):
    """
    Implements Algorithm 1 from pod_paper.tex.
    snapshots_per_layer: list of (N, T, T) arrays
    Returns: dict with keys per layer:
      'scalogram': (n_scales, T) ensemble-averaged energy
      'dominant_scales': top scale indices
      'pod_modes': dict[scale_idx] -> (n_modes, T, T) POD modes
      'eigenvalues': dict[scale_idx] -> (K,) eigenvalue array
      'all_eigenvalues': (K,) full POD spectrum for Reynolds fit
      'beta': spectral decay exponent
      'reynolds': 1/beta
    """
    n_layers = len(snapshots_per_layer)
    results = {}

    for l in range(n_layers):
        snaps = snapshots_per_layer[l]  # (N, T, T)
        N, T, _ = snaps.shape
        print(f"  Layer {l+1}: POD on {N} snapshots of size {T}×{T}")

        # Step 2: ensemble-averaged scalogram
        print(f"    Computing scalogram...")
        ensemble_scalogram = np.zeros((len(scales), T))
        for s in range(N):
            sc = compute_attention_scalogram(snaps[s], scales, omega0)
            ensemble_scalogram += sc
        ensemble_scalogram /= N

        # identify dominant scales (top 4 by total energy)
        scale_energy = ensemble_scalogram.sum(axis=1)
        dominant_idx = np.argsort(-scale_energy)[:4]

        # Step 3: POD at each dominant scale
        pod_modes_per_scale = {}
        eigenvalues_per_scale = {}

        for midx in dominant_idx:
            a_star = scales[midx]
            # scale-filter: extract attention near this lag scale
            filtered = np.zeros_like(snaps)
            for s in range(N):
                # apply Gaussian window of width a_star along diagonals
                for b in range(T):
                    for lag in range(T):
                        j = b + lag
                        if 0 <= j < T:
                            weight = np.exp(-lag**2 / (2 * a_star**2))
                            filtered[s, b, j] = snaps[s, b, j] * weight

            # snapshot POD: form N×N correlation matrix
            U = filtered.reshape(N, -1)          # (N, T²)
            U_mean = U.mean(axis=0, keepdims=True)
            U_centered = U - U_mean
            C = (U_centered @ U_centered.T) / N  # (N, N)
            eigvals, eigvecs = np.linalg.eigh(C)
            # sort descending
            order = np.argsort(-eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            eigvals = np.maximum(eigvals, 0)      # numerical safety

            # POD modes: phi_k = U^T v_k / sqrt(N * lambda_k)
            modes = []
            for k in range(min(n_modes, N)):
                if eigvals[k] > 1e-12:
                    mode = (U_centered.T @ eigvecs[:, k]) / (
                        N * eigvals[k])**0.5
                    modes.append(mode.reshape(T, T))
                else:
                    modes.append(np.zeros((T, T)))

            pod_modes_per_scale[midx] = np.stack(modes)  # (n_modes, T, T)
            eigenvalues_per_scale[midx] = eigvals

        # Step 4: Reynolds number — fit full spectrum
        # use the dominant scale's eigenvalues
        top_scale_idx = dominant_idx[0]
        eigs = eigenvalues_per_scale[top_scale_idx]
        K_fit = min(30, len(eigs))
        eigs_fit = eigs[:K_fit]
        mask = eigs_fit > 1e-12
        if mask.sum() >= 3:
            k_arr = np.arange(1, K_fit+1)[mask]
            log_k = np.log(k_arr)
            log_e = np.log(eigs_fit[mask])
            slope, intercept, r, p, se = linregress(log_k, log_e)
            beta = max(-slope, 0.01)  # beta = -slope, must be positive
        else:
            beta = 1.0
        reynolds = 1.0 / beta

        # Step 5: Kolmogorov n-width — H*(epsilon)
        all_eigs = eigs
        total_energy = all_eigs.sum()

        results[l] = {
            'scalogram': ensemble_scalogram,
            'dominant_idx': dominant_idx,
            'pod_modes': pod_modes_per_scale,
            'eigenvalues': eigenvalues_per_scale,
            'all_eigenvalues': all_eigs,
            'beta': beta,
            'reynolds': reynolds,
            'total_energy': total_energy,
        }
        print(f"    β={beta:.3f}  Re_attn={reynolds:.3f}")

    return results


def compute_min_heads(eigenvalues, total_energy, epsilons=(0.10, 0.05, 0.01)):
    """
    Compute H*(epsilon) for each tolerance using tail-sum criterion.

    H*_l(epsilon) = min{n : sum_{k>n} lambda_k / sum_j lambda_j <= epsilon}

    This is the correct criterion from Theorem 1 (pod_paper.tex):
    average relative reconstruction error below epsilon.

    Note: the old criterion (lambda_{n+1} <= epsilon^2) is WRONG
    for slow-decaying spectra — see paper appendix for explanation.
    """
    heads = {}
    eigs = np.maximum(eigenvalues, 0)
    total = total_energy + 1e-12
    for eps in epsilons:
        tail_sum = np.cumsum(eigs[::-1])[::-1]  # tail sums
        # find first n where tail_sum[n] / total <= eps
        relative_tail = tail_sum / total
        idx = np.searchsorted(-relative_tail, -eps)  # first idx where <= eps
        heads[eps] = min(int(idx) + 1, len(eigs))
    return heads


# ── Cross-coherency ──────────────────────────────────────────────
def compute_coherency_map(snapshots, scale, T, omega0=5.0):
    """
    Compute cross-coherency map |γ_ij(a)| for given scale.
    Returns (T, T) array.
    """
    N = len(snapshots)
    W_cross = np.zeros((T, T), dtype=complex)
    W_auto_i = np.zeros((T, T))
    W_auto_j = np.zeros((T, T))

    for s in range(N):
        attn = snapshots[s]  # (T, T)
        # wavelet transform along each row
        for i in range(T):
            row = attn[i]
            W_row = wavelet_transform_1d(row, [scale], omega0)[0]  # (T,)
            for j in range(T):
                W_cross[i, j] += W_row[i] * np.conj(W_row[j])
                W_auto_i[i, j] += np.abs(W_row[i])**2
                W_auto_j[i, j] += np.abs(W_row[j])**2

    W_cross /= N
    W_auto_i /= N
    W_auto_j /= N
    denom = np.sqrt(W_auto_i * W_auto_j) + 1e-12
    coherency = np.abs(W_cross) / denom
    return np.clip(coherency, 0, 1)


# ── Figure 1: Attention Scalogram ────────────────────────────────
def plot_scalogram(results_base, results_ega, scales, out_path):
    n_layers = len(results_base)
    fig, axes = plt.subplots(2, n_layers,
                             figsize=(n_layers * 2.0, 4.5),
                             constrained_layout=True)

    models = [('BASE', results_base, 'Blues'),
              ('EGA-1', results_ega, 'Oranges')]

    for row, (name, results, cmap) in enumerate(models):
        for l in range(n_layers):
            ax = axes[row, l]
            sc = results[l]['scalogram']  # (n_scales, T)
            T = sc.shape[1]
            im = ax.imshow(
                np.log1p(sc),
                aspect='auto',
                origin='lower',
                extent=[0, T, np.log2(scales[0]), np.log2(scales[-1])],
                cmap=cmap,
                vmin=0,
            )
            ax.set_title(f'L{l+1}', fontsize=8)
            if l == 0:
                ax.set_ylabel(f'{name}\nlog₂(scale)', fontsize=8)
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel('Position $b$', fontsize=8)
            else:
                ax.set_xticklabels([])

    fig.suptitle(
        'Ensemble-averaged attention scalogram\n'
        'BASE (top) vs EGA-1 (bottom), layers 1–6',
        fontsize=9, fontweight='bold')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Figure 2: POD Modes ──────────────────────────────────────────
def plot_pod_modes(results_ega, scales, layers=(1, 3, 5), n_modes=5,
                  out_path=None):
    """
    Plot top-n POD modes as 1D lag profiles for multiple layers.

    Uses layers with non-degenerate eigenvalue structure.
    Layers 1,3,5 (0-indexed) = L2,L4,L6 (1-indexed).
    Each panel shows the lag profile phi_k(tau) for tau=0..63 tokens.
    Red vertical line marks the peak lag of each mode.

    Note: Layer 1 (L1) is excluded because its eigenvalues
    are near-zero beyond Mode 1 (numerical underflow due to
    near-flat spectrum, T_spec=1.00, N=150 snapshots).
    """
    layer_labels = {
        0: 'Layer 1 (early)',
        1: 'Layer 2 (early)',
        2: 'Layer 3 (middle)',
        3: 'Layer 4 (middle)',
        4: 'Layer 5 (late)',
        5: 'Layer 6 (final)',
    }

    n_layers_show = len(layers)
    lags = np.arange(256)

    fig, axes = plt.subplots(n_layers_show, n_modes,
                             figsize=(n_modes * 2.6, n_layers_show * 2.5),
                             constrained_layout=True)

    for row, l in enumerate(layers):
        res   = results_ega[l]
        # use lag profiles from the dominant scale
        top_midx = res['dominant_idx'][0]
        modes = res['pod_modes'][top_midx]   # (n_modes, T, T)
        eigs  = res['eigenvalues'][top_midx]
        total = eigs.sum() + 1e-12

        for col in range(n_modes):
            ax = axes[row, col]

            if col >= len(modes):
                ax.axis('off')
                continue

            # Extract lag profile: mean along query positions
            mode_2d = modes[col]  # (T, T)
            # diagonal mean: for each lag tau, average mode[b, b+tau]
            lag_profile = np.array([
                np.mean([mode_2d[b, b+tau]
                         for b in range(256 - tau)
                         if 0 <= b+tau < 256])
                for tau in range(64)
            ])

            norm = np.linalg.norm(lag_profile)
            if norm < 1e-10:
                # degenerate mode — near zero
                ax.text(0.5, 0.5,
                        f'degenerate\n(norm={norm:.2e})',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=7, color='red')
                ax.set_xlim(0, 63)
            else:
                ax.plot(lags[:64], lag_profile, color='steelblue', lw=1.2)
                ax.axhline(0, color='gray', lw=0.5, ls='--')
                peak = int(np.argmax(np.abs(lag_profile)))
                ax.axvline(peak, color='red', lw=0.8, alpha=0.6)
                ax.set_xlim(0, 63)

            pct = 100 * eigs[col] / total if col < len(eigs) else 0
            ax.set_title(f'Mode {col+1}\n({pct:.1f}%)', fontsize=8)
            if col == 0:
                ax.set_ylabel(layer_labels.get(l, f'Layer {l+1}'),
                              fontsize=8)
            if row == n_layers_show - 1:
                ax.set_xlabel('Lag $\\tau$ (tok)', fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        'Scale-selective POD modes (lag profiles)\n'
        f'EGA-1, layers {[l+1 for l in layers]}  —  '
        f'top-{n_modes} modes per layer',
        fontsize=9, fontweight='bold')

    if out_path:
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved {out_path}")
    else:
        plt.show()


def _plot_pod_modes_legacy(results_ega, scales, layer=2, n_modes=3,
                           out_path=None):
    """Legacy version: 2D heatmap at one layer (kept for reference)."""
    res = results_ega[layer]
    dominant_idx = res['dominant_idx'][:3]  # top 3 scales
    n_scales = len(dominant_idx)

    fig, axes = plt.subplots(n_scales, n_modes,
                             figsize=(n_modes * 2.2, n_scales * 2.2),
                             constrained_layout=True)
    if n_scales == 1:
        axes = axes[np.newaxis, :]

    scale_labels = {0: 'Fine', 1: 'Medium', 2: 'Coarse'}

    for row, midx in enumerate(dominant_idx):
        modes = res['pod_modes'][midx]    # (n_modes, T, T)
        eigs  = res['eigenvalues'][midx]
        a_star = scales[midx]
        total  = eigs.sum() + 1e-12

        for col in range(n_modes):
            ax = axes[row, col]
            mode = modes[col]
            vmax = np.abs(mode).max() + 1e-12
            im = ax.imshow(mode, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           aspect='auto', origin='upper')
            pct = 100 * eigs[col] / total
            ax.set_title(f'Mode {col+1}\n({pct:.1f}%)', fontsize=7)
            if col == 0:
                label = scale_labels.get(row, f'a={a_star:.0f}')
                ax.set_ylabel(f'{label}\n$a^*$={a_star:.0f}', fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

    fig.suptitle(
        f'Top-{n_modes} POD modes per dominant scale\n'
        f'EGA-1, Layer {layer+1}',
        fontsize=9, fontweight='bold')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Figure 3: Coherency Map ──────────────────────────────────────
def plot_coherency(snapshots_ega, scales, layers=(0, 2, 5),
                   scale_idx=1, out_path=None):
    """Cross-coherency at word scale for three layers."""
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers,
                             figsize=(n_layers * 3.0, 3.2),
                             constrained_layout=True)

    a_star = scales[scale_idx]
    T = snapshots_ega[0].shape[1]

    for col, l in enumerate(layers):
        ax = axes[col]
        print(f"    Computing coherency for layer {l+1}...")
        coh = compute_coherency_map(snapshots_ega[l], a_star, T)
        im = ax.imshow(coh, cmap='hot', vmin=0, vmax=1,
                       aspect='auto', origin='upper')
        ax.set_title(f'Layer {l+1}', fontsize=8)
        ax.set_xlabel('Token $j$', fontsize=8)
        if col == 0:
            ax.set_ylabel('Token $i$', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Cross-coherency $|\\gamma_{{ij}}(a^*)|$ at scale $a^*={a_star:.0f}$\n'
        f'EGA-1, layers {[l+1 for l in layers]}',
        fontsize=9, fontweight='bold')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


# ── Tables ───────────────────────────────────────────────────────
def print_reynolds_table(results_base, results_ega):
    n_layers = len(results_base)
    lines = []
    lines.append("% Table 2: Attention Reynolds Number")
    lines.append("% Paste into pod_paper.tex replacing [val] placeholders")
    lines.append("")
    lines.append("BASE:")
    vals = [f"{results_base[l]['reynolds']:.3f}" for l in range(n_layers)]
    lines.append("  " + " & ".join(vals) + " \\\\")
    lines.append("")
    lines.append("EGA-1:")
    vals = [f"{results_ega[l]['reynolds']:.3f}" for l in range(n_layers)]
    lines.append("  " + " & ".join(vals) + " \\\\")
    return "\n".join(lines)


def print_heads_table(results_ega, epsilons=(0.10, 0.05, 0.01)):
    n_layers = len(results_ega)
    lines = []
    lines.append("% Table 3: Minimum Heads H*(epsilon)")
    lines.append("% Paste into pod_paper.tex replacing [val] placeholders")
    lines.append("")
    for eps in epsilons:
        vals = []
        for l in range(n_layers):
            eigs = results_ega[l]['all_eigenvalues']
            total = results_ega[l]['total_energy']
            h = compute_min_heads(eigs, total, (eps,))[eps]
            vals.append(str(h))
        lines.append(f"$\\epsilon={eps}$: " + " & ".join(vals) + " \\\\")
    return "\n".join(lines)


# ── Model loader ─────────────────────────────────────────────────
def load_model(path, config):
    """Load a model checkpoint — handles both raw state_dict and wrapped."""
    model = GPT(config)
    ckpt = torch.load(path, map_location='cpu')
    # handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
    # strip 'module.' prefix if DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}")
    model.eval()
    return model


def inspect_checkpoint(path):
    """Print checkpoint structure to help diagnose loading issues."""
    ckpt = torch.load(path, map_location='cpu')
    print(f"\nCheckpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())[:10]}")
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"  '{k}' -> dict with {len(v)} keys, first: {list(v.keys())[:5]}")
            elif isinstance(v, torch.Tensor):
                print(f"  '{k}' -> Tensor {v.shape}")
            else:
                print(f"  '{k}' -> {type(v).__name__}: {v}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='BASE model .pt path')
    parser.add_argument('--ega',  required=True, help='EGA-1 model .pt path')
    parser.add_argument('--data', default=None,
                        help='Text data file (uses random tokens if not provided)')
    parser.add_argument('--n_snapshots', type=int, default=500)
    parser.add_argument('--block_size',  type=int, default=256)
    parser.add_argument('--n_layer',     type=int, default=6)
    parser.add_argument('--n_head',      type=int, default=8)
    parser.add_argument('--n_embd',      type=int, default=256)
    parser.add_argument('--vocab_size',  type=int, default=65)
    parser.add_argument('--inspect',     action='store_true',
                        help='Just inspect checkpoint structure and exit')
    parser.add_argument('--outdir', default='/mnt/user-data/outputs')
    args = parser.parse_args()

    config = {
        'n_layer':    args.n_layer,
        'n_head':     args.n_head,
        'n_embd':     args.n_embd,
        'vocab_size': args.vocab_size,
        'block_size': args.block_size,
    }

    os.makedirs(args.outdir, exist_ok=True)

    # Inspect mode
    if args.inspect:
        print("=== BASE checkpoint ===")
        inspect_checkpoint(args.base)
        print("\n=== EGA checkpoint ===")
        inspect_checkpoint(args.ega)
        return

    # ── Data ────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}...")
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data_tokens = [stoi[c] for c in text]
        config['vocab_size'] = len(chars)
        print(f"  Vocab size: {len(chars)}, tokens: {len(data_tokens)}")
    else:
        print("No data file provided — using random tokens for testing")
        data_tokens = np.random.randint(0, config['vocab_size'],
                                        size=100000).tolist()

    # ── Wavelet scales ───────────────────────────────────────────
    # Dyadic scales from 2 to 64 tokens, 20 levels
    scales = np.array([2**i for i in np.linspace(1, 6, 20)])
    print(f"Scales: {scales[:5].round(1)} ... {scales[-5:].round(1)}")

    # ── Load models ──────────────────────────────────────────────
    print(f"\nLoading BASE from {args.base}...")
    model_base = load_model(args.base, config)
    print(f"Loading EGA-1 from {args.ega}...")
    model_ega  = load_model(args.ega,  config)

    np.random.seed(42)

    # ── Collect snapshots ────────────────────────────────────────
    print(f"\nCollecting snapshots (n={args.n_snapshots})...")
    print("BASE:")
    snaps_base = collect_snapshots(model_base, data_tokens,
                                   args.n_snapshots, args.block_size)
    print("EGA-1:")
    snaps_ega  = collect_snapshots(model_ega,  data_tokens,
                                   args.n_snapshots, args.block_size)

    # ── Run POD ──────────────────────────────────────────────────
    print("\nRunning scale-selective POD — BASE...")
    results_base = scale_selective_pod(snaps_base, scales)
    print("\nRunning scale-selective POD — EGA-1...")
    results_ega  = scale_selective_pod(snaps_ega,  scales)

    # ── Figures ──────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_scalogram(
        results_base, results_ega, scales,
        out_path=os.path.join(args.outdir, 'attn_scalogram.png'))

    # Figure 2: lag-profile POD modes for layers 2,4,6 (1-indexed)
    # Layers 1,3,5 (0-indexed) have non-degenerate structure
    plot_pod_modes(
        results_ega, scales,
        layers=(1, 3, 5),   # L2, L4, L6 (0-indexed)
        n_modes=5,
        out_path=os.path.join(args.outdir, 'pod_modes.png'))

    print("  Computing coherency map (this takes a few minutes)...")
    # use a subset of snapshots for coherency (speed)
    snaps_ega_sub = [s[:100] for s in snaps_ega]
    plot_coherency(
        snaps_ega_sub, scales, layers=(0, 2, 5), scale_idx=4,
        out_path=os.path.join(args.outdir, 'coherency_map.png'))

    # ── Tables ───────────────────────────────────────────────────
    print("\nGenerating table values...")
    reynolds_txt = print_reynolds_table(results_base, results_ega)
    heads_txt    = print_heads_table(results_ega)

    re_path = os.path.join(args.outdir, 'reynolds_table.txt')
    hd_path = os.path.join(args.outdir, 'heads_table.txt')
    with open(re_path, 'w') as f: f.write(reynolds_txt)
    with open(hd_path, 'w') as f: f.write(heads_txt)
    print(f"  Saved {re_path}")
    print(f"  Saved {hd_path}")

    print("\n" + "="*50)
    print("REYNOLDS TABLE:")
    print(reynolds_txt)
    print("\nHEADS TABLE:")
    print(heads_txt)
    print("="*50)
    print("\nAll outputs saved to", args.outdir)
    print("Upload the .txt files and I will fill in the paper tables.")


if __name__ == '__main__':
    main()
