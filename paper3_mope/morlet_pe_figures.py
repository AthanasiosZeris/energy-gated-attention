"""
Morlet PE — Uncertainty Plane Figure
=====================================
Upload exp2_egam.pt to Colab first:

  from google.colab import files
  import shutil, os
  os.makedirs("/content/gpt_phase4", exist_ok=True)
  uploaded = files.upload()
  for fname in uploaded:
      shutil.move(fname, f"/content/gpt_phase4/{fname}")

Then run this cell.
No training, no GPU needed — CPU is fine.
Runtime: ~2 minutes.

Output files:
  uncertainty_plane.png     ← main paper figure
  omega_spectrum.png        ← learned frequency spectrum
  sigma_spectrum.png        ← learned bandwidth spectrum
  morlet_pe_params.txt      ← numerical values
"""

import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

CKPT_DIR = "/content/gpt_phase4"

print("=" * 55)
print("  Morlet PE — Uncertainty Plane Analysis")
print("=" * 55)

# ================================================================
# STEP 1 — DEFINE MORLET PE CLASS
# (must match exactly the class used during training)
# ================================================================
class MorletPE(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        n = d_model // 2
        freqs = torch.exp(torch.linspace(
            math.log(1.0),
            math.log(math.pi * 0.99), n))
        self.log_omega = nn.Parameter(torch.log(freqs))
        self.log_sigma = nn.Parameter(torch.log(5.0/freqs))
        self.register_buffer(
            "pos", torch.arange(max_len).float())

    def forward(self, T):
        pos   = self.pos[:T]
        omega = torch.exp(self.log_omega
                          ).clamp(max=math.pi*0.95)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)
        omega = torch.where(omega*sigma < 5.0,
                             5.0/sigma.clamp(min=1e-6),
                             omega)
        env   = torch.exp(
            -pos.unsqueeze(1)**2 /
            (2.0*sigma.unsqueeze(0)**2 + 1e-8))
        phase = pos.unsqueeze(1) * omega.unsqueeze(0)
        pe    = torch.zeros(T, 2*len(omega))
        pe[:, 0::2] = torch.cos(phase) * env
        pe[:, 1::2] = torch.sin(phase) * env
        return pe

    def get_learned_params(self):
        """Extract learned ω and σ after admissibility."""
        with torch.no_grad():
            omega = torch.exp(self.log_omega
                              ).clamp(max=math.pi*0.95)
            sigma = torch.exp(self.log_sigma).clamp(min=1e-3)
            # Apply admissibility correction
            omega = torch.where(omega*sigma < 5.0,
                                 5.0/sigma.clamp(min=1e-6),
                                 omega)
            return omega.numpy(), sigma.numpy()

# ================================================================
# STEP 2 — MINIMAL GPT TO LOAD CHECKPOINT
# ================================================================
N_EMBED=256; N_HEAD=8; N_LAYER=6
DROPOUT=0.1; BLOCK_SIZE=256; VOCAB=65

class EGA1Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.hs    = head_size
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        self.proj  = nn.Linear(N_EMBED, 1, bias=True)
        self.tau   = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1)*2.0)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    def forward(self, x):
        B,T,_ = x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        mu=x.mean(-1,keepdim=True).transpose(-2,-1)
        std=x.std(-1,keepdim=True,correction=0
                   ).clamp(min=1e-8).transpose(-2,-1)
        e=(self.proj(x).transpose(-2,-1)-mu)/std
        g=torch.sigmoid(self.alpha*(e-self.tau))
        att=self.drop(F.softmax(sc,dim=-1))
        att=att*g
        att=att/att.sum(-1,keepdim=True).clamp(min=1e-8)
        return att@v

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        hs=N_EMBED//N_HEAD
        self.heads=nn.ModuleList([
            EGA1Attention(hs) for _ in range(N_HEAD)])
        self.proj=nn.Linear(N_EMBED,N_EMBED)
        self.drop=nn.Dropout(DROPOUT)
        self.ff=nn.Sequential(
            nn.Linear(N_EMBED,4*N_EMBED),nn.GELU(),
            nn.Linear(4*N_EMBED,N_EMBED),nn.Dropout(DROPOUT))
        self.ln1=nn.LayerNorm(N_EMBED)
        self.ln2=nn.LayerNorm(N_EMBED)
    def forward(self,x):
        xn=self.ln1(x)
        ao=torch.cat([h(xn) for h in self.heads],dim=-1)
        x=x+self.drop(self.proj(ao))
        return x+self.ff(self.ln2(x))

class EGA_MORLET_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB,N_EMBED)
        self.pe=MorletPE(N_EMBED,BLOCK_SIZE)
        self.drop=nn.Dropout(DROPOUT)
        self.blocks=nn.Sequential(*[
            Block() for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBED)
        self.head=nn.Linear(N_EMBED,VOCAB,bias=False)
        self.tok_emb.weight=self.head.weight
    def forward(self,idx,targets=None):
        B,T=idx.shape
        x=self.drop(
            self.tok_emb(idx)+
            self.pe(T).unsqueeze(0))
        x=self.blocks(x); x=self.ln_f(x)
        logits=self.head(x)
        loss=(F.cross_entropy(
                  logits.view(-1,VOCAB),targets.view(-1))
              if targets is not None else None)
        return logits,loss

# ================================================================
# STEP 3 — LOAD CHECKPOINT
# ================================================================
ckpt_path = os.path.join(CKPT_DIR, "exp2_egam.pt")
if not os.path.exists(ckpt_path):
    print(f"  ERROR: {ckpt_path} not found")
    print("  Upload exp2_egam.pt first")
    raise FileNotFoundError(ckpt_path)

print(f"\n  Loading {ckpt_path} ...")
ck     = torch.load(ckpt_path, map_location="cpu")
model  = EGA_MORLET_GPT()

# Load compatible weights
state  = ck["model_state"]
ms     = model.state_dict()
compat = {k:v for k,v in state.items()
           if k in ms and v.shape==ms[k].shape}
ms.update(compat)
model.load_state_dict(ms, strict=False)
model.eval()

val_loss = ck["history"]["val"][-1]
print(f"  ✓ Loaded  val={val_loss:.4f}  "
      f"({len(compat)}/{len(ms)} weights)")

# ================================================================
# STEP 4 — EXTRACT LEARNED PARAMETERS
# ================================================================
omega, sigma = model.pe.get_learned_params()
n_pairs      = len(omega)   # = N_EMBED // 2 = 128

print(f"\n  Extracted {n_pairs} (ω, σ) pairs")
print(f"  ω range: [{omega.min():.4f}, {omega.max():.4f}]")
print(f"  σ range: [{sigma.min():.4f}, {sigma.max():.4f}]")

# Compute ω·σ products (admissibility check)
products = omega * sigma
print(f"  ω·σ range: [{products.min():.4f}, "
      f"{products.max():.4f}]")
print(f"  Dims with ω·σ < 5: "
      f"{(products < 5.0).sum()} / {n_pairs}")

# Sin/cos initialization frequencies for comparison
omega_sincos = np.exp(np.linspace(
    math.log(1.0), math.log(math.pi*0.99), n_pairs))
sigma_sincos = np.full(n_pairs, np.inf)  # σ=∞ for sin/cos

# Compare to initialization
omega_init = omega_sincos.copy()
sigma_init = 5.0 / omega_init

print(f"\n  Comparison to initialization:")
print(f"  ω shift: mean={np.mean(omega-omega_init):+.4f}  "
      f"std={np.std(omega-omega_init):.4f}")
print(f"  σ shift: mean={np.mean(sigma-sigma_init):+.4f}  "
      f"std={np.std(sigma-sigma_init):.4f}")

# ================================================================
# STEP 5 — IDENTIFY LINGUISTIC SCALE CLUSTERS
# ================================================================
# Character scale:  σ ≈ 2-4 tokens
# Word scale:       σ ≈ 8-15 tokens
# Clause scale:     σ ≈ 25-50 tokens
# Sentence scale:   σ ≈ 60-120 tokens

scales = {
    "Character\n(σ≈2-4)":  (sigma >= 1)   & (sigma < 6),
    "Word\n(σ≈8-15)":      (sigma >= 6)   & (sigma < 20),
    "Clause\n(σ≈25-50)":   (sigma >= 20)  & (sigma < 55),
    "Sentence\n(σ≥60)":    (sigma >= 55),
}

print(f"\n  Linguistic scale distribution:")
for scale_name, mask in scales.items():
    n    = mask.sum()
    pct  = 100*n/n_pairs
    name = scale_name.replace("\n", " ")
    print(f"    {name:<22} {n:>3} dims  ({pct:.0f}%)")

# ================================================================
# STEP 6 — GENERATE FIGURES
# ================================================================

# ── Figure 1: Main uncertainty plane ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Morlet Positional Encoding — Learned Parameters\n"
    "EGA-MORLET model  |  TinyShakespeare  |  "
    "5000 training steps",
    fontsize=12, fontweight="bold"
)

# Panel A — Uncertainty plane
ax = axes[0]

# Color by dimension index (fine → coarse)
dim_idx = np.arange(n_pairs)
sc = ax.scatter(sigma, omega,
                c=dim_idx, cmap="plasma",
                s=40, alpha=0.8, zorder=3,
                label="Learned (σᵢ, ωᵢ)")

# Admissibility bound: ω·σ = 5
sigma_range = np.logspace(
    math.log10(max(sigma.min()*0.5, 0.1)),
    math.log10(sigma.max()*1.5), 200)
ax.plot(sigma_range, 5.0/sigma_range,
        "r--", linewidth=2,
        label=r"Admissibility: $\omega\sigma=5$",
        zorder=2)

# Sin/cos reference line (σ→∞, shown as right boundary)
ax.axvline(sigma.max()*3, color="gray",
           linewidth=1.5, linestyle=":",
           alpha=0.5, label="Sin/cos: σ→∞")

# Heisenberg bound annotation
ax.fill_between(sigma_range,
                5.0/sigma_range,
                ax.get_ylim()[0] if ax.get_ylim()[0] > 0
                else 0.01,
                alpha=0.08, color="red",
                label="Inadmissible region")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("σᵢ (bandwidth / position locality)",
              fontsize=10)
ax.set_ylabel("ωᵢ (center frequency)", fontsize=10)
ax.set_title(
    "Uncertainty Plane\n"
    "Each point = one embedding dimension pair\n"
    r"Heisenberg bound: $\Delta b \cdot \Delta\omega \geq \frac{1}{2}$",
    fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

plt.colorbar(sc, ax=ax,
             label="Dimension index (0=fine, 127=coarse)",
             fraction=0.05)

# Annotate linguistic scales
scale_positions = {
    "Character": (3.0,  omega[sigma < 6].mean()
                         if (sigma < 6).any() else 2.0),
    "Word":      (11.0, omega[(sigma>=6)&(sigma<20)].mean()
                         if ((sigma>=6)&(sigma<20)).any()
                         else 0.5),
    "Clause":    (37.0, omega[(sigma>=20)&(sigma<55)].mean()
                         if ((sigma>=20)&(sigma<55)).any()
                         else 0.2),
    "Sentence":  (80.0, omega[sigma>=55].mean()
                         if (sigma>=55).any() else 0.05),
}
for name, (sx, oy) in scale_positions.items():
    if not np.isnan(oy):
        ax.annotate(name,
                    xy=(sx, oy),
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="navy",
                              alpha=0.7))

# Panel B — ω spectrum comparison
ax = axes[1]
dim_axis = np.arange(n_pairs)

ax.plot(dim_axis, omega,
        color="#E91E63", linewidth=2,
        label="Learned ωᵢ", zorder=3)
ax.plot(dim_axis, omega_init,
        color="#2196F3", linewidth=1.5,
        linestyle="--", label="Init ωᵢ (dyadic)",
        alpha=0.7, zorder=2)
ax.fill_between(dim_axis, omega, omega_init,
                alpha=0.2, color="#E91E63",
                label="Deviation from init")

ax.set_xlabel("Embedding dimension index", fontsize=10)
ax.set_ylabel("ωᵢ (center frequency)", fontsize=10)
ax.set_title(
    "Learned vs Initial Frequencies\n"
    "Deviations show corpus-specific adaptation\n"
    "Sin/cos uses fixed dyadic spacing (dashed)",
    fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Annotate scale regions
for label, lo, hi, col in [
    ("Char", 0, 10, "#FF9800"),
    ("Word", 10, 40, "#4CAF50"),
    ("Clause", 40, 90, "#9C27B0"),
    ("Sentence", 90, 128, "#00BCD4"),
]:
    ax.axvspan(lo, min(hi, n_pairs),
               alpha=0.08, color=col)
    mid = (lo + min(hi, n_pairs)) / 2
    ax.text(mid, omega.max()*0.7, label,
            ha="center", fontsize=7,
            color=col, fontweight="bold")

# Panel C — σ spectrum comparison
ax = axes[2]

ax.plot(dim_axis, sigma,
        color="#FF9800", linewidth=2,
        label="Learned σᵢ", zorder=3)
ax.plot(dim_axis, sigma_init,
        color="#2196F3", linewidth=1.5,
        linestyle="--", label="Init σᵢ (=5/ω)",
        alpha=0.7, zorder=2)
ax.fill_between(dim_axis, sigma, sigma_init,
                alpha=0.2, color="#FF9800")

# Reference lines for linguistic scales
for val, label, col in [
    (3,   "Char (3 tok)",     "#FF9800"),
    (10,  "Word (10 tok)",    "#4CAF50"),
    (35,  "Clause (35 tok)",  "#9C27B0"),
    (80,  "Sentence (80 tok)","#00BCD4"),
]:
    ax.axhline(val, color=col, linewidth=1,
               linestyle=":", alpha=0.7)
    ax.text(n_pairs*0.98, val*1.05, label,
            ha="right", fontsize=7, color=col)

ax.set_xlabel("Embedding dimension index", fontsize=10)
ax.set_ylabel("σᵢ (bandwidth / locality)", fontsize=10)
ax.set_title(
    "Learned vs Initial Bandwidths\n"
    "σᵢ controls spatial extent of positional influence\n"
    "Horizontal lines = linguistic temporal scales",
    fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
out1 = os.path.join(CKPT_DIR, "uncertainty_plane.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Saved → {out1}")

# ── Figure 2: The spiral diagram ────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    "MoPE Complex Plane Geometry — The Inward Spiral\n"
    "Comparing MoPE, RoPE, and sin/cos in complex plane",
    fontsize=12, fontweight="bold"
)

# Show spiral for 3 representative dimensions
dim_examples = [
    (0,   "Dim 0 (fine scale)"),
    (64,  "Dim 64 (medium scale)"),
    (127, "Dim 127 (coarse scale)"),
]

T_plot = 256
b_vals  = np.arange(T_plot)

for pi, (dim, label) in enumerate(dim_examples):
    ax = axes2[pi]
    wi = omega[dim]
    si = sigma[dim]

    # MoPE spiral
    re_mope = np.cos(wi * b_vals) * np.exp(
        -b_vals**2 / (2*si**2))
    im_mope = np.sin(wi * b_vals) * np.exp(
        -b_vals**2 / (2*si**2))

    # RoPE circle (same ω, no envelope)
    wi_rope = omega_sincos[dim]
    re_rope = np.cos(wi_rope * b_vals)
    im_rope = np.sin(wi_rope * b_vals)

    # Color by position
    colors = plt.cm.viridis(b_vals / T_plot)

    # Plot MoPE spiral
    for i in range(len(b_vals)-1):
        ax.plot(re_mope[i:i+2], im_mope[i:i+2],
                color=colors[i], linewidth=1.5, alpha=0.8)

    # Plot RoPE circle (unit circle)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            "b--", linewidth=1, alpha=0.4,
            label="RoPE / sin/cos (unit circle)")

    # Mark start and some positions
    ax.scatter([re_mope[0]], [im_mope[0]],
               color="green", s=80, zorder=5,
               label="b=0 (start)")
    for b_mark in [10, 30, 60, 100]:
        if b_mark < T_plot:
            ax.scatter([re_mope[b_mark]],
                       [im_mope[b_mark]],
                       color="red", s=30,
                       alpha=0.7, zorder=4)
            ax.annotate(f"b={b_mark}",
                        (re_mope[b_mark], im_mope[b_mark]),
                        fontsize=6, alpha=0.8)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(
        f"{label}\n"
        f"ω={wi:.3f}  σ={si:.1f} tokens\n"
        f"MoPE: inward spiral | RoPE: unit circle",
        fontsize=9)
    ax.set_xlabel("Re (cos component)", fontsize=8)
    ax.set_ylabel("Im (sin component)", fontsize=8)
    if pi == 0:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Add colorbar for position
    sm = ScalarMappable(norm=Normalize(0, T_plot),
                         cmap="viridis")
    sm.set_array([])
    plt.colorbar(sm, ax=ax,
                 label="Token position b",
                 fraction=0.05)

plt.tight_layout()
out2 = os.path.join(CKPT_DIR, "morlet_spiral.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {out2}")

# ── Figure 3: Cross-correlation comparison ───────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    "Proposition 3: MoPE Cross-Correlation = RoPE × Gaussian\n"
    r"$C_\mathrm{MoPE}(\tau) = \cos(\omega_i\tau) "
    r"\cdot e^{-\tau^2/4\sigma_i^2}$",
    fontsize=11, fontweight="bold"
)

tau_vals = np.arange(-100, 101)

for pi, dim in enumerate([0, 64]):
    ax = axes3[pi]
    wi = omega[dim]
    si = sigma[dim]

    # Full MoPE cross-correlation
    xcorr_mope = (np.cos(wi * tau_vals) *
                  np.exp(-tau_vals**2 / (4*si**2)))

    # RoPE component (no envelope)
    xcorr_rope = np.cos(wi * tau_vals)

    # Gaussian envelope only
    envelope = np.exp(-tau_vals**2 / (4*si**2))

    ax.plot(tau_vals, xcorr_rope,
            "b--", linewidth=1.5, alpha=0.6,
            label=f"RoPE: cos(ω·τ), ω={wi:.3f}")
    ax.fill_between(tau_vals, envelope, -envelope,
                    alpha=0.15, color="orange",
                    label=f"Gaussian: exp(-τ²/4σ²), σ={si:.1f}")
    ax.plot(tau_vals, xcorr_mope,
            color="#E91E63", linewidth=2.5,
            label="MoPE = RoPE × Gaussian")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5,
               linestyle="--")

    ax.set_xlabel("Lag τ (token distance)", fontsize=10)
    ax.set_ylabel("Cross-correlation C(τ)", fontsize=10)
    ax.set_title(
        f"Dim {dim}: ω={wi:.3f}, σ={si:.1f} tokens\n"
        f"Locality: 95% of influence within "
        f"±{2*si:.0f} tokens",
        fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-80, 80)

plt.tight_layout()
out3 = os.path.join(CKPT_DIR, "xcorr_comparison.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved → {out3}")

# ================================================================
# STEP 7 — PRINT NUMERICAL SUMMARY
# ================================================================
print(f"\n{'='*55}")
print("  LEARNED PARAMETER SUMMARY")
print(f"{'='*55}")

print(f"\n  Model: EGA-MORLET  val={val_loss:.4f}")
print(f"  Parameters: {n_pairs} dimension pairs (ω, σ)")

print(f"\n  ω (center frequency):")
print(f"    min:  {omega.min():.4f}  (coarsest scale)")
print(f"    max:  {omega.max():.4f}  (finest scale)")
print(f"    mean: {omega.mean():.4f}")
print(f"    vs init mean: {omega_init.mean():.4f}")

print(f"\n  σ (bandwidth):")
print(f"    min:  {sigma.min():.2f} tokens  (most local)")
print(f"    max:  {sigma.max():.2f} tokens  (most global)")
print(f"    mean: {sigma.mean():.2f} tokens")

print(f"\n  Admissibility (ω·σ):")
print(f"    min:  {products.min():.4f}  "
      f"(should be ≥ 5.0)")
print(f"    max:  {products.max():.4f}")
print(f"    mean: {products.mean():.4f}")
n_viol = (products < 4.99).sum()
print(f"    Violations < 5.0: {n_viol} / {n_pairs}")

print(f"\n  Linguistic scale clusters:")
for scale_name, mask in scales.items():
    n   = mask.sum()
    name= scale_name.replace("\n"," ")
    if n > 0:
        w_mean = omega[mask].mean()
        s_mean = sigma[mask].mean()
        print(f"    {name:<25} {n:>3} dims  "
              f"ω̄={w_mean:.3f}  σ̄={s_mean:.1f} tok")

# Save numerical values
txt_lines = [
    "Morlet PE Learned Parameters",
    "="*40,
    f"Model: EGA-MORLET  val={val_loss:.4f}",
    f"Dimension pairs: {n_pairs}",
    "",
    "dim, omega, sigma, omega*sigma",
]
for i in range(n_pairs):
    txt_lines.append(
        f"{i:>3}, {omega[i]:.6f}, "
        f"{sigma[i]:.6f}, {products[i]:.6f}"
    )
txt_path = os.path.join(CKPT_DIR, "morlet_pe_params.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(txt_lines))
print(f"\n  Numerical values saved → {txt_path}")

# ================================================================
# STEP 8 — DOWNLOAD ALL
# ================================================================
print("\nDownloading figures …")
from google.colab import files
files.download(out1)   # uncertainty_plane.png
files.download(out2)   # morlet_spiral.png
files.download(out3)   # xcorr_comparison.png
files.download(txt_path)

print("\nDone. Files in your Downloads folder:")
print("  uncertainty_plane.png   ← main paper figure")
print("  morlet_spiral.png       ← complex plane geometry")
print("  xcorr_comparison.png    ← Proposition 3 illustration")
print("  morlet_pe_params.txt    ← all (ω,σ) values")
