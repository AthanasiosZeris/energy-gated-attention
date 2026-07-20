"""
GPT: Baseline vs Spectral Energy-Gated Attention  — v2
=======================================================
Compares two attention variants on TinyShakespeare:
  - Baseline : standard scaled dot-product attention
  - EGA      : energy-gated attention with multi-scale
                spectral energy estimated from embeddings

Fixes vs original code
-----------------------
1.  Energy computed from KEY embeddings, not from QK scores
2.  Attention renormalized after gating (sum-to-1 preserved)
3.  Energy z-normalized before thresholding
4.  tau and alpha are learnable nn.Parameters (per scale)
5.  Pre-LayerNorm ordering (LN before each sublayer)
6.  Validation loss averaged over N_EVAL_BATCHES batches
7.  Both models trained simultaneously on identical batches
8.  Full results table + convergence summary printed at end

Fixes vs v1 (from observed runtime errors)
-------------------------------------------
BUG-A  std() warning + RuntimeError during generation:
       When sequence length T=1 (autoregressive generation
       starts from a single token), e_s has shape [B,1,1]
       and std over a single element is undefined → nan/inf
       cascade → multinomial RuntimeError.
       Fix: skip z-normalization when T==1; set e_s=0 so
       the sigmoid gate defaults to 0.5 (neutral / no gate).
       Also use correction=0 in std() for numerical safety.

BUG-B  nan/inf guard missing in generate():
       Added torch.nan_to_num() before softmax and a
       fallback to uniform distribution if probs are invalid.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# ──────────────────────────────────────────────────────────────
# 1.  DATASET
# ──────────────────────────────────────────────────────────────
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn"
       "/master/data/tinyshakespeare/input.txt")
print("Downloading TinyShakespeare …")
text = requests.get(URL).text
print(f"  {len(text):,} characters loaded.")

chars   = sorted(set(text))
VOCAB   = len(chars)
stoi    = {ch: i for i, ch in enumerate(chars)}
itos    = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join(itos[i] for i in l)

data       = torch.tensor(encode(text), dtype=torch.long)
n_split    = int(0.9 * len(data))
train_data = data[:n_split]
val_data   = data[n_split:]

print(f"  Train tokens : {len(train_data):,}")
print(f"  Val   tokens : {len(val_data):,}")

# ──────────────────────────────────────────────────────────────
# 2.  HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
BLOCK_SIZE     = 128
N_EMBED        = 128
N_HEAD         = 4
N_LAYER        = 4
N_SCALES       = 4      # energy scales in EGA head
DROPOUT        = 0.1
LR             = 3e-4
MAX_ITERS      = 3000
EVAL_INTERVAL  = 300
N_EVAL_BATCHES = 50     # batches averaged for stable loss estimate
GRAD_CLIP      = 1.0
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice : {DEVICE}")
print(f"Vocab  : {VOCAB} chars\n")

# ──────────────────────────────────────────────────────────────
# 3.  DATA LOADER
# ──────────────────────────────────────────────────────────────
def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i : i + BLOCK_SIZE]         for i in ix])
    y   = torch.stack([src[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    """
    Average cross-entropy loss over N_EVAL_BATCHES random batches
    for both train and val splits.  Much more stable than single-
    batch loss reported during the training loop.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(N_EVAL_BATCHES)
        for k in range(N_EVAL_BATCHES):
            xb, yb    = get_batch(split)
            _, loss   = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ──────────────────────────────────────────────────────────────
# 4.  ATTENTION MODULES
# ──────────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    """Vanilla causal scaled dot-product attention head."""

    def __init__(self, head_size: int):
        super().__init__()
        self.hs    = head_size
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)                                      # [B,T,hs]
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hs)
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        att = self.drop(F.softmax(scores, dim=-1))           # [B,T,T]
        return att @ v                                       # [B,T,hs]


class SpectralEnergyGatedAttention(nn.Module):
    """
    Causal attention with multi-scale spectral energy gating.

    Design
    ──────
    FIX 1  : Energy estimated from token embeddings (x), not QK scores.
    FIX 2  : Attention renormalized after gating — rows still sum to 1.
    FIX 3  : Energy z-normalized per batch before thresholding.
    FIX 4  : tau (threshold) and alpha (sharpness) learned per scale.
    BUG-A  : T==1 guard — z-normalization skipped when sequence length
              is 1 (autoregressive generation first step); gate set to
              neutral (0.5) via e_s = 0.

    N_SCALES learned linear projections map the full embedding to a
    scalar energy estimate for each token at each scale.  A sigmoid
    gate (smooth threshold) suppresses low-energy keys.  Gates from
    all scales are combined with learned softmax weights.
    """

    def __init__(self, head_size: int, n_scales: int = N_SCALES):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales

        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)

        # FIX 1 — energy projections operating on embeddings
        self.scale_proj = nn.ModuleList([
            nn.Linear(N_EMBED, 1, bias=True)
            for _ in range(n_scales)
        ])

        # FIX 4 — learnable threshold & sharpness, one per scale
        self.tau   = nn.Parameter(torch.zeros(n_scales))
        self.alpha = nn.Parameter(torch.ones(n_scales) * 2.0)

        # Learned combination weights across scales
        self.scale_w = nn.Parameter(torch.ones(n_scales) / n_scales)

        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # ── standard causal attention scores ───────────────────
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hs)
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )

        # ── multi-scale energy gate ─────────────────────────────
        gates = []
        for s, proj in enumerate(self.scale_proj):

            # FIX 1: energy from embedding, shape [B,T,1] → [B,1,T]
            e_s = proj(x).transpose(-2, -1)                # [B,1,T]

            # BUG-A FIX: z-normalization is undefined for T==1
            # (single element → std=0 → division by ~0 → nan/inf)
            # When T==1 we set e_s=0 so sigmoid gives 0.5 (neutral gate)
            if T > 1:
                # FIX 3: z-normalize; correction=0 avoids Bessel issue
                mu  = e_s.mean(dim=-1, keepdim=True)
                std = e_s.std(dim=-1, keepdim=True,
                              correction=0).clamp(min=1e-8)
                e_s = (e_s - mu) / std
            else:
                # Neutral: gate = sigmoid(alpha*(0 - tau))
                # tau initialised at 0 → gate ≈ 0.5, fully transparent
                e_s = torch.zeros_like(e_s)

            g_s = torch.sigmoid(self.alpha[s] * (e_s - self.tau[s]))
            gates.append(g_s)                              # [B,1,T]

        # Combine scales with softmax-normalized weights
        sw   = F.softmax(self.scale_w, dim=0)              # [n_scales]
        gate = sum(sw[s] * gates[s] for s in range(self.n_scales))
        # gate: [B,1,T] → broadcasts over queries → [B,T,T]

        # ── gate + renormalize ──────────────────────────────────
        att = self.drop(F.softmax(scores, dim=-1))         # [B,T,T]
        att = att * gate                                   # energy gate

        # FIX 2: renormalize — preserve sum-to-1 after gating
        att = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return att @ v                                     # [B,T,hs]


# ──────────────────────────────────────────────────────────────
# 5.  TRANSFORMER BLOCK  (pre-LayerNorm — FIX 5)
# ──────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    Transformer block with pre-LayerNorm (FIX 5).
    LayerNorm is applied BEFORE each sublayer — more stable
    gradients than the original post-norm ordering.
    """

    def __init__(self, use_energy_gate: bool = True):
        super().__init__()
        head_size = N_EMBED // N_HEAD
        AttnClass = (SpectralEnergyGatedAttention
                     if use_energy_gate else StandardAttention)

        self.heads = nn.ModuleList(
            [AttnClass(head_size) for _ in range(N_HEAD)]
        )
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.drop = nn.Dropout(DROPOUT)

        self.ff = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.GELU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

        # FIX 5 — one LN before each sublayer
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        # Pre-norm attention sublayer
        xn      = self.ln1(x)
        a_out   = torch.cat([h(xn) for h in self.heads], dim=-1)
        x       = x + self.drop(self.proj(a_out))

        # Pre-norm feed-forward sublayer
        x = x + self.ff(self.ln2(x))
        return x


# ──────────────────────────────────────────────────────────────
# 6.  GPT MODEL
# ──────────────────────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self, use_energy_gate: bool = True):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBED)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.drop    = nn.Dropout(DROPOUT)

        self.blocks = nn.Sequential(*[
            Block(use_energy_gate) for _ in range(N_LAYER)
        ])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, VOCAB, bias=False)

        # Weight tying: share embedding ↔ output projection
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)                              # [B,T,VOCAB]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, VOCAB), targets.view(-1)
            )
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 7.  GENERATION  (BUG-B fix — nan/inf guard)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt: str = "\n", max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 40):
    """
    Autoregressive text generation with temperature + top-k sampling.

    BUG-B FIX: torch.nan_to_num() sanitises logits before softmax.
    A uniform-distribution fallback guards against any residual
    invalid probability tensor that would crash torch.multinomial.
    """
    model.eval()
    idx = torch.tensor(
        encode(prompt), dtype=torch.long, device=DEVICE
    ).unsqueeze(0)                                         # [1, T_prompt]

    for _ in range(max_new_tokens):
        # Crop to last BLOCK_SIZE tokens
        idx_cond  = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / temperature         # [1, VOCAB]

        # BUG-B FIX: sanitise before any operation that may amplify nan
        logits = torch.nan_to_num(
            logits, nan=0.0, posinf=1e4, neginf=-1e4
        )

        # Top-k filtering
        if top_k is not None:
            v, _                     = torch.topk(logits, min(top_k, VOCAB))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)                  # [1, VOCAB]

        # BUG-B FIX: last-resort fallback to uniform if still invalid
        if torch.isnan(probs).any() or (probs < 0).any():
            probs = torch.ones(1, VOCAB, device=DEVICE) / VOCAB

        next_tok = torch.multinomial(probs, num_samples=1) # [1, 1]
        idx      = torch.cat([idx, next_tok], dim=1)

    model.train()
    return decode(idx[0].tolist())


# ──────────────────────────────────────────────────────────────
# 8.  PARALLEL COMPARISON TRAINING  (FIX 6, 7)
# ──────────────────────────────────────────────────────────────

def run_comparison():
    """
    Train Baseline and EGA simultaneously on identical mini-batches.
    Using the same batch at every step means any difference in
    validation loss is attributable to architecture, not data order.
    """
    base_model = GPT(use_energy_gate=False).to(DEVICE)
    ega_model  = GPT(use_energy_gate=True ).to(DEVICE)

    print(f"Baseline params : {base_model.num_parameters():,}")
    print(f"EGA      params : {ega_model.num_parameters():,}\n")

    opt_base = torch.optim.AdamW(
        base_model.parameters(), lr=LR,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_ega = torch.optim.AdamW(
        ega_model.parameters(), lr=LR,
        betas=(0.9, 0.95), weight_decay=0.1
    )

    history = {
        "step":            [],
        "base_train":      [], "base_val": [],
        "ega_train":       [], "ega_val":  [],
        "base_loss_curve": [],
        "ega_loss_curve":  [],
    }

    # ── print header ────────────────────────────────────────────
    sep    = "─" * 72
    header = (f"{'Step':>6}  {'BASE-train':>10}  {'BASE-val':>9}"
              f"  {'EGA-train':>10}  {'EGA-val':>9}"
              f"  {'Delta-val':>10}  {'Who':>6}")
    print(header)
    print(sep)

    for step in range(MAX_ITERS + 1):

        # FIX 7 — identical batch for both models
        xb, yb = get_batch("train")

        # ── baseline step ────────────────────────────────────────
        _, loss_base = base_model(xb, yb)
        opt_base.zero_grad(set_to_none=True)
        loss_base.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP)
        opt_base.step()

        # ── EGA step ─────────────────────────────────────────────
        _, loss_ega = ega_model(xb, yb)
        opt_ega.zero_grad(set_to_none=True)
        loss_ega.backward()
        torch.nn.utils.clip_grad_norm_(ega_model.parameters(), GRAD_CLIP)
        opt_ega.step()

        history["base_loss_curve"].append(loss_base.item())
        history["ega_loss_curve"].append(loss_ega.item())

        # FIX 6 — evaluation averaged over N_EVAL_BATCHES
        if step % EVAL_INTERVAL == 0:
            l_base = estimate_loss(base_model)
            l_ega  = estimate_loss(ega_model)
            delta  = l_base["val"] - l_ega["val"]
            who    = ("EGA↑"  if delta >  0.001 else
                      "BASE↑" if delta < -0.001 else "TIE")

            print(f"{step:>6}  "
                  f"{l_base['train']:>10.4f}  {l_base['val']:>9.4f}  "
                  f"{l_ega['train']:>10.4f}  {l_ega['val']:>9.4f}  "
                  f"{delta:>+10.4f}  {who:>6}")

            history["step"].append(step)
            history["base_train"].append(l_base["train"])
            history["base_val"].append(l_base["val"])
            history["ega_train"].append(l_ega["train"])
            history["ega_val"].append(l_ega["val"])

    print(sep)
    return base_model, ega_model, history


# ──────────────────────────────────────────────────────────────
# 9.  FINAL SUMMARY  (FIX 8)
# ──────────────────────────────────────────────────────────────

def print_summary(history, base_model, ega_model):

    final_base = history["base_val"][-1]
    final_ega  = history["ega_val"][-1]
    delta      = final_base - final_ega

    print(f"\n{'═'*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"  Baseline  final val loss : {final_base:.4f}")
    print(f"  EGA       final val loss : {final_ega:.4f}")
    print(f"  Δ val  (BASE − EGA)      : {delta:+.4f}")
    print(f"  Baseline  params         : {base_model.num_parameters():,}")
    print(f"  EGA       params         : {ega_model.num_parameters():,}")

    if delta > 0.005:
        verdict = "EGA wins — energy gating improved generalisation."
    elif delta < -0.005:
        verdict = "Baseline wins — energy gating hurt generalisation."
    else:
        verdict = "Models are approximately equivalent."
    print(f"\n  Verdict : {verdict}")

    # Convergence speed — first step below midpoint threshold
    threshold = (history["base_val"][0] + final_base) / 2.0

    def first_below(vals, thr):
        for i, v in enumerate(vals):
            if v < thr:
                return history["step"][i]
        return None

    sb = first_below(history["base_val"], threshold)
    se = first_below(history["ega_val"],  threshold)
    print(f"\n  Convergence to val-loss < {threshold:.4f}:")
    print(f"    Baseline : step {sb}")
    print(f"    EGA      : step {se}")
    if sb and se and sb != 0:
        speedup = (sb - se) / sb * 100.0
        print(f"    EGA speedup : {speedup:+.1f}%")

    # Generalisation gap at final checkpoint
    gap_base = history["base_val"][-1] - history["base_train"][-1]
    gap_ega  = history["ega_val"][-1]  - history["ega_train"][-1]
    print(f"\n  Generalisation gap (val − train):")
    print(f"    Baseline : {gap_base:+.4f}")
    print(f"    EGA      : {gap_ega:+.4f}")

    print(f"{'═'*60}\n")


# ──────────────────────────────────────────────────────────────
# 10.  OPTIONAL PLOT  (requires matplotlib)
# ──────────────────────────────────────────────────────────────

def plot_results(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    steps = history["step"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        "GPT: Baseline vs Spectral Energy-Gated Attention\n"
        f"TinyShakespeare  |  {N_LAYER}L × {N_HEAD}H × {N_EMBED}d  |  "
        f"block={BLOCK_SIZE}  batch={BATCH_SIZE}",
        fontsize=11, fontweight="bold"
    )

    # Validation loss
    ax = axes[0]
    ax.plot(steps, history["base_val"], "b-o",
            label="Baseline", markersize=4)
    ax.plot(steps, history["ega_val"],  "r-o",
            label="EGA",      markersize=4)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Training loss
    ax = axes[1]
    ax.plot(steps, history["base_train"], "b-o",
            label="Baseline", markersize=4)
    ax.plot(steps, history["ega_train"],  "r-o",
            label="EGA",      markersize=4)
    ax.set_title("Training Loss (eval batches)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Delta val
    ax    = axes[2]
    delta = [b - e for b, e in
             zip(history["base_val"], history["ega_val"])]
    cols  = ["green" if d > 0 else "red" for d in delta]
    ax.bar(steps, delta, color=cols, alpha=0.7,
           width=EVAL_INTERVAL * 0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Δ Val Loss (BASE − EGA)\nGreen = EGA better")
    ax.set_xlabel("Step"); ax.set_ylabel("Δ Loss")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("energy_attention_results.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved → energy_attention_results.png")


# ──────────────────────────────────────────────────────────────
# 11.  MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    torch.manual_seed(42)

    # Train both models in parallel on identical batches
    base_model, ega_model, history = run_comparison()

    # Summary table
    print_summary(history, base_model, ega_model)

    # Optional loss plots
    plot_results(history)

    # Text generation — both models
    # BUG-B fix active inside generate()
    print("── Baseline sample " + "─" * 40)
    print(generate(base_model, prompt="\n",
                   max_new_tokens=200, temperature=0.8, top_k=40))

    print("\n── EGA sample " + "─" * 45)
    print(generate(ega_model,  prompt="\n",
                   max_new_tokens=200, temperature=0.8, top_k=40))
