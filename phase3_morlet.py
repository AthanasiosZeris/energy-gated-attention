# ================================================================
# EGA-M Fixed (τ=0) — The Central Question of Phase 3
# ================================================================
# Fresh Colab session — everything self-contained in one cell.
#
# Scientific question:
#   Can parametric Morlet wavelets beat a simple linear
#   projection (EGA-1) when properly initialized?
#
# The only change from original EGA-M:
#   tau initialized at 0.0  (was +0.35 from EGA-C values)
#
# Why this matters:
#   EGA-M original: tau=+0.35 → gate suppresses 84% of tokens
#                   → over-aggressive, model learns nothing new
#   EGA-M fixed:    tau=0.0   → gate starts neutral at sigmoid(0)=0.5
#                   → learns the right threshold from data
#
# Reference results to beat:
#   BASE   val=1.4742  (control)
#   EGA-1  val=1.3712  (best so far — 1 linear projection)
#   EGA-C  val=1.3745  (causal conv filter bank)
#   EGA-M  val=1.4800  (original Morlet — failed due to high tau)
#
# Expected runtime: ~60 min on T4
# ================================================================

import math, os, gc, warnings, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(42)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    p = torch.cuda.get_device_properties(0)

print("=" * 58)
print("  EGA-M Fixed (τ=0) — Phase 3 Central Question")
print("=" * 58)
print(f"  Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU    : {p.name}")
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  Free   : {free:.2f} GB")
print("=" * 58 + "\n")

# ================================================================
# 1.  DATASET
# ================================================================
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn"
       "/master/data/tinyshakespeare/input.txt")
print("Downloading TinyShakespeare …")
text  = requests.get(URL).text
chars = sorted(set(text))
VOCAB = len(chars)
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join(itos[i] for i in l)

data       = torch.tensor(encode(text), dtype=torch.long)
n_split    = int(0.9 * len(data))
train_data = data[:n_split]
val_data   = data[n_split:]
print(f"  {len(text):,} chars  |  "
      f"train={len(train_data):,}  val={len(val_data):,}\n")

# ================================================================
# 2.  HYPERPARAMETERS
# ================================================================
BATCH_SIZE     = 64
BLOCK_SIZE     = 256
N_EMBED        = 256
N_HEAD         = 8
N_LAYER        = 6
DROPOUT        = 0.1
LR             = 3e-4
MAX_ITERS      = 5000
EVAL_INTERVAL  = 500
N_EVAL_BATCHES = 50
WARMUP_ITERS   = 300
GRAD_CLIP      = 1.0
FILTER_LENGTHS = [3, 7, 15, 31]
CKPT_DIR       = "/content/gpt_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ================================================================
# 3.  DATA UTILITIES
# ================================================================
def get_batch(split):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i:i+BLOCK_SIZE]     for i in ix])
    y   = torch.stack([src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(N_EVAL_BATCHES, device=DEVICE)
        for k in range(N_EVAL_BATCHES):
            xb, yb = get_batch(split)
            with autocast(enabled=USE_AMP):
                _, loss = model(xb, yb)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step):
    if step < WARMUP_ITERS:
        return LR * step / max(1, WARMUP_ITERS)
    t = (step-WARMUP_ITERS) / max(1, MAX_ITERS-WARMUP_ITERS)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))

def znorm(t, T):
    if T > 1:
        mu  = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True,
                    correction=0).clamp(min=1e-8)
        return (t - mu) / std
    return torch.zeros_like(t)

# ================================================================
# 4.  FLOAT32 MORLET  (prevents AMP underflow)
# ================================================================
def apply_causal_morlet_f32(x, omega0, sigma, length):
    """
    Morlet wavelet convolution in float32.
    Prevents exp() underflow when running under AMP (float16).
    Cast back to input dtype at the end.
    """
    B, T, _ = x.shape
    device  = x.device

    t   = torch.arange(length, dtype=torch.float32, device=device)
    env = torch.exp(-t**2 / (2.0 * sigma.float()**2 + 1e-6))
    rk  = torch.nan_to_num(torch.cos(omega0.float()*t) * env)
    ik  = torch.nan_to_num(torch.sin(omega0.float()*t) * env)

    # Project embedding → scalar signal
    x_1d  = x.float().mean(dim=-1, keepdim=True).transpose(1,2)
    x_pad = F.pad(x_1d, (length-1, 0))

    real = torch.nan_to_num(
        F.conv1d(x_pad, rk.view(1,1,-1))[:,:,:T]
    )
    imag = torch.nan_to_num(
        F.conv1d(x_pad, ik.view(1,1,-1))[:,:,:T]
    )
    return torch.cat([real, imag], dim=1).to(x.dtype)

# ================================================================
# 5.  ATTENTION MODULES
# ================================================================

class StandardAttention(nn.Module):
    """Baseline — vanilla causal dot-product attention."""
    def __init__(self, head_size):
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
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        return self.drop(F.softmax(sc, dim=-1)) @ v


class LinearEnergyGatedAttention(nn.Module):
    """EGA-1 — single learned linear projection energy gate."""
    def __init__(self, head_size, n_scales=1):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        self.scale_proj = nn.ModuleList([
            nn.Linear(N_EMBED, 1, bias=True)
            for _ in range(n_scales)
        ])
        self.tau     = nn.Parameter(torch.zeros(n_scales))
        self.alpha   = nn.Parameter(torch.ones(n_scales) * 2.0)
        self.scale_w = nn.Parameter(torch.ones(n_scales)/n_scales)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        gates = []
        for s, proj in enumerate(self.scale_proj):
            e_s = znorm(proj(x).transpose(-2,-1), T)
            gates.append(
                torch.sigmoid(self.alpha[s]*(e_s-self.tau[s]))
            )
        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s]*gates[s] for s in range(self.n_scales))
        att  = self.drop(F.softmax(sc, dim=-1))
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v


class MorletEnergyGatedAttentionFixed(nn.Module):
    """
    EGA-M Fixed — parametric Morlet wavelet energy gate.

    THE FIX vs original EGA-M:
      tau initialized at 0.0 (not +0.35 from EGA-C)

    Why original failed:
      tau=+0.35 → sigmoid(2*(e-0.35)) where e~N(0,1)
      → P(gate>0.5) = P(e>0.35) ≈ 36% of tokens pass
      → 64% suppressed from the start, model stuck

    With tau=0.0:
      → sigmoid(2*(e-0.0)) → P(gate>0.5) = P(e>0) = 50%
      → neutral start, model learns optimal threshold

    Additional fix: float32 Morlet computation (AMP safety)
    """
    FILTER_LENGTHS = [3, 7, 15, 31]

    def __init__(self, head_size, n_scales=4):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales
        fl            = self.FILTER_LENGTHS[:n_scales]
        self.filter_lengths = fl

        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)

        # Morlet parameters — log space for positivity
        omega0_init = [math.pi / l for l in fl]
        sigma_init  = [5.0 / w for w in omega0_init]
        self.log_omega0 = nn.Parameter(
            torch.tensor([math.log(w) for w in omega0_init],
                          dtype=torch.float32)
        )
        self.log_sigma  = nn.Parameter(
            torch.tensor([math.log(s) for s in sigma_init],
                          dtype=torch.float32)
        )

        # ── THE FIX ──────────────────────────────────────────────
        # tau = 0.0  →  neutral gate at initialization
        # Original had tau = [0.354, 0.344, 0.341, 0.323]
        # which over-suppressed from step 0
        self.tau   = nn.Parameter(torch.zeros(n_scales))
        # ─────────────────────────────────────────────────────────

        self.alpha   = nn.Parameter(torch.ones(n_scales) * 2.0)
        self.scale_w = nn.Parameter(torch.ones(n_scales)/n_scales)

        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def _get_morlet_params(self):
        omega0 = torch.exp(
            self.log_omega0.float()
        ).clamp(max=math.pi*0.95)
        sigma  = torch.exp(
            self.log_sigma.float()
        ).clamp(min=1e-3)
        product = omega0 * sigma
        return torch.where(
            product < 5.0,
            5.0 / sigma.clamp(min=1e-6),
            omega0
        ), sigma

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)

        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T]==0, float("-inf"))

        omega0, sigma = self._get_morlet_params()
        gates = []
        for s in range(self.n_scales):
            c  = apply_causal_morlet_f32(
                x, omega0[s], sigma[s], self.filter_lengths[s]
            )
            e  = torch.nan_to_num(
                c[:,0:1,:]**2 + c[:,1:2,:]**2, nan=0.0
            )
            es = znorm(e.to(x.dtype), T)
            gates.append(
                torch.sigmoid(self.alpha[s]*(es-self.tau[s]))
            )

        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s]*gates[s] for s in range(self.n_scales))

        att  = self.drop(F.softmax(sc, dim=-1))
        att  = torch.nan_to_num(att, nan=0.0)
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v

    def get_morlet_params(self):
        with torch.no_grad():
            o, s = self._get_morlet_params()
            return (o.cpu(), s.cpu(),
                    self.tau.cpu(), self.alpha.cpu(),
                    F.softmax(self.scale_w, dim=0).cpu())

# ================================================================
# 6.  TRANSFORMER BLOCK + GPT
# ================================================================

class Block(nn.Module):
    def __init__(self, attn_class, attn_kwargs):
        super().__init__()
        hs         = N_EMBED // N_HEAD
        self.heads = nn.ModuleList([
            attn_class(hs, **attn_kwargs) for _ in range(N_HEAD)
        ])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.drop = nn.Dropout(DROPOUT)
        self.ff   = nn.Sequential(
            nn.Linear(N_EMBED, 4*N_EMBED), nn.GELU(),
            nn.Linear(4*N_EMBED, N_EMBED), nn.Dropout(DROPOUT),
        )
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.ln2  = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        xn = self.ln1(x)
        ao = torch.cat([h(xn) for h in self.heads], dim=-1)
        x  = x + self.drop(self.proj(ao))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, attn_class, attn_kwargs, label=""):
        super().__init__()
        self.label   = label
        self.tok_emb = nn.Embedding(VOCAB, N_EMBED)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[
            Block(attn_class, attn_kwargs)
            for _ in range(N_LAYER)
        ])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, VOCAB, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) +
            self.pos_emb(torch.arange(T, device=idx.device))
        )
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = (F.cross_entropy(
                      logits.view(-1,VOCAB), targets.view(-1))
                  if targets is not None else None)
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

# ================================================================
# 7.  GENERATION
# ================================================================
@torch.no_grad()
def generate(model, prompt="\n", max_new_tokens=200,
             temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor(encode(prompt),
                        dtype=torch.long,
                        device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits    = logits[:,-1,:] / temperature
        logits    = torch.nan_to_num(logits,
                                      nan=0.0,
                                      posinf=1e4,
                                      neginf=-1e4)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, VOCAB))
            logits[logits < v[:,[-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or (probs < 0).any():
            probs = torch.ones(1, VOCAB, device=DEVICE) / VOCAB
        idx = torch.cat([idx,
                          torch.multinomial(probs, 1)], dim=1)
    model.train()
    return decode(idx[0].tolist())

# ================================================================
# 8.  TRAIN ONE MODEL
# ================================================================
def train_model(label, model, save_name):
    opt    = torch.optim.AdamW(
        model.parameters(), lr=LR,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = GradScaler(enabled=USE_AMP)
    h      = {"train":[], "val":[], "curve":[],
              "step": list(range(0, MAX_ITERS+1, EVAL_INTERVAL))}
    ckpt   = os.path.join(CKPT_DIR, f"{save_name}.pt")

    pbar = tqdm(range(MAX_ITERS+1),
                desc=f"  {label}", leave=True)

    for step in pbar:
        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch("train")
        with autocast(enabled=USE_AMP):
            _, loss = model(xb, yb)

        if torch.isnan(loss):
            tqdm.write(f"  nan at step {step} — skipping")
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )
        scaler.step(opt)
        scaler.update()

        h["curve"].append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        if step % EVAL_INTERVAL == 0:
            ev = estimate_loss(model)
            h["train"].append(ev["train"])
            h["val"].append(ev["val"])
            tqdm.write(
                f"  {label}  step={step:>5}  "
                f"train={ev['train']:.4f}  "
                f"val={ev['val']:.4f}"
            )
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "history":         h,
                "step":            step,
            }, ckpt)

    return h, model

# ================================================================
# 9.  BUILD AND TRAIN — THREE MODELS
# ================================================================
# We train BASE, EGA-1, and EGA-M-Fixed in sequence.
# BASE and EGA-1 are fast (~20 min each).
# EGA-M-Fixed is the key experiment (~20 min).
# Total: ~60 min.
#
# If BASE and EGA-1 checkpoints already exist from a
# previous session in /content/gpt_checkpoints/, they
# will be loaded instead of retrained.

def load_or_train(label, save_name, attn_class, attn_kwargs):
    ckpt = os.path.join(CKPT_DIR, f"{save_name}.pt")
    m    = GPT(attn_class, attn_kwargs, label=label).to(DEVICE)
    print(f"\n── {label}  ({m.num_parameters():,} params) ──────────")

    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=DEVICE)
        if ck.get("step", 0) >= MAX_ITERS:
            m.load_state_dict(ck["model_state"])
            h = ck["history"]
            print(f"  ✓ Loaded from checkpoint  "
                  f"val={h['val'][-1]:.4f}")
            del ck
            return h, m

    print(f"  Training from scratch …")
    h, m = train_model(label, m, save_name)
    return h, m

configs = [
    ("BASE",          "BASE",      StandardAttention,             {}),
    ("EGA-1",         "EGA-1",     LinearEnergyGatedAttention,    {"n_scales":1}),
    ("EGA-M-Fixed",   "EGA-M-Fix", MorletEnergyGatedAttentionFixed,{"n_scales":4}),
]

all_history = {}
all_models  = {}

print("\n" + "="*58)
print("  TRAINING SEQUENCE")
print("  BASE → EGA-1 → EGA-M-Fixed")
print("="*58)

for label, save_name, cls, kwargs in configs:
    h, m = load_or_train(label, save_name, cls, kwargs)
    all_history[label] = h
    all_models[label]  = m
    # Free GPU memory between models
    m.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    # Move back to GPU for next use
    m.to(DEVICE)

# ================================================================
# 10.  RESULTS
# ================================================================
print(f"\n{'═'*58}")
print("  FINAL RESULTS — THE CENTRAL QUESTION")
print(f"{'═'*58}")

bv = all_history["BASE"]["val"][-1]
print(f"\n  {'Model':<14} {'Val':>8} {'ΔBASE':>8} "
      f"{'Gap':>8}  Conclusion")
print("  " + "─"*54)

best = min(all_history[lb]["val"][-1] for lb in all_history)
conclusions = {
    "BASE":        "control",
    "EGA-1":       "best linear projection",
    "EGA-M-Fixed": "parametric Morlet (τ=0 fix)",
}
for label, _, _, _ in configs:
    v   = all_history[label]["val"][-1]
    t   = all_history[label]["train"][-1]
    db  = bv - v
    gap = v - t
    mrk = " ◄BEST" if v == best else ""
    print(f"  {label:<14} {v:>8.4f} {db:>+8.4f} "
          f"{gap:>8.4f}  {conclusions[label]}{mrk}")

v1 = all_history["EGA-1"]["val"][-1]
vm = all_history["EGA-M-Fixed"]["val"][-1]

print(f"\n  THE ANSWER:")
print(f"  EGA-1 (linear):      val = {v1:.4f}")
print(f"  EGA-M-Fixed (Morlet): val = {vm:.4f}")
print(f"  Δ = {v1-vm:+.4f}")

if vm < v1:
    diff = v1 - vm
    print(f"\n  ✓ Morlet wavelets BEAT linear projection by {diff:.4f}")
    print(f"  Parametric wavelet structure is genuinely useful.")
    print(f"  The admissibility constraint and learned ω₀/σ provide")
    print(f"  information that a linear projection cannot capture.")
elif abs(vm - v1) < 0.005:
    print(f"\n  → Morlet ≈ Linear (within noise)")
    print(f"  Wavelet structure matches linear projection.")
    print(f"  The energy gate concept is validated but the specific")
    print(f"  wavelet form gives no additional benefit.")
else:
    diff = vm - v1
    print(f"\n  → Linear projection still wins by {diff:.4f}")
    print(f"  Simple learned energy direction beats structured wavelet.")
    print(f"  Suggests the optimal energy basis for LLM embeddings")
    print(f"  is not sinusoidal — consistent with non-sinusoidal")
    print(f"  kernels found by Verma & Pilanci (2024).")

# Learned Morlet parameters
print(f"\n  EGA-M-Fixed learned Morlet parameters:")
m_egam = all_models["EGA-M-Fixed"]
m_egam.to(DEVICE)
for block in m_egam.blocks:
    head0 = block.heads[0]
    if hasattr(head0, "get_morlet_params"):
        o, s, tau, alpha, sw = head0.get_morlet_params()
        print(f"  {'fl':>4}  {'ω₀':>7}  {'σ':>7}  "
              f"{'ω₀·σ':>7}  {'τ learned':>10}  "
              f"{'α':>7}  {'w':>7}")
        print("  " + "─"*58)
        for i, fl in enumerate(FILTER_LENGTHS):
            prod = o[i].item() * s[i].item()
            adm  = "✓" if prod >= 5.0 else "✗"
            print(f"  {fl:>4}  {o[i]:.4f}  {s[i]:.4f}  "
                  f"{prod:.4f}{adm}  {tau[i]:>+10.4f}  "
                  f"{alpha[i]:.4f}  {sw[i]:.4f}")
        print(f"\n  τ > 0: gate learned to suppress low-energy tokens")
        print(f"  τ < 0: gate learned to pass most tokens")
        print(f"  τ ≈ 0: gate stayed neutral (no strong preference)")
        break

# ================================================================
# 11.  PLOT
# ================================================================
steps = all_history["BASE"]["step"]
colors = {
    "BASE":        "#2196F3",
    "EGA-1":       "#FF9800",
    "EGA-M-Fixed": "#E91E63",
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Phase 3 — Can Morlet Wavelets Beat Linear Projection?\n"
    f"BASE vs EGA-1 vs EGA-M-Fixed (τ=0)",
    fontsize=12, fontweight="bold"
)

# Val loss curves
ax = axes[0]
for label, _, _, _ in configs:
    ls = "--" if label == "EGA-M-Fixed" else "-"
    ax.plot(steps, all_history[label]["val"],
            color=colors[label], linestyle=ls,
            marker="o", markersize=4,
            label=label, linewidth=2.2)
ax.set_title("Validation Loss")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Final bar
ax   = axes[1]
lbls = [lb for lb, *_ in configs]
vals = [all_history[lb]["val"][-1] for lb in lbls]
cols = [colors[lb] for lb in lbls]
bars = ax.bar(lbls, vals, color=cols, alpha=0.85, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.001, f"{v:.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.set_title("Final Validation Loss")
ax.set_ylabel("Val Loss")
ax.set_ylim(min(vals)*0.97, max(vals)*1.02)
ax.grid(True, alpha=0.3, axis="y")
bi = vals.index(min(vals))
bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(3)

# Generalisation gap
ax = axes[2]
for label, _, _, _ in configs:
    ls  = "--" if label == "EGA-M-Fixed" else "-"
    gap = [v-t for v,t in
           zip(all_history[label]["val"],
               all_history[label]["train"])]
    ax.plot(steps, gap,
            color=colors[label], linestyle=ls,
            marker="o", markersize=4,
            label=label, linewidth=2.2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Generalisation Gap (val − train)")
ax.set_xlabel("Step"); ax.set_ylabel("Gap")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(CKPT_DIR, "egam_fixed_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 12.  GENERATE TEXT SAMPLES
# ================================================================
print("\n── BASE ─────────────────────────────────────────")
print(generate(all_models["BASE"], prompt="HAMLET:\n"))

print("\n── EGA-1 ────────────────────────────────────────")
print(generate(all_models["EGA-1"], prompt="HAMLET:\n"))

print("\n── EGA-M-Fixed ──────────────────────────────────")
print(generate(all_models["EGA-M-Fixed"], prompt="HAMLET:\n"))

# ================================================================
# 13.  SAVE TO LOCAL DISK
# ================================================================
print("\nSaving to local disk …")

# Text summary
summary = [
    "Phase 3 — EGA-M-Fixed Results",
    "=" * 40,
    f"BASE         val={all_history['BASE']['val'][-1]:.4f}",
    f"EGA-1        val={all_history['EGA-1']['val'][-1]:.4f}",
    f"EGA-M-Fixed  val={all_history['EGA-M-Fixed']['val'][-1]:.4f}",
    "",
    f"Morlet vs Linear: Δ={v1-vm:+.4f}",
    ("Morlet wins" if vm < v1 else
     "Equivalent" if abs(vm-v1) < 0.005 else
     "Linear wins"),
    "",
    "Learned Morlet parameters:",
]
for block in all_models["EGA-M-Fixed"].blocks:
    for head in block.heads:
        if hasattr(head, "get_morlet_params"):
            o,s,tau,alpha,sw = head.get_morlet_params()
            for i, fl in enumerate(FILTER_LENGTHS):
                prod = o[i].item()*s[i].item()
                summary.append(
                    f"  len={fl}  ω₀={o[i]:.4f}  σ={s[i]:.4f}  "
                    f"ω₀·σ={prod:.4f}  τ={tau[i]:+.4f}  "
                    f"w={sw[i]:.4f}"
                )
            break
    break

txt_path = os.path.join(CKPT_DIR, "egam_fixed_summary.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(summary))

from google.colab import files
print("  Downloading plot …")
files.download(plot_path)
print("  Downloading summary …")
files.download(txt_path)
print("  Downloading EGA-M-Fixed checkpoint …")
files.download(os.path.join(CKPT_DIR, "EGA-M-Fix.pt"))
print("\nDone. Check your Downloads folder.")
print(f"{'='*58}")
