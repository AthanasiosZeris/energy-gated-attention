# ================================================================
# GPT — Spectral Energy Ablation + Morlet Wavelet Attention
# Phase 2 + Phase 3  |  Colab GPU  |  No Google Drive required
# ================================================================
#
# STORAGE OPTIONS (choose one at the top of the file):
#
# OPTION A — Local Colab storage (default, no auth needed)
#   Saves to /content/gpt_checkpoints/
#   Survives within session but lost on runtime restart.
#   Good for uninterrupted runs.
#
# OPTION B — Google Drive (persistent across restarts)
#   Uncomment the Drive section below if Drive auth works.
#
# OPTION C — No saving (fastest start, no checkpoints)
#   Set SAVE_CHECKPOINTS = False below.
#
# HOW TO RUN
# ----------
# Runtime → Change runtime type → T4 GPU → Run cell
#
# ================================================================

# ── Storage configuration ────────────────────────────────────────
SAVE_CHECKPOINTS = True       # set False to disable all saving

# OPTION A — local Colab storage (works without any auth)
SAVE_DIR = "/content/gpt_checkpoints"

# OPTION B — Google Drive (uncomment these 4 lines if Drive works)
# from google.colab import drive
# drive.mount("/content/drive", force_remount=False)
# SAVE_DIR = "/content/drive/MyDrive/gpt_spectral_energy"

import os
if SAVE_CHECKPOINTS:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Checkpoint directory: {SAVE_DIR}")
else:
    print("Checkpointing disabled.")

# ================================================================
# 1.  IMPORTS
# ================================================================
import math, time, warnings, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
torch.manual_seed(42)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

print("=" * 60)
print("  GPT Spectral Energy — Phase 2 + Phase 3")
print("=" * 60)
print(f"  Device  : {DEVICE}")
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {p.name}")
    print(f"  VRAM    : {p.total_memory/1e9:.1f} GB")
print("=" * 60)

# ================================================================
# 2.  DATASET
# ================================================================
URL  = ("https://raw.githubusercontent.com/karpathy/char-rnn"
        "/master/data/tinyshakespeare/input.txt")
print("\nDownloading TinyShakespeare …")
text = requests.get(URL).text
print(f"  {len(text):,} characters loaded.")

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
print(f"  Train : {len(train_data):,} | Val : {len(val_data):,}\n")

# ================================================================
# 3.  HYPERPARAMETERS
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

print("Hyperparameters")
for k, v in [("BATCH_SIZE",BATCH_SIZE),("BLOCK_SIZE",BLOCK_SIZE),
             ("N_EMBED",N_EMBED),("N_HEAD",N_HEAD),
             ("N_LAYER",N_LAYER),("MAX_ITERS",MAX_ITERS),
             ("AMP",USE_AMP)]:
    print(f"  {k:<14} = {v}")
print()

# ================================================================
# 4.  DATA LOADER + UTILITIES
# ================================================================
def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i : i + BLOCK_SIZE]         for i in ix])
    y   = torch.stack([src[i+1 : i + BLOCK_SIZE + 1]   for i in ix])
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

def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR * step / max(1, WARMUP_ITERS)
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

def znorm(t, T):
    if T > 1:
        mu  = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True, correction=0).clamp(min=1e-8)
        return (t - mu) / std
    return torch.zeros_like(t)

# ================================================================
# 5.  CHECKPOINT UTILITIES
# ================================================================
def ckpt_path(label):
    return os.path.join(SAVE_DIR, f"{label}.pt")

def save_checkpoint(label, model, optimizer, h, step):
    if not SAVE_CHECKPOINTS:
        return
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history":         h,
        "step":            step,
    }, ckpt_path(label))

def checkpoint_exists(label):
    if not SAVE_CHECKPOINTS:
        return False
    p = ckpt_path(label)
    if not os.path.exists(p):
        return False
    ck = torch.load(p, map_location="cpu")
    return ck.get("step", 0) >= MAX_ITERS

def load_history(label):
    return torch.load(ckpt_path(label),
                      map_location="cpu")["history"]

def load_model_weights(label, model):
    ck = torch.load(ckpt_path(label), map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    print(f"    Loaded {label} weights from checkpoint.")
    return model

# ================================================================
# 6.  ATTENTION MODULES — PHASE 2
# ================================================================

class StandardAttention(nn.Module):
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
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        return self.drop(F.softmax(sc, dim=-1)) @ v


class LinearEnergyGatedAttention(nn.Module):
    def __init__(self, head_size: int, n_scales: int):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        self.scale_proj = nn.ModuleList([
            nn.Linear(N_EMBED, 1, bias=True) for _ in range(n_scales)
        ])
        self.tau     = nn.Parameter(torch.zeros(n_scales))
        self.alpha   = nn.Parameter(torch.ones(n_scales) * 2.0)
        self.scale_w = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        gates = []
        for s, proj in enumerate(self.scale_proj):
            e_s = proj(x).transpose(-2,-1)
            e_s = znorm(e_s, T)
            g_s = torch.sigmoid(self.alpha[s] * (e_s - self.tau[s]))
            gates.append(g_s)
        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s] * gates[s] for s in range(self.n_scales))
        att  = self.drop(F.softmax(sc, dim=-1))
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v


class ConvEnergyGatedAttention(nn.Module):
    def __init__(self, head_size: int, filter_lengths=None):
        super().__init__()
        self.hs = head_size
        if filter_lengths is None:
            filter_lengths = FILTER_LENGTHS
        self.filter_lengths = filter_lengths
        n_scales = len(filter_lengths)
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        self.conv_filters = nn.ModuleList([
            nn.Conv1d(N_EMBED, 2, kernel_size=fl,
                      padding=fl-1, bias=True)
            for fl in filter_lengths
        ])
        self.tau     = nn.Parameter(torch.zeros(n_scales))
        self.alpha   = nn.Parameter(torch.ones(n_scales) * 2.0)
        self.scale_w = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        x_conv = x.transpose(1, 2)
        gates  = []
        for s, conv in enumerate(self.conv_filters):
            fl  = self.filter_lengths[s]
            out = conv(x_conv)[:, :, :T]
            energy = out[:,0:1,:]**2 + out[:,1:2,:]**2
            e_s = znorm(energy, T)
            g_s = torch.sigmoid(self.alpha[s] * (e_s - self.tau[s]))
            gates.append(g_s)
        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s] * gates[s] for s in range(len(gates)))
        att  = self.drop(F.softmax(sc, dim=-1))
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v

    def get_learned_scales(self):
        return F.softmax(self.scale_w, dim=0).detach().cpu()

    def get_learned_thresholds(self):
        return self.tau.detach().cpu(), self.alpha.detach().cpu()

# ================================================================
# 7.  ATTENTION MODULES — PHASE 3 (Morlet)
# ================================================================

def morlet_filter(omega0, sigma, length, device):
    t    = torch.arange(length, dtype=torch.float32, device=device)
    env  = torch.exp(-t**2 / (2.0 * sigma**2 + 1e-8))
    real = torch.cos(omega0 * t) * env
    imag = torch.sin(omega0 * t) * env
    return torch.stack([real, imag], dim=0)

def apply_causal_morlet(x, omega0, sigma, length):
    B, T, C = x.shape
    device  = x.device
    kernel  = morlet_filter(omega0, sigma, length, device)
    x_1d    = x.mean(dim=-1, keepdim=True).transpose(1, 2)
    x_pad   = F.pad(x_1d, (length - 1, 0))
    k_real  = kernel[0].view(1, 1, -1)
    k_imag  = kernel[1].view(1, 1, -1)
    real    = F.conv1d(x_pad, k_real)[:, :, :T]
    imag    = F.conv1d(x_pad, k_imag)[:, :, :T]
    return torch.cat([real, imag], dim=1)


class MorletEnergyGatedAttention(nn.Module):
    FILTER_LENGTHS = [3, 7, 15, 31]
    TAU_INIT       = [0.3538, 0.3443, 0.3414, 0.3233]
    ALPHA_INIT     = [2.2260, 2.2453, 2.2537, 2.2596]

    def __init__(self, head_size: int, n_scales: int = 4):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales
        fl            = self.FILTER_LENGTHS[:n_scales]
        self.filter_lengths = fl
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        omega0_init = [math.pi / l for l in fl]
        sigma_init  = [5.0 / w for w in omega0_init]
        self.log_omega0 = nn.Parameter(
            torch.tensor([math.log(w) for w in omega0_init])
        )
        self.log_sigma  = nn.Parameter(
            torch.tensor([math.log(s) for s in sigma_init])
        )
        self.tau     = nn.Parameter(torch.tensor(self.TAU_INIT[:n_scales]))
        self.alpha   = nn.Parameter(torch.tensor(self.ALPHA_INIT[:n_scales]))
        self.scale_w = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def _get_morlet_params(self):
        omega0 = torch.exp(self.log_omega0).clamp(max=math.pi * 0.95)
        sigma  = torch.exp(self.log_sigma ).clamp(min=1e-3)
        product = omega0 * sigma
        omega0_safe = torch.where(
            product < 5.0, 5.0 / sigma.clamp(min=1e-8), omega0
        )
        return omega0_safe, sigma

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        sc = sc.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        omega0, sigma = self._get_morlet_params()
        gates = []
        for s in range(self.n_scales):
            coeffs = apply_causal_morlet(
                x, omega0[s], sigma[s], self.filter_lengths[s]
            )
            energy = coeffs[:,0:1,:]**2 + coeffs[:,1:2,:]**2
            e_s = znorm(energy, T)
            g_s = torch.sigmoid(self.alpha[s] * (e_s - self.tau[s]))
            gates.append(g_s)
        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s] * gates[s] for s in range(self.n_scales))
        att  = self.drop(F.softmax(sc, dim=-1))
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v

    def get_morlet_params(self):
        with torch.no_grad():
            o, s = self._get_morlet_params()
            return (o.cpu(), s.cpu(),
                    self.tau.cpu(), self.alpha.cpu(),
                    F.softmax(self.scale_w, dim=0).cpu())


class MorletMultiQuantityAttention(nn.Module):
    FILTER_LENGTHS = [3, 7, 15, 31]
    TAU_INIT       = [0.3538, 0.3443, 0.3414, 0.3233]
    ALPHA_INIT     = [2.2260, 2.2453, 2.2537, 2.2596]

    def __init__(self, head_size: int, n_scales: int = 4):
        super().__init__()
        self.hs       = head_size
        self.n_scales = n_scales
        fl            = self.FILTER_LENGTHS[:n_scales]
        self.filter_lengths = fl
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)
        omega0_init = [math.pi / l for l in fl]
        sigma_init  = [5.0 / w for w in omega0_init]
        self.log_omega0 = nn.Parameter(
            torch.tensor([math.log(w) for w in omega0_init])
        )
        self.log_sigma  = nn.Parameter(
            torch.tensor([math.log(s) for s in sigma_init])
        )
        self.tau          = nn.Parameter(torch.tensor(self.TAU_INIT[:n_scales]))
        self.alpha        = nn.Parameter(torch.tensor(self.ALPHA_INIT[:n_scales]))
        self.scale_w      = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.lambda_sim   = nn.Parameter(torch.tensor(1.0))
        self.lambda_energy= nn.Parameter(torch.tensor(0.1))
        self.lambda_phase = nn.Parameter(torch.tensor(0.1))
        self.lambda_flux  = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def _get_morlet_params(self):
        omega0 = torch.exp(self.log_omega0).clamp(max=math.pi * 0.95)
        sigma  = torch.exp(self.log_sigma ).clamp(min=1e-3)
        product = omega0 * sigma
        omega0_safe = torch.where(
            product < 5.0, 5.0 / sigma.clamp(min=1e-8), omega0
        )
        return omega0_safe, sigma

    def _spectral_quantities(self, x, T):
        omega0, sigma = self._get_morlet_params()
        all_e, all_p, all_f = [], [], []
        for s in range(self.n_scales):
            coeffs = apply_causal_morlet(
                x, omega0[s], sigma[s], self.filter_lengths[s]
            )
            real = coeffs[:,0:1,:]; imag = coeffs[:,1:2,:]
            energy = real**2 + imag**2
            phase  = torch.atan2(imag, real)
            flux   = (torch.diff(energy, dim=-1,
                                  prepend=energy[:,:,:1]).abs()
                      if T > 1 else torch.zeros_like(energy))
            all_e.append(energy)
            all_p.append(phase)
            all_f.append(flux)
        return (znorm(torch.stack(all_e).mean(0), T),
                torch.stack(all_p).mean(0),
                znorm(torch.stack(all_f).mean(0), T))

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x);  q = self.query(x);  v = self.value(x)
        sc_sim = q @ k.transpose(-2,-1) / math.sqrt(self.hs)
        energy, phase, flux = self._spectral_quantities(x, T)
        e_energy = torch.log((energy.abs()+1e-8).expand(B,T,T))
        e_phase  = torch.cos(phase).expand(B,T,T)
        e_flux   = flux.expand(B,T,T)
        combined = (self.lambda_sim    * sc_sim   +
                    self.lambda_energy * e_energy  +
                    self.lambda_phase  * e_phase   +
                    self.lambda_flux   * e_flux)
        combined = combined.masked_fill(
            self.tril[:T,:T] == 0, float("-inf")
        )
        omega0, sigma = self._get_morlet_params()
        gates = []
        for s in range(self.n_scales):
            coeffs = apply_causal_morlet(
                x, omega0[s], sigma[s], self.filter_lengths[s]
            )
            energy_s = coeffs[:,0:1,:]**2 + coeffs[:,1:2,:]**2
            e_s = znorm(energy_s, T)
            g_s = torch.sigmoid(self.alpha[s] * (e_s - self.tau[s]))
            gates.append(g_s)
        sw   = F.softmax(self.scale_w, dim=0)
        gate = sum(sw[s] * gates[s] for s in range(self.n_scales))
        att  = self.drop(F.softmax(combined, dim=-1))
        att  = att * gate
        att  = att / att.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return att @ v

    def get_lambda_weights(self):
        return {"similarity": self.lambda_sim.item(),
                "energy":     self.lambda_energy.item(),
                "phase":      self.lambda_phase.item(),
                "flux":       self.lambda_flux.item()}

    def get_morlet_params(self):
        with torch.no_grad():
            o, s = self._get_morlet_params()
            return (o.cpu(), s.cpu(),
                    self.tau.cpu(), self.alpha.cpu(),
                    F.softmax(self.scale_w, dim=0).cpu())

# ================================================================
# 8.  TRANSFORMER BLOCK + GPT
# ================================================================

class Block(nn.Module):
    def __init__(self, attn_class, attn_kwargs):
        super().__init__()
        head_size  = N_EMBED // N_HEAD
        self.heads = nn.ModuleList([
            attn_class(head_size, **attn_kwargs) for _ in range(N_HEAD)
        ])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.drop = nn.Dropout(DROPOUT)
        self.ff   = nn.Sequential(
            nn.Linear(N_EMBED, 4*N_EMBED), nn.GELU(),
            nn.Linear(4*N_EMBED, N_EMBED), nn.Dropout(DROPOUT),
        )
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

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
            Block(attn_class, attn_kwargs) for _ in range(N_LAYER)
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
        x    = self.drop(
            self.tok_emb(idx) +
            self.pos_emb(torch.arange(T, device=idx.device))
        )
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = (F.cross_entropy(logits.view(-1,VOCAB), targets.view(-1))
                  if targets is not None else None)
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

# ================================================================
# 9.  GENERATION
# ================================================================

@torch.no_grad()
def generate(model, prompt="\n", max_new_tokens=250,
             temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor(encode(prompt),
                        dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits    = logits[:,-1,:] / temperature
        logits    = torch.nan_to_num(logits, nan=0.0,
                                      posinf=1e4, neginf=-1e4)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, VOCAB))
            logits[logits < v[:,[-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or (probs<0).any():
            probs = torch.ones(1,VOCAB,device=DEVICE)/VOCAB
        idx = torch.cat([idx,
                          torch.multinomial(probs, 1)], dim=1)
    model.train()
    return decode(idx[0].tolist())

# ================================================================
# 10.  TRAINING FUNCTION  (single model, with checkpointing)
# ================================================================

def train_model(label, model, optimizer, scaler):
    """Train one model, save checkpoint every EVAL_INTERVAL steps."""
    h = {"train": [], "val": [], "curve": [],
         "step":  list(range(0, MAX_ITERS+1, EVAL_INTERVAL))}

    pbar = tqdm(range(MAX_ITERS+1), desc=f"  {label}", leave=True)

    for step in pbar:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch("train")
        with autocast(enabled=USE_AMP):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        h["curve"].append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        if step % EVAL_INTERVAL == 0:
            ev = estimate_loss(model)
            h["train"].append(ev["train"])
            h["val"].append(ev["val"])
            tqdm.write(
                f"  {label}  step={step:>5}  "
                f"train={ev['train']:.4f}  val={ev['val']:.4f}"
            )
            save_checkpoint(label, model, optimizer, h, step)

    return h

# ================================================================
# 11.  BUILD + TRAIN / LOAD ALL MODELS
# ================================================================

ALL_CONFIGS = [
    # label      class                         kwargs         phase
    ("BASE",   StandardAttention,          {},              "P2"),
    ("EGA-1",  LinearEnergyGatedAttention, {"n_scales":1},  "P2"),
    ("EGA-2",  LinearEnergyGatedAttention, {"n_scales":2},  "P2"),
    ("EGA-4",  LinearEnergyGatedAttention, {"n_scales":4},  "P2"),
    ("EGA-C",  ConvEnergyGatedAttention,   {},              "P2"),
    ("EGA-M",  MorletEnergyGatedAttention, {"n_scales":4},  "P3"),
    ("EGA-M+", MorletMultiQuantityAttention,{"n_scales":4}, "P3"),
]

# Global dicts
models  = {}   # label → (model, optimizer, scaler)
history = {}   # label → history dict

print("\n" + "=" * 60)
print("  TRAINING ALL 7 MODELS")
print("  Checkpoints saved every 500 steps.")
print("  Re-run this cell after any restart —")
print("  completed models will be loaded, not retrained.")
print("=" * 60)

for label, cls, kwargs, phase in ALL_CONFIGS:
    print(f"\n── {label} ({phase}) ──────────────────────────────────")
    m      = GPT(cls, kwargs, label=label).to(DEVICE)
    opt    = torch.optim.AdamW(m.parameters(), lr=LR,
                                betas=(0.9,0.95), weight_decay=0.1)
    scaler = GradScaler(enabled=USE_AMP)
    print(f"  Params: {m.num_parameters():,}")

    if checkpoint_exists(label):
        print(f"  ✓ Checkpoint found — loading (no retraining)")
        m = load_model_weights(label, m)
        h = load_history(label)
        print(f"  Final val = {h['val'][-1]:.4f}")
    else:
        print(f"  ✗ No checkpoint — training now …")
        h = train_model(label, m, opt, scaler)
        print(f"  Final val = {h['val'][-1]:.4f}  ✓ saved")

    models[label]  = (m, opt, scaler)
    history[label] = h

# ================================================================
# 12.  COMPLETE SUMMARY
# ================================================================

print("\n" + "═"*68)
print("  COMPLETE SUMMARY — ALL 7 MODELS")
print("═"*68)

base_val = history["BASE"]["val"][-1]
egac_val = history["EGA-C"]["val"][-1]
best_val = min(history[lb]["val"][-1] for lb,*_ in ALL_CONFIGS)

print(f"\n  {'Model':<10} {'Val':>8} {'ΔBASE':>8} "
      f"{'ΔEGA-C':>8} {'Gap':>8} {'Params':>12}  Ph")
print("  " + "─"*62)

for label, cls, kwargs, phase in ALL_CONFIGS:
    m, _, _ = models[label]
    v   = history[label]["val"][-1]
    t   = history[label]["train"][-1]
    db  = base_val - v
    dc  = egac_val - v
    gap = v - t
    prm = m.num_parameters()
    mrk = " ◄BEST" if v == best_val else ""
    print(f"  {label:<10} {v:>8.4f} {db:>+8.4f} "
          f"{dc:>+8.4f} {gap:>8.4f} {prm:>12,}  {phase}{mrk}")

# Key conclusions
vm  = history["EGA-M"]["val"][-1]
vmp = history["EGA-M+"]["val"][-1]
vc  = history["EGA-C"]["val"][-1]
print(f"\n  Q1 Morlet vs Conv  : EGA-M={vm:.4f}  EGA-C={vc:.4f}  "
      f"{'Morlet wins ✓' if vm<vc else 'Conv wins'}")
print(f"  Q2 Phase+Flux      : EGA-M={vm:.4f}  EGA-M+={vmp:.4f}  "
      f"{'Multi-qty wins ✓' if vmp<vm else 'Energy alone sufficient'}")

# EGA-M Morlet params (first head, first block)
print(f"\n  EGA-M learned Morlet parameters:")
m_egam, _, _ = models["EGA-M"]
for block in m_egam.blocks:
    head0 = block.heads[0]
    if hasattr(head0, "get_morlet_params"):
        o,s,tau,alpha,sw = head0.get_morlet_params()
        print(f"  {'Scale':<8} {'ω₀':>7} {'σ':>7} "
              f"{'ω₀·σ':>7} {'τ':>7} {'α':>7} {'w':>7}")
        print(f"  {'─'*52}")
        for i, fl in enumerate(MorletEnergyGatedAttention.FILTER_LENGTHS):
            prod = o[i].item() * s[i].item()
            adm  = "✓" if prod >= 5.0 else "✗"
            print(f"  len={fl:<4}  "
                  f"{o[i].item():>7.4f} {s[i].item():>7.4f} "
                  f"{prod:>7.4f}{adm} {tau[i].item():>+7.4f} "
                  f"{alpha[i].item():>7.4f} {sw[i].item():>7.4f}")
        break

# EGA-M+ lambda weights
print(f"\n  EGA-M+ learned λ weights:")
m_egamp, _, _ = models["EGA-M+"]
for block in m_egamp.blocks:
    head0 = block.heads[0]
    if hasattr(head0, "get_lambda_weights"):
        lw    = head0.get_lambda_weights()
        total = sum(abs(v) for v in lw.values())
        for k, v in lw.items():
            bar = "█" * int(abs(v)/total*30)
            print(f"    λ_{k:<12} = {v:>7.4f}  "
                  f"({abs(v)/total*100:>5.1f}%)  {bar}")
        break
print("═"*68)

# ================================================================
# 13.  PLOTS
# ================================================================

colors = {"BASE":"#2196F3","EGA-1":"#FF9800","EGA-2":"#9C27B0",
          "EGA-4":"#F44336","EGA-C":"#4CAF50",
          "EGA-M":"#00BCD4","EGA-M+":"#E91E63"}
p2_labels = ["BASE","EGA-1","EGA-2","EGA-4","EGA-C"]
p3_labels = ["EGA-M","EGA-M+"]
steps     = history["BASE"]["step"]

fig = plt.figure(figsize=(22,14))
fig.suptitle(
    "GPT Phase 2+3 — Complete Ablation: Baseline → Morlet Wavelet\n"
    f"TinyShakespeare  |  {N_LAYER}L×{N_HEAD}H×{N_EMBED}d  |"
    f"  block={BLOCK_SIZE}  batch={BATCH_SIZE}  steps={MAX_ITERS}\n"
    "Solid = Phase 2  |  Dashed = Phase 3 (Morlet)",
    fontsize=11, fontweight="bold"
)
gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.40,wspace=0.32)

def vhist(lb): return history[lb]["val"]
def thist(lb): return history[lb]["train"]

# Panel 1 — Validation loss
ax = fig.add_subplot(gs[0,0])
for lb in p2_labels:
    ax.plot(steps, vhist(lb), color=colors[lb],
            linestyle="-", marker="o", markersize=3,
            label=lb, linewidth=1.8)
for lb in p3_labels:
    ax.plot(steps, vhist(lb), color=colors[lb],
            linestyle="--", marker="s", markersize=4,
            label=lb, linewidth=2.2)
ax.set_title("Validation Loss")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)

# Panel 2 — Delta vs BASE
ax = fig.add_subplot(gs[0,1])
bv = vhist("BASE")
for lb in [l for l in (p2_labels+p3_labels) if l!="BASE"]:
    ls = "--" if lb in p3_labels else "-"
    ms = "s"  if lb in p3_labels else "o"
    delta = [b-v for b,v in zip(bv, vhist(lb))]
    ax.plot(steps, delta, color=colors[lb], linestyle=ls,
            marker=ms, markersize=3, label=f"Δ{lb}", linewidth=1.8)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Δ Val Loss (BASE − model)")
ax.set_xlabel("Step"); ax.set_ylabel("Δ Loss")
ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)

# Panel 3 — Final bar chart
ax   = fig.add_subplot(gs[0,2])
lbls = [lb for lb,*_ in ALL_CONFIGS]
vals = [vhist(lb)[-1] for lb in lbls]
cols = [colors[lb] for lb in lbls]
bars = ax.bar(lbls, vals, color=cols, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.001, f"{v:.4f}",
            ha="center",va="bottom",fontsize=7,rotation=40)
ax.set_title("Final Val Loss — All 7 Models")
ax.set_ylabel("Val Loss")
ax.set_ylim(min(vals)*0.97, max(vals)*1.02)
ax.tick_params(axis="x",labelsize=8,rotation=20)
ax.grid(True,alpha=0.3,axis="y")
bi = vals.index(min(vals))
bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(3)

# Panel 4 — Generalisation gap
ax = fig.add_subplot(gs[1,0])
for lb in p2_labels:
    gap = [v-t for v,t in zip(vhist(lb),thist(lb))]
    ax.plot(steps, gap, color=colors[lb], linestyle="-",
            marker="o", markersize=3, label=lb, linewidth=1.8)
for lb in p3_labels:
    gap = [v-t for v,t in zip(vhist(lb),thist(lb))]
    ax.plot(steps, gap, color=colors[lb], linestyle="--",
            marker="s", markersize=4, label=lb, linewidth=2.2)
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_title("Generalisation Gap (val − train)")
ax.set_xlabel("Step"); ax.set_ylabel("Gap")
ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)

# Panel 5 — EGA-M learned ω₀ and σ
ax  = fig.add_subplot(gs[1,1])
ax2 = ax.twinx()
all_o, all_s = [], []
for block in m_egam.blocks:
    for head in block.heads:
        if hasattr(head,"get_morlet_params"):
            o,s,_,_,_ = head.get_morlet_params()
            all_o.append(o.numpy()); all_s.append(s.numpy())
mean_o = np.mean(all_o,axis=0)
mean_s = np.mean(all_s,axis=0)
fl_lab = [f"len={fl}" for fl in MorletEnergyGatedAttention.FILTER_LENGTHS]
xp     = range(len(fl_lab))
ax.bar(list(xp), mean_o, color="#00BCD4", alpha=0.7, label="ω₀")
ax2.plot(list(xp), mean_s, "r-o", markersize=7,
         linewidth=2, label="σ")
ax.set_xticks(list(xp)); ax.set_xticklabels(fl_lab,fontsize=9)
ax.set_title("EGA-M: Learned Morlet Parameters\nω₀ (bars) & σ (line)")
ax.set_ylabel("ω₀",color="#00BCD4")
ax2.set_ylabel("σ",color="red")
ax2.tick_params(axis="y",labelcolor="red")
l1,n1=ax.get_legend_handles_labels()
l2,n2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,n1+n2,fontsize=8,loc="upper right")
ax.grid(True,alpha=0.3,axis="y")

# Panel 6 — EGA-M+ lambda weights
ax  = fig.add_subplot(gs[1,2])
all_lw = {"similarity":[],"energy":[],"phase":[],"flux":[]}
for block in m_egamp.blocks:
    for head in block.heads:
        if hasattr(head,"get_lambda_weights"):
            lw = head.get_lambda_weights()
            for k in all_lw: all_lw[k].append(abs(lw[k]))
means = {k:sum(v)/len(v) for k,v in all_lw.items()}
total = sum(means.values())
pcts  = {k:v/total*100 for k,v in means.items()}
qcols = ["#2196F3","#4CAF50","#E91E63","#FF9800"]
bars2 = ax.bar(list(means.keys()),list(pcts.values()),
               color=qcols,alpha=0.85)
for bar,(k,pct) in zip(bars2,pcts.items()):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.5,f"{pct:.1f}%",
            ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.set_title("EGA-M+: Learned |λ| Weights\nContribution to attention score")
ax.set_ylabel("|λ|/Σ|λ| (%)")
ax.tick_params(axis="x",labelsize=9)
ax.grid(True,alpha=0.3,axis="y")

plt.savefig("phase2_phase3_results.png",dpi=150,bbox_inches="tight")
if SAVE_CHECKPOINTS:
    plt.savefig(os.path.join(SAVE_DIR,"phase2_phase3_results.png"),
                dpi=150,bbox_inches="tight")
plt.show()
print("Plot saved.")

# ================================================================
# 14.  TEXT SAMPLES
# ================================================================
for label in ["BASE","EGA-C","EGA-M","EGA-M+"]:
    print(f"\n── {label} {'─'*(50-len(label))}")
    m, _, _ = models[label]
    print(generate(m, prompt="HAMLET:\n"))

print(f"\nCheckpoints at: {SAVE_DIR if SAVE_CHECKPOINTS else 'disabled'}")
