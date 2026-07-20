"""
Phase 4 Exp 1+2 — Memory Fixed
================================
CONV-L16 replaced with CONV-L4 (max_lag=4)
Reason: [B,T,T] x 32 lags exceeds T4 VRAM at max_lag=16

Scientific justification:
  Most linguistic lag structure is local (±1 to ±4 tokens)
  Long-range dependencies handled by multi-layer attention
  CONV-L4 tests the hypothesis cleanly with less memory

Models:
  BASE-DOT   ✓ loaded from checkpoint
  CONV-L8    ✓ loaded from checkpoint
  CONV-L4    ← replaces CONV-L16 (more memory efficient)
  EGA-1      ← trains fresh
  Exp 2      ← all 4 PE models
"""

import math, os, gc, warnings, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(42)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP  = DEVICE == "cuda"
CKPT_DIR = "/content/gpt_phase4"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── memory clear ─────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()
free = torch.cuda.mem_get_info()[0]/1e9
print(f"GPU: {free:.2f} GB free\n")

# ================================================================
# DATASET
# ================================================================
URL  = ("https://raw.githubusercontent.com/karpathy/char-rnn"
        "/master/data/tinyshakespeare/input.txt")
text = requests.get(URL).text
chars= sorted(set(text)); VOCAB=len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
def encode(s): return [stoi[c] for c in s]
data = torch.tensor(encode(text),dtype=torch.long)
n    = int(0.9*len(data))
train_data=data[:n]; val_data=data[n:]

# ================================================================
# HYPERPARAMETERS
# ================================================================
BATCH_SIZE=64; BLOCK_SIZE=256; N_EMBED=256
N_HEAD=8; N_LAYER=6; DROPOUT=0.1
LR=3e-4; MAX_ITERS=5000; EVAL_INTERVAL=500
N_EVAL_BATCHES=50; WARMUP_ITERS=300; GRAD_CLIP=1.0

# ================================================================
# UTILITIES
# ================================================================
def get_batch(split):
    src=train_data if split=="train" else val_data
    ix=torch.randint(len(src)-BLOCK_SIZE,(BATCH_SIZE,))
    x=torch.stack([src[i:i+BLOCK_SIZE]     for i in ix])
    y=torch.stack([src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE),y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    model.eval(); out={}
    for split in ("train","val"):
        ls=torch.zeros(N_EVAL_BATCHES,device=DEVICE)
        for k in range(N_EVAL_BATCHES):
            xb,yb=get_batch(split)
            with autocast(enabled=USE_AMP):
                _,loss=model(xb,yb)
            ls[k]=loss.detach()
        out[split]=ls.mean().item()
    model.train(); return out

def get_lr(step):
    if step<WARMUP_ITERS:
        return LR*step/max(1,WARMUP_ITERS)
    t=(step-WARMUP_ITERS)/max(1,MAX_ITERS-WARMUP_ITERS)
    return LR*0.5*(1.0+math.cos(math.pi*t))

def znorm(t,T):
    if T>1:
        mu=t.mean(dim=-1,keepdim=True)
        std=t.std(dim=-1,keepdim=True,correction=0).clamp(min=1e-8)
        return (t-mu)/std
    return torch.zeros_like(t)

def ckpt_done(name):
    p=os.path.join(CKPT_DIR,f"{name}.pt")
    if not os.path.exists(p): return False
    return torch.load(p,map_location="cpu").get("step",0)>=MAX_ITERS

def save_ckpt(name,model,opt,h,step):
    torch.save({"model_state":model.state_dict(),
                "optimizer_state":opt.state_dict(),
                "history":h,"step":step},
               os.path.join(CKPT_DIR,f"{name}.pt"))

def load_ckpt(name,model):
    ck=torch.load(os.path.join(CKPT_DIR,f"{name}.pt"),
                   map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    return ck["history"]

def train_model(label,model,save_name):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,
                           betas=(0.9,0.95),weight_decay=0.1)
    scaler=GradScaler(enabled=USE_AMP)
    h={"train":[],"val":[],"curve":[],
       "step":list(range(0,MAX_ITERS+1,EVAL_INTERVAL))}
    pbar=tqdm(range(MAX_ITERS+1),desc=f"  {label}",leave=True)
    for step in pbar:
        lr=get_lr(step)
        for pg in opt.param_groups: pg["lr"]=lr
        xb,yb=get_batch("train")
        with autocast(enabled=USE_AMP):
            _,loss=model(xb,yb)
        if torch.isnan(loss): continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        scaler.step(opt); scaler.update()
        h["curve"].append(loss.item())
        pbar.set_postfix({"loss":f"{loss.item():.3f}"})
        if step%EVAL_INTERVAL==0:
            ev=estimate_loss(model)
            h["train"].append(ev["train"])
            h["val"].append(ev["val"])
            tqdm.write(f"  {label}  step={step:>5}  "
                       f"train={ev['train']:.4f}  "
                       f"val={ev['val']:.4f}")
            save_ckpt(save_name,model,opt,h,step)
    return h

def load_or_train(label,save_name,model):
    print(f"\n── {label} "
          f"({sum(p.numel() for p in model.parameters()):,}"
          f" params) ──")
    free=torch.cuda.mem_get_info()[0]/1e9
    print(f"  GPU free: {free:.2f} GB")
    if ckpt_done(save_name):
        h=load_ckpt(save_name,model)
        print(f"  ✓ Loaded — val={h['val'][-1]:.4f}")
    else:
        print("  Training …")
        h=train_model(label,model,save_name)
    # Full cleanup
    model.cpu(); del model
    gc.collect(); torch.cuda.empty_cache()
    free=torch.cuda.mem_get_info()[0]/1e9
    print(f"  GPU after: {free:.2f} GB free")
    return h

# ================================================================
# TRANSFORMER COMPONENTS
# ================================================================
class Block(nn.Module):
    def __init__(self,attn_class,attn_kwargs={}):
        super().__init__()
        hs=N_EMBED//N_HEAD
        self.heads=nn.ModuleList([
            attn_class(hs,**attn_kwargs) for _ in range(N_HEAD)])
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

class GPT(nn.Module):
    def __init__(self,attn_class,attn_kwargs={},
                 pe_class=None,label=""):
        super().__init__()
        self.label=label
        self.tok_emb=nn.Embedding(VOCAB,N_EMBED)
        self.pe=(pe_class(N_EMBED,BLOCK_SIZE) if pe_class else None)
        self.pos_emb=(None if pe_class else
                      nn.Embedding(BLOCK_SIZE,N_EMBED))
        self.drop=nn.Dropout(DROPOUT)
        self.blocks=nn.Sequential(*[
            Block(attn_class,attn_kwargs) for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBED)
        self.head=nn.Linear(N_EMBED,VOCAB,bias=False)
        self.tok_emb.weight=self.head.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok=self.tok_emb(idx)
        if self.pe is not None:
            x=self.drop(tok+self.pe(T).unsqueeze(0))
        else:
            pos=torch.arange(T,device=idx.device)
            x=self.drop(tok+self.pos_emb(pos))
        x=self.blocks(x); x=self.ln_f(x)
        logits=self.head(x)
        loss=(F.cross_entropy(logits.view(-1,VOCAB),
                               targets.view(-1))
              if targets is not None else None)
        return logits,loss

# ================================================================
# ATTENTION MODULES
# ================================================================
class DotProductAttention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.hs=head_size
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)
        self.register_buffer(
            "tril",torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        return self.drop(F.softmax(sc,dim=-1))@v


class ConvolutionAttention(nn.Module):
    """
    Cross-correlation attention — sequential lag computation.

    Memory strategy: compute one lag at a time, accumulate
    into scores, immediately delete the intermediate tensor.
    Peak memory = ONE [B,T,T] matrix at a time, not max_lag
    matrices simultaneously.

    This is the key fix: the original code accumulated all
    lag scores before masking, keeping all [B,T,T] tensors
    alive simultaneously. Now each is computed and immediately
    added to the running sum then deleted.
    """
    def __init__(self,head_size,max_lag=4):
        super().__init__()
        self.hs=head_size
        self.max_lag=max_lag
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)
        n_lags=2*max_lag+1
        self.lag_weights=nn.Parameter(torch.zeros(n_lags))
        nn.init.normal_(self.lag_weights,0.0,0.02)
        self.register_buffer(
            "tril",torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))

    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        lag_w=F.softmax(self.lag_weights,dim=0)
        scale=math.sqrt(self.hs)

        # Allocate score accumulator once — reuse for all lags
        # This avoids creating multiple [B,T,T] tensors
        scores=torch.zeros(B,T,T,device=x.device,dtype=x.dtype)

        for lag_idx,lag in enumerate(
            range(-self.max_lag,self.max_lag+1)
        ):
            w=lag_w[lag_idx].item()
            if abs(w)<1e-4: continue  # skip negligible lags

            if lag==0:
                # Zero lag: standard dot product
                scores.add_(
                    (q@k.transpose(-2,-1)/scale)*w
                )
            elif lag>0 and lag<T:
                # Key leads query: position j attends to j+lag
                # q[i] matches k[i+lag]
                q_part=q[:,:T-lag,:]        # [B,T-lag,hs]
                k_part=k[:,lag:,:]           # [B,T-lag,hs]
                sc_lag=q_part@k_part.transpose(-2,-1)/scale
                # Place in correct position of score matrix
                scores[:,:T-lag,lag:].add_(sc_lag*w)
                del q_part,k_part,sc_lag

            elif lag<0 and -lag<T:
                # Query leads key: position i attends to i+|lag|
                ab=-lag
                q_part=q[:,ab:,:]           # [B,T-ab,hs]
                k_part=k[:,:T-ab,:]         # [B,T-ab,hs]
                sc_lag=q_part@k_part.transpose(-2,-1)/scale
                scores[:,ab:,:T-ab].add_(sc_lag*w)
                del q_part,k_part,sc_lag

        scores=scores.masked_fill(
            self.tril[:T,:T]==0,float("-inf"))
        att=self.drop(F.softmax(scores,dim=-1))
        del scores
        return att@v

    def get_lag_weights(self):
        w=F.softmax(self.lag_weights,dim=0).detach().cpu()
        return list(range(-self.max_lag,self.max_lag+1)),w.numpy()


class EGA1Attention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.hs=head_size
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)
        self.proj=nn.Linear(N_EMBED,1,bias=True)
        self.tau=nn.Parameter(torch.zeros(1))
        self.alpha=nn.Parameter(torch.ones(1)*2.0)
        self.register_buffer(
            "tril",torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        e=znorm(self.proj(x).transpose(-2,-1),T)
        g=torch.sigmoid(self.alpha*(e-self.tau))
        att=self.drop(F.softmax(sc,dim=-1))
        att=att*g
        att=att/att.sum(-1,keepdim=True).clamp(min=1e-8)
        return att@v


# Positional encodings
class SinCosPE(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*
                       (-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe)
    def forward(self,T): return self.pe[:T]

class MorletPE(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        n=d_model//2
        freqs=torch.exp(torch.linspace(
            math.log(1.0),math.log(math.pi*0.99),n))
        self.log_omega=nn.Parameter(torch.log(freqs))
        self.log_sigma=nn.Parameter(torch.log(5.0/freqs))
        self.register_buffer("pos",torch.arange(max_len).float())
    def forward(self,T):
        pos=self.pos[:T]
        omega=torch.exp(self.log_omega).clamp(max=math.pi*0.95)
        sigma=torch.exp(self.log_sigma).clamp(min=1e-3)
        omega=torch.where(omega*sigma<5.0,
                           5.0/sigma.clamp(min=1e-6),omega)
        env=torch.exp(-pos.unsqueeze(1)**2/
                       (2.0*sigma.unsqueeze(0)**2+1e-8))
        phase=pos.unsqueeze(1)*omega.unsqueeze(0)
        pe=torch.zeros(T,2*len(omega),device=omega.device)
        pe[:,0::2]=torch.cos(phase)*env
        pe[:,1::2]=torch.sin(phase)*env
        return pe
    def get_learned_params(self):
        with torch.no_grad():
            return (torch.exp(self.log_omega).clamp(
                        max=math.pi*0.95).cpu().numpy(),
                    torch.exp(self.log_sigma).clamp(
                        min=1e-3).cpu().numpy())

class RoPEAttention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.hs=head_size
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)
        theta=1.0/(10000**(
            torch.arange(0,head_size,2).float()/head_size))
        self.register_buffer("theta",theta)
        self.register_buffer(
            "tril",torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
    def _rotate(self,x,T):
        pos=torch.arange(T,device=x.device).float()
        freqs=torch.outer(pos,self.theta)
        out=torch.zeros_like(x)
        out[...,0::2]=(x[...,0::2]*torch.cos(freqs)-
                        x[...,1::2]*torch.sin(freqs))
        out[...,1::2]=(x[...,0::2]*torch.sin(freqs)+
                        x[...,1::2]*torch.cos(freqs))
        return out
    def forward(self,x):
        B,T,_=x.shape
        k=self._rotate(self.key(x),T)
        q=self._rotate(self.query(x),T)
        v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        return self.drop(F.softmax(sc,dim=-1))@v

# ================================================================
# EXPERIMENT 1
# ================================================================
print("="*55)
print("  EXP 1 — Convolution Attention")
print("  CONV-L16 → CONV-L4 (memory fix)")
print("="*55)

exp1_hist={}

# Load completed models
for lb,sn in [("BASE-DOT","exp1_base"),("CONV-L8","exp1_conv8")]:
    if ckpt_done(sn):
        ck=torch.load(os.path.join(CKPT_DIR,f"{sn}.pt"),
                       map_location="cpu")
        exp1_hist[lb]=ck["history"]
        print(f"  ✓ {lb}  val={exp1_hist[lb]['val'][-1]:.4f}")

# CONV-L4 — replaces CONV-L16
m=GPT(ConvolutionAttention,{"max_lag":4},label="CONV-L4").to(DEVICE)
exp1_hist["CONV-L4"]=load_or_train("CONV-L4","exp1_conv4",m)

# EGA-1
m=GPT(EGA1Attention,{},label="EGA-1").to(DEVICE)
exp1_hist["EGA-1"]=load_or_train("EGA-1","exp1_ega1",m)

# ================================================================
# EXPERIMENT 2
# ================================================================
print("\n"+"="*55)
print("  EXP 2 — Morlet Positional Encoding")
print("="*55)

exp2_hist={}
exp2_configs=[
    ("PE-SINCOS", "exp2_sincos", DotProductAttention, SinCosPE),
    ("PE-ROPE",   "exp2_rope",   RoPEAttention,       None),
    ("PE-MORLET", "exp2_morlet", DotProductAttention, MorletPE),
    ("EGA-MORLET","exp2_egam",   EGA1Attention,       MorletPE),
]
for label,sname,attn_cls,pe_cls in exp2_configs:
    m=GPT(attn_cls,{},pe_cls,label=label).to(DEVICE)
    exp2_hist[label]=load_or_train(label,sname,m)

# ================================================================
# RESULTS
# ================================================================
bv=exp1_hist["BASE-DOT"]["val"][-1]

print("\n"+"="*55)
print("  EXPERIMENT 1 RESULTS — Convolution Attention")
print("="*55)
print(f"  {'Model':<14} {'Val':>8} {'ΔBASE':>8}  Note")
print("  "+"─"*45)
notes={"BASE-DOT":"zero-lag dot product",
       "CONV-L8": "±8 lag cross-correlation",
       "CONV-L4": "±4 lag cross-correlation",
       "EGA-1":   "energy gate (Phase 1-3)"}
for lb in ["BASE-DOT","CONV-L4","CONV-L8","EGA-1"]:
    if lb in exp1_hist:
        v=exp1_hist[lb]["val"][-1]
        print(f"  {lb:<14} {v:>8.4f} {bv-v:>+8.4f}  "
              f"{notes.get(lb,'')}")

print("\n  KEY QUESTION: Do CONV models beat BASE-DOT?")
for lb in ["CONV-L4","CONV-L8"]:
    if lb in exp1_hist:
        v=exp1_hist[lb]["val"][-1]
        ans="✓ YES — nonzero lags carry information" \
            if v<bv else "✗ NO — zero-lag dot product sufficient"
        print(f"  {lb}: {ans}")

print("\n"+"="*55)
print("  EXPERIMENT 2 RESULTS — Positional Encoding")
print("="*55)
print(f"  {'Model':<14} {'Val':>8} {'ΔBASE':>8}  Encoding")
print("  "+"─"*52)
enc_notes={"PE-SINCOS":"fixed sin/cos",
            "PE-ROPE":  "rotary (relative)",
            "PE-MORLET":"learned Morlet wavelet",
            "EGA-MORLET":"EGA-1 + Morlet PE"}
for lb,sn,_,_ in exp2_configs:
    if lb in exp2_hist:
        v=exp2_hist[lb]["val"][-1]
        print(f"  {lb:<14} {v:>8.4f} {bv-v:>+8.4f}  "
              f"{enc_notes.get(lb,'')}")

print("\n  KEY QUESTION: Does Morlet PE beat sin/cos?")
if "PE-SINCOS" in exp2_hist and "PE-MORLET" in exp2_hist:
    vs=exp2_hist["PE-SINCOS"]["val"][-1]
    vm=exp2_hist["PE-MORLET"]["val"][-1]
    delta=vs-vm
    if delta>0.005:
        print(f"  ✓ YES — Morlet PE wins by {delta:.4f}")
        print(f"    Gaussian locality helps at T=256")
    elif abs(delta)<0.005:
        print(f"  → EQUIVALENT — delta={delta:+.4f}")
        print(f"    Locality provides no benefit at this scale")
    else:
        print(f"  ✗ NO — sin/cos wins by {abs(delta):.4f}")
        print(f"    Fixed uniform tiling better than Morlet")

print("\n  KEY QUESTION: Does EGA-MORLET beat EGA-1?")
if "EGA-MORLET" in exp2_hist and "EGA-1" in exp1_hist:
    ve=exp1_hist["EGA-1"]["val"][-1]
    vem=exp2_hist["EGA-MORLET"]["val"][-1]
    delta=ve-vem
    if delta>0.005:
        print(f"  ✓ YES — combining EGA + Morlet PE wins by {delta:.4f}")
        print(f"    Phase 1-3 + Phase 4 combination is additive")
    elif abs(delta)<0.005:
        print(f"  → EQUIVALENT — delta={delta:+.4f}")
    else:
        print(f"  ✗ NO — EGA-1 alone better by {abs(delta):.4f}")

# ================================================================
# PLOT
# ================================================================
steps=list(range(0,MAX_ITERS+1,EVAL_INTERVAL))
colors={"BASE-DOT":"#2196F3","CONV-L4":"#009688",
        "CONV-L8":"#4CAF50","EGA-1":"#FF9800",
        "PE-SINCOS":"#2196F3","PE-ROPE":"#9C27B0",
        "PE-MORLET":"#E91E63","EGA-MORLET":"#FF5722"}

fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle(
    "Phase 4 — Exp 1 (Convolution) + Exp 2 (Morlet PE)\n"
    "TinyShakespeare | 6L×8H×256d | 5000 steps",
    fontsize=11,fontweight="bold"
)

# Exp 1 val loss
ax=axes[0]
for lb in ["BASE-DOT","CONV-L4","CONV-L8","EGA-1"]:
    if lb in exp1_hist:
        ax.plot(steps,exp1_hist[lb]["val"],
                color=colors[lb],marker="o",
                markersize=3,label=lb,linewidth=1.8)
ax.set_title("Exp 1: Convolution Attention\n"
             "Does cross-correlation beat dot product?")
ax.set_xlabel("Step"); ax.set_ylabel("Val Loss")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# Exp 2 val loss
ax=axes[1]
for lb,sn,_,_ in exp2_configs:
    if lb in exp2_hist:
        ls="--" if "MORLET" in lb else "-"
        ax.plot(steps,exp2_hist[lb]["val"],
                color=colors[lb],marker="o",
                markersize=3,label=lb,
                linewidth=1.8,linestyle=ls)
ax.set_title("Exp 2: Positional Encoding\n"
             "Does Morlet PE beat sin/cos?")
ax.set_xlabel("Step"); ax.set_ylabel("Val Loss")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# Final bar — all models
ax=axes[2]
all_h={**exp1_hist}
for lb,sn,_,_ in exp2_configs:
    if lb in exp2_hist and lb not in all_h:
        all_h[lb]=exp2_hist[lb]
lbls=list(all_h.keys())
vals=[all_h[lb]["val"][-1] for lb in lbls]
cols=[colors.get(lb,"#888") for lb in lbls]
bars=ax.bar(range(len(lbls)),vals,color=cols,alpha=0.85)
ax.set_xticks(range(len(lbls)))
ax.set_xticklabels(lbls,rotation=45,ha="right",fontsize=8)
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.001,f"{v:.4f}",
            ha="center",va="bottom",
            fontsize=7,fontweight="bold")
ax.axhline(bv,color="black",linewidth=1.5,
           linestyle="--",label=f"BASE ({bv:.4f})")
ax.set_title("Final Val Loss — All Models")
ax.set_ylabel("Val Loss")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis="y")
bi=vals.index(min(vals))
bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(3)

plt.tight_layout()
out=os.path.join(CKPT_DIR,"phase4_exp12.png")
plt.savefig(out,dpi=150,bbox_inches="tight")
plt.show()

# ================================================================
# DOWNLOAD
# ================================================================
print("\nDownloading …")
from google.colab import files
files.download(out)

# Also download all new checkpoints
for sname in ["exp1_conv4","exp1_ega1",
               "exp2_sincos","exp2_rope",
               "exp2_morlet","exp2_egam"]:
    p=os.path.join(CKPT_DIR,f"{sname}.pt")
    if os.path.exists(p):
        files.download(p)
        print(f"  Downloaded {sname}.pt")

print("\nDone. Next: run Exp 3-5 cell.")
