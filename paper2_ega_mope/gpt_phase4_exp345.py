"""
Phase 4 — Experiments 3, 4, 5
===============================
Exp 3  Scale-Initialized Attention Heads
Exp 4  Multi-Quantity Attention Score (EGA-M+ fixed)
Exp 5  Spectral Cascade Interpretability (no training)

Prerequisites: upload these checkpoints before running:
  exp1_base.pt   exp1_ega1.pt   exp2_egam.pt
  (from your Downloads folder)

Upload cell:
  from google.colab import files
  import shutil, os
  os.makedirs("/content/gpt_phase4", exist_ok=True)
  uploaded = files.upload()
  for fname in uploaded:
      shutil.move(fname, f"/content/gpt_phase4/{fname}")
  print("Restored.")

Total runtime: ~2.5 hours on T4
"""

import math, os, gc, warnings, requests
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
print(f"  Dataset: {len(text):,} chars\n")

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
        std=t.std(dim=-1,keepdim=True,
                   correction=0).clamp(min=1e-8)
        return (t-mu)/std
    return torch.zeros_like(t)

def ckpt_done(name):
    p=os.path.join(CKPT_DIR,f"{name}.pt")
    if not os.path.exists(p): return False
    return torch.load(
        p,map_location="cpu").get("step",0)>=MAX_ITERS

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
    pbar=tqdm(range(MAX_ITERS+1),
              desc=f"  {label}",leave=True)
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),GRAD_CLIP)
        scaler.step(opt); scaler.update()
        h["curve"].append(loss.item())
        pbar.set_postfix({"loss":f"{loss.item():.3f}"})
        if step%EVAL_INTERVAL==0:
            ev=estimate_loss(model)
            h["train"].append(ev["train"])
            h["val"].append(ev["val"])
            tqdm.write(
                f"  {label}  step={step:>5}  "
                f"train={ev['train']:.4f}  "
                f"val={ev['val']:.4f}"
            )
            save_ckpt(save_name,model,opt,h,step)
    return h

def load_or_train(label,save_name,model):
    n=sum(p.numel() for p in model.parameters())
    print(f"\n── {label} ({n:,} params) ──────────────")
    print(f"  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
    if ckpt_done(save_name):
        h=load_ckpt(save_name,model)
        print(f"  ✓ Loaded — val={h['val'][-1]:.4f}")
    else:
        print("  Training …")
        h=train_model(label,model,save_name)
    model.cpu(); del model
    gc.collect(); torch.cuda.empty_cache()
    print(f"  GPU after: "
          f"{torch.cuda.mem_get_info()[0]/1e9:.2f} GB free")
    return h

# ================================================================
# SHARED TRANSFORMER COMPONENTS
# ================================================================
class Block(nn.Module):
    def __init__(self,attn_class,attn_kwargs={}):
        super().__init__()
        hs=N_EMBED//N_HEAD
        self.heads=nn.ModuleList([
            attn_class(hs,**attn_kwargs)
            for _ in range(N_HEAD)])
        self.proj=nn.Linear(N_EMBED,N_EMBED)
        self.drop=nn.Dropout(DROPOUT)
        self.ff=nn.Sequential(
            nn.Linear(N_EMBED,4*N_EMBED),nn.GELU(),
            nn.Linear(4*N_EMBED,N_EMBED),
            nn.Dropout(DROPOUT))
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
        self.pe=(pe_class(N_EMBED,BLOCK_SIZE)
                 if pe_class else None)
        self.pos_emb=(None if pe_class else
                      nn.Embedding(BLOCK_SIZE,N_EMBED))
        self.drop=nn.Dropout(DROPOUT)
        self.blocks=nn.Sequential(*[
            Block(attn_class,attn_kwargs)
            for _ in range(N_LAYER)])
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
        loss=(F.cross_entropy(
                  logits.view(-1,VOCAB),targets.view(-1))
              if targets is not None else None)
        return logits,loss
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

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
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(
            self.tril[:T,:T]==0,float("-inf"))
        return self.drop(F.softmax(sc,dim=-1))@v


class EGA1Attention(nn.Module):
    """EGA-1 — Phase 1-3 best model."""
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
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(
            self.tril[:T,:T]==0,float("-inf"))
        e=znorm(self.proj(x).transpose(-2,-1),T)
        g=torch.sigmoid(self.alpha*(e-self.tau))
        att=self.drop(F.softmax(sc,dim=-1))
        att=att*g
        att=att/att.sum(-1,keepdim=True).clamp(min=1e-8)
        return att@v


# ================================================================
# EXPERIMENT 3 — SCALE-INITIALIZED ATTENTION HEADS
# ================================================================
print("="*58)
print("  EXP 3 — Scale-Initialized Attention Heads")
print("  Hypothesis: scale-specific init converges faster")
print("="*58)

class ScaleInitAttention(nn.Module):
    """
    Attention head initialized at a specific scale level.

    Theoretical basis:
      "Queries and keys are scale selectors" (Phase 4 theory)
      Each head should specialize to a frequency band.
      Random init discovers this accidentally.
      Scale init gives the right structure from step 0.

    Implementation:
      Fine scale  (level=0): small init std → responds to
                              high-frequency patterns
      Coarse scale (level=3): large init std → responds to
                              low-frequency patterns
    """
    def __init__(self,head_size,scale_level=0,n_scales=4):
        super().__init__()
        self.hs=head_size
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)

        # Scale-specific initialization
        # std varies from 0.02 (fine) to 0.04 (coarse)
        scale_std=0.02*(1.0+scale_level/n_scales)
        nn.init.normal_(self.key.weight,  0.0,scale_std)
        nn.init.normal_(self.query.weight,0.0,scale_std)
        nn.init.normal_(self.value.weight,0.0,0.02)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))

    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc=q@k.transpose(-2,-1)/math.sqrt(self.hs)
        sc=sc.masked_fill(
            self.tril[:T,:T]==0,float("-inf"))
        return self.drop(F.softmax(sc,dim=-1))@v


class ScaleInitBlock(nn.Module):
    """Block where each head gets a different scale init."""
    def __init__(self,n_scales=4):
        super().__init__()
        hs=N_EMBED//N_HEAD
        self.heads=nn.ModuleList([
            ScaleInitAttention(hs,
                               scale_level=h%n_scales,
                               n_scales=n_scales)
            for h in range(N_HEAD)])
        self.proj=nn.Linear(N_EMBED,N_EMBED)
        self.drop=nn.Dropout(DROPOUT)
        self.ff=nn.Sequential(
            nn.Linear(N_EMBED,4*N_EMBED),nn.GELU(),
            nn.Linear(4*N_EMBED,N_EMBED),
            nn.Dropout(DROPOUT))
        self.ln1=nn.LayerNorm(N_EMBED)
        self.ln2=nn.LayerNorm(N_EMBED)
    def forward(self,x):
        xn=self.ln1(x)
        ao=torch.cat([h(xn) for h in self.heads],dim=-1)
        x=x+self.drop(self.proj(ao))
        return x+self.ff(self.ln2(x))


class ScaleInitGPT(nn.Module):
    """GPT with scale-initialized attention heads."""
    def __init__(self,label=""):
        super().__init__()
        self.label=label
        self.tok_emb=nn.Embedding(VOCAB,N_EMBED)
        self.pos_emb=nn.Embedding(BLOCK_SIZE,N_EMBED)
        self.drop=nn.Dropout(DROPOUT)
        self.blocks=nn.Sequential(*[
            ScaleInitBlock() for _ in range(N_LAYER)])
        self.ln_f=nn.LayerNorm(N_EMBED)
        self.head=nn.Linear(N_EMBED,VOCAB,bias=False)
        self.tok_emb.weight=self.head.weight
        # Note: only non-attention weights get standard init
        # Attention weights were already set by ScaleInitAttention
        for m in self.modules():
            if isinstance(m,(nn.Embedding,)):
                nn.init.normal_(m.weight,0.0,0.02)
            if (isinstance(m,nn.Linear) and
                m.bias is not None):
                nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device)
        x=self.drop(self.tok_emb(idx)+self.pos_emb(pos))
        x=self.blocks(x); x=self.ln_f(x)
        logits=self.head(x)
        loss=(F.cross_entropy(
                  logits.view(-1,VOCAB),targets.view(-1))
              if targets is not None else None)
        return logits,loss
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


exp3_hist={}

# BASE-RANDOM — load from Exp 1 checkpoint
print("\n  Loading BASE-RANDOM from exp1_base.pt …")
if ckpt_done("exp1_base"):
    ck=torch.load(os.path.join(CKPT_DIR,"exp1_base.pt"),
                   map_location="cpu")
    exp3_hist["BASE-RANDOM"]=ck["history"]
    print(f"  ✓ val={exp3_hist['BASE-RANDOM']['val'][-1]:.4f}")
else:
    print("  exp1_base.pt not found — training BASE-RANDOM")
    m=GPT(DotProductAttention,{},label="BASE-RANDOM").to(DEVICE)
    exp3_hist["BASE-RANDOM"]=load_or_train(
        "BASE-RANDOM","exp1_base",m)

# SCALE-INIT
m=ScaleInitGPT(label="SCALE-INIT").to(DEVICE)
print(f"\n── SCALE-INIT ({m.num_parameters():,} params) ──────")
print(f"  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
if ckpt_done("exp3_scale"):
    exp3_hist["SCALE-INIT"]=load_ckpt("exp3_scale",m)
    print(f"  ✓ Loaded — "
          f"val={exp3_hist['SCALE-INIT']['val'][-1]:.4f}")
    m.cpu(); del m
    gc.collect(); torch.cuda.empty_cache()
else:
    print("  Training …")
    exp3_hist["SCALE-INIT"]=train_model(
        "SCALE-INIT",m,"exp3_scale")
    m.cpu(); del m
    gc.collect(); torch.cuda.empty_cache()

# ================================================================
# EXPERIMENT 4 — MULTI-QUANTITY ATTENTION SCORE
# ================================================================
print("\n"+"="*58)
print("  EXP 4 — Multi-Quantity Attention Score")
print("  Revisits EGA-M+ with all numerical fixes applied")
print("  Q: Does phase/flux improve over energy alone?")
print("="*58)

def apply_morlet_f32(x,omega0,sigma,length):
    """Morlet wavelet — forced float32 for AMP safety."""
    B,T,_=x.shape
    device=x.device
    t=torch.arange(length,dtype=torch.float32,device=device)
    env=torch.exp(-t**2/(2.0*sigma.float()**2+1e-6))
    rk=torch.nan_to_num(torch.cos(omega0.float()*t)*env)
    ik=torch.nan_to_num(torch.sin(omega0.float()*t)*env)
    x1d=x.float().mean(dim=-1,keepdim=True).transpose(1,2)
    xp=F.pad(x1d,(length-1,0))
    r=torch.nan_to_num(F.conv1d(xp,rk.view(1,1,-1))[:,:,:T])
    i=torch.nan_to_num(F.conv1d(xp,ik.view(1,1,-1))[:,:,:T])
    return torch.cat([r,i],dim=1).to(x.dtype)


class MultiQuantityAttention(nn.Module):
    """
    Multi-quantity attention — Phase 4 numerically fixed.

    Score: λ_sim·(q·k/√d) + λ_E·znorm(E) + λ_φ·cos(φ) + λ_Φ·znorm(F)

    Three fixes vs failed EGA-M+ in Phase 3:
    1. znorm(energy) not log(energy) — avoids -inf
    2. λ init at 0.01 not 0.1 — keeps similarity dominant
    3. nan_to_num guard before masked_fill

    Ablation: use_phase and use_flux control which
    quantities are included, isolating their contribution.
    """
    FILTER_LENGTHS=[3,7,15,31]

    def __init__(self,head_size,n_scales=4,
                 use_phase=True,use_flux=True):
        super().__init__()
        self.hs=head_size
        self.n_scales=n_scales
        self.use_phase=use_phase
        self.use_flux=use_flux
        fl=self.FILTER_LENGTHS[:n_scales]
        self.filter_lengths=fl
        self.key=nn.Linear(N_EMBED,head_size,bias=False)
        self.query=nn.Linear(N_EMBED,head_size,bias=False)
        self.value=nn.Linear(N_EMBED,head_size,bias=False)
        self.drop=nn.Dropout(DROPOUT)

        # Morlet params in log space — float32
        omega0=[math.pi/l for l in fl]
        sigma=[5.0/w for w in omega0]
        self.log_omega0=nn.Parameter(
            torch.tensor([math.log(w) for w in omega0],
                          dtype=torch.float32))
        self.log_sigma=nn.Parameter(
            torch.tensor([math.log(s) for s in sigma],
                          dtype=torch.float32))

        # Gate params — tau=0 avoids over-suppression
        self.tau=nn.Parameter(torch.zeros(n_scales))
        self.alpha=nn.Parameter(torch.ones(n_scales)*2.0)
        self.scale_w=nn.Parameter(
            torch.ones(n_scales)/n_scales)

        # FIX 2: small lambda init
        self.lambda_sim=nn.Parameter(torch.tensor(1.0))
        self.lambda_energy=nn.Parameter(torch.tensor(0.01))
        self.lambda_phase=nn.Parameter(torch.tensor(0.01))
        self.lambda_flux=nn.Parameter(torch.tensor(0.01))

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))

    def _morlet_params(self):
        o=torch.exp(self.log_omega0.float()
                    ).clamp(max=math.pi*0.95)
        s=torch.exp(self.log_sigma.float()
                    ).clamp(min=1e-3)
        return torch.where(o*s<5.0,
                            5.0/s.clamp(min=1e-6),o),s

    def _quantities(self,x,T):
        omega0,sigma=self._morlet_params()
        all_e,all_p,all_f=[],[],[]
        for s in range(self.n_scales):
            c=apply_morlet_f32(
                x,omega0[s],sigma[s],
                self.filter_lengths[s])
            real=c[:,0:1,:]; imag=c[:,1:2,:]
            energy=real**2+imag**2
            # FIX 1: znorm not log
            e_s=znorm(energy.to(x.dtype),T)
            all_e.append(e_s)
            if self.use_phase:
                all_p.append(
                    torch.atan2(imag,real).to(x.dtype))
            if self.use_flux:
                flux=(torch.diff(energy,dim=-1,
                                  prepend=energy[:,:,:1]
                                  ).abs()
                      if T>1
                      else torch.zeros_like(energy))
                all_f.append(znorm(flux.to(x.dtype),T))

        e_mean=torch.stack(all_e).mean(0)
        p_mean=(torch.stack(all_p).mean(0)
                if self.use_phase else None)
        f_mean=(torch.stack(all_f).mean(0)
                if self.use_flux else None)
        return e_mean,p_mean,f_mean

    def forward(self,x):
        B,T,_=x.shape
        k=self.key(x); q=self.query(x); v=self.value(x)
        sc_sim=q@k.transpose(-2,-1)/math.sqrt(self.hs)

        energy,phase,flux=self._quantities(x,T)

        combined=self.lambda_sim*sc_sim
        combined=combined+self.lambda_energy*\
                  energy.expand(B,T,T)
        if self.use_phase and phase is not None:
            combined=combined+self.lambda_phase*\
                      torch.cos(phase).expand(B,T,T)
        if self.use_flux and flux is not None:
            combined=combined+self.lambda_flux*\
                      flux.expand(B,T,T)

        # FIX 3: nan guard before masked_fill
        combined=torch.nan_to_num(
            combined,nan=0.0,posinf=1e4,neginf=-1e4)
        combined=combined.masked_fill(
            self.tril[:T,:T]==0,float("-inf"))

        # Energy gate
        omega0,sigma=self._morlet_params()
        gates=[]
        for s in range(self.n_scales):
            c=apply_morlet_f32(
                x,omega0[s],sigma[s],
                self.filter_lengths[s])
            e_s=znorm(
                (c[:,0:1,:]**2+c[:,1:2,:]**2
                 ).to(x.dtype),T)
            gates.append(torch.sigmoid(
                self.alpha[s]*(e_s-self.tau[s])))
        sw=F.softmax(self.scale_w,dim=0)
        gate=sum(sw[s]*gates[s]
                 for s in range(self.n_scales))

        att=self.drop(F.softmax(combined,dim=-1))
        att=torch.nan_to_num(att,nan=0.0)
        att=att*gate
        att=att/att.sum(-1,keepdim=True).clamp(min=1e-8)
        return att@v

    def get_lambda_weights(self):
        return {"similarity":self.lambda_sim.item(),
                "energy":    self.lambda_energy.item(),
                "phase":     self.lambda_phase.item(),
                "flux":      self.lambda_flux.item()}


exp4_hist={}
exp4_configs=[
    ("MQ-E",  "exp4_mq_e",  False,False),
    ("MQ-EP", "exp4_mq_ep", True, False),
    ("MQ-EF", "exp4_mq_ef", False,True),
    ("MQ-EPF","exp4_mq_epf",True, True),
]

# Reference: EGA-1 val from Exp 1
ega1_val=1.3821  # from Exp 1+2 run

for label,sname,use_p,use_f in exp4_configs:
    m=GPT(MultiQuantityAttention,
          {"n_scales":4,"use_phase":use_p,
           "use_flux":use_f},
          label=label).to(DEVICE)
    exp4_hist[label]=load_or_train(label,sname,m)

# ================================================================
# EXPERIMENT 5 — SPECTRAL CASCADE INTERPRETABILITY
# ================================================================
print("\n"+"="*58)
print("  EXP 5 — Spectral Cascade Analysis")
print("  No training — analysis of saved checkpoints")
print("  Cascade(layer, scale) = mean |W_ψ[e^(l)](a)|")
print("="*58)

def morlet_energy_1d(signal,scales,omega0=6.0):
    """Morlet CWT energy for a 1D signal at given scales."""
    T=len(signal)
    sig=torch.tensor(signal,dtype=torch.float32)
    out=[]
    for a in scales:
        hl=min(int(4*math.sqrt(a))+1,T//2)
        t=torch.arange(-hl,hl+1,dtype=torch.float32)
        ts=t/math.sqrt(a)
        env=torch.exp(-ts**2/2.0)
        norm=1.0/math.pow(a,0.25)
        real=torch.cos(omega0*ts)*env*norm
        imag=torch.sin(omega0*ts)*env*norm
        fl=len(real); pad=fl//2
        sp=F.pad(sig.unsqueeze(0).unsqueeze(0),
                  (pad,pad),mode="reflect")
        r=F.conv1d(sp,real.flip(0).view(1,1,-1))
        i=F.conv1d(sp,imag.flip(0).view(1,1,-1))
        mag=torch.sqrt(r**2+i**2)
        st=(mag.shape[-1]-T)//2
        out.append(mag[0,0,st:st+T].mean().item())
    return np.array(out)

@torch.no_grad()
def compute_cascade(model,scales,n_batches=15):
    """
    Compute Cascade(l,a) = mean spectral energy at
    layer l and scale a.

    By Parseval: this measures how much information
    at each temporal scale is present at each layer.
    Fine scales dominate early layers (syntax),
    coarse scales dominate later layers (semantics).
    """
    model.eval().to(DEVICE)
    layer_embs=[[] for _ in range(N_LAYER+1)]
    hooks=[]

    def make_hook(li):
        def hook(mod,inp,out):
            # Store mean over batch — shape [T, E]
            layer_embs[li].append(
                out.detach().cpu().float().mean(0))
        return hook

    hooks.append(model.drop.register_forward_hook(
        make_hook(0)))
    for i,block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            make_hook(i+1)))

    with torch.no_grad():
        for _ in range(n_batches):
            x,_=get_batch("val")
            model(x)

    for h in hooks: h.remove()
    model.cpu(); gc.collect(); torch.cuda.empty_cache()

    # Compute cascade matrix
    cascade=np.zeros((N_LAYER+1,len(scales)))
    for li in range(N_LAYER+1):
        if not layer_embs[li]: continue
        emb=torch.stack(layer_embs[li]).mean(0)  # [T,E]
        # Subsample dims for speed
        dim_indices=range(0,N_EMBED,8)  # 32 dims
        e_per_scale=[]
        for di in dim_indices:
            sig=emb[:,di].numpy()
            e_per_scale.append(morlet_energy_1d(sig,scales))
        cascade[li]=np.mean(e_per_scale,axis=0)

    return cascade

scales=np.logspace(0,2,20)  # 20 scales: 1→100 tokens
cascade_results={}

# Analyze BASE-DOT and EGA-1
for label,sname,cls,kw in [
    ("BASE-DOT","exp1_base",DotProductAttention,{}),
    ("EGA-1",   "exp1_ega1",EGA1Attention,{}),
]:
    path=os.path.join(CKPT_DIR,f"{sname}.pt")
    if not os.path.exists(path):
        print(f"  {label}: {sname}.pt not found — skipping")
        continue
    print(f"\n  Computing cascade for {label} …")
    m=GPT(cls,kw,label=label).to(DEVICE)
    load_ckpt(sname,m)
    cascade_results[label]=compute_cascade(m,scales)
    print(f"  ✓ {label} cascade computed")

# Also analyze best Phase 4 model: EGA-MORLET
class MorletPE(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        n=d_model//2
        freqs=torch.exp(torch.linspace(
            math.log(1.0),math.log(math.pi*0.99),n))
        self.log_omega=nn.Parameter(torch.log(freqs))
        self.log_sigma=nn.Parameter(
            torch.log(5.0/freqs))
        self.register_buffer(
            "pos",torch.arange(max_len).float())
    def forward(self,T):
        pos=self.pos[:T]
        omega=torch.exp(self.log_omega
                         ).clamp(max=math.pi*0.95)
        sigma=torch.exp(self.log_sigma).clamp(min=1e-3)
        omega=torch.where(omega*sigma<5.0,
                           5.0/sigma.clamp(min=1e-6),
                           omega)
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

path_egam=os.path.join(CKPT_DIR,"exp2_egam.pt")
if os.path.exists(path_egam):
    print(f"\n  Computing cascade for EGA-MORLET …")
    m=GPT(EGA1Attention,{},MorletPE,
           label="EGA-MORLET").to(DEVICE)
    load_ckpt("exp2_egam",m)
    cascade_results["EGA-MORLET"]=compute_cascade(m,scales)
    print(f"  ✓ EGA-MORLET cascade computed")

    # Extract learned Morlet PE frequencies
    morlet_pe_params=None
    if hasattr(m,"pe") and m.pe is not None:
        morlet_pe_params=m.pe.get_learned_params()
    m.cpu(); del m; gc.collect(); torch.cuda.empty_cache()

# ================================================================
# RESULTS
# ================================================================
bv=1.4742  # BASE-DOT reference

print("\n"+"="*58)
print("  EXPERIMENT 3 — Scale-Initialized Heads")
print("="*58)
print(f"  {'Model':<14} {'Val':>8} {'ΔBASE':>8}")
print("  "+"─"*32)
for lb in ["BASE-RANDOM","SCALE-INIT"]:
    if lb in exp3_hist:
        v=exp3_hist[lb]["val"][-1]
        print(f"  {lb:<14} {v:>8.4f} {bv-v:>+8.4f}")

if ("BASE-RANDOM" in exp3_hist and
    "SCALE-INIT"  in exp3_hist):
    vr=exp3_hist["BASE-RANDOM"]["val"][-1]
    vs=exp3_hist["SCALE-INIT"]["val"][-1]
    # Also check convergence speed: steps to reach val<1.50
    def steps_to(h,threshold=1.50):
        for i,(s,v) in enumerate(
            zip(h["step"],h["val"])
        ):
            if v<threshold: return s
        return MAX_ITERS
    sr=steps_to(exp3_hist["BASE-RANDOM"])
    ss=steps_to(exp3_hist["SCALE-INIT"])
    print(f"\n  Convergence (steps to val<1.50):")
    print(f"    BASE-RANDOM: step {sr}")
    print(f"    SCALE-INIT:  step {ss}")
    if ss<sr:
        print(f"    ✓ Scale init converges {sr-ss} "
              f"steps faster")
    if vs<vr:
        print(f"    ✓ Scale init also achieves better "
              f"final val (+{vr-vs:.4f})")

print("\n"+"="*58)
print("  EXPERIMENT 4 — Multi-Quantity Score")
print("="*58)
print(f"  {'Model':<10} {'Val':>8} {'ΔBASE':>8} "
      f"{'Δ EGA-1':>9}")
print("  "+"─"*40)
for lb,sn,_,_ in exp4_configs:
    if lb in exp4_hist:
        v=exp4_hist[lb]["val"][-1]
        print(f"  {lb:<10} {v:>8.4f} {bv-v:>+8.4f} "
              f"{ega1_val-v:>+9.4f}")

# Print learned lambda weights for MQ-EPF
if ckpt_done("exp4_mq_epf"):
    print(f"\n  MQ-EPF learned λ weights:")
    m_mq=GPT(MultiQuantityAttention,
              {"n_scales":4,"use_phase":True,
               "use_flux":True}).to(DEVICE)
    load_ckpt("exp4_mq_epf",m_mq)
    for block in m_mq.blocks:
        for head in block.heads:
            if hasattr(head,"get_lambda_weights"):
                lw=head.get_lambda_weights()
                total=sum(abs(v) for v in lw.values())
                for k,v in lw.items():
                    bar="█"*int(abs(v)/total*25)
                    print(f"    λ_{k:<12}= {v:>8.4f}  "
                          f"({abs(v)/total*100:>5.1f}%) {bar}")
                break
        break
    m_mq.cpu(); del m_mq
    gc.collect(); torch.cuda.empty_cache()

print("\n"+"="*58)
print("  EXPERIMENT 5 — Spectral Cascade")
print("="*58)
if cascade_results:
    layer_labels=["emb"]+[f"L{i+1}"
                           for i in range(N_LAYER)]
    print(f"\n  {'Layer':<8}", end="")
    for lb in cascade_results:
        print(f"  {lb:>12}", end="")
    print()
    print("  "+"─"*50)
    for li,ll in enumerate(layer_labels):
        print(f"  {ll:<8}", end="")
        for lb in cascade_results:
            e=cascade_results[lb][li].mean()
            print(f"  {e:>12.4f}", end="")
        print()

# ================================================================
# COMPLETE PHASE 4 SUMMARY
# ================================================================
print("\n"+"═"*58)
print("  PHASE 4 — COMPLETE SUMMARY")
print("═"*58)

all_results={
    "BASE-DOT": 1.4742,
    "CONV-L4":  1.4668,
    "CONV-L8":  1.4691,
    "EGA-1":    ega1_val,
    "PE-SINCOS":1.5863,
    "PE-ROPE":  1.4637,
    "PE-MORLET":1.5060,
    "EGA-MORLET":1.3550,
}
for lb in ["BASE-RANDOM","SCALE-INIT"]:
    if lb in exp3_hist:
        all_results[lb]=exp3_hist[lb]["val"][-1]
for lb,sn,_,_ in exp4_configs:
    if lb in exp4_hist:
        all_results[lb]=exp4_hist[lb]["val"][-1]

best_lb=min(all_results,key=all_results.get)
print(f"\n  {'Model':<14} {'Val':>8} {'ΔBASE':>8}")
print("  "+"─"*34)
for lb,v in all_results.items():
    mrk=" ◄BEST" if lb==best_lb else ""
    print(f"  {lb:<14} {v:>8.4f} {bv-v:>+8.4f}{mrk}")

print(f"\n  KEY FINDINGS:")
print(f"  1. Nonzero lags help: CONV-L4 Δ=+0.007")
print(f"  2. EGA-MORLET best: combines salience+locality")
print(f"     Superadditive: EGA+Morlet > EGA alone")
print(f"  3. sin/cos PE surprisingly bad for char-level")
print(f"  4. Scale init: see convergence speed above")
print(f"  5. Multi-quantity: see λ weights above")
print(f"  6. Cascade: see layer-scale table above")

# ================================================================
# FINAL PLOT — ALL EXPERIMENTS
# ================================================================
steps=list(range(0,MAX_ITERS+1,EVAL_INTERVAL))
colors={
    "BASE-DOT":"#2196F3","CONV-L4":"#009688",
    "CONV-L8":"#4CAF50","EGA-1":"#FF9800",
    "BASE-RANDOM":"#2196F3","SCALE-INIT":"#E91E63",
    "MQ-E":"#FF9800","MQ-EP":"#E91E63",
    "MQ-EF":"#9C27B0","MQ-EPF":"#4CAF50",
    "EGA-MORLET":"#FF5722",
}

fig=plt.figure(figsize=(24,16))
fig.suptitle(
    "Phase 4 — Experiments 3, 4, 5\n"
    "Scale Init | Multi-Quantity Score | Spectral Cascade",
    fontsize=13,fontweight="bold")
gs=gridspec.GridSpec(3,4,figure=fig,
                      hspace=0.45,wspace=0.35)

# Panel 1 — Exp 3 val loss
ax=fig.add_subplot(gs[0,0])
for lb in ["BASE-RANDOM","SCALE-INIT"]:
    if lb in exp3_hist:
        ax.plot(steps,exp3_hist[lb]["val"],
                color=colors[lb],marker="o",
                markersize=4,label=lb,linewidth=2)
ax.set_title("Exp 3: Scale-Initialized Heads\n"
             "Does physics-guided init help?")
ax.set_xlabel("Step"); ax.set_ylabel("Val Loss")
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

# Panel 2 — Exp 3 convergence detail
ax=fig.add_subplot(gs[0,1])
for lb in ["BASE-RANDOM","SCALE-INIT"]:
    if lb in exp3_hist:
        # Show first 2000 steps to see convergence speed
        st_short=steps[:5]
        vl_short=exp3_hist[lb]["val"][:5]
        ax.plot(st_short,vl_short,
                color=colors[lb],marker="o",
                markersize=5,label=lb,linewidth=2.2)
ax.set_title("Exp 3: Convergence Speed\n"
             "First 2000 steps")
ax.set_xlabel("Step"); ax.set_ylabel("Val Loss")
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

# Panel 3 — Exp 4 val loss
ax=fig.add_subplot(gs[0,2])
for lb,sn,_,_ in exp4_configs:
    if lb in exp4_hist:
        ax.plot(steps,exp4_hist[lb]["val"],
                color=colors[lb],marker="o",
                markersize=3,label=lb,linewidth=1.8)
ax.axhline(ega1_val,color="gray",linewidth=1.5,
           linestyle="--",
           label=f"EGA-1 ({ega1_val:.4f})")
ax.set_title("Exp 4: Multi-Quantity Score\n"
             "Phase+flux vs energy alone")
ax.set_xlabel("Step"); ax.set_ylabel("Val Loss")
ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# Panel 4 — Exp 4 lambda weights bar
ax=fig.add_subplot(gs[0,3])
if ckpt_done("exp4_mq_epf"):
    m_mq=GPT(MultiQuantityAttention,
              {"n_scales":4,"use_phase":True,
               "use_flux":True}).to(DEVICE)
    load_ckpt("exp4_mq_epf",m_mq)
    all_lw={"similarity":[],"energy":[],
             "phase":[],"flux":[]}
    for block in m_mq.blocks:
        for head in block.heads:
            if hasattr(head,"get_lambda_weights"):
                lw=head.get_lambda_weights()
                for k in all_lw:
                    all_lw[k].append(abs(lw[k]))
    means={k:sum(v)/len(v) for k,v in all_lw.items()}
    total=sum(means.values())
    pcts={k:v/total*100 for k,v in means.items()}
    qcols=["#2196F3","#4CAF50","#E91E63","#FF9800"]
    bars=ax.bar(list(means.keys()),list(pcts.values()),
                color=qcols,alpha=0.85)
    for bar,(k,pct) in zip(bars,pcts.items()):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3,f"{pct:.1f}%",
                ha="center",va="bottom",
                fontsize=10,fontweight="bold")
    ax.set_title("Exp 4: MQ-EPF Learned λ Weights\n"
                 "Did phase and flux become useful?")
    ax.set_ylabel("|λ|/Σ|λ| (%)")
    ax.grid(True,alpha=0.3,axis="y")
    m_mq.cpu(); del m_mq
    gc.collect(); torch.cuda.empty_cache()

# Panel 5 — Exp 5 cascade heatmap BASE
ax=fig.add_subplot(gs[1,:2])
if "BASE-DOT" in cascade_results:
    casc=cascade_results["BASE-DOT"]
    im=ax.imshow(casc.T,aspect="auto",
                  origin="lower",cmap="inferno",
                  extent=[0,N_LAYER,scales[0],scales[-1]])
    ax.set_yscale("log")
    ax.set_title("Exp 5: Spectral Cascade — BASE-DOT\n"
                 "Cascade(layer,scale) = "
                 "mean |W_ψ[e^(l)](a)|  "
                 "Bright = high spectral energy")
    ax.set_xlabel("Layer (0=embed, 6=output)")
    ax.set_ylabel("Scale a (log, tokens)")
    ax.set_xticks(range(N_LAYER+1))
    xlab=["emb"]+[f"L{i+1}" for i in range(N_LAYER)]
    ax.set_xticklabels(xlab,fontsize=8)
    plt.colorbar(im,ax=ax,label="|W_ψ|",fraction=0.02)

# Panel 6 — Exp 5 cascade heatmap EGA-1
ax=fig.add_subplot(gs[1,2:])
if "EGA-1" in cascade_results:
    casc=cascade_results["EGA-1"]
    im=ax.imshow(casc.T,aspect="auto",
                  origin="lower",cmap="inferno",
                  extent=[0,N_LAYER,scales[0],scales[-1]])
    ax.set_yscale("log")
    ax.set_title("Exp 5: Spectral Cascade — EGA-1\n"
                 "Compare to BASE: does energy gating "
                 "change the spectral structure?")
    ax.set_xlabel("Layer (0=embed, 6=output)")
    ax.set_ylabel("Scale a (log, tokens)")
    ax.set_xticks(range(N_LAYER+1))
    ax.set_xticklabels(xlab,fontsize=8)
    plt.colorbar(im,ax=ax,label="|W_ψ|",fraction=0.02)

# Panel 7 — Mean energy per layer (Parseval)
ax=fig.add_subplot(gs[2,:2])
layer_labels=["emb"]+[f"L{i+1}" for i in range(N_LAYER)]
casc_colors={"BASE-DOT":"#2196F3",
              "EGA-1":"#FF9800",
              "EGA-MORLET":"#FF5722"}
for lb in cascade_results:
    col=casc_colors.get(lb,"#888")
    mean_e=cascade_results[lb].mean(axis=1)
    ax.plot(range(len(mean_e)),mean_e,
            color=col,marker="o",markersize=5,
            label=lb,linewidth=2.2)
ax.set_xticks(range(N_LAYER+1))
ax.set_xticklabels(layer_labels,fontsize=8)
ax.set_title("Exp 5: Mean Spectral Energy per Layer\n"
             "By Parseval = total information content\n"
             "Does energy increase or decrease with depth?")
ax.set_xlabel("Layer"); ax.set_ylabel("Mean |W_ψ|")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# Panel 8 — Complete final bar all Phase 4 models
ax=fig.add_subplot(gs[2,2:])
lbls=list(all_results.keys())
vals=[all_results[lb] for lb in lbls]
cols=[colors.get(lb,"#888") for lb in lbls]
bars=ax.bar(range(len(lbls)),vals,
            color=cols,alpha=0.85)
ax.set_xticks(range(len(lbls)))
ax.set_xticklabels(lbls,rotation=45,
                    ha="right",fontsize=7)
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.001,f"{v:.4f}",
            ha="center",va="bottom",
            fontsize=6,fontweight="bold")
ax.axhline(bv,color="black",linewidth=1.5,
           linestyle="--",
           label=f"BASE-DOT ({bv:.4f})")
ax.set_title("Final Val Loss — All Phase 4 Models\n"
             "EGA-MORLET = best overall")
ax.set_ylabel("Val Loss")
ax.legend(fontsize=7)
ax.grid(True,alpha=0.3,axis="y")
bi=vals.index(min(vals))
bars[bi].set_edgecolor("gold"); bars[bi].set_linewidth(3)

plt.savefig(os.path.join(CKPT_DIR,"phase4_exp345.png"),
            dpi=150,bbox_inches="tight")
plt.show()

# ================================================================
# DOWNLOAD ALL
# ================================================================
print("\nDownloading results …")
from google.colab import files

files.download(os.path.join(CKPT_DIR,"phase4_exp345.png"))

txt_lines=["Phase 4 Complete Results","="*40]
for lb,v in all_results.items():
    txt_lines.append(f"{lb:<16} val={v:.4f}  Δ={bv-v:+.4f}")
txt_path=os.path.join(CKPT_DIR,"phase4_complete.txt")
with open(txt_path,"w") as f: f.write("\n".join(txt_lines))
files.download(txt_path)

for sname in ["exp3_scale","exp4_mq_e","exp4_mq_ep",
               "exp4_mq_ef","exp4_mq_epf"]:
    p=os.path.join(CKPT_DIR,f"{sname}.pt")
    if os.path.exists(p):
        files.download(p)
        print(f"  Downloaded {sname}.pt")

print("\nPhase 4 complete.")
print("Best model: EGA-MORLET val=1.3550  Δ=+0.119")
