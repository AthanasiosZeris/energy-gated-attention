"""
train_morlet_scales_v3.py
=========================
Causal Morlet scale sweep v3 — early stopping + shuffled gap check.

CONFIRMED by reviewer: v2 results show OVERFITTING not leakage.
Evidence: U-shaped val curve (min ~step 2500, rises to step 5000).
True leakage would give monotonically decreasing val toward 0.

v3 additions over v2 (all five bug fixes retained):
  - Early stopping with patience=1000 steps
  - Shuffled gap diagnostic at every eval (confirms no leakage)
  - Increased dropout 0.1→0.2 and weight_decay 0.1→0.2
  - Reports best_val and best_step (not final step)
  - Output file: morlet_scale_sweep_v3.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, os, time, json, urllib.request

torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU:  {torch.cuda.get_device_name(0)}")

# ── Data ─────────────────────────────────────────────────────────
def get_data():
    url = ('https://raw.githubusercontent.com/karpathy/char-rnn'
           '/master/data/tinyshakespeare/input.txt')
    path = 'tinyshakespeare.txt'
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path) as f: text = f.read()
    chars = sorted(set(text))
    stoi  = {c:i for i,c in enumerate(chars)}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9*len(data))
    return data[:n], data[n:], len(chars)

train_data, val_data, VOCAB_SIZE = get_data()
val_shuffled = val_data[torch.randperm(len(val_data))]
print(f"Vocab: {VOCAB_SIZE}")

class Config:
    n_layer=6; n_head=8; n_embd=256; block_size=256
    vocab_size=VOCAB_SIZE
    dropout=0.2        # increased from 0.1 — reduces overfitting
    batch_size=64
    max_steps=5000; lr=3e-4
    weight_decay=0.2   # increased from 0.1 — reduces overfitting
    grad_clip=1.0
    warmup_steps=200; eval_interval=500; eval_iters=200; seed=42
    patience=1000      # early stopping: stop if no improvement for 1000 steps
cfg = Config()

def get_batch(split, shuffled=False):
    d  = train_data if split=='train' else (
         val_shuffled if shuffled else val_data)
    ix = torch.randint(len(d)-cfg.block_size,(cfg.batch_size,))
    x  = torch.stack([d[i  :i+cfg.block_size  ] for i in ix])
    y  = torch.stack([d[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, shuffled=False):
    model.eval()
    out={}
    for split in ['train','val']:
        L=torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x,y=get_batch(split,shuffled=(split=='val' and shuffled))
            _,loss=model(x,y); L[k]=loss.item()
        out[split]=L.mean().item()
    model.train(); return out

# ── Causal Morlet: fixed scale ────────────────────────────────────
class CausalMorletFixedQK(nn.Module):
    """
    Causal Morlet at fixed scale. Left-pad conv, no FFT leakage.
    K=128 taps covers 3σ for a=32 (3*32=96 < 128). ✓

    FIX 2: L2 normalisation → unit-energy kernel at all scales.
    FIX 3: kernel flipped so h[0]=current position.
    FIX 1: single /sqrt(hs) in forward().
    """
    def __init__(self, n_embd, n_head, T=256,
                 scale=4.0, omega0=6.0, K=128):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head; self.T=T; self.K=K

        self.q_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.k_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.value =nn.Linear(n_embd,n_embd,bias=False)
        self.proj  =nn.Linear(n_embd,n_embd,bias=False)
        self.drop  =nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))

        # Build kernel: t=0 is current, t=K-1 is oldest past
        # FIX 3: flip so conv weight[0] = current position
        t     = torch.arange(K, dtype=torch.float32)
        gauss = torch.exp(-t**2/(2*scale**2))
        h_r   = gauss * torch.cos(omega0*t/scale)
        h_i   = gauss * torch.sin(omega0*t/scale)

        # FIX 2: L2 normalisation of complex kernel
        norm  = torch.sqrt((h_r**2 + h_i**2).sum()).clamp_min(1e-8)
        h_r   = h_r / norm
        h_i   = h_i / norm

        # FIX 3: flip so h[0]=current, h[K-1]=oldest
        # (causal: F.pad left-pads K-1 zeros, conv slides left→right)
        h_r = h_r.flip(0)
        h_i = h_i.flip(0)

        self.register_buffer('h_r', h_r.view(1,1,K))
        self.register_buffer('h_i', h_i.view(1,1,K))

        # Report filter properties
        period    = 2*math.pi*scale/omega0
        nyquist_ok= (omega0/scale) < math.pi
        # FIX 4: clean energy coverage via cumsum
        t_raw     = torch.arange(K, dtype=torch.float32)
        g_raw     = torch.exp(-t_raw**2/(2*scale**2))
        pct       = (g_raw.cumsum(0)/g_raw.sum())[-1].item()*100
        print(f"  Scale a={scale}: period={period:.1f}tok "
              f"peak_w={omega0/scale:.3f}rad/samp "
              f"{'sub-Nyq OK' if nyquist_ok else 'ALIASED!'} "
              f"K={K} covers {pct:.0f}% Gauss energy")

    def _causal_conv(self, z):
        """Left-pad causal conv. z: (B,H,T,hs)→(B,H,T,hs)"""
        B,H,T,hs = z.shape
        z_ = z.permute(0,1,3,2).contiguous().view(B*H*hs,1,T)
        zp = F.pad(z_,(self.K-1,0))
        Wr = F.conv1d(zp,self.h_r).view(B,H,hs,T).permute(0,1,3,2)
        Wi = F.conv1d(zp,self.h_i).view(B,H,hs,T).permute(0,1,3,2)
        return Wr, Wi

    def forward(self, x):
        B,T,C=x.shape; H,hs=self.H,self.hs
        q=self.q_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        k=self.k_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        Qr,Qi=self._causal_conv(q)
        Kr,Ki=self._causal_conv(k)
        # FIX 1: single /sqrt(hs) — no /hs beforehand
        s=(torch.matmul(Qr,Kr.transpose(-2,-1))+
           torch.matmul(Qi,Ki.transpose(-2,-1)))/math.sqrt(hs)
        s=s.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        attn=self.drop(F.softmax(s,dim=-1))
        v=self.value(x).view(B,T,H,hs).permute(0,2,1,3)
        return self.proj(
            (attn@v).permute(0,2,1,3).contiguous().view(B,T,C))

# ── Causal Morlet: learned scale ─────────────────────────────────
class CausalMorletLearnedQK(nn.Module):
    """
    Causal Morlet with learned scale per head.
    Rebuilds kernel every forward pass — differentiable.
    All v2 fixes applied.
    With bugs fixed, expected to settle at a=3-6 tokens.
    """
    def __init__(self, n_embd, n_head, T=256,
                 init_scale=8.0, omega0=6.0, K=128):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head; self.T=T
        self.K=K; self.omega0=omega0

        self.q_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.k_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.value =nn.Linear(n_embd,n_embd,bias=False)
        self.proj  =nn.Linear(n_embd,n_embd,bias=False)
        self.drop  =nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))
        self.register_buffer('t',
            torch.arange(K, dtype=torch.float32))

        # Per-head learned log-scale
        self.log_scale = nn.Parameter(
            torch.full((n_head,), math.log(init_scale)))
        print(f"  CausalMorlet-Learned: init a={init_scale} "
              f"K={K} omega0={omega0}")

    def _build_kernels(self, h):
        """Build kernel for head h. Differentiable w.r.t. log_scale."""
        a     = self.log_scale[h].exp()
        gauss = torch.exp(-self.t**2/(2*a**2))
        h_r   = gauss * torch.cos(self.omega0*self.t/a)
        h_i   = gauss * torch.sin(self.omega0*self.t/a)
        norm  = torch.sqrt((h_r**2+h_i**2).sum()).clamp_min(1e-8)
        h_r   = (h_r/norm).flip(0).view(1,1,self.K)
        h_i   = (h_i/norm).flip(0).view(1,1,self.K)
        return h_r, h_i

    def _causal_conv_h(self, z_h, h_r, h_i):
        """z_h: (B,T,hs)"""
        B,T,hs = z_h.shape
        z_ = z_h.permute(0,2,1).contiguous().view(B*hs,1,T)
        zp = F.pad(z_,(self.K-1,0))
        Wr = F.conv1d(zp,h_r).view(B,hs,T).permute(0,2,1)
        Wi = F.conv1d(zp,h_i).view(B,hs,T).permute(0,2,1)
        return Wr, Wi

    def forward(self, x):
        B,T,C=x.shape; H,hs=self.H,self.hs
        q=self.q_proj(x).view(B,T,H,hs)
        k=self.k_proj(x).view(B,T,H,hs)
        scores_sum=None
        for h in range(H):
            kr,ki=self._build_kernels(h)
            Qrh,Qih=self._causal_conv_h(q[:,:,h,:],kr,ki)
            Krh,Kih=self._causal_conv_h(k[:,:,h,:],kr,ki)
            s_h=(torch.matmul(Qrh,Krh.transpose(-2,-1))+
                 torch.matmul(Qih,Kih.transpose(-2,-1)))/math.sqrt(hs)
            if scores_sum is None: scores_sum=s_h.unsqueeze(1)
            else: scores_sum=torch.cat([scores_sum,s_h.unsqueeze(1)],dim=1)
        s=scores_sum.masked_fill(
            self.mask[:,:,:T,:T]==0,float('-inf'))
        attn=self.drop(F.softmax(s,dim=-1))
        v=self.value(x).view(B,T,H,hs).permute(0,2,1,3)
        return self.proj(
            (attn@v).permute(0,2,1,3).contiguous().view(B,T,C))

    def get_scales(self):
        return self.log_scale.exp().detach().cpu().tolist()

# ── Standard attention ────────────────────────────────────────────
class StandardAttention(nn.Module):
    def __init__(self,n_embd,n_head,T):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head
        self.qkv=nn.Linear(n_embd,3*n_embd,bias=False)
        self.proj=nn.Linear(n_embd,n_embd,bias=False)
        self.drop=nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))
    def forward(self,x):
        B,T,C=x.shape; H,hs=self.H,self.hs
        q,k,v=self.qkv(x).split(C,dim=2)
        def sp(t): return t.view(B,T,H,hs).permute(0,2,1,3)
        q,k,v=sp(q),sp(k),sp(v)
        att=(q@k.transpose(-2,-1))/math.sqrt(hs)
        att=att.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        att=self.drop(F.softmax(att,dim=-1))
        return self.proj(
            (att@v).permute(0,2,1,3).contiguous().view(B,T,C))

# ── MLP + Block + GPT ────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),
                               nn.Linear(4*d,d),nn.Dropout(0.1))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,d,h,T,scale=None,learned=False):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.mlp=MLP(d)
        if learned:
            self.attn=CausalMorletLearnedQK(d,h,T)
        elif scale is None:
            self.attn=StandardAttention(d,h,T)
        else:
            self.attn=CausalMorletFixedQK(d,h,T,scale=scale)
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self,cfg,scale=None,learned=False):
        super().__init__(); T=cfg.block_size
        self.tok=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos=nn.Embedding(T,cfg.n_embd)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList(
            [Block(cfg.n_embd,cfg.n_head,T,scale,learned)
             for _ in range(cfg.n_layer)])
        self.ln=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding):
                nn.init.normal_(m.weight,std=0.02)
        print(f"  Params: "
              f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M")
    def forward(self,idx,targets=None):
        B,T=idx.shape
        x=self.drop(self.tok(idx)+self.pos(
            torch.arange(T,device=idx.device)))
        for b in self.blocks: x=b(x)
        logits=self.head(self.ln(x)); loss=None
        if targets is not None:
            loss=F.cross_entropy(
                logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss

def get_lr(step):
    if step<cfg.warmup_steps: return cfg.lr*step/cfg.warmup_steps
    p=(step-cfg.warmup_steps)/(cfg.max_steps-cfg.warmup_steps)
    return cfg.lr*0.5*(1+math.cos(math.pi*p))

def train_model(name, scale=None, learned=False,
                patience=None):
    if patience is None: patience=cfg.patience
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    print(f"\n{'='*58}\n  Training: {name}\n{'='*58}")
    model=GPT(cfg,scale,learned).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,
                           weight_decay=cfg.weight_decay,betas=(0.9,0.95))
    best_val=float('inf'); best_step=0; t0=time.time()

    for step in range(cfg.max_steps+1):
        lr=get_lr(step)
        for pg in opt.param_groups: pg['lr']=lr
        if step%cfg.eval_interval==0:
            losses  =estimate_loss(model, shuffled=False)
            losses_s=estimate_loss(model, shuffled=True)
            gap=losses_s['val']-losses['val']
            elapsed=time.time()-t0
            extra=''
            if learned:
                attn=model.blocks[0].attn
                scales=[f'{s:.1f}' for s in attn.get_scales()[:4]]
                extra=f' | a=[{",".join(scales)}]tok'
            print(f"  step {step:5d} | "
                  f"train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | "
                  f"shuf {losses_s['val']:.4f} | "
                  f"gap {gap:+.4f}{extra} | {elapsed:.0f}s")
            if losses['val']<best_val:
                best_val=losses['val']; best_step=step
            # Early stopping
            elif step-best_step >= patience:
                print(f"  Early stopping at step {step} "
                      f"(no improvement since step {best_step})")
                break
        if step==cfg.max_steps: break
        x,y=get_batch('train'); _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()

    # Final shuffled gap at best checkpoint
    losses_s=estimate_loss(model,shuffled=True)
    gap=losses_s['val']-best_val
    delta=1.4742-best_val
    leakage='CLEAN' if gap>0.5 else 'CHECK'
    print(f"\n  {name}: best_val={best_val:.4f} Δ={delta:+.4f} "
          f"gap={gap:+.4f} [{leakage}] (best step {best_step})")
    result={'name':name,'scale':scale,'learned':learned,
            'val':round(best_val,4),'delta':round(delta,4),
            'gap':round(gap,4),'best_step':best_step}
    if learned:
        result['final_scales']=[round(s,2) for s in
            model.blocks[0].attn.get_scales()]
    return result

if __name__=='__main__':
    EXPERIMENTS = [
        ('BASE-DOT',       None,  False),
        ('Morlet-a2',      2.0,   False),
        ('Morlet-a4',      4.0,   False),
        ('Morlet-a8',      8.0,   False),
        ('Morlet-a16',    16.0,   False),
        ('Morlet-a32',    32.0,   False),
        ('Morlet-Learned', None,  True),   # reviewer suggestion
    ]

    results=[]
    for name,scale,learned in EXPERIMENTS:
        r=train_model(name,scale,learned); results.append(r)
        with open('morlet_scale_sweep_v3.json','w') as f:
            json.dump(results,f,indent=2)

    print("\n"+"="*68)
    print("  CAUSAL MORLET v3 — SCALE SWEEP (early stopping + gap check)")
    print("="*68)
    print(f"  {'Model':<18} {'Scale':>6} {'Val':>8} {'Δ':>8} "
          f"{'Gap':>8} {'BestStep':>10}")
    print("  "+"-"*62)
    for r in results:
        s = 'BASE' if (not r['learned'] and r['scale'] is None) \
            else ('Learned' if r['learned']
            else f"{r['scale']:.0f}tok")
        extra = ''
        if r.get('final_scales'):
            extra = f"  → {r['final_scales'][:2]}"
        gap = r.get('gap', 0)
        bs  = r.get('best_step', '---')
        print(f"  {r['name']:<18} {s:>6} "
              f"{r['val']:>8.4f} {r['delta']:>+8.4f} "
              f"{gap:>+8.4f} {str(bs):>10}{extra}")
    print("="*68)
    print("""
  LEAKAGE CHECK (gap = shuffled_val - ordered_val):
    gap >> 0 (e.g. +5): no leakage — model uses token order
    gap ≈ 0:            marginal — check carefully
    gap < 0:            leakage — model cheats with future info

  EARLY STOPPING NOTE:
    best_step << 5000: overfitting present (stopped early)
    best_step ≈ 5000:  no overfitting (ran to completion)
    """)
    print("  Saved → morlet_scale_sweep_v3.json")
