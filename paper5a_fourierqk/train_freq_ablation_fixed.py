"""
train_freq_ablation_fixed.py
============================
Fixed-frequency ablation — five bugs corrected.

FIXES vs train_freq_ablation.py:

FIX 1 (critical): Single attention scaling
  v1: scores = (...)/hs  then  /sqrt(hs)  [181× too soft]
  v2: scores = (...)/math.sqrt(hs)         [correct]

FIX 2 (documented): FFT bilateral leakage
  1j*(Qf*w) = Hilbert transform of filtered signal.
  All fixed bins leak equally → relative ordering valid.
  Absolute vals are bilateral-FFT numbers (not causal).
  Same regime as all Paper 5a/5b results.

FIX 3 (critical — NEW): Weight decay excluded from log_freq
  v1: AdamW decays log_freq toward 0 → bin=1 (period=256tok)
      The "migration toward low freq" result was confounded.
  v2: Separate param groups — log_freq has weight_decay=0.
      Any migration now reflects the loss, not the optimiser.

FIX 4 (bookkeeping): Consistent frequency/period logging
  v1: Header said Low=bin1, Mid=bin32, High=bin128
      Actual code: bin1, bin4, bin16, bin64
      Init bin was log(F_bins/4.0) ≈ bin32, reported as bin16
      Per-head report was layer 1 only
  v2: Header matches code exactly
      Init bin = 4 (period=64tok, paragraph — middle of sweep)
      All layers averaged in frequency report

FIX 5 (medium): L2 filter normalisation
  v1: w = w/w.sum()  [L1 norm — boosts narrow filters]
  v2: w = w/w.norm() [L2 norm — consistent energy across bins]

EXPERIMENTS (complete ladder: DC → low → mid → high → Nyquist):
  DC-QK:         bin=0   period=inf  (pure mean — null test)
  LowFreq-QK:    bin=1   period=256tok (global)
  LowFreq2-QK:   bin=4   period=64tok  (paragraph)
  MidFreq-QK:    bin=16  period=16tok  (phrase)
  HighFreq-QK:   bin=64  period=4tok   (character)
  Nyquist-QK:    bin=128 period=2tok   (Nyquist — null test)
  Fourier-QK:    learned (init bin=4, period=64tok)
  BASE-DOT:      standard attention reference

NOTE: DC and Nyquist connect to train_spectral_compression_fixed.py:
  DC-QK fixed val=2.017 (null), Nyquist fixed val=1.808 (null).
  Any fixed-bin val between these is the "spectral ladder".

Expected time: ~3.5 hours (8 runs × ~25min).
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
    url  = ('https://raw.githubusercontent.com/karpathy/char-rnn'
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
    vocab_size=VOCAB_SIZE; dropout=0.1; batch_size=64
    max_steps=5000; lr=3e-4; weight_decay=0.1; grad_clip=1.0
    warmup_steps=200; eval_interval=500; eval_iters=200; seed=42
cfg = Config()
T = cfg.block_size
F_BINS = T//2+1   # 129 bins for T=256

def get_batch(split, shuffled=False):
    d  = train_data if split=='train' else (
         val_shuffled if shuffled else val_data)
    ix = torch.randint(len(d)-T,(cfg.batch_size,))
    x  = torch.stack([d[i  :i+T  ] for i in ix])
    y  = torch.stack([d[i+1:i+T+1] for i in ix])
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

# ── Spectral QK (fixed) ───────────────────────────────────────────
class SpectralQKFixed(nn.Module):
    """
    Frequency-collapse attention at a fixed frequency bin.
    FIX 1: single /sqrt(hs).
    FIX 5: L2 filter normalisation.
    FIX 2: documented — bilateral FFT, same leakage as Paper 5a/5b.
    """
    def __init__(self, n_embd, n_head, T=256, fixed_bin=4,
                 sigma=2.0):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head; self.T=T
        self.fixed_bin=fixed_bin; self.sigma2=sigma**2

        self.q_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.k_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.value =nn.Linear(n_embd,n_embd,bias=False)
        self.proj  =nn.Linear(n_embd,n_embd,bias=False)
        self.drop  =nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))

        period = T/fixed_bin if fixed_bin > 0 else float('inf')
        if fixed_bin == 0:
            label = 'DC (mean) — null'
        elif fixed_bin == 1:
            label = 'global (256tok)'
        elif fixed_bin <= 4:
            label = 'paragraph'
        elif fixed_bin <= 16:
            label = 'phrase/word'
        elif fixed_bin <= 64:
            label = 'character'
        else:
            label = 'Nyquist — null'
        print(f"  Fixed bin={fixed_bin}: period={period:.1f}tok [{label}]")

    def _weights(self):
        bins = torch.arange(F_BINS, device=self.mask.device).float()
        w = torch.exp(-(bins - self.fixed_bin)**2 / (2*self.sigma2))
        # FIX 5: L2 normalisation
        return w / w.norm().clamp_min(1e-8)

    def forward(self, x):
        B,T,C=x.shape; H,hs=self.H,self.hs
        q=self.q_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        k=self.k_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        Qf=torch.fft.rfft(q, dim=2)
        Kf=torch.fft.rfft(k, dim=2)
        w  = self._weights().view(1,1,-1,1)
        # FIX 2: real-only for narrow fixed-bin filters
        # (1j branch gives Hilbert of filtered signal —
        #  for narrow sigma=2 bins, leakage is mild, same as Init4)
        Qr = torch.fft.irfft(Qf*w, n=T, dim=2)
        Kr = torch.fft.irfft(Kf*w, n=T, dim=2)
        # FIX 1: single /sqrt(hs)
        s  = torch.matmul(Qr, Kr.transpose(-2,-1)) / math.sqrt(hs)
        s  = s.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        attn=self.drop(F.softmax(s, dim=-1))
        v  = self.value(x).view(B,T,H,hs).permute(0,2,1,3)
        return self.proj(
            (attn@v).permute(0,2,1,3).contiguous().view(B,T,C))


# ── Spectral QK (learned) ─────────────────────────────────────────
class SpectralQKLearned(nn.Module):
    """
    Frequency-collapse with learned per-head centre frequency.

    FIX 1: single /sqrt(hs).
    FIX 3: log_freq excluded from weight decay (see train_model).
    FIX 4: init bin=4 (period=64tok, paragraph — middle of sweep).
            All layers averaged in frequency logging.
    FIX 5: L2 normalisation.
    """
    def __init__(self, n_embd, n_head, T=256,
                 init_bin=4, sigma=2.0):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head; self.T=T
        self.sigma2=sigma**2

        self.q_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.k_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.value =nn.Linear(n_embd,n_embd,bias=False)
        self.proj  =nn.Linear(n_embd,n_embd,bias=False)
        self.drop  =nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))

        # FIX 4: init at bin=4 (paragraph, middle of sweep)
        # FIX 3: this param gets weight_decay=0 in train_model
        self.log_freq = nn.Parameter(
            torch.full((n_head,), math.log(float(init_bin))))
        period = T / init_bin
        print(f"  Fourier-QK learned: init bin={init_bin} "
              f"period={period:.1f}tok [paragraph]")
        print(f"  (log_freq excluded from weight decay — FIX 3)")

    def _get_freq(self):
        return self.log_freq.exp()

    def _weights(self):
        f    = self._get_freq()           # (H,)
        bins = torch.arange(F_BINS, device=f.device).float()
        w    = torch.exp(-(bins.view(1,-1)-f.view(-1,1))**2
                         / (2*self.sigma2))
        # FIX 5: L2 normalisation
        return w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x):
        B,T,C=x.shape; H,hs=self.H,self.hs
        q=self.q_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        k=self.k_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        Qf=torch.fft.rfft(q, dim=2)
        Kf=torch.fft.rfft(k, dim=2)
        w  = self._weights()              # (H, F_BINS)
        we = w.unsqueeze(0).unsqueeze(-1) # (1,H,F,1)
        # FIX 1: single /sqrt(hs), real-only
        Qr = torch.fft.irfft(Qf*we, n=T, dim=2)
        Kr = torch.fft.irfft(Kf*we, n=T, dim=2)
        s  = torch.matmul(Qr, Kr.transpose(-2,-1)) / math.sqrt(hs)
        s  = s.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        attn=self.drop(F.softmax(s, dim=-1))
        v  = self.value(x).view(B,T,H,hs).permute(0,2,1,3)
        return self.proj(
            (attn@v).permute(0,2,1,3).contiguous().view(B,T,C))

    def get_freq_info(self):
        """FIX 4: average across ALL heads and ALL layers."""
        f = self._get_freq().detach().cpu()
        mean_bin = f.mean().item()
        period   = T / mean_bin
        per_head = [round(x,2) for x in f.tolist()]
        return {'mean_bin':round(mean_bin,2),
                'period_tok':round(period,2),
                'per_head':per_head}


# ── Standard attention ────────────────────────────────────────────
class StandardAttention(nn.Module):
    def __init__(self, n_embd, n_head, T):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head
        self.qkv =nn.Linear(n_embd,3*n_embd,bias=False)
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
    def __init__(self,d,h,T,attn):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.mlp=MLP(d); self.attn=attn
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self,cfg,attn_class,attn_kwargs=None,standard=False):
        super().__init__(); T=cfg.block_size
        self.tok=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos=nn.Embedding(T,cfg.n_embd)
        self.drop=nn.Dropout(cfg.dropout)
        attn_kwargs = attn_kwargs or {}
        self.blocks=nn.ModuleList(
            [Block(cfg.n_embd,cfg.n_head,T,
                   StandardAttention(cfg.n_embd,cfg.n_head,T)
                   if standard else
                   attn_class(cfg.n_embd,cfg.n_head,T,**attn_kwargs))
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

    def get_freq_info(self):
        """FIX 4: aggregate ALL layers for learned model."""
        all_bins = []
        for b in self.blocks:
            if hasattr(b.attn,'log_freq'):
                all_bins.append(b.attn._get_freq().detach().cpu())
        if not all_bins:
            return None
        mean_bin = torch.stack(all_bins).mean().item()
        return {'mean_bin':round(mean_bin,2),
                'period_tok':round(T/mean_bin,2),
                'per_layer':[round(f.mean().item(),2)
                              for f in all_bins]}


# ── Training ─────────────────────────────────────────────────────
def get_lr(step):
    if step<cfg.warmup_steps: return cfg.lr*step/cfg.warmup_steps
    p=(step-cfg.warmup_steps)/(cfg.max_steps-cfg.warmup_steps)
    return cfg.lr*0.5*(1+math.cos(math.pi*p))

def train_model(name, attn_class=None, attn_kwargs=None,
                standard=False, is_learned=False):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)

    print(f"\n{'='*58}\n  Training: {name}\n{'='*58}")
    model=GPT(cfg,attn_class,attn_kwargs,standard).to(DEVICE)

    # FIX 3: separate weight decay for log_freq parameters
    decay_params, nodecay_params = [], []
    for pname, param in model.named_parameters():
        if 'log_freq' in pname or param.ndim < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    opt=torch.optim.AdamW(
        [{'params': decay_params,   'weight_decay': cfg.weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=cfg.lr, betas=(0.9,0.95))

    best_val=float('inf'); t0=time.time(); freq_history=[]

    for step in range(cfg.max_steps+1):
        lr=get_lr(step)
        for pg in opt.param_groups: pg['lr']=lr
        if step%cfg.eval_interval==0:
            losses  =estimate_loss(model,shuffled=False)
            losses_s=estimate_loss(model,shuffled=True)
            gap     =losses_s['val']-losses['val']
            elapsed =time.time()-t0
            freq_str=''
            if is_learned:
                fi=model.get_freq_info()
                if fi:
                    freq_str=(f" | f={fi['mean_bin']:.1f} "
                              f"({fi['period_tok']:.1f}tok)")
                    freq_history.append({'step':step,**fi})
            print(f"  step {step:5d} | "
                  f"train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | "
                  f"shuf {losses_s['val']:.4f} | "
                  f"gap {gap:+.4f}"
                  f"{freq_str} | {elapsed:.0f}s")
            if losses['val']<best_val: best_val=losses['val']
        if step==cfg.max_steps: break
        x,y=get_batch('train'); _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()

    losses_s=estimate_loss(model,shuffled=True)
    gap=losses_s['val']-best_val; delta=1.4742-best_val
    print(f"\n  {name}: val={best_val:.4f} "
          f"Δ={delta:+.4f} gap={gap:+.4f}")

    if is_learned and freq_history:
        fi=freq_history[-1]
        init_period = T/4  # init bin=4
        print(f"  Freq migration: init bin=4 ({init_period:.0f}tok) "
              f"→ final bin={fi['mean_bin']:.1f} "
              f"({fi['period_tok']:.1f}tok)")
        print(f"  Per-layer bins: {fi['per_layer']}")
        direction = ('toward low freq (paragraph/global)'
                     if fi['mean_bin'] < 4 else
                     'stable at paragraph' if fi['mean_bin'] <= 6 else
                     'toward higher freq')
        print(f"  Direction: {direction}")

    return {'name':name,'val':round(best_val,4),
            'delta':round(delta,4),'gap':round(gap,4),
            'freq_history':freq_history}


# ── Main ─────────────────────────────────────────────────────────
if __name__=='__main__':
    # Complete DC → low → mid → high → Nyquist ladder
    # DC and Nyquist are null controls (from spectral_compression_fixed)
    EXPERIMENTS = [
        # (name,           class,             kwargs,            std,   learned)
        ('DC-QK',          SpectralQKFixed,  {'fixed_bin':0},   False, False),
        ('LowFreq-QK',     SpectralQKFixed,  {'fixed_bin':1},   False, False),
        ('LowFreq2-QK',    SpectralQKFixed,  {'fixed_bin':4},   False, False),
        ('MidFreq-QK',     SpectralQKFixed,  {'fixed_bin':16},  False, False),
        ('HighFreq-QK',    SpectralQKFixed,  {'fixed_bin':64},  False, False),
        ('Nyquist-QK',     SpectralQKFixed,  {'fixed_bin':128}, False, False),
        ('Fourier-QK',     SpectralQKLearned,{'init_bin':4},    False, True),
        ('BASE-DOT',       None,             {},                True,  False),
    ]

    results=[]
    for name,cls,kwargs,std,learned in EXPERIMENTS:
        r=train_model(name,cls,kwargs,std,learned)
        results.append(r)
        with open('freq_ablation_fixed_results.json','w') as f:
            json.dump(results,f,indent=2,default=str)

    print("\n"+"="*68)
    print("  FREQUENCY ABLATION FIXED — SPECTRAL LADDER")
    print("="*68)
    # Reference from spectral_compression_fixed
    refs = {
        'DC-QK':     2.017,
        'Nyquist-QK':1.808,
    }
    print(f"\n  {'Model':<16} {'Bin':>6} {'Period':>10} "
          f"{'Val':>8} {'Delta':>8} {'Gap':>8}")
    print("  "+"-"*60)
    bins   = [0,1,4,16,64,128,None,None]
    labels = ['DC (null)','global','paragraph','phrase',
              'char','Nyq (null)','learned','---']
    for r,(b,lbl) in zip(results,zip(bins,labels)):
        period = f'{T/b:.0f}tok' if b and b>0 else ('inf' if b==0 else '---')
        leaky  = ' LEAKY' if r['gap']<2 else ' CLEAN' if r['gap']>4 else ''
        print(f"  {r['name']:<16} {str(b):>6} {period:>10} "
              f"{r['val']:>8.4f} {r['delta']:>+8.4f} "
              f"{r['gap']:>+8.4f}{leaky}")
    print("="*68)
    print(f"""
  SPECTRAL LADDER INTERPRETATION:
  DC (bin=0):     → bandpass essential (null result)
  bin=1 (256tok): → global mean-free component
  bin=4 (64tok):  → paragraph scale (Init4 optimum)
  bin=16 (16tok): → phrase scale
  bin=64 (4tok):  → character scale
  Nyquist (128):  → alternating characters (null)
  Learned:        → where does gradient descent go?
  
  KEY QUESTION: does migration go toward paragraph scale
  (bin~4) when weight decay no longer biases toward bin=1?
  
  If learned val ≈ LowFreq2-QK (bin=4):
    Gradient descent finds the same optimum as Init4.
    The frequency hierarchy is robust to optimiser bias.
  
  If learned val < LowFreq2-QK:
    The model learns something beyond fixed paragraph scale.
    Multi-scale structure is being captured.
    """)
    print("  Saved → freq_ablation_fixed_results.json")
