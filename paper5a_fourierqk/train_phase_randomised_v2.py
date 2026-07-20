"""
train_phase_randomised_v2.py
=============================
Phase coherence decomposition via eval-time Theiler surrogates.

Based on reviewer's v2 design with two critical fixes:
  FIX A: log_freq properly excluded from weight decay
         (reviewer's code had comment but didn't implement it)
  FIX B: causal conv uses correct loop over frequencies
         (reviewer's grouped conv had channel/group mismatch)
  FIX C: n_freq=1 for clean single-frequency phase test
         (n_freq=4 is causal MultiFourier — different experiment)
  FIX D: phase_shift surrogate implemented (was no-op in v2)

CORRECT THEILER DESIGN (reviewer's key insight):
  1. Train a normal causal SpectralQK to convergence
  2. At EVAL TIME ONLY, evaluate with 5 surrogates:
     a. 'none'              — standard (baseline)
     b. 'phase_rand'        — Theiler: shared phi Q/K, DC/Nyq real
     c. 'phase_rand_indep'  — stress test: independent phi Q/K
     d. 'amp_only'          — |Qf|→irfft, keeps time index
     e. 'phase_shift'       — constant pi/4 phase offset (control)
  3. Compute pct_amplitude / pct_phase:
     pct_phase = (val_phrand - val_std) / (val_base - val_std) * 100

WHY EVAL-TIME ONLY:
  Train-time phase noise (original script) destroyed learning:
  the model could not learn phase structure because it was
  randomised every batch. Conclusion "phase essential" was
  trivially forced. Eval-time surrogates test a TRAINED model.

CAUSAL IMPLEMENTATION:
  Uses left-pad conv1d (no bilateral FFT leakage).
  Directly comparable to train_morlet_scales_v2/v3 results.
  Surrogates applied to Q/K BEFORE the causal conv.

Expected time: ~35-40 minutes (single model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, os, time, json, urllib.request

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)
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
    K=128        # causal kernel taps
    init_bin=4   # paragraph scale (period=64tok)
    sigma_f=2.0  # frequency bandwidth (bins)
cfg = Config()
T = cfg.block_size
F_BINS = T//2+1

def get_batch(split, shuffled=False):
    d  = train_data if split=='train' else (
         val_shuffled if shuffled else val_data)
    ix = torch.randint(len(d)-T,(cfg.batch_size,))
    x  = torch.stack([d[i  :i+T  ] for i in ix])
    y  = torch.stack([d[i+1:i+T+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Theiler surrogate perturbations ──────────────────────────────
def perturb(z, mode, shared_phi=None):
    """
    Apply surrogate perturbation to z: (B,H,T,hs).
    Returns perturbed z and the phi used (for shared Q/K).

    modes:
      'none'            — identity
      'phase_rand'      — shared phi, DC/Nyq real (Theiler)
      'phase_rand_indep'— independent phi for Q and K
      'amp_only'        — irfft(|Zf|, n=T) — keeps time index
      'phase_shift'     — constant pi/4 offset (control)
    """
    if mode == 'none':
        return z, None

    B,H,T_,hs = z.shape
    # Reshape to (N, T) for FFT
    z_flat = z.permute(0,1,3,2).reshape(-1, T_)   # (B*H*hs, T)
    Zf     = torch.fft.rfft(z_flat, dim=-1)
    amp    = Zf.abs()
    N, F   = Zf.shape

    if mode == 'phase_rand':
        if shared_phi is None:
            # One phi per (B,H,F) — shared across hs dimension
            phi = torch.rand(B*H, F, device=z.device) * 2*math.pi
            phi = phi.unsqueeze(1).expand(B*H, hs, F).reshape(N, F)
            # FIX: DC and Nyquist must be real
            phi[:, 0] = 0.0
            if T_ % 2 == 0: phi[:, -1] = 0.0
        else:
            phi = shared_phi
        Zf_p = amp * torch.exp(1j * phi)

    elif mode == 'phase_rand_indep':
        # Independent phi for Q and K — stress test (not Theiler)
        phi  = torch.rand(N, F, device=z.device) * 2*math.pi
        phi[:, 0] = 0.0
        if T_ % 2 == 0: phi[:, -1] = 0.0
        Zf_p = amp * torch.exp(1j * phi)

    elif mode == 'amp_only':
        # Amplitude envelope only — keeps time index
        # Use real amplitude spectrum, irfft gives time-localised signal
        Zf_p = amp.to(torch.complex64)

    elif mode == 'phase_shift':
        # FIX D: constant pi/4 phase offset (control for absolute phase)
        # If model is invariant to constant phase shifts, val ≈ standard
        phase_offset = math.pi / 4
        Zf_p = Zf * torch.exp(torch.tensor(1j * phase_offset,
                                            device=z.device))
        # DC and Nyquist: keep real
        Zf_p[:, 0]  = Zf[:, 0].real.to(torch.complex64)
        if T_ % 2 == 0:
            Zf_p[:, -1] = Zf[:, -1].real.to(torch.complex64)
    else:
        Zf_p = Zf

    z_p_flat = torch.fft.irfft(Zf_p, n=T_, dim=-1)
    z_p = z_p_flat.reshape(B, H, hs, T_).permute(0,1,3,2)
    phi_out = phi if mode in ('phase_rand','phase_rand_indep') else None
    return z_p, phi_out


# ── Causal SpectralQK (single frequency per head) ────────────────
class SpectralQK(nn.Module):
    """
    Causal Gaussian-windowed sinusoidal filter.
    Single learned frequency per head (n_freq=1).
    
    FIX A: log_freq in no-decay param group (see make_optimizer).
    FIX B: loop over n_freq (=1 here) — no grouped conv bug.
    FIX C: n_freq=1 for clean single-frequency Theiler test.
    FIX D: phase_shift surrogate implemented.
    """
    def __init__(self, n_embd, n_head, T=256,
                 init_bin=4, sigma_f=2.0, K=128):
        super().__init__()
        self.H=n_head; self.hs=n_embd//n_head; self.T=T; self.K=K

        self.q_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.k_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.value =nn.Linear(n_embd,n_embd,bias=False)
        self.proj  =nn.Linear(n_embd,n_embd,bias=False)
        self.drop  =nn.Dropout(0.1)
        self.register_buffer('mask',
            torch.tril(torch.ones(T,T)).view(1,1,T,T))
        self.register_buffer('t',
            torch.arange(K, dtype=torch.float32))

        # FIX A: log_freq — needs no weight decay (see make_optimizer)
        self.log_freq = nn.Parameter(
            torch.full((n_head,), math.log(float(init_bin))))

        # sigma in time domain: sigma_t = T / (2*pi*sigma_f_bins)
        self.sigma_t = T / (2*math.pi*sigma_f)

        # Perturbation mode (eval only)
        self.perturb_mode = 'none'

        period = T/init_bin
        print(f"  SpectralQK: causal, K={K}, init bin={init_bin} "
              f"(period={period:.0f}tok), sigma_t={self.sigma_t:.1f}")

    def _build_kernel(self, h):
        """Build L2-normalised causal kernel for head h."""
        f_bin = self.log_freq[h].exp().clamp(2.0, F_BINS-2)
        omega  = 2*math.pi * f_bin / self.T
        gauss  = torch.exp(-self.t**2 / (2*self.sigma_t**2))
        h_r    = gauss * torch.cos(omega*self.t)
        h_i    = gauss * torch.sin(omega*self.t)
        norm   = (h_r**2+h_i**2).sum().sqrt().clamp_min(1e-8)
        return (h_r/norm).flip(0).view(1,1,self.K), \
               (h_i/norm).flip(0).view(1,1,self.K)

    def _causal_conv(self, z_h, h_r, h_i):
        """z_h: (B,T,hs) → Wr, Wi each (B,T,hs)."""
        B,T,hs = z_h.shape
        z_ = z_h.permute(0,2,1).contiguous().view(B*hs,1,T)
        zp = F.pad(z_,(self.K-1,0))
        Wr = F.conv1d(zp,h_r).view(B,hs,T).permute(0,2,1)
        Wi = F.conv1d(zp,h_i).view(B,hs,T).permute(0,2,1)
        return Wr, Wi

    def forward(self, x):
        B,T,C = x.shape; H,hs = self.H, self.hs
        q = self.q_proj(x).view(B,T,H,hs).permute(0,2,1,3)
        k = self.k_proj(x).view(B,T,H,hs).permute(0,2,1,3)

        # Eval-time Theiler surrogates applied to Q and K
        # BEFORE causal conv (operating on raw Q/K vectors)
        if self.perturb_mode != 'none':
            mode = self.perturb_mode
            if mode == 'phase_rand':
                # Shared phi: same phase for Q and K (Theiler)
                q, phi = perturb(q, mode)
                k, _   = perturb(k, mode, shared_phi=phi)
            else:
                # Independent perturbation
                q, _ = perturb(q, mode)
                k, _ = perturb(k, mode)

        scores_sum = None
        for h in range(H):
            h_r, h_i = self._build_kernel(h)
            Wr_q, Wi_q = self._causal_conv(q[:,:,h,:], h_r, h_i)
            Wr_k, Wi_k = self._causal_conv(k[:,:,h,:], h_r, h_i)
            # Single /sqrt(hs) — FIX 1
            s_h = (torch.matmul(Wr_q, Wr_k.transpose(-2,-1)) +
                   torch.matmul(Wi_q, Wi_k.transpose(-2,-1))
                   ) / math.sqrt(hs)
            s_h = s_h.unsqueeze(1)
            scores_sum = s_h if scores_sum is None \
                         else torch.cat([scores_sum,s_h],dim=1)

        scores = scores_sum.masked_fill(
            self.mask[:,:,:T,:T]==0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))
        v    = self.value(x).view(B,T,H,hs).permute(0,2,1,3)
        return self.proj(
            (attn@v).permute(0,2,1,3).contiguous().view(B,T,C))

    def get_freq(self):
        return self.log_freq.exp().detach().cpu().tolist()


# ── MLP + Block + GPT ────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),
                               nn.Linear(4*d,d),nn.Dropout(0.1))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,d,h,T,init_bin=4,sigma_f=2.0,K=128):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.mlp=MLP(d)
        self.attn=SpectralQK(d,h,T,init_bin=init_bin,
                             sigma_f=sigma_f,K=K)
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__(); T=cfg.block_size
        self.tok=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        self.pos=nn.Embedding(T,cfg.n_embd)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList(
            [Block(cfg.n_embd,cfg.n_head,T,
                   cfg.init_bin,cfg.sigma_f,cfg.K)
             for _ in range(cfg.n_layer)])
        self.ln=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding):
                nn.init.normal_(m.weight,std=0.02)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"  Params: {n_all/1e6:.2f}M")
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
    def set_perturb(self, mode):
        for b in self.blocks:
            b.attn.perturb_mode = mode
    def get_freqs(self):
        return [b.attn.get_freq() for b in self.blocks]


# ── FIX A: make_optimizer with no-decay for log_freq ─────────────
def make_optimizer(model):
    """Separate param groups: log_freq and 1D params → no decay."""
    decay, nodecay = [], []
    for name,param in model.named_parameters():
        if 'log_freq' in name or param.ndim < 2:
            nodecay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{'params':decay,   'weight_decay':cfg.weight_decay},
         {'params':nodecay, 'weight_decay':0.0}],
        lr=cfg.lr, betas=(0.9,0.95))


# ── Evaluation with surrogates ────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, mode='none', shuffled=False):
    model.eval(); model.set_perturb(mode)
    out={}
    for split in ['train','val']:
        L=torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x,y=get_batch(split,shuffled=(split=='val' and shuffled))
            _,loss=model(x,y); L[k]=loss.item()
        out[split]=L.mean().item()
    model.set_perturb('none'); model.train()
    return out


# ── Learning rate schedule ────────────────────────────────────────
def get_lr(step):
    if step<cfg.warmup_steps: return cfg.lr*step/cfg.warmup_steps
    p=(step-cfg.warmup_steps)/(cfg.max_steps-cfg.warmup_steps)
    return cfg.lr*0.5*(1+math.cos(math.pi*p))


# ── Main ─────────────────────────────────────────────────────────
if __name__=='__main__':
    set_seed(cfg.seed)
    print(f"\n{'='*58}")
    print(f"  Phase coherence decomposition — causal SpectralQK")
    print(f"  init_bin={cfg.init_bin} (period={T/cfg.init_bin:.0f}tok)")
    print(f"{'='*58}")

    model=GPT(cfg).to(DEVICE)
    opt=make_optimizer(model)

    best_val=float('inf'); best_step=0; t0=time.time()
    history=[]

    for step in range(cfg.max_steps+1):
        lr=get_lr(step)
        for pg in opt.param_groups: pg['lr']=lr

        if step%cfg.eval_interval==0:
            # Standard eval (train mode OFF)
            std   = estimate_loss(model,'none',   shuffled=False)
            shuf  = estimate_loss(model,'none',   shuffled=True)
            phrs  = estimate_loss(model,'phase_rand',       shuffled=False)
            phin  = estimate_loss(model,'phase_rand_indep', shuffled=False)
            ampo  = estimate_loss(model,'amp_only',         shuffled=False)
            phsh  = estimate_loss(model,'phase_shift',      shuffled=False)

            elapsed=time.time()-t0
            freqs_l0=model.get_freqs()[0]
            f_mean = sum(freqs_l0)/len(freqs_l0)

            gap = shuf['val']-std['val']
            print(f"  step {step:5d} | "
                  f"val {std['val']:.4f} | "
                  f"shuf_gap {gap:+.5f} | "
                  f"phrand {phrs['val']:.4f} | "
                  f"phinp {phin['val']:.4f} | "
                  f"amp {ampo['val']:.4f} | "
                  f"phsh {phsh['val']:.4f} | "
                  f"f={f_mean:.1f}bin | {elapsed:.0f}s")

            if std['val']<best_val:
                best_val=std['val']; best_step=step

            history.append({
                'step':step,
                'val':round(std['val'],4),
                'gap':round(gap,4),
                'phase_rand':round(phrs['val'],4),
                'phase_rand_indep':round(phin['val'],4),
                'amp_only':round(ampo['val'],4),
                'phase_shift':round(phsh['val'],4),
                'freq_l0':round(f_mean,2),
            })
            with open('phase_rand_v2_results.json','w') as f:
                json.dump({'history':history},f,indent=2)

        if step==cfg.max_steps: break
        x,y=get_batch('train'); _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        opt.step()

    # ── Final Theiler decomposition ──────────────────────────────
    BASE_VAL = 1.4742   # BASE-DOT reference
    val_std  = best_val
    val_phr  = history[-1]['phase_rand']
    val_amp  = history[-1]['amp_only']
    val_phs  = history[-1]['phase_shift']

    total_gain  = BASE_VAL - val_std
    phase_loss  = val_phr - val_std  # gain lost when phase randomised
    amp_survive = BASE_VAL - val_amp  # gain that survives amplitude-only

    pct_phase   = phase_loss / total_gain * 100 if total_gain > 0 else 0
    pct_amp     = amp_survive / total_gain * 100 if total_gain > 0 else 0

    print(f"\n{'='*65}")
    print(f"  PHASE COHERENCE DECOMPOSITION")
    print(f"{'='*65}")
    print(f"  BASE-DOT:          val = {BASE_VAL:.4f}")
    print(f"  SpectralQK:        val = {val_std:.4f}  "
          f"(gain = {total_gain:+.4f})")
    print(f"  PhaseRand (shared):val = {val_phr:.4f}  "
          f"(phase loss = {phase_loss:+.4f})")
    print(f"  AmpOnly:           val = {val_amp:.4f}  "
          f"(amp survives = {amp_survive:+.4f})")
    print(f"  PhaseShift (pi/4): val = {val_phs:.4f}  "
          f"(shift sensitivity)")
    print(f"\n  Fraction of gain from PHASE coherence: {pct_phase:.0f}%")
    print(f"  Fraction of gain from AMPLITUDE only:  {pct_amp:.0f}%")
    print(f"\n  Interpretation:")
    if pct_phase > 50:
        print(f"  Phase coherence dominates (>50%)")
    elif pct_amp > 50:
        print(f"  Amplitude spectrum dominates (>50%)")
    else:
        print(f"  Both contribute: {pct_amp:.0f}% amplitude, "
              f"{pct_phase:.0f}% phase coherence")
    if abs(val_phs - val_std) < 0.05:
        print(f"  Phase shift control: invariant to constant phase "
              f"offset (as expected for relative Q-K scoring)")
    print(f"\n  Final learned freq (L0 mean): "
          f"{history[-1]['freq_l0']:.1f} bins "
          f"(period={T/history[-1]['freq_l0']:.0f}tok)")
    print(f"\n  Saved → phase_rand_v2_results.json")

    result = {
        'base_val':BASE_VAL,
        'val_standard':round(val_std,4),
        'val_phase_rand':round(val_phr,4),
        'val_amp_only':round(val_amp,4),
        'val_phase_shift':round(val_phs,4),
        'total_gain':round(total_gain,4),
        'pct_phase':round(pct_phase,1),
        'pct_amplitude':round(pct_amp,1),
        'best_step':best_step,
        'history':history,
    }
    with open('phase_rand_v2_results.json','w') as f:
        json.dump(result,f,indent=2)
