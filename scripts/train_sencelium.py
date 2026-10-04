from __future__ import annotations
"""
Sencelium -- training script.

Trains Sencelium, a non-Transformer language model architecture built from
gated-delta-rule blocks with a content-blind "nucleus" channel, a memory
pool that keeps spawning, merging and reorganizing itself during training
and at inference, a set of live self-monitoring signals the model uses to
track its own confidence and effort, and an ESN (echo state network)
residual path.

Scale is set by a config file, not hardcoded -- see configs/.

Data:
  Train -- FineWebEdu (HuggingFaceFW/fineweb-edu, streaming)
  Val   -- WikiText-103 test (Salesforce/wikitext, wikitext-103-raw-v1)

Usage:
  python train_sencelium.py --config configs/30M.yaml
  python train_sencelium.py --config configs/65M.yaml --lr=2e-4

The config file overrides Config's defaults; any trailing --key=value
arguments override the config file. `tag` (used to name the
checkpoints/logs directory) defaults to "sencelium_<config filename stem>"
unless set explicitly in the config file or on the command line.

Requires: torch, pytorch-lightning, transformers, datasets, pyyaml.
See the project README for environment setup and the full architecture
write-up.
"""

import argparse
import math
import re
import sys
import time
from ast import literal_eval
from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets import load_dataset
from transformers import GPT2TokenizerFast


SIGNAL_NAMES = [
    # LOCAL (6) — per-block, per-token, live
    "phi1_entropy", "phi2_norm", "phi3_alpha_mean", "phi4_beta_std",
    "phi5_beta_mean", "delta_cryst",
    # GLOBAL (6) — slow EMA of colony_pheromone (long-term hidden-state memory)
    "glob_phi1_slow", "glob_phi2_slow", "glob_phi3_slow",
    "glob_phi4_slow", "glob_phi5_slow", "glob_dcryst_slow",
    # COLONY (7) — inter-layer pheromone, causal per-position live update
    "col_phi1_mean", "col_phi2_mean", "col_phi3_mean", "col_phi4_mean",
    "col_phi5_mean", "col_dcryst_mean", "col_diversity",
]
N_SIG    = len(SIGNAL_NAMES)   # 19
N_LOCAL  = 6
N_GLOBAL = 6
N_COLONY = 7


@dataclass
class Config:
    vocab_size:      int   = 50257

    # Backbone
    d_model:         int   = 256
    n_blocks:        int   = 15
    n_heads:         int   = 4
    d_k:             int   = 32
    d_v:             int   = 64
    ffn_hidden:      int   = 1024
    chunk_size:      int   = 128

    # Sencelium feature flags
    use_router_distill: bool  = True
    n_slots_esn:         int  = 8
    lambda_router:       float = 0.02
    router_lr_mult:      float = 10.0

    # Nucleus side-channel
    nucleus_frac:    float = 0.5
    n_heads_n:       int   = 0   # derived
    d_hn:            int   = 0   # derived
    use_checkpoint:  bool  = False

    signal_expand:   int   = 1

    # EmotionHead
    n_emotion_heads_mult: int   = 2
    n_emotion:            int   = 4   # derived

    # Memory pool
    mem_max_slots:        int   = 100_000
    mem_d_k_mult:         int   = 1
    voice_rounds:         int   = 3
    emo_tau:              float = 1.0

    mem_novelty_percentile:    float = 0.1
    mem_novelty_history_size:  int   = 2048
    mem_novelty_warmup:        int   = 32_000
    mem_max_lr:                float = 0.5
    mem_usage_decay:           float = 0.999

    mem_cat_select_margin:   float = 1.0
    cat_load_bias_k:         float = 0.35
    cat_load_bias_clamp:     float = 3.0
    cat_transform_margin_k:  float = 1.5

    mem_underdog_bonus: float = 1.0

    mem_merge_idle_percentile:    float = 0.9
    mem_merge_redundancy_margin:  float = 1.0
    mem_merge_nn_sim_warmup:      int   = 100
    mem_merge_scan_k:             int   = 5

    mem_spawn_bias_margin_seed: float = 1.0
    mem_spawn_bias_lr:          float = 0.01
    mem_spawn_bias_clamp:       float = 5.0

    mem_plasticity_percentile: float = 0.9

    mem_max_lr_infer:           float = 0.08
    mem_write_mass_floor_infer: float = 0.05

    mem_occupied_u_threshold:  float = 1.0
    mem_usage_mass_fraction:   float = 0.9

    mem_neardup_cos_threshold: float = 0.9
    mem_neardup_max_slots:     int   = 4096

    carry_state: bool = False

    mem_trunk_frac: float = 0.25

    lambda_div:  float = 0.02
    lambda_emo:  float = 0.02

    max_steps:    int   = 36_621
    warmup_steps: int   = 1_000
    lr:           float = 1e-4
    grad_clip:    float = 0.5
    log_every:    int   = 500
    val_interval: int   = 9_000

    seq_len:      int   = 512
    batch_size:   int   = 32

    ema_alpha:          float = 0.99
    ema_norm:           float = 0.99
    sig_norm_warmup:    int   = 100
    colony_slow_alpha:  float = 0.999
    colony_live_alpha:  float = 0.9

    tag: str = "sencelium"   # checkpoint/log directory name; normally set from the config filename

    def __post_init__(self) -> None:
        """Derives n_emotion/n_heads_n/d_hn from n_heads/d_model/nucleus_frac."""
        self.n_emotion = self.n_heads * self.n_emotion_heads_mult
        self.n_heads_n = self.n_heads
        self.d_hn      = max(1, round(self.d_model * self.nucleus_frac / self.n_heads_n))
        assert self.n_heads * self.d_v == self.d_model, (
            f"n_heads*d_v ({self.n_heads}*{self.d_v}={self.n_heads*self.d_v}) must "
            f"equal d_model ({self.d_model}) -- the DELTA path is meant to be "
            f"full-width. Set n_heads/d_v explicitly for this scale.")
        assert self.ffn_hidden == 4 * self.d_model, (
            f"ffn_hidden ({self.ffn_hidden}) must equal 4*d_model "
            f"({4*self.d_model}) -- the fixed FFN expansion ratio this file "
            f"uses at every scale. Set ffn_hidden explicitly for this scale.")
        assert self.seq_len % self.chunk_size == 0, (
            f"seq_len ({self.seq_len}) must be divisible by chunk_size "
            f"({self.chunk_size}) -- required by the chunked delta/nucleus scans.")


def load_config(argv: List[str]) -> Config:
    """Builds a Config from --config <path.yaml> plus any trailing
    --key=value overrides (highest priority, same --key=value syntax
    nanoGPT's own configurator uses). `tag` defaults to
    "sencelium_<config filename stem>" unless set explicitly in the
    config file or on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str,
                         help="Path to a YAML file overriding Config's defaults.")
    args, rest = parser.parse_known_args(argv)

    config_path = Path(args.config)
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    overrides.setdefault("tag", f"sencelium_{config_path.stem}")

    valid_fields = {f.name for f in fields(Config)}
    for arg in rest:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"Unrecognized argument {arg!r}, expected --key=value")
        key, raw_value = arg[2:].split("=", 1)
        try:
            value = literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value   # plain string (e.g. --tag=my_run)
        overrides[key] = value

    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown Config field(s) {unknown} (from {config_path} or the command line)")

    return replace(Config(), **overrides)


# ─── Gated Delta Rule kernel ──────────────────────────────────────────────

def _gated_delta_chunked(q, k, v, log_a, beta, S0, C: int = 128):
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"

    eyeC  = torch.eye(C, device=q.device, dtype=q.dtype)
    smask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
    neg_inf = float("-inf")

    S = S0
    out = []
    for i in range(T // C):
        sl = slice(i * C, (i + 1) * C)
        qc, kc, vc = q[:, :, sl], k[:, :, sl], v[:, :, sl]
        lac, bc    = log_a[:, :, sl], beta[:, :, sl]

        g  = lac.cumsum(-1)
        gp = g - lac

        kk = torch.einsum("bhtd,bhsd->bhts", kc, kc)
        D  = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~smask, neg_inf)
        A  = bc.unsqueeze(-1) * torch.exp(D) * kk

        S0k = torch.einsum("bhvd,bhtd->bhtv", S, kc)
        rhs = bc.unsqueeze(-1) * (vc - torch.exp(g).unsqueeze(-1) * S0k)

        W = torch.linalg.solve_triangular(eyeC + A, rhs, upper=False, unitriangular=True)

        qk        = torch.einsum("bhtd,bhsd->bhts", qc, kc)
        Dq        = (gp.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~smask, neg_inf)
        ctx_intra = (torch.exp(Dq) * qk) @ W
        ctx_inter = torch.exp(gp).unsqueeze(-1) * torch.einsum("bhvd,bhtd->bhtv", S, qc)
        out.append(ctx_inter + ctx_intra)

        g_last = g[:, :, -1]
        Wd = W * torch.exp(g_last.unsqueeze(-1) - g).unsqueeze(-1)
        S  = torch.exp(g_last)[..., None, None] * S \
             + torch.einsum("bhtv,bhtd->bhvd", Wd, kc)

    return torch.cat(out, dim=2), S


# ─── NUCLEUS kernel — content-blind decay-EMA, chunked ───────────────────────
#   S_t = a_t * S_{t-1} + (1 - a_t) * v_t     (a_t: scalar per head per token)
# No q/k — pure temporal integration of the value stream. Verified numerically
# against a brute-force sequential loop (valence tiny series, max err ~4e-16 f64).

def _nucleus_chunked(v, log_a, S0, C: int = 128):
    B, H, T, Dh = v.shape
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"

    causal = torch.tril(torch.ones(C, C, device=v.device, dtype=torch.bool))
    S = S0
    out = []
    for i in range(T // C):
        sl  = slice(i * C, (i + 1) * C)
        vc  = v[:, :, sl]
        lac = log_a[:, :, sl]
        g   = lac.cumsum(-1)
        beta = 1 - torch.exp(lac)

        D = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~causal, float("-inf"))
        W = torch.exp(D)

        intra = torch.einsum("bhts,bhsd->bhtd", W, beta.unsqueeze(-1) * vc)
        carry = torch.exp(g).unsqueeze(-1) * S.unsqueeze(2)
        St = carry + intra
        out.append(St)
        S = St[:, :, -1]

    return torch.cat(out, dim=2), S


def _decay_scan_chunked(b: torch.Tensor, log_a: torch.Tensor, S0: torch.Tensor,
                        C: int = 128) -> torch.Tensor:
    """General first-order linear recurrence with a scalar decay per
    (batch, head/slot, time) and a vector input:
        S_t = a_t * S_{t-1} + b_t,     S_{-1} = S0,   a_t = exp(log_a_t)
    b: [B,H,T,D], log_a: [B,H,T], S0: [B,H,D] -> all states [B,H,T,D].
    Same chunked construction as `_nucleus_chunked` (whose recurrence is the
    special case b_t = (1-a_t) v_t), kept separate so that kernel stays
    untouched. T need not be a multiple of C: the tail is padded with
    (log_a=0, b=0), i.e. "carry the state unchanged", and sliced off.
    Used by ScanESNContext."""
    B, H, T, D = b.shape
    pad = (-T) % C
    if pad:
        b = F.pad(b, (0, 0, 0, pad))
        log_a = F.pad(log_a, (0, pad))
    Tp = T + pad
    causal = torch.tril(torch.ones(C, C, device=b.device, dtype=torch.bool))
    S = S0
    out = []
    for i in range(Tp // C):
        sl = slice(i * C, (i + 1) * C)
        g = log_a[:, :, sl].cumsum(-1)                                     # [B,H,C]
        W = torch.exp((g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~causal, float("-inf")))
        St = torch.exp(g).unsqueeze(-1) * S.unsqueeze(2) \
             + torch.einsum("bhts,bhsd->bhtd", W, b[:, :, sl])
        out.append(St)
        S = St[:, :, -1]
    return torch.cat(out, dim=2)[:, :, :T]


# ─── MetaController ────────────────────────────────────────────────────────

class MetaController(nn.Module):
    """Reads the model's self-monitoring signals and writes a learned
    correction into the hidden state."""

    def __init__(self, d_model: int, signal_expand: int = 1) -> None:
        super().__init__()
        _L, _G, _C   = (d_model // 16) * signal_expand, (d_model * 3 // 64) * signal_expand, (d_model // 16) * signal_expand
        _MERGE_HIDDEN = (d_model // 2) * signal_expand
        self.local_proj  = nn.Sequential(nn.Linear(N_LOCAL,  _L, bias=True), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(N_GLOBAL, _G, bias=True), nn.GELU())
        self.colony_proj = nn.Sequential(nn.Linear(N_COLONY, _C, bias=True), nn.GELU())
        self.merge = nn.Sequential(
            nn.Linear(_L + _G + _C, _MERGE_HIDDEN, bias=True),
            nn.GELU(),
            nn.Linear(_MERGE_HIDDEN, d_model, bias=False),
        )
        nn.init.zeros_(self.merge[2].weight)

        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        self.register_buffer('sig_mean', torch.zeros(N_SIG))
        self.register_buffer('sig_std',  torch.ones(N_SIG))
        _bcast = [False] * N_LOCAL + [True] * N_GLOBAL + [True] * N_COLONY
        self.register_buffer('is_broadcast', torch.tensor(_bcast))
        self.register_buffer('sig_temporal_var', torch.ones(N_SIG))

        self._last_signals:     torch.Tensor | None = None
        self._last_signals_std: torch.Tensor | None = None
        self._last_gate_mag:    float = 0.0
        self._last_corr_signed: torch.Tensor | None = None
        self._diag: bool = True

    def forward(self, h: torch.Tensor, signals: torch.Tensor,
                layer_sens: torch.Tensor) -> torch.Tensor:
        s_norm = ((signals - self.sig_mean) / (self.sig_std + 1e-6)).clamp(-5.0, 5.0)
        s_norm = (s_norm + layer_sens).clamp(-5.0, 5.0)

        s_local  = s_norm[..., :N_LOCAL]
        s_global = s_norm[..., N_LOCAL:N_LOCAL + N_GLOBAL]
        s_colony = s_norm[..., N_LOCAL + N_GLOBAL:]

        m = self.merge(torch.cat([
            self.local_proj(s_local),
            self.global_proj(s_global),
            self.colony_proj(s_colony),
        ], dim=-1))

        g    = torch.sigmoid(self.gate_proj(h))
        corr = g * m

        sig_f = signals.detach().float()
        self._last_signals     = sig_f.mean(dim=(0, 1))
        self._last_signals_std = sig_f.std(dim=(0, 1))
        self._last_gate_mag    = corr.detach().float().abs().mean()
        if self._diag:
            self._last_corr_signed = corr.detach().float().mean(dim=-1)

        return h + corr


class EmotionHead(nn.Module):
    """Projects the introspection signals down to a small emotion vector."""

    def __init__(self, n_sig: int, n_emotion: int = 4, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sig, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, n_emotion, bias=False),
        )

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return self.net(sig)  # [B,T,n_emotion]


class InnerVoiceLoop(nn.Module):
    """Shared-weight iterative refinement after the backbone, before the head.
    Each round proposes a candidate revision, gates how much to move toward
    it, and repeats for n_rounds."""

    def __init__(self, d_model: int, d_mem_v: int, n_emotion: int = 4,
                 n_rounds: int = 3, propose_hidden: int | None = None) -> None:
        super().__init__()
        self.n_rounds = n_rounds
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        input_width = d_model + n_emotion + d_mem_v
        if propose_hidden is None:
            propose_hidden = -(-input_width // d_model) * d_model
        self.propose = nn.Sequential(
            nn.Linear(d_model + n_emotion + d_mem_v, propose_hidden, bias=True),
            nn.GELU(),
            nn.Linear(propose_hidden, d_model, bias=False),
        )
        nn.init.zeros_(self.propose[-1].weight)
        self.conflict_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.conflict_gate.weight)
        nn.init.constant_(self.conflict_gate.bias, -2.0)

        self._last_gate_mean: float = 0.0
        self._last_gate_std_ch:  float = 0.0
        self._last_gate_std_pos: float = 0.0
        self._last_round_deltas: List[float] = []
        self._last_gate_effort: torch.Tensor | None = None
        self._last_round_gate_means: List[float] = []
        self._last_round_gate_change: List[float] = []
        self._diag: bool = True

    def forward(self, h: torch.Tensor, e_t: torch.Tensor, read_fn
                ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        u = h
        last_w: torch.Tensor | None = None
        self._last_round_deltas = []
        self._last_round_gate_means = []
        self._last_round_gate_change = []
        _prev_gate_mean: float | None = None
        gate_effort_acc = torch.zeros(u.shape[:-1], device=u.device, dtype=torch.float32)
        for _ in range(self.n_rounds):
            u_n = self.ln(u)
            m_read, last_w = read_fn(u_n)
            cand   = u + self.propose(torch.cat([u_n, e_t, m_read], dim=-1))
            gate   = torch.sigmoid(self.conflict_gate(u_n))
            gate_effort_acc += gate.detach().float().mean(dim=-1)
            u_prev = u.detach()
            u      = self.ln(u + gate * (cand - u))
            if self._diag:
                _g_mean_now = gate.detach().float().mean()
                self._last_round_gate_means.append(_g_mean_now)
                self._last_round_gate_change.append(
                    torch.zeros_like(_g_mean_now) if _prev_gate_mean is None
                    else (_g_mean_now - _prev_gate_mean).abs())
                _prev_gate_mean = _g_mean_now
                delta_norm = (u.detach() - u_prev).float().norm(dim=-1)
                rel_delta  = (delta_norm / (u_prev.float().norm(dim=-1) + 1e-6)).mean()
                self._last_round_deltas.append(rel_delta)
        self._last_gate_effort = gate_effort_acc / self.n_rounds
        if self._diag:
            g = gate.detach().float()
            self._last_gate_mean   = g.mean()
            self._last_gate_std_ch = g.std(dim=-1).mean()
            self._last_gate_std_pos = g.std(dim=tuple(range(g.dim() - 1))).mean()
        return u, last_w


def cat_max_slots(cfg: "Config") -> int:
    """Ceiling on the number of categories."""
    return math.ceil(math.sqrt(cfg.mem_max_slots))


def _select_categories(cat_sim: torch.Tensor, margin: float,
                        empty_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Which categories are close enough to the best match to be worth
    reading/routing to."""
    if empty_mask is not None and not bool(empty_mask.all()):
        real_mask = ~empty_mask
        cat_sim_for_max = cat_sim.masked_fill(empty_mask, float('-inf'))
        best = cat_sim_for_max.max(dim=-1, keepdim=True).values
        n_real = real_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_real = (cat_sim * real_mask).sum(dim=-1, keepdim=True) / n_real
        var_real = ((cat_sim - mean_real) ** 2 * real_mask).sum(dim=-1, keepdim=True) / n_real
        std = var_real.clamp(min=0).sqrt()
        return cat_sim_for_max >= (best - margin * std)
    std = cat_sim.std(unbiased=False, dim=-1, keepdim=True)
    best = cat_sim.max(dim=-1, keepdim=True).values
    return cat_sim >= (best - margin * std)


@torch.no_grad()
def _pool_near_dup_rates(content: torch.Tensor, cos_threshold: float,
                          max_slots: int, seed: int = 0
                          ) -> Tuple[float, float, int, float, float]:
    """Pairwise near-duplicate rate of the slot pool, raw and with the pool's
    mean direction removed. Returns
    `(raw_rate, mean_removed_rate, n_used, raw_mean_cos, mean_removed_mean_cos)`."""
    n = int(content.shape[0])
    if n < 2:
        return 0.0, 0.0, n, 0.0, 0.0
    c = content.detach().float()
    if n > max_slots:
        g = torch.Generator(device='cpu').manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_slots].to(c.device)
        c = c[idx]
        n = max_slots

    def _rates(x: torch.Tensor) -> Tuple[float, float]:
        xn = F.normalize(x, dim=-1)
        hits = 0
        cos_sum = 0.0
        pairs = 0
        chunk = max(1, min(1024, n))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = xn[start:stop] @ xn.T                      # [chunk, n]
            rows = torch.arange(start, stop, device=xn.device).unsqueeze(1)
            cols = torch.arange(n, device=xn.device).unsqueeze(0)
            keep = cols > rows
            vals = block[keep]
            hits += int((vals > cos_threshold).sum().item())
            cos_sum += float(vals.sum().item())
            pairs += int(vals.numel())
        if pairs == 0:
            return 0.0, 0.0
        return hits / pairs, cos_sum / pairs

    raw_rate, raw_mean = _rates(c)
    dm_rate, dm_mean   = _rates(c - c.mean(dim=0, keepdim=True))
    return raw_rate, dm_rate, n, raw_mean, dm_mean


@torch.no_grad()
def _pool_usage_concentration(U: torch.Tensor, mass_fraction: float
                               ) -> Tuple[int, float]:
    """Scale-free measure of how many slots the pool is actually using.
    Returns `(n_mass, eff)`: slots holding `mass_fraction` of total usage,
    and the effective-number-of-components entropy measure."""
    n = int(U.numel())
    if n == 0:
        return 0, 0.0
    u = U.detach().float().clamp(min=0)
    total = float(u.sum().item())
    if total <= 0.0:
        return 0, 0.0
    srt, _ = torch.sort(u, descending=True)
    csum = torch.cumsum(srt, dim=0) / total
    n_mass = int((csum < mass_fraction).sum().item()) + 1
    n_mass = min(n_mass, n)
    p = u / total
    ent = float(-(p * (p + 1e-12).log()).sum().item())
    return n_mass, math.exp(ent)


class MemoryPool(nn.Module):
    """Content-addressed persistent slot memory."""

    def __init__(self, d_model: int, d_k: int, d_v: int,
                 spawn_bias_margin_seed: float = 1.0, spawn_bias_clamp: float = 5.0) -> None:
        super().__init__()
        self.content_ln = nn.LayerNorm(d_model, elementwise_affine=False)
        assert d_v == d_model, "value = stored content: d_v must equal d_model"
        self.d_k, self.d_v = d_k, d_v
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.read_temp  = nn.Parameter(torch.tensor(4.0))
        self.k_null = nn.Parameter(torch.zeros(d_k))
        self.b_null = nn.Parameter(torch.zeros(()))
        self._diag: bool = True
        self._last_null_mass: torch.Tensor | None = None
        self.spawn_bias = SpawnBiasHead(spawn_bias_margin_seed, spawn_bias_clamp)
        self.register_buffer('_merge_nn_sim_mean',  torch.zeros(()))
        self.register_buffer('_merge_nn_sim_var',   torch.ones(()))
        self.register_buffer('_merge_nn_sim_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('_merge_nn_sim_nonfinite', torch.zeros((), dtype=torch.long))

    def observe_merge_sim(self, raw: float, alpha: float, warmup: int) -> Tuple[bool, float]:
        """Updates the running mean/var of a candidate's nearest-neighbor
        similarity. Returns (ready, z)."""
        with torch.no_grad():
            if not math.isfinite(raw):
                self._merge_nn_sim_nonfinite += 1
                return False, 0.0
            delta = raw - self._merge_nn_sim_mean.item()
            self._merge_nn_sim_mean.add_(alpha * delta)
            self._merge_nn_sim_var.mul_(1.0 - alpha).add_(alpha * delta * delta)
            self._merge_nn_sim_count += 1
            ready = self._merge_nn_sim_count.item() >= warmup
            z = (raw - self._merge_nn_sim_mean.item()) / (self._merge_nn_sim_var.item() ** 0.5 + 1e-6)
            if not math.isfinite(z):
                self._merge_nn_sim_nonfinite += 1
                return False, 0.0
            return ready, z

    def project(self, content: torch.Tensor, dtype: torch.dtype
                ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Computes live K,V from raw content. Returns (None, None) at genesis."""
        if content.shape[0] == 0:
            return None, None
        c = content.to(dtype)
        K = self.Wk(c)   # [C,d_k] -- live, gets gradient
        V = c            # [C,d_model] -- value = stored content
        return K, V

    def attend(self, u: torch.Tensor, K: torch.Tensor | None, V: torch.Tensor | None,
               row_mask: torch.Tensor | None = None
               ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """The per-round read over the memory pool. Also returns the
        attention weights `w` [B,T,C] (or None at genesis)."""
        if K is None:
            zeros = torch.zeros(*u.shape[:-1], self.d_v, device=u.device, dtype=u.dtype)
            if self._diag:
                self._last_null_mass = torch.ones((), device=u.device)
            return zeros, None
        q   = self.Wq(u)                                                   # [B,T,d_k]
        C   = K.shape[0]
        sim = torch.einsum('btd,cd->btc', q, K) / math.sqrt(self.d_k)
        temp   = self.read_temp.clamp(max=8.0)
        logits = (sim * temp).float()                                     # [B,T,C]
        # Null-slot logit, appended as the last column:
        #   null = sg(mean_c logit_c) + log C + g,   g = b_null + temp*q.k_null/sqrt(d_k)
        g = self.b_null.float() + temp.float() * F.linear(
                q.float(), self.k_null.float().unsqueeze(0)) / math.sqrt(self.d_k)       # [B,T,1]
        if row_mask is None:
            null = logits.detach().mean(-1, keepdim=True) + math.log(C) + g
        else:
            m = row_mask.unsqueeze(1)                                      # [B,1,C]
            n = m.sum(-1, keepdim=True).clamp_min(1).float()               # [B,1,1]
            mean = (logits.detach() * m).sum(-1, keepdim=True) / n
            logits = logits.masked_fill(~m, float("-inf"))
            null = mean + n.log() + g
        w_full = torch.softmax(torch.cat([logits, null], dim=-1), dim=-1).to(V.dtype)   # [B,T,C+1]
        w      = w_full[..., :C]              # real slots only; rows sum to 1 - null mass
        if self._diag and w_full.shape[1] > 0:
            self._last_null_mass = w_full[..., C].detach().float().mean()
        return torch.einsum('btc,cd->btd', w, V), w                       # [B,T,d_v]

    def read(self, u: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """Convenience one-shot wrapper (project+attend together)."""
        K, V = self.project(content, u.dtype)
        out, _ = self.attend(u, K, V)
        return out

    def write_targets(self, u_flat: torch.Tensor, content: torch.Tensor,
                       U: torch.Tensor | None = None, underdog_bonus: float = 0.0
                       ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Returns (w_join, sim, u_flat_n, k_new). Returns (None, None,
        u_flat_n, None) at genesis (C=0)."""
        u_flat_n = self.content_ln(u_flat)
        if content.shape[0] == 0:
            return None, None, u_flat_n, None
        k_new      = self.Wk(u_flat_n)                                      # comparison only, borrows live Wk
        K_existing = self.Wk(content.to(u_flat_n.dtype))                    # content already normalized (stored that way)
        sim    = (k_new @ K_existing.T) / math.sqrt(self.d_k)               # [N,C] -- unbiased, returned as-is
        temp   = self.read_temp.clamp(max=8.0)
        join_logits = sim * temp
        if underdog_bonus != 0.0 and U is not None and U.shape[0] > 0:
            bonus_scale   = sim.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-6)  # [N,1]
            underdog_frac = 1.0 - U / U.max().clamp(min=1e-6)                  # [C]
            join_logits   = join_logits + underdog_bonus * bonus_scale * underdog_frac.unsqueeze(0)
        w_join = torch.softmax(join_logits, dim=-1)                        # [N,C] -- content-similarity + underdog nudge
        return w_join, sim, u_flat_n, k_new

class SpawnBiasHead(nn.Module):
    """Small learned additive shift on the spawn-vs-join novelty threshold.
    5 scalar weights + 1 bias, frozen from autograd and updated only via
    `apply_update`'s manual, error-driven correction."""

    def __init__(self, margin_seed: float, clamp: float) -> None:
        super().__init__()
        self.clamp = clamp
        self.w_margin  = nn.Parameter(torch.tensor(margin_seed))
        self.w_std     = nn.Parameter(torch.zeros(()))
        self.w_emo     = nn.Parameter(torch.zeros(()))
        self.w_effort  = nn.Parameter(torch.zeros(()))
        self.bias      = nn.Parameter(torch.zeros(()))
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('n_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('_w_margin_prior', torch.tensor(margin_seed))
        self.register_buffer('_ss_mean',  torch.zeros(()))
        self.register_buffer('_ss_var',   torch.ones(()))
        self.register_buffer('_ss_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('_emo_mean',  torch.zeros(()))
        self.register_buffer('_emo_var',   torch.ones(()))
        self.register_buffer('_emo_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('_ss_nonfinite',  torch.zeros((), dtype=torch.long))
        self.register_buffer('_emo_nonfinite', torch.zeros((), dtype=torch.long))

    def forward(self, hist_std: float, sim_std_z: float,
                emo_mag: float, gate_effort: float) -> float:
        """Returns b_spawn as a plain float."""
        return (- self.w_margin.item()  * hist_std
                - self.w_std.item()     * sim_std_z
                + self.w_emo.item()     * emo_mag
                + self.w_effort.item()  * gate_effort
                + self.bias.item())

    def observe_sim_std(self, raw: float, n_candidates: int,
                         alpha: float, warmup: int) -> float:
        """Updates the running mean/var of raw sim_std and returns the
        clamped z-score. Returns 0.0 until `warmup` real samples accrue."""
        with torch.no_grad():
            if not math.isfinite(raw):
                self._ss_nonfinite += 1
                return 0.0
            if n_candidates >= 2:
                delta = raw - self._ss_mean.item()
                self._ss_mean.add_(alpha * delta)
                self._ss_var.mul_(1.0 - alpha).add_(alpha * delta * delta)
                self._ss_count += 1
            if self._ss_count.item() < warmup:
                return 0.0
            z = (raw - self._ss_mean.item()) / (self._ss_var.item() ** 0.5 + 1e-6)
            if not math.isfinite(z):
                self._ss_nonfinite += 1
                return 0.0
            return max(-5.0, min(5.0, z))

    def observe_emo_mag(self, raw: float, alpha: float, warmup: int) -> float:
        """Same running z-score tracker as observe_sim_std, for emo_mag."""
        with torch.no_grad():
            if not math.isfinite(raw):
                self._emo_nonfinite += 1
                return 0.0
            delta = raw - self._emo_mean.item()
            self._emo_mean.add_(alpha * delta)
            self._emo_var.mul_(1.0 - alpha).add_(alpha * delta * delta)
            self._emo_count += 1
            if self._emo_count.item() < warmup:
                return 0.0
            z = (raw - self._emo_mean.item()) / (self._emo_var.item() ** 0.5 + 1e-6)
            if not math.isfinite(z):
                self._emo_nonfinite += 1
                return 0.0
            return max(-5.0, min(5.0, z))

    def apply_update(self, features: Tuple[float, float, float, float],
                      effective_signal: float, lr: float) -> None:
        """Perceptron-style correction from a sparse, delayed reward signal."""
        with torch.no_grad():
            effective_lr = lr / (1.0 + self.n_updates.item()) ** 0.5
            fm, fz, fe, fg = features
            self.w_margin.add_(-effective_lr * effective_signal * fm).clamp_(-self.clamp, self.clamp)
            self.w_std.add_(-effective_lr * effective_signal * fz).clamp_(-self.clamp, self.clamp)
            self.w_emo.add_(effective_lr * effective_signal * fe).clamp_(-self.clamp, self.clamp)
            self.w_effort.add_(effective_lr * effective_signal * fg).clamp_(-self.clamp, self.clamp)
            self.bias.add_(effective_lr * effective_signal).clamp_(-self.clamp, self.clamp)
            self.n_updates += 1

    def decay_toward_neutral(self, rate: float) -> None:
        """Small unconditional pull back toward each weight's resting value,
        so an unreinforced drift washes out over time."""
        with torch.no_grad():
            if self.n_updates.item() == 0:
                return
            self.w_margin.add_(rate * (self._w_margin_prior - self.w_margin))
            self.w_std.mul_(1.0 - rate)
            self.w_emo.mul_(1.0 - rate)
            self.w_effort.mul_(1.0 - rate)
            self.bias.mul_(1.0 - rate)

class _PendingSpawnDecision:
    """One queued spawn-or-join decision, waiting until its delayed outcome
    is known so it can update SpawnBiasHead. Not persisted across checkpoint
    resume."""
    __slots__ = ("features", "decision_sign", "slot_idx", "u_before", "due_at",
                 "relevance", "g_at_queue", "n_at_queue", "cat_at_queue")

    def __init__(self, features: Tuple[float, float, float, float],
                 decision_sign: float, slot_idx: int, u_before: float, due_at: int,
                 relevance: float = 1.0, g_at_queue: float = 0.0, n_at_queue: int = 0,
                 cat_at_queue: int = -1) -> None:
        self.features      = features
        self.decision_sign = decision_sign   # +1.0 = was a spawn, -1.0 = was a join
        self.slot_idx       = slot_idx
        self.u_before       = u_before
        self.due_at         = due_at
        self.relevance      = relevance
        self.g_at_queue     = g_at_queue
        self.n_at_queue     = n_at_queue
        self.cat_at_queue   = cat_at_queue


def _process_due_decisions(pending: deque, bias_head: SpawnBiasHead,
                            U: torch.Tensor, cat_id: torch.Tensor, cat_member_shadow: torch.Tensor,
                            total_docs_seen: int, cfg: "Config", commit_count_now: int) -> int:
    """Evaluates every queued decision whose delay window has elapsed and
    applies the error-driven correction to `bias_head`. Returns the number
    of decisions dropped because their slot was reassigned to a different
    category since queueing."""
    if not pending:
        return 0
    still_pending: List[_PendingSpawnDecision] = []
    dropped_reassigned = 0
    for dec in pending:
        if dec.due_at > total_docs_seen:
            still_pending.append(dec)
            continue
        if dec.slot_idx >= U.shape[0]:
            continue
        if not (0 <= dec.cat_at_queue < cat_member_shadow.shape[0]):
            continue
        if int(cat_id[dec.slot_idx].item()) != dec.cat_at_queue:
            dropped_reassigned += 1
            continue
        decay      = cfg.mem_usage_decay
        k          = max(commit_count_now - dec.n_at_queue, 0)
        decay_k    = decay ** k
        shadow_now = cat_member_shadow[dec.cat_at_queue].item()
        baseline_u = dec.u_before * decay_k + (shadow_now - dec.g_at_queue * decay_k)
        healthy    = U[dec.slot_idx].item() >= baseline_u
        reward     = 1.0 if healthy else -1.0
        effective_signal = reward if dec.decision_sign > 0 else -reward * dec.relevance
        bias_head.apply_update(dec.features, effective_signal, cfg.mem_spawn_bias_lr)
    pending.clear()
    pending.extend(still_pending)
    return dropped_reassigned


class CategoryState:
    """Groups all category-tier state -- the coarse routing level above
    individual memory slots -- so it threads through `apply_write_event` as
    one object. `content`/`U`/`emotion` are exact recomputed aggregates of
    live member slots, not EMAs."""
    __slots__ = ("content", "U", "member_shadow", "join_hist",
                 "join_hist_count", "top_shadow", "emotion")

    def __init__(self, content: torch.Tensor, U: torch.Tensor, member_shadow: torch.Tensor,
                 join_hist: torch.Tensor, join_hist_count: torch.Tensor,
                 top_shadow: torch.Tensor, emotion: torch.Tensor) -> None:
        self.content         = content
        self.U               = U
        self.member_shadow   = member_shadow
        self.join_hist       = join_hist
        self.join_hist_count = join_hist_count
        self.top_shadow      = top_shadow
        self.emotion         = emotion


class WriteStats:
    """Plain result object for apply_write_event."""
    __slots__ = ("spawn_share", "total_write_mass", "spawned", "n_slots",
                 "novelty_margin", "c_sub", "k_nov", "drought_boost", "merged",
                 "cat_load_bias", "cat_load_bias_spread", "cat_transform_events",
                 "decisions_dropped_reassigned")
    def __init__(self, spawn_share: float, total_write_mass: float,
                 spawned: bool, n_slots: int, novelty_margin: float = 0.0,
                 c_sub: int = 0, k_nov: int = 0, drought_boost: float = 0.0,
                 merged: bool = False, cat_load_bias: float = 0.0,
                 cat_load_bias_spread: float = 0.0, cat_transform_events: int = 0,
                 decisions_dropped_reassigned: int = 0) -> None:
        self.spawn_share      = spawn_share
        self.total_write_mass = total_write_mass
        self.spawned          = spawned
        self.n_slots          = n_slots
        self.c_sub            = c_sub
        self.k_nov            = k_nov
        self.novelty_margin   = novelty_margin
        self.drought_boost    = drought_boost
        self.cat_load_bias                = cat_load_bias
        self.cat_load_bias_spread         = cat_load_bias_spread
        self.cat_transform_events         = cat_transform_events
        self.decisions_dropped_reassigned = decisions_dropped_reassigned
        self.merged            = merged


def _recompute_categories(content: torch.Tensor, U: torch.Tensor, cat_id: torch.Tensor,
                           touched: torch.Tensor, cat_content: torch.Tensor, cat_U: torch.Tensor,
                           emotion: torch.Tensor | None = None, cat_emotion: torch.Tensor | None = None,
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Exact member-U-weighted centroid/U_cat recompute for the categories
    listed in `touched`. A touched category with zero members keeps its
    existing (frozen) content/U."""
    if touched.numel() == 0:
        return cat_content, cat_U, cat_emotion
    M = cat_content.shape[0]
    d = content.shape[1]
    mask = torch.isin(cat_id, touched)
    member_idx = mask.nonzero(as_tuple=True)[0]
    if member_idx.numel() == 0:
        return cat_content, cat_U, cat_emotion
    member_cat = cat_id[member_idx]
    w = U[member_idx].clamp(min=1e-6)
    wsum = torch.zeros(M, device=content.device, dtype=U.dtype)
    wsum.index_add_(0, member_cat, w)
    csum = torch.zeros(M, d, device=content.device, dtype=content.dtype)
    csum.index_add_(0, member_cat, content[member_idx] * w.unsqueeze(-1))
    fresh_centroid = csum / wsum.clamp(min=1e-6).unsqueeze(-1)
    nonempty = wsum[touched] > 0
    touched_nonempty = touched[nonempty]
    new_cat_content = cat_content.clone()
    new_cat_U = cat_U.clone()
    new_cat_content[touched_nonempty] = fresh_centroid[touched_nonempty].to(cat_content.dtype)
    new_cat_U[touched_nonempty] = wsum[touched_nonempty].to(cat_U.dtype)
    new_cat_emotion = cat_emotion
    if emotion is not None and cat_emotion is not None:
        n_emo = emotion.shape[1]
        esum = torch.zeros(M, n_emo, device=content.device, dtype=emotion.dtype)
        esum.index_add_(0, member_cat, emotion[member_idx] * w.unsqueeze(-1))
        fresh_emotion = esum / wsum.clamp(min=1e-6).unsqueeze(-1)
        new_cat_emotion = cat_emotion.clone()
        new_cat_emotion[touched_nonempty] = fresh_emotion[touched_nonempty].to(cat_emotion.dtype)
    return new_cat_content, new_cat_U, new_cat_emotion


def apply_write_event(pool: "MemoryPool", content: torch.Tensor, U: torch.Tensor,
                       emotion: torch.Tensor, idle_steps: torch.Tensor,
                       join_hist: torch.Tensor, join_hist_count: torch.Tensor,
                       bias_head: SpawnBiasHead, pending: deque, total_docs_seen: torch.Tensor,
                       commit_count: torch.Tensor, cat_id: torch.Tensor, cat: "CategoryState",
                       u_flat: torch.Tensor, e_flat: torch.Tensor, gate_effort_flat: torch.Tensor,
                       cfg: "Config", lr_cap: float, emo_threshold: float,
                       docs_since_spawn: torch.Tensor,
                       mass_floor: float = 0.0
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, WriteStats]:
    """Blends new content into the memory pool: joins it into existing
    slots, or spawns a new one if nothing matches closely enough."""
    with torch.no_grad():
        C = content.shape[0]
        M = cat.content.shape[0]
        N = u_flat.shape[0]
        strength = torch.sigmoid(e_flat.abs().sum(-1) - emo_threshold)  # [N]

        total_docs_seen += 1
        docs_since_spawn += 1
        drought_norm = min(docs_since_spawn.item() / cfg.mem_novelty_warmup, 5.0)
        neutral_decay_rate = 1.0 / max(cfg.mem_novelty_warmup, 1)
        bias_head.decay_toward_neutral(neutral_decay_rate)
        decisions_dropped_reassigned = _process_due_decisions(
            pending, bias_head, U, cat_id, cat.member_shadow,
            int(total_docs_seen.item()), cfg, int(commit_count.item()))

        if C == 0:
            u_flat_n = pool.content_ln(u_flat)
            total_write_mass = strength.sum().clamp(min=1e-6)
            if total_write_mass.item() < mass_floor:
                return content, U, emotion, idle_steps, cat_id, WriteStats(0.0, total_write_mass.item(), False, C)
            commit_count += 1
            new_content = ((strength.unsqueeze(-1) * u_flat_n).sum(0, keepdim=True)
                           / total_write_mass)
            new_emotion = ((strength.unsqueeze(-1) * e_flat).sum(0, keepdim=True)
                           / total_write_mass).to(emotion.dtype)
            new_U = torch.tensor([1.0], device=U.device, dtype=U.dtype)
            new_cat_id = torch.zeros(1, dtype=torch.long, device=content.device)
            new_idle_steps = torch.zeros(1, device=idle_steps.device, dtype=idle_steps.dtype)
            cat.content = new_content.clone()
            cat.U = torch.tensor([1.0], device=U.device, dtype=U.dtype)
            cat.member_shadow = torch.tensor([1.0], device=U.device, dtype=U.dtype)
            cat.emotion = new_emotion.clone()
            cat.top_shadow.mul_(cfg.mem_usage_decay).add_(1.0)
            docs_since_spawn.zero_()
            return new_content, new_U, new_emotion, new_idle_steps, new_cat_id, WriteStats(1.0, total_write_mass.item(), True, 1, 0.0)

        u_flat_n = pool.content_ln(u_flat)
        w_str_route = strength / strength.sum().clamp(min=1e-6)
        doc_query = (u_flat_n * w_str_route.unsqueeze(-1)).sum(0)  # [d_model]

        cat_k   = pool.Wk(cat.content.to(u_flat_n.dtype))
        q_cat   = pool.Wk(doc_query.unsqueeze(0).to(u_flat_n.dtype))
        cat_sim = F.cosine_similarity(q_cat, cat_k, dim=-1)
        cat_sizes       = torch.bincount(cat_id, minlength=M)
        cat_empty_mask  = cat_sizes == 0
        if M > 1:
            if not bool(cat_empty_mask.all()):
                cat_real_mask = (~cat_empty_mask).to(cat_sim.dtype)
                cat_n_real    = int(cat_real_mask.sum().item())
                cat_mean_real = (cat_sim * cat_real_mask).sum() / max(cat_n_real, 1)
                cat_var_real  = ((cat_sim - cat_mean_real) ** 2 * cat_real_mask).sum() / max(cat_n_real, 1)
                cat_sim_std   = cat_var_real.clamp(min=0).sqrt().item()
            else:
                cat_n_real  = M
                cat_sim_std = cat_sim.std(unbiased=False).item()
        else:
            cat_n_real  = M
            cat_sim_std = 0.0
        cat_M_live   = max(int((~cat_empty_mask).sum().item()), 1)
        total_slots  = max(int(cat_id.shape[0]), 1)
        load_ratio   = cat_sizes[:M].to(cat_sim.dtype) * cat_M_live / total_slots
        load_bias    = (cat_sim_std * (load_ratio - 1.0).clamp(
                            -cfg.cat_load_bias_clamp, cfg.cat_load_bias_clamp)
                         * cfg.cat_load_bias_k)
        cat_sim_raw  = cat_sim
        cat_sim      = cat_sim_raw - load_bias
        cat_sim_masked = (cat_sim.masked_fill(cat_empty_mask, float('-inf'))
                           if not bool(cat_empty_mask.all()) else cat_sim)
        cat_novelty_stat = cat_sim_masked.max()
        selected_idx = _select_categories(cat_sim, cfg.mem_cat_select_margin,
                                           cat_empty_mask).nonzero(as_tuple=True)[0]

        member_mask = torch.isin(cat_id, selected_idx)
        cand_idx    = member_mask.nonzero(as_tuple=True)[0]
        C_sub       = cand_idx.shape[0]

        if C_sub == 0:
            total_write_mass = strength.sum().clamp(min=1e-6)
            if total_write_mass.item() < mass_floor:
                return content, U, emotion, idle_steps, cat_id, WriteStats(0.0, total_write_mass.item(), False, C)
            commit_count += 1
            content_spawn = ((strength.unsqueeze(-1) * u_flat_n).sum(0)
                              / strength.sum().clamp(min=1e-6))
            emotion_spawn = ((strength.unsqueeze(-1) * e_flat).sum(0)
                              / strength.sum().clamp(min=1e-6)).to(emotion.dtype)
            new_share   = (strength.sum() / max(N, 1)).clamp(0.0, 1.0)
            new_content = torch.cat([content, content_spawn.unsqueeze(0).to(content.dtype)], dim=0)
            new_emotion = torch.cat([emotion, emotion_spawn.unsqueeze(0)], dim=0)
            new_U       = torch.cat([U, new_share.view(1).to(U.dtype)], dim=0)
            new_idle_steps = torch.cat(
                [idle_steps, torch.zeros(1, device=idle_steps.device, dtype=idle_steps.dtype)], dim=0)
            best_cat    = selected_idx[cat_sim[selected_idx].argmax()]
            new_cat_id  = torch.cat([cat_id, best_cat.view(1)], dim=0)
            cat.content[best_cat] = content_spawn.to(cat.content.dtype)
            cat.U[best_cat]       = new_share
            cat.emotion[best_cat] = emotion_spawn.to(cat.emotion.dtype)
            cat.member_shadow.mul_(cfg.mem_usage_decay)
            cat.member_shadow[best_cat] = 1.0
            cat.top_shadow.mul_(cfg.mem_usage_decay).add_(1.0 / M)
            docs_since_spawn.zero_()
            return new_content, new_U, new_emotion, new_idle_steps, new_cat_id, WriteStats(
                1.0, total_write_mass.item(), True, new_content.shape[0], 0.0)

        content_sub = content[cand_idx]
        U_sub       = U[cand_idx]
        emotion_sub = emotion[cand_idx]
        w_join, sim, _, k_new = pool.write_targets(u_flat, content_sub, U_sub, cfg.mem_underdog_bonus)  # w_join,sim: [N,C_sub]
        m = w_join * strength.unsqueeze(-1)  # [N,C_sub]
        total_mass_sub = m.sum(0)  # [C_sub]
        total_write_mass = total_mass_sub.sum().clamp(min=1e-6)

        if total_write_mass.item() < mass_floor:
            return content, U, emotion, idle_steps, cat_id, WriteStats(0.0, total_write_mass.item(), False, C)

        content_sum = torch.einsum('nc,nd->cd', m, u_flat_n)  # [C_sub,d_model]
        emotion_sum = torch.einsum('nc,nd->cd', m, e_flat)  # [C_sub,n_emotion]
        denom       = total_mass_sub.clamp(min=1e-6).unsqueeze(-1)
        avg_content = content_sum / denom
        avg_emotion = emotion_sum / denom
        anchor      = torch.quantile(U, cfg.mem_plasticity_percentile)
        plasticity  = 1.0 / (1.0 + U_sub / (anchor + 1e-6))  # [C_sub]
        raw_lr      = total_mass_sub / N
        lr          = (raw_lr * plasticity).clamp(max=lr_cap).unsqueeze(-1)
        blended_sub = content_sub * (1 - lr) + lr * avg_content
        blended_emotion_sub = emotion_sub * (1 - lr) + lr * avg_emotion

        new_content = content.clone()
        new_content[cand_idx] = blended_sub
        new_emotion = emotion.clone()
        new_emotion[cand_idx] = blended_emotion_sub.to(new_emotion.dtype)
        new_U = U.clone()
        new_U.mul_(cfg.mem_usage_decay)
        mass_share_sub = total_mass_sub / total_write_mass
        new_U[cand_idx] += mass_share_sub

        new_idle_steps = idle_steps + 1.0
        new_idle_steps[cand_idx] = new_idle_steps[cand_idx] * (1.0 - mass_share_sub.clamp(0.0, 1.0))

        cat.member_shadow.mul_(cfg.mem_usage_decay)
        cat.member_shadow[selected_idx] += 1.0 / C_sub
        cat.top_shadow.mul_(cfg.mem_usage_decay).add_(1.0 / M)
        commit_count += 1

        touch_mask  = mass_share_sub >= (1.0 / max(C_sub, 1))
        touched_idx = cand_idx[touch_mask]
        cat_transform_events = 0
        transform_touched = selected_idx
        if touched_idx.numel() > 0:
            q_touched   = pool.Wk(blended_sub[touch_mask].to(cat_k.dtype))  # [T,d_k]
            sim_touched = (q_touched @ cat_k.T) / math.sqrt(pool.d_k) - load_bias.unsqueeze(0)  # [T,M]
            sim_touched_sel = (sim_touched.masked_fill(cat_empty_mask.unsqueeze(0), float('-inf'))
                                if not bool(cat_empty_mask.all()) else sim_touched)
            current_cat_t = cat_id[touched_idx]  # [T]
            current_sim   = sim_touched.gather(1, current_cat_t.unsqueeze(1)).squeeze(1)
            best_sim, best_cat_t = sim_touched_sel.max(dim=1)
            row_std = sim_touched.std(dim=1, unbiased=False)  # [T]
            transform_margin = cfg.cat_transform_margin_k * row_std
            move_mask = (best_sim > current_sim + transform_margin) & (best_cat_t != current_cat_t)
            cat_transform_events = int(move_mask.sum().item())
            if cat_transform_events > 0:
                moving_slots = touched_idx[move_mask]
                old_cats     = current_cat_t[move_mask]
                new_cats     = best_cat_t[move_mask]
                cat_id = cat_id.clone()
                cat_id[moving_slots] = new_cats
                transform_touched = torch.unique(torch.cat([selected_idx, old_cats, new_cats]))

        cat.content, cat.U, cat.emotion = _recompute_categories(
            new_content, new_U, cat_id, transform_touched, cat.content, cat.U,
            new_emotion, cat.emotion)

        unsel_mask = torch.ones(M, dtype=torch.bool, device=cat_id.device)
        unsel_mask[selected_idx] = False
        unsel_mask &= ~cat_empty_mask
        unsel_idx = unsel_mask.nonzero(as_tuple=True)[0]  # [M_unsel]
        if unsel_idx.numel() > 0:
            K_cent   = pool.Wk(cat.content[unsel_idx].to(k_new.dtype))  # [M_unsel,d_k]
            sim_cent = (k_new @ K_cent.T) / math.sqrt(pool.d_k)  # [N,M_unsel]
            sim_nov  = torch.cat([sim, sim_cent], dim=-1)  # [N, K_nov]
        else:
            sim_nov = sim
        K_nov = C_sub + int(unsel_idx.numel())

        doc_max_sim      = sim_nov.max(dim=-1).values  # [N]
        w_str            = strength / strength.sum().clamp(min=1e-6)
        doc_novelty_stat = (doc_max_sim * w_str).sum()
        sim_std_feat   = (sim_nov.std(dim=-1, unbiased=False) * w_str).sum().item()
        emo_mag_feat   = (e_flat.abs().sum(-1) * w_str).sum().item()
        gate_effort_feat = (gate_effort_flat * w_str).sum().item()
        sim_std_z = bias_head.observe_sim_std(sim_std_feat, K_nov,
                                               1.0 / cfg.mem_novelty_history_size,
                                               cfg.mem_novelty_history_size)
        emo_mag_z = bias_head.observe_emo_mag(emo_mag_feat,
                                               1.0 / cfg.mem_novelty_history_size,
                                               cfg.mem_novelty_history_size)

        valid_hist = int(join_hist_count.item())
        spawned = False
        merge_this_event = False
        novelty_margin = 0.0
        hist_std_feat = 0.0
        drought_boost = 0.0
        if valid_hist < cfg.mem_novelty_warmup:
            spawn_this = False
            hist_valid = None
        else:
            H = join_hist.shape[0]
            hist_valid = join_hist[:min(valid_hist, H)]
            hist_std_feat = hist_valid.std(unbiased=False).item() if hist_valid.numel() > 1 else 0.0
            fullness_factor = 1.0 - C_sub / max(cfg.mem_max_slots / M, 1.0)
            effective_percentile = cfg.mem_novelty_percentile * max(fullness_factor, 0.0)
            thresh = torch.quantile(hist_valid, max(effective_percentile, 0.0))
            b_spawn = bias_head(hist_std_feat, sim_std_z,
                                 emo_mag_z, gate_effort_feat)
            drought_boost = hist_std_feat * (drought_norm / 5.0)
            bar = thresh + b_spawn + drought_boost
            if drought_norm >= 5.0:
                bar = torch.maximum(bar, hist_valid.min())
            spawn_this = bool((doc_novelty_stat < bar).item())
            novelty_margin = (bar - doc_novelty_stat).item()

        features = (hist_std_feat, sim_std_z, emo_mag_z, gate_effort_feat)
        due_at = int(total_docs_seen.item()) + cfg.mem_novelty_history_size
        new_cat_id = cat_id

        recycle_idx: int | None = None
        donee_idx:   int | None = None
        if spawn_this and C >= 2:
            protected = {d.slot_idx for d in pending}
            idle_bar = torch.quantile(new_idle_steps, cfg.mem_merge_idle_percentile).item()
            above_idle_bar = (new_idle_steps >= idle_bar) & (new_idle_steps > 0.0)
            u_bar = torch.quantile(new_U, 1.0 - cfg.mem_merge_idle_percentile).item()
            below_u_bar = new_U <= u_bar
            cand_pool = (above_idle_bar | below_u_bar).nonzero(as_tuple=True)[0]
            if cand_pool.numel() > 0:
                cand_pool = cand_pool[torch.tensor(
                    [c.item() not in protected for c in cand_pool], dtype=torch.bool)]
            if cand_pool.numel() > 0:
                denom = max(C - 1, 1)
                idle_rank = new_idle_steps.argsort().argsort().float() / denom
                u_rank    = (-new_U).argsort().argsort().float() / denom
                priority  = torch.maximum(idle_rank, u_rank)
                rank = priority[cand_pool].argsort(descending=True)
                cand_pool = cand_pool[rank][:cfg.mem_merge_scan_k]
            if cand_pool.numel() > 0:
                K_cands = pool.Wk(new_content[cand_pool].to(u_flat_n.dtype))  # [K,d_k]
                K_all   = pool.Wk(new_content.to(u_flat_n.dtype))  # [C,d_k]
                sim_mat = (K_cands @ K_all.T) / math.sqrt(pool.d_k)  # [K,C]
                for row_i, cand in enumerate(cand_pool.tolist()):
                    sim_mat[row_i, cand] = float('-inf')
                for row_i, cand in enumerate(cand_pool.tolist()):
                    row = sim_mat[row_i]
                    nearest_val, nearest_idx = row.max(dim=0)
                    ready, z = pool.observe_merge_sim(
                        nearest_val.item(), 1.0 / cfg.mem_merge_nn_sim_warmup,
                        cfg.mem_merge_nn_sim_warmup)
                    if ready and z > cfg.mem_merge_redundancy_margin:
                        recycle_idx, donee_idx = cand, int(nearest_idx.item())
                        break

        cat_valid_hist = int(cat.join_hist_count.item())
        cmax = cat_max_slots(cfg)
        cat_spawn_this = False
        cat_novelty_margin = 0.0
        cat_hist_valid = None
        cat_hist_std_feat = 0.0
        cat_drought_boost = 0.0
        if cat_valid_hist >= cfg.mem_novelty_warmup and M < cmax:
            CH = cat.join_hist.shape[0]
            cat_hist_valid = cat.join_hist[:min(cat_valid_hist, CH)]
            cat_hist_std_feat = cat_hist_valid.std(unbiased=False).item() if cat_hist_valid.numel() > 1 else 0.0
            cat_fullness = 1.0 - M / cmax
            cat_eff_pctile = cfg.mem_novelty_percentile * max(cat_fullness, 0.0)
            cat_thresh = torch.quantile(cat_hist_valid, max(cat_eff_pctile, 0.0))
            cat_drought_boost = cat_hist_std_feat * (drought_norm / 5.0)
            cat_spawn_this = bool((cat_novelty_stat < cat_thresh + cat_drought_boost).item())
            cat_novelty_margin = (cat_thresh + cat_drought_boost - cat_novelty_stat).item()

        if spawn_this and (recycle_idx is not None or C < cfg.mem_max_slots):

            content_spawn = ((strength.unsqueeze(-1) * u_flat_n).sum(0)
                              / strength.sum().clamp(min=1e-6))
            emotion_spawn = ((strength.unsqueeze(-1) * e_flat).sum(0)
                              / strength.sum().clamp(min=1e-6)).to(new_emotion.dtype)
            new_share = (strength.sum() / max(N, 1)).clamp(0.0, 1.0)

            if recycle_idx is not None:
                U_dead, U_donee = new_U[recycle_idx], new_U[donee_idx]
                denom_m = (U_dead + U_donee).clamp(min=1e-6)
                merged_content = (U_dead * new_content[recycle_idx]
                                   + U_donee * new_content[donee_idx]) / denom_m
                merged_emotion = (U_dead * new_emotion[recycle_idx]
                                   + U_donee * new_emotion[donee_idx]) / denom_m
                new_content[donee_idx] = merged_content.to(new_content.dtype)
                new_emotion[donee_idx] = merged_emotion.to(new_emotion.dtype)
                new_U[donee_idx] = U_dead + U_donee
                donee_cat = int(cat_id[donee_idx].item())
                pending.append(_PendingSpawnDecision(
                    features, -1.0, donee_idx, new_U[donee_idx].item(), due_at,
                    relevance=1.0,
                    g_at_queue=cat.member_shadow[donee_cat].item(), n_at_queue=int(commit_count.item()),
                    cat_at_queue=donee_cat))

                new_content[recycle_idx] = content_spawn.to(new_content.dtype)
                new_emotion[recycle_idx] = emotion_spawn.to(new_emotion.dtype)
                new_U[recycle_idx] = new_share.to(new_U.dtype)
                new_idle_steps[recycle_idx] = 0.0
                spawned_slot_idx = recycle_idx
                merge_this_event = True
            else:
                new_content = torch.cat(
                    [new_content, content_spawn.unsqueeze(0).to(new_content.dtype)], dim=0)
                new_emotion = torch.cat([new_emotion, emotion_spawn.unsqueeze(0)], dim=0)
                new_U = torch.cat([new_U, new_share.view(1).to(new_U.dtype)], dim=0)
                new_idle_steps = torch.cat(
                    [new_idle_steps, torch.zeros(1, device=new_idle_steps.device,
                                                  dtype=new_idle_steps.dtype)], dim=0)
                spawned_slot_idx = new_content.shape[0] - 1
                merge_this_event = False
            spawned = True
            docs_since_spawn.zero_()

            if cat_spawn_this:
                cat_sizes = torch.bincount(cat_id, minlength=M)
                empty_cats = (cat_sizes == 0).nonzero(as_tuple=True)[0]
                if empty_cats.numel() > 0:
                    best_cat = int(empty_cats[0].item())
                    cat.content[best_cat] = content_spawn.to(cat.content.dtype)
                    cat.U[best_cat] = new_share.to(cat.U.dtype)
                    cat.emotion[best_cat] = emotion_spawn.to(cat.emotion.dtype)
                    cat.member_shadow[best_cat] = 1.0
                else:
                    new_cat_content = torch.cat(
                        [cat.content, content_spawn.unsqueeze(0).to(cat.content.dtype)], dim=0)
                    new_cat_U_arr = torch.cat([cat.U, new_share.view(1).to(cat.U.dtype)], dim=0)
                    new_cat_emotion = torch.cat(
                        [cat.emotion, emotion_spawn.unsqueeze(0).to(cat.emotion.dtype)], dim=0)
                    new_cat_shadow = torch.cat(
                        [cat.member_shadow, torch.zeros(1, device=cat.member_shadow.device,
                                                         dtype=cat.member_shadow.dtype)], dim=0)
                    best_cat = new_cat_content.shape[0] - 1
                    cat.content = new_cat_content
                    cat.U = new_cat_U_arr
                    cat.emotion = new_cat_emotion
                    cat.member_shadow = new_cat_shadow
                    cat.member_shadow[best_cat] = 1.0
                if recycle_idx is not None:
                    new_cat_id = cat_id.clone()
                    new_cat_id[spawned_slot_idx] = best_cat
                else:
                    new_cat_id = torch.cat([cat_id, torch.tensor([best_cat], dtype=torch.long,
                                                                  device=cat_id.device)], dim=0)
            else:
                best_cat = int(selected_idx[cat_sim[selected_idx].argmax()].item())
                if recycle_idx is not None:
                    new_cat_id = cat_id.clone()
                    new_cat_id[spawned_slot_idx] = best_cat
                else:
                    new_cat_id = torch.cat([cat_id, torch.tensor([best_cat], dtype=torch.long,
                                                                  device=cat_id.device)], dim=0)
                cat.content, cat.U, cat.emotion = _recompute_categories(
                    new_content, new_U, new_cat_id, torch.tensor([best_cat], device=cat_id.device),
                    cat.content, cat.U, new_emotion, cat.emotion)

            pending.append(_PendingSpawnDecision(
                features, +1.0, spawned_slot_idx, new_share.item(), due_at,
                g_at_queue=cat.member_shadow[best_cat].item(), n_at_queue=int(commit_count.item()),
                cat_at_queue=int(best_cat)))
        else:
            if hist_valid is not None and K_nov >= 2:
                relevance = 1.0 / (1.0 + max(-novelty_margin, 0.0) / (hist_std_feat + 1e-6))
                target_slot = int(cand_idx[total_mass_sub.argmax()].item())
                target_cat  = int(cat_id[target_slot].item())
                pending.append(_PendingSpawnDecision(
                    features, -1.0, target_slot, new_U[target_slot].item(), due_at,
                    relevance=relevance,
                    g_at_queue=cat.member_shadow[target_cat].item(), n_at_queue=int(commit_count.item()),
                    cat_at_queue=target_cat))


        idx = valid_hist % join_hist.shape[0]
        join_hist[idx] = doc_novelty_stat.detach()
        join_hist_count += 1

        idxc = int(cat.join_hist_count.item()) % cat.join_hist.shape[0]
        cat.join_hist[idxc] = cat_novelty_stat.detach()
        cat.join_hist_count += 1

        return new_content, new_U, new_emotion, new_idle_steps, new_cat_id, WriteStats(
            1.0 if spawned else 0.0, total_write_mass.item(),
            spawned, new_content.shape[0], novelty_margin,
            c_sub=C_sub, k_nov=K_nov, drought_boost=drought_boost,
            merged=merge_this_event, cat_load_bias=load_bias.abs().mean().item(),
            cat_load_bias_spread=(load_bias.max() - load_bias.min()).item(),
            cat_transform_events=cat_transform_events,
            decisions_dropped_reassigned=decisions_dropped_reassigned)


class ComboBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 n_heads_n: int, d_hn: int,
                 ffn_hidden: int, chunk_size: int = 128,
                 signal_expand: int = 1, ema_norm: float = 0.99) -> None:
        super().__init__()
        self.H, self.dk, self.dv, self.C = n_heads, d_k, d_v, chunk_size
        self.Hn, self.Dhn = n_heads_n, d_hn  # v8.13 NUCLEUS
        self.delta_dim   = n_heads * d_v
        self.nucleus_dim = n_heads_n * d_hn

        self.ln = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * d_v, bias=False)
        nn.init.orthogonal_(self.Wq.weight)
        nn.init.orthogonal_(self.Wk.weight)

        self.Wa = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.Wa.weight)
        a_vals = torch.exp(-torch.linspace(-math.log(0.999), -math.log(0.85), n_heads))
        self.Wa.bias.data.copy_(torch.log(a_vals / (1 - a_vals)))

        self.Wb = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.Wb.weight)
        nn.init.constant_(self.Wb.bias, -1.0)

        _RET_HIDDEN = d_model * signal_expand
        self.register_buffer('prev_local_sig', torch.zeros(N_LOCAL))
        self.retention_mod = nn.Sequential(
            nn.Linear(N_LOCAL + N_GLOBAL + N_COLONY, _RET_HIDDEN, bias=True),
            nn.GELU(),
            nn.Linear(_RET_HIDDEN, n_heads * 2, bias=False),
        )
        nn.init.zeros_(self.retention_mod[0].weight)
        nn.init.zeros_(self.retention_mod[0].bias)

        self.layer_sens = nn.Parameter(torch.zeros(N_SIG))

        self.S_init = nn.Parameter(torch.randn(n_heads, d_v, d_k) * 0.02)
        self.ln_ctx = nn.LayerNorm(d_v)

        self.Wv_n = nn.Linear(d_model, self.nucleus_dim, bias=False)
        self.Wa_n = nn.Linear(d_model, self.Hn, bias=True)
        nn.init.zeros_(self.Wa_n.weight)
        _a_n = torch.exp(-torch.linspace(-math.log(0.999), -math.log(0.95), self.Hn))
        self.Wa_n.bias.data.copy_(torch.log(_a_n / (1 - _a_n)))
        self.S0_n = nn.Parameter(torch.randn(self.Hn, self.Dhn) * 0.02)
        self.ln_nucleus = nn.LayerNorm(self.Dhn)
        self.Wm_a = nn.Linear(self.delta_dim, self.Hn, bias=False)
        nn.init.zeros_(self.Wm_a.weight)
        self.register_buffer('_wm_a_shift_rms', torch.tensor(1.0))
        self._wm_a_ema = ema_norm
        self._wm_a_rms_pending: torch.Tensor | None = None

        _combined = self.delta_dim + self.nucleus_dim
        self.Wg = nn.Linear(d_model, _combined, bias=False)
        self.Wo = nn.Linear(_combined, d_model, bias=False)

        self.meta = MetaController(d_model, signal_expand=signal_expand)
        self.register_buffer('phi6_slow', torch.tensor(0.90))

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_dn = nn.Linear(ffn_hidden, d_model, bias=False)

        self._last_local_sig: torch.Tensor | None = None
        self._last_all_sig: torch.Tensor | None = None
        self._last_gate_delta_mag: float = 0.0
        self._last_da_signed: torch.Tensor | None = None
        self._last_nuc_half_life: float = 0.0
        self._last_wm_a_rms: float = 0.0
        self._last_wm_a_shift_rms_tracker: float = 0.0
        self._last_gate_delta_frac: float = 0.0
        self._last_gate_nuc_frac: float = 0.0
        self._last_nuc_half_life_per_head: torch.Tensor | None = None
        self._diag: bool = True
        self._state_out: Tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def commit_trackers(self) -> None:
        """Applies the Wm_a RMS EMA update stashed by forward()."""
        if self._wm_a_rms_pending is not None:
            self._wm_a_shift_rms.mul_(self._wm_a_ema).add_(
                (1 - self._wm_a_ema) * self._wm_a_rms_pending)
            self._wm_a_rms_pending = None

    def _local_signals(self, ctx: torch.Tensor, log_a: torch.Tensor,
                       beta: torch.Tensor) -> torch.Tensor:
        B, T, N = ctx.shape
        flat = ctx.abs()
        p    = flat / (flat.sum(-1, keepdim=True) + 1e-9)
        phi1 = -(p * (p + 1e-9).log()).sum(-1) / math.log(N)
        phi2 = ctx.norm(dim=-1) / math.sqrt(N)
        alpha = torch.exp(log_a).transpose(1, 2)
        phi3  = alpha.mean(-1)
        phi4  = beta.transpose(1, 2).std(-1)
        phi5  = beta.transpose(1, 2).mean(-1)
        delta_cryst = phi3 - self.phi6_slow.float().view(1, 1).expand(B, T)
        return torch.stack([phi1, phi2, phi3, phi4, phi5, delta_cryst], dim=-1)

    def forward(self, h: torch.Tensor, global_ctx: torch.Tensor,
                colony_ctx: torch.Tensor, use_checkpoint: bool = False,
                S_delta0: torch.Tensor | None = None, S_nuc0: torch.Tensor | None = None,
                carry: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional carried recurrent state across windows."""
        if use_checkpoint:
            return grad_checkpoint(self._forward, h, global_ctx, colony_ctx,
                                   S_delta0, S_nuc0, carry, use_reentrant=False)
        return self._forward(h, global_ctx, colony_ctx, S_delta0, S_nuc0, carry)

    def _forward(self, h: torch.Tensor, global_ctx: torch.Tensor,
                 colony_ctx: torch.Tensor, S_delta0: torch.Tensor | None = None,
                 S_nuc0: torch.Tensor | None = None,
                 carry: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = h.shape
        H, dk, dv = self.H, self.dk, self.dv
        h_ln = self.ln(h)

        q = F.normalize(self.Wq(h_ln).view(B, T, H, dk), dim=-1).transpose(1, 2)
        k = F.normalize(self.Wk(h_ln).view(B, T, H, dk), dim=-1).transpose(1, 2)
        v = self.Wv(h_ln).view(B, T, H, dv).transpose(1, 2)

        prev_local = self.prev_local_sig.to(h_ln.dtype).view(1, 1, N_LOCAL).expand(B, T, N_LOCAL)
        retention_input = torch.cat([
            prev_local,
            global_ctx.to(h_ln.dtype),
            colony_ctx.to(h_ln.dtype),
        ], dim=-1)
        gate_delta = self.retention_mod(retention_input)
        da = gate_delta[..., :H]
        db = gate_delta[..., H:]
        if self._diag:
            self._last_gate_delta_mag = gate_delta.detach().float().abs().mean()
            self._last_da_signed = da.detach().float().mean(dim=-1)

        log_a = F.logsigmoid(self.Wa(h_ln) + da).transpose(1, 2)
        beta  = torch.sigmoid(self.Wb(h_ln) + db).transpose(1, 2)

        S0 = self.S_init.unsqueeze(0).expand(B, H, dv, dk)
        if S_delta0 is not None:
            S0 = torch.where(carry.view(B, 1, 1, 1), S_delta0.to(S0.dtype), S0)

        with torch.autocast(device_type=h.device.type, enabled=False):
            ctx, S_delta_T = _gated_delta_chunked(
                q.float(), k.float(), v.float(),
                log_a.float(), beta.float(), S0.float(), C=self.C)
        ctx = ctx.to(h.dtype)

        ctx_view = self.ln_ctx(ctx.transpose(1, 2))
        ctx_flat = ctx_view.reshape(B, T, H * dv)  # [B,T,delta_dim]

        Hn, Dhn = self.Hn, self.Dhn
        v_n       = self.Wv_n(h_ln).view(B, T, Hn, Dhn).transpose(1, 2)
        _wm_a_raw = self.Wm_a(ctx_flat)  # [B,T,Hn], zero-init
        if self.training and torch.is_grad_enabled():
            self._wm_a_rms_pending = _wm_a_raw.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
        _wm_a_shift = _wm_a_raw / self._wm_a_shift_rms.clamp_min(1e-6).to(_wm_a_raw.dtype)
        log_a_n = F.logsigmoid(self.Wa_n(h_ln) + _wm_a_shift).transpose(1, 2)
        S0_n    = self.S0_n.unsqueeze(0).expand(B, Hn, Dhn)
        if S_nuc0 is not None:
            S0_n = torch.where(carry.view(B, 1, 1), S_nuc0.to(S0_n.dtype), S0_n)
        with torch.autocast(device_type=h.device.type, enabled=False):
            nuc, S_nuc_T = _nucleus_chunked(v_n.float(), log_a_n.float(), S0_n.float(), C=self.C)
        nuc = nuc.to(h.dtype)
        self._state_out = (S_delta_T.detach(), S_nuc_T.detach())
        nuc_flat = self.ln_nucleus(nuc.transpose(1, 2)).reshape(B, T, Hn * Dhn)

        combined = torch.cat([ctx_flat, nuc_flat], dim=-1)
        g     = torch.sigmoid(self.Wg(h_ln))
        h_ctx = h + self.Wo(g * combined)

        if self._diag:
          with torch.no_grad():
            _nl = (-log_a_n.detach().float()).clamp_min(1e-8)
            _hl = math.log(2.0) / _nl  # [B,Hn,T]
            self._last_nuc_half_life = _hl.mean()
            self._last_nuc_half_life_per_head = _hl.mean(dim=(0, 2))
            self._last_wm_a_rms = _wm_a_shift.detach().float().std()
            self._last_wm_a_shift_rms_tracker = self._wm_a_shift_rms.detach().clone()
            _gd = g.detach().float()
            self._last_gate_delta_frac = _gd[..., :self.delta_dim].mean()
            self._last_gate_nuc_frac   = _gd[..., self.delta_dim:].mean()

        local_sig = self._local_signals(ctx_flat.float(), log_a.float(), beta.float())
        self._last_local_sig = local_sig.detach()
        all_sig   = torch.cat([
            local_sig,
            global_ctx.to(local_sig.dtype),
            colony_ctx.to(local_sig.dtype),
        ], dim=-1)
        self._last_all_sig = all_sig.detach()

        h_mod = self.meta(h_ctx, all_sig, self.layer_sens.to(h.dtype))
        return h_mod + self.ffn_dn(F.gelu(self.ffn_up(self.ffn_ln(h_mod))))


def _select_read_slots(h: torch.Tensor, cat_id: torch.Tensor, cat_content: torch.Tensor,
                        mem_content: torch.Tensor, Wq: nn.Linear, Wk: nn.Linear,
                        content_ln: nn.LayerNorm,
                        cat_select_margin: float, chunk_size: int,
                        return_row_mask: bool = False,
                        doc_query: torch.Tensor | None = None):
    """Selects which memory slots are candidates for this read, via category
    routing. `doc_query` ([B,d_model], optional) overrides the per-position query."""
    C = mem_content.shape[0]
    M = cat_content.shape[0]
    full = torch.arange(C, device=mem_content.device)
    if M == 0 or M == 1:
        return (full, None, None) if return_row_mask else full
    if doc_query is None:
        T = h.shape[1]
        k = min(chunk_size, T)
        prefix = h[:, :k]
        doc_query = content_ln(prefix.mean(dim=1))  # [B,d_model]
    cat_k     = Wk(cat_content.to(doc_query.dtype))  # [M,d_k]
    q_cat     = Wq(doc_query)
    cat_sim   = F.cosine_similarity(q_cat.unsqueeze(1), cat_k.unsqueeze(0), dim=-1)  # [B,M]
    cat_empty_mask = torch.bincount(cat_id, minlength=M) == 0
    sel_mask  = _select_categories(cat_sim, cat_select_margin, cat_empty_mask)  # [B,M]
    selected  = sel_mask.any(dim=0)
    selected_idx = selected.nonzero(as_tuple=True)[0]
    member_mask  = torch.isin(cat_id, selected_idx)
    cand_idx     = member_mask.nonzero(as_tuple=True)[0]
    if cand_idx.numel() == 0:
        return (full, None, None) if return_row_mask else full
    if not return_row_mask:
        return cand_idx
    return cand_idx, sel_mask[:, cat_id[cand_idx]], sel_mask


class ReadRouter(nn.Module):
    """Reader-to-router distillation: trains a lightweight router to imitate
    the memory reader's own category choice."""

    def __init__(self, d_model: int, d_k: int) -> None:
        super().__init__()
        self.Wq_router = nn.Linear(d_model, d_k, bias=False)
        nn.init.normal_(self.Wq_router.weight, std=0.02)
        self.temp = nn.Parameter(torch.tensor(8.0))
        self.register_buffer('_lift_ema', torch.zeros(2))  # [incumbent, router]
        self.register_buffer('_n_obs', torch.zeros((), dtype=torch.long))
        self.register_buffer('_in_control', torch.zeros((), dtype=torch.bool))

    @staticmethod
    def assert_inputs_detached(*tensors: torch.Tensor) -> None:
        """Structural guard: router inputs must be detached from the main graph."""
        for t in tensors:
            if t.requires_grad:
                raise RuntimeError(
                    "ReadRouter isolation violated: an input to the router's "
                    "KL loss carries autograd history into shared parameters")

    def cos_scores(self, doc_query: torch.Tensor, cat_k_n: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between a detached query and detached category keys."""
        q = F.normalize(self.Wq_router(doc_query.float()), dim=-1)
        return q @ cat_k_n.float().T


class ScanESNContext(nn.Module):
    """ESN-style context from a bank of leaky-integrator slots."""

    def __init__(self, d_model: int, n_slots: int, chunk: int = 128) -> None:
        super().__init__()
        self.n_slots      = n_slots
        self.d_model      = d_model
        self.chunk        = chunk
        self._diag: bool  = True
        self.Wd           = nn.Linear(d_model, n_slots,           bias=False)
        self.Ww           = nn.Linear(d_model, n_slots,           bias=False)
        self.Wv           = nn.Linear(d_model, n_slots * d_model, bias=False)
        self.ln_e         = nn.LayerNorm(d_model)
        self.tau          = nn.Parameter(torch.tensor(1.0))
        self.ln_ctx       = nn.LayerNorm(d_model)
        self.essence_init = nn.Parameter(torch.zeros(n_slots, d_model))  # learned init
        self._last_diag: Dict[str, torch.Tensor] | None = None
        self._state_out: torch.Tensor | None = None

    def forward(self, h_in: torch.Tensor, s0: torch.Tensor | None = None,
                carry: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass. `s0` [B,K,d] is the raw prior state, if any."""
        out_dtype = h_in.dtype
        with torch.autocast(device_type=h_in.device.type, enabled=False):
            h = h_in.float()
            B, T, _ = h.shape

            write = torch.sigmoid(self.Ww(h))  # [B,T,K]
            v_all = self.Wv(h).view(B, T, self.n_slots, self.d_model)  # [B,T,K,d]

            b = write.unsqueeze(-1) * v_all  # [B,T,K,d]

            log_a = F.logsigmoid(self.Wd(h) + 3.0)
            h_init      = self.essence_init.view(1, 1, self.n_slots, self.d_model)
            S0 = self.essence_init.unsqueeze(0).expand(B, -1, -1)  # [B,K,d]
            init_slice = h_init.expand(B, 1, -1, -1)  # [B,1,K,d]
            if s0 is not None:
                cmask = carry.view(B, 1, 1)
                S0 = torch.where(cmask, s0.float(), S0)
                init_slice = torch.where(cmask.unsqueeze(1),
                                         self.ln_e(s0.float()).unsqueeze(1), init_slice)
            essence_raw = _decay_scan_chunked(
                b.permute(0, 2, 1, 3), log_a.transpose(1, 2), S0, self.chunk
            ).permute(0, 2, 1, 3)  # [B,T,K,d]
            self._state_out = essence_raw[:, -1].detach()  # raw end state
            essence_all = self.ln_e(essence_raw)

            essence_prev = torch.cat([init_slice, essence_all[:, :-1]], dim=1)  # [B,T,K,d]

            tau    = self.tau.clamp(0.1, 10.0)
            h_norm = F.normalize(h, dim=-1)
            scores = tau * torch.einsum('btd,btkd->btk', h_norm, essence_prev)
            attn   = torch.softmax(scores, dim=-1)  # [B,T,K]
            ctx    = torch.einsum('btk,btkd->btd', attn, essence_prev)  # [B,T,d]
            out    = self.ln_ctx(ctx)  # [B,T,d]

            if self._diag:
              with torch.no_grad():
                ent = -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean()
                decay = log_a.exp()
                self._last_diag = {
                    "esn/slot_entropy_norm": ent / math.log(max(2, self.n_slots)),
                    "esn/decay_mean":        decay.mean(),
                    "esn/horizon_mean":      (1.0 / (1.0 - decay).clamp_min(1e-6)).mean(),
                    "esn/tau":               tau.detach(),
                }
        return out.to(out_dtype)


def mem_trunk_k(cfg: "Config") -> int:
    """Depth (number of ComboBlocks) of the shared memory-write trunk."""
    return max(1, round(cfg.n_blocks * cfg.mem_trunk_frac))


class SenceliumModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg    = cfg
        self.emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln_in  = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([
            ComboBlock(cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v,
                       cfg.n_heads_n, cfg.d_hn,
                       cfg.ffn_hidden, cfg.chunk_size,
                       signal_expand=cfg.signal_expand)
            for _ in range(cfg.n_blocks)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

        self.emotion_head = EmotionHead(N_SIG, cfg.n_emotion, hidden=cfg.d_model * 4)
        mem_d_k = cfg.d_model * cfg.mem_d_k_mult
        mem_d_v = cfg.d_model
        propose_input_width = cfg.d_model + cfg.n_emotion + mem_d_v
        propose_hidden = max(4 * cfg.d_model,
                             -(-propose_input_width // cfg.d_model) * cfg.d_model)
        self.voice_loop    = InnerVoiceLoop(cfg.d_model, mem_d_v,
                                            cfg.n_emotion, cfg.voice_rounds,
                                            propose_hidden=propose_hidden)
        self.mem_pool      = MemoryPool(cfg.d_model, mem_d_k, mem_d_v,
                                        spawn_bias_margin_seed=cfg.mem_spawn_bias_margin_seed,
                                        spawn_bias_clamp=cfg.mem_spawn_bias_clamp)
        self._last_read_entropy      = torch.zeros(())
        self._last_read_entropy_norm = torch.zeros(())

        if cfg.use_router_distill:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + 1)
                self.read_router = ReadRouter(cfg.d_model, mem_d_k)
        else:
            self.read_router = None
        self._last_router_loss: torch.Tensor | None = None
        self._last_router_diag: Dict[str, float] | None = None
        self._router_per_doc_diag: bool = False
        self._diag: bool = True
        self._router_degenerate_skips: int = 0

        self.esn_residual = ScanESNContext(cfg.d_model, cfg.n_slots_esn, chunk=cfg.chunk_size)
        self.esn_alpha    = nn.Parameter(torch.zeros(1))
        self._last_esn_rel_norm: torch.Tensor | None = None
        self._trunk_k = mem_trunk_k(cfg)
        self._last_trunk_h: torch.Tensor | None = None
        self._last_trunk_sig: torch.Tensor | None = None

    def set_diagnostics(self, on: bool) -> None:
        """Turns diagnostic bookkeeping on/off across all submodules."""
        for m in self.modules():
            if hasattr(m, "_diag"):
                m._diag = on

    def _route_with_router(self, h: torch.Tensor, cat_id: torch.Tensor,
                            cat_content: torch.Tensor, mem_content: torch.Tensor,
                            cand_idx_inc: torch.Tensor,
                            doc_query_rows: torch.Tensor | None = None):
        """Routes via the lightweight ReadRouter instead of the full reader."""
        C = mem_content.shape[0]
        M = cat_content.shape[0]
        if C == 0 or M <= 1:
            return cand_idx_inc, None
        router = self.read_router
        k = min(self.cfg.chunk_size, h.shape[1])
        cat_empty = torch.bincount(cat_id, minlength=M) == 0
        if int((~cat_empty).sum().item()) <= 1:
            self._router_degenerate_skips += 1
            return cand_idx_inc, None
        with torch.no_grad():
            if doc_query_rows is not None:
                doc_query = doc_query_rows.detach().float()
            else:
                doc_query = self.mem_pool.content_ln(h[:, :k].mean(dim=1)).detach().float()
            cat_k_n = F.normalize(
                self.mem_pool.Wk(cat_content.to(h.dtype)).float(), dim=-1)  # [M,d_k], no graph
        ReadRouter.assert_inputs_detached(doc_query, cat_k_n)
        with torch.autocast(device_type=h.device.type, enabled=False):
            cos = router.cos_scores(doc_query, cat_k_n)
        with torch.no_grad():
            sel = _select_categories(cos.detach(), self.cfg.mem_cat_select_margin, cat_empty)  # [B,M]
            union = sel.any(dim=0)
            member = union[cat_id]  # [C]
            cand_r = member.nonzero(as_tuple=True)[0]
            if cand_r.numel() == 0:
                cand_r = torch.arange(C, device=mem_content.device)
                row_mask_r = None
                sel_used_r = None
            else:
                row_mask_r = sel[:, cat_id[cand_r]]
                sel_used_r = sel
        use_router = bool(router._in_control.item())
        state = dict(cos=cos, sel=sel, cat_empty=cat_empty, cand_r=cand_r,
                     cand_inc=cand_idx_inc, used_router=use_router,
                     row_mask_r=row_mask_r, sel_used_r=sel_used_r,
                     doc_query=doc_query)
        return (cand_r if use_router else cand_idx_inc), state

    def _router_distill(self, state: dict, teacher_w: torch.Tensor, cat_id: torch.Tensor,
                         h: torch.Tensor, mem_content: torch.Tensor, cat_content: torch.Tensor,
                         rows: torch.Tensor | None = None) -> None:
        """Trains the router via KL distillation against the real reader;
        also logs recall/lift diagnostics."""
        router = self.read_router
        C = mem_content.shape[0]
        M = cat_content.shape[0]
        if rows is not None:
            teacher_w = teacher_w[rows]
            h = h[rows]
            state = dict(state, cos=state["cos"][rows], sel=state["sel"][rows],
                         doc_query=state["doc_query"][rows])
        with torch.no_grad():
            p_slot = teacher_w.detach().float().sum(dim=1)  # [B,C]
            p_slot = p_slot / p_slot.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            p_cat = torch.zeros(p_slot.shape[0], M, device=p_slot.device)
            p_cat.index_add_(1, cat_id, p_slot)  # [B,M]
        ReadRouter.assert_inputs_detached(p_slot, p_cat)
        cat_empty = state["cat_empty"]
        cos = state["cos"]
        temp = router.temp.clamp(1.0, 32.0)
        logits = (cos * temp).masked_fill(cat_empty.unsqueeze(0), float('-inf'))
        log_q = F.log_softmax(logits, dim=-1).masked_fill(cat_empty.unsqueeze(0), 0.0)
        ce = -(p_cat * log_q).sum(dim=-1).mean()
        self._last_router_loss = ce

        with torch.no_grad():
            H_p = -(p_cat * p_cat.clamp_min(1e-12).log()).sum(dim=-1).mean()
            cat_sizes = torch.bincount(cat_id, minlength=M).float()
            cand_inc, cand_r = state["cand_inc"], state["cand_r"]
            rec_inc = p_slot[:, cand_inc].sum(dim=-1).mean()
            rec_r   = p_slot[:, cand_r].sum(dim=-1).mean()
            frac_inc = cand_inc.numel() / C
            frac_r   = cand_r.numel() / C
            lift = torch.stack([rec_inc - frac_inc, rec_r - frac_r])
            a = self.cfg.ema_norm
            router._lift_ema.copy_(torch.where(router._n_obs == 0, lift,
                                               a * router._lift_ema + (1 - a) * lift))
            router._n_obs += 1
            warm = math.ceil(1.0 / (1.0 - a))
            router._in_control.copy_((router._n_obs >= warm)
                                     & (router._lift_ema[1] > router._lift_ema[0]))
            if not self._diag:
                self._last_router_diag = None
                return
            diag = {
                "router/kl": (ce.detach() - H_p).item(),
                "router/recall_union_incumbent": rec_inc.item(),
                "router/recall_union_router": rec_r.item(),
                "router/cand_frac_incumbent": frac_inc,
                "router/cand_frac_router": frac_r,
                "router/lift_ema_incumbent": router._lift_ema[0].item(),
                "router/lift_ema_router": router._lift_ema[1].item(),
                "router/used_router": float(state["used_router"]),
                "router/in_control_next": float(router._in_control.item()),
                "router/temp": temp.item(),
                "router/teacher_entropy_norm": (H_p / math.log(max(int((~cat_empty).sum().item()), 2))).item(),
            }
            if self._router_per_doc_diag:
                rec_r_doc = (p_cat * state["sel"].float()).sum(dim=-1)  # [B]
                frac_r_doc = (state["sel"].float() @ cat_sizes) / C  # [B]
                rec_i_doc, frac_i_doc = [], []
                for b in range(h.shape[0]):
                    ci = _select_read_slots(h[b:b + 1], cat_id, cat_content, mem_content,
                                            self.mem_pool.Wq, self.mem_pool.Wk,
                                            self.mem_pool.content_ln,
                                            self.cfg.mem_cat_select_margin, self.cfg.chunk_size,
                                            doc_query=state["doc_query"][b:b + 1])
                    rec_i_doc.append(p_slot[b, ci].sum())
                    frac_i_doc.append(ci.numel() / C)
                diag["router/recall_doc_incumbent"] = torch.stack(rec_i_doc).mean().item()
                diag["router/recall_doc_router"] = rec_r_doc.mean().item()
                diag["router/cand_frac_doc_incumbent"] = sum(frac_i_doc) / len(frac_i_doc)
                diag["router/cand_frac_doc_router"] = frac_r_doc.mean().item()
            self._last_router_diag = diag

    def forward(self, idx: torch.Tensor, global_ctx: torch.Tensor,
                colony_ctx: torch.Tensor, mem_content: torch.Tensor,
                cat_id: torch.Tensor, cat_content: torch.Tensor,
                state: Dict[str, object] | None = None,
                carry: torch.Tensor | None = None,
                return_state: bool = False,
                ):
        """Main model forward pass: backbone, inner voice loop, memory read,
        optional carried recurrent state."""
        h = self.ln_in(self.emb(idx))
        B, T = idx.shape
        stateful = state is not None
        if stateful:
            assert carry is not None and carry.dtype == torch.bool and carry.shape == (B,), \
                "stateful forward needs a [B] bool carry mask"
            cnt0 = torch.where(carry, state["col_cnt"].to(carry.device).float(),
                               torch.zeros((), device=carry.device))  # [B]
        new_delta: List[torch.Tensor] = []
        new_nuc: List[torch.Tensor] = []
        new_col_sum: List[torch.Tensor] = []
        is_lm_train = self.training and torch.is_grad_enabled()

        current_colony = colony_ctx
        α = self.cfg.colony_live_alpha

        for i, blk in enumerate(self.blocks):
            if stateful:
                h = blk(h, global_ctx, current_colony,
                        use_checkpoint=self.cfg.use_checkpoint,
                        S_delta0=state["delta"][i], S_nuc0=state["nuc"][i], carry=carry)
            else:
                h = blk(h, global_ctx, current_colony,
                        use_checkpoint=self.cfg.use_checkpoint)
            if return_state:
                new_delta.append(blk._state_out[0])
                new_nuc.append(blk._state_out[1])
            if i == self._trunk_k - 1 and is_lm_train:
                self._last_trunk_h = h.detach()
                self._last_trunk_sig = blk._last_all_sig

            if blk._last_local_sig is not None:
                local_full  = blk._last_local_sig
                counts      = torch.arange(
                    1, T + 1, device=local_full.device, dtype=local_full.dtype
                ).view(1, T, 1)
                csum        = local_full.cumsum(dim=1)
                if stateful:
                    sum0   = torch.where(carry.view(B, 1),
                                         state["col_sum"][i].to(local_full.dtype),
                                         torch.zeros((), device=local_full.device,
                                                     dtype=local_full.dtype))
                    csum   = csum + sum0.unsqueeze(1)
                    counts = counts + cnt0.to(local_full.dtype).view(B, 1, 1)
                if return_state:
                    new_col_sum.append(csum[:, -1].detach().float())
                causal_mean = csum / counts
                diversity   = causal_mean.std(dim=-1, keepdim=True)
                colony_new  = torch.cat([causal_mean, diversity], dim=-1)
                colony_new  = colony_new.to(dtype=current_colony.dtype,
                                            device=current_colony.device)
                current_colony = α * current_colony + (1 - α) * colony_new

        meta  = self.blocks[-1].meta
        sig_t = self.blocks[-1]._last_all_sig
        sig_n = ((sig_t - meta.sig_mean) / (meta.sig_std + 1e-6)).clamp(-5.0, 5.0)
        e_t   = self.emotion_head(sig_n.to(h.dtype))  # [B,T,n_emotion]

        k = min(self.cfg.chunk_size, T)
        doc_query = self.mem_pool.content_ln(h[:, :k].mean(dim=1))  # [B,d_model]
        if stateful and state.get("query") is not None:
            doc_query = torch.where(carry.view(B, 1),
                                    state["query"].to(doc_query.dtype), doc_query)
        cand_idx, row_mask, sel_mask = _select_read_slots(
            h, cat_id, cat_content, mem_content,
            self.mem_pool.Wq, self.mem_pool.Wk, self.mem_pool.content_ln,
            self.cfg.mem_cat_select_margin, self.cfg.chunk_size,
            return_row_mask=True, doc_query=doc_query)
        router_state = None
        teacher_box: List[torch.Tensor | None] = [None]
        if self.read_router is not None:
            self._last_router_loss = None
            self._last_router_diag = None
            cand_idx, router_state = self._route_with_router(
                h, cat_id, cat_content, mem_content, cand_idx, doc_query_rows=doc_query)
            if router_state is not None and router_state["used_router"]:
                row_mask = router_state["row_mask_r"]
                sel_mask = router_state["sel_used_r"]
        prefix_mask = None
        if stateful and sel_mask is not None and mem_content.shape[0] > 0:
            prefix_mask = torch.where(carry.view(B, 1), sel_mask[:, cat_id],
                                      torch.ones((), dtype=torch.bool, device=carry.device))

        C_pool = mem_content.shape[0]
        mem_K_full, mem_V_full = self.mem_pool.project(mem_content, h.dtype)
        if cand_idx.numel() == mem_content.shape[0]:
            mem_K_sub, mem_V_sub = mem_K_full, mem_V_full
        else:
            mem_K_sub, mem_V_sub = self.mem_pool.project(mem_content[cand_idx], h.dtype)

        def _read(u: torch.Tensor):
            out_prefix, w_prefix = self.mem_pool.attend(u[:, :k], mem_K_full, mem_V_full,
                                                        row_mask=prefix_mask)
            out_rest,   w_rest   = self.mem_pool.attend(u[:, k:], mem_K_sub,  mem_V_sub,
                                                        row_mask=row_mask)
            if router_state is not None:
                teacher_box[0] = w_prefix[..., :C_pool]
            out = torch.cat([out_prefix, out_rest], dim=1)
            return out, w_rest
        u, last_w = self.voice_loop(h, e_t, _read)  # [B,T,d_model]

        if (router_state is not None and teacher_box[0] is not None
                and self.training and torch.is_grad_enabled()):
            rows = None
            if stateful:
                rows = (~carry).nonzero(as_tuple=True)[0]
            if rows is None or rows.numel() > 0:
                self._router_distill(router_state, teacher_box[0], cat_id, h,
                                     mem_content, cat_content, rows=rows)

        L_div = torch.zeros((), device=h.device, dtype=torch.float32)
        if last_w is not None and B > 1 and last_w.shape[1] > 0:
            w_doc = last_w.mean(dim=1).float()  # [B,C_sub]
            w_doc_n = F.normalize(w_doc, dim=-1, eps=1e-6)
            sim_matrix = w_doc_n @ w_doc_n.T  # [B,B], cosine sims
            off_diag = ~torch.eye(B, dtype=torch.bool, device=h.device)
            L_div = sim_matrix[off_diag].mean()

        if not self._diag:
            pass
        elif last_w is not None and last_w.shape[1] > 0 and last_w.shape[-1] > 1:
            w_ent = last_w.float()
            w_ent = (w_ent / w_ent.sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9)
            ent = -(w_ent * w_ent.log()).sum(-1)  # [B,T-k]
            self._last_read_entropy = ent.mean().detach()
            if row_mask is None:
                self._last_read_entropy_norm = (
                    self._last_read_entropy / math.log(last_w.shape[-1])
                ).detach()
            else:
                _n = row_mask.sum(-1).clamp_min(2).float().log()  # [B]
                self._last_read_entropy_norm = (ent / _n.unsqueeze(-1)).mean().detach()
        else:
            self._last_read_entropy = torch.zeros((), device=h.device)
            self._last_read_entropy_norm = torch.zeros((), device=h.device)

        u_ln = self.ln_out(u)
        if stateful:
            esn_out = self.esn_alpha * self.esn_residual(u_ln, s0=state["esn"], carry=carry)
        else:
            esn_out = self.esn_alpha * self.esn_residual(u_ln)
        if self._diag:
            with torch.no_grad():
                self._last_esn_rel_norm = (esn_out.float().norm()
                                           / u_ln.float().norm().clamp_min(1e-6))
        u_ln = u_ln + esn_out
        logits = self.head(u_ln)
        if not return_state:
            return logits, u, e_t, L_div
        col_cnt = (cnt0 if stateful else torch.zeros(B, device=idx.device)) + float(T)
        new_state = {
            "delta": new_delta, "nuc": new_nuc,
            "col_sum": new_col_sum, "col_cnt": col_cnt.detach(),
            "esn": self.esn_residual._state_out,
            "query": doc_query.detach().float(),
        }
        return logits, u, e_t, L_div, new_state

    def forward_trunk(self, idx: torch.Tensor, global_ctx: torch.Tensor,
                       colony_ctx: torch.Tensor, k_blocks: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs only the first `k_blocks` of the backbone (no voice loop/head),
        used to produce the representation memory writes are based on."""
        h = self.ln_in(self.emb(idx))
        current_colony = colony_ctx
        alpha = self.cfg.colony_live_alpha
        T = idx.shape[1]
        for blk in self.blocks[:k_blocks]:
            h = blk(h, global_ctx, current_colony, use_checkpoint=False)
            if blk._last_local_sig is not None:
                local_full  = blk._last_local_sig
                counts      = torch.arange(
                    1, T + 1, device=local_full.device, dtype=local_full.dtype
                ).view(1, T, 1)
                causal_mean = local_full.cumsum(dim=1) / counts
                diversity   = causal_mean.std(dim=-1, keepdim=True)
                colony_new  = torch.cat([causal_mean, diversity], dim=-1)
                colony_new  = colony_new.to(dtype=current_colony.dtype,
                                            device=current_colony.device)
                current_colony = alpha * current_colony + (1 - alpha) * colony_new

        last_blk = self.blocks[k_blocks - 1]
        meta  = last_blk.meta
        sig_t = last_blk._last_all_sig
        sig_n = ((sig_t - meta.sig_mean) / (meta.sig_std + 1e-6)).clamp(-5.0, 5.0)
        e_t   = self.emotion_head(sig_n.to(h.dtype))
        return h, e_t


class LaneStream:
    """Packs an iterator of token-id documents into `n_lanes` continuous."""

    def __init__(self, docs: Iterator[List[int]], n_lanes: int, seq_len: int,
                 lane_offset: int = 0) -> None:
        self.docs = docs
        self.B, self.T = n_lanes, seq_len
        self.lane_offset = lane_offset
        self._buf: List[List[int]] = [[] for _ in range(n_lanes)]
        self._dbuf: List[List[int]] = [[] for _ in range(n_lanes)]
        self._prev_start: List[int | None] = [None] * n_lanes
        self._win = [0] * n_lanes
        self._next_doc = 0

    def _fill(self, b: int) -> None:
        while len(self._buf[b]) < self.T + 1:
            ids = next(self.docs)
            self._buf[b].extend(ids)
            self._dbuf[b].extend([self._next_doc] * len(ids))
            self._next_doc += 1

    def next_batch(self) -> Tuple[torch.Tensor, ...]:
        T = self.T
        xs, ys, ds, carry, win = [], [], [], [], []
        for b in range(self.B):
            self._fill(b)
            buf, dbuf = self._buf[b], self._dbuf[b]
            xs.append(buf[:T])
            ys.append(buf[1:T + 1])
            ds.append(dbuf[:T])
            carry.append(self._prev_start[b] is not None and dbuf[0] == self._prev_start[b])
            self._prev_start[b] = dbuf[0]
            win.append(self._win[b])
            self._win[b] += 1
            del buf[:T]
            del dbuf[:T]
        return (torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long),
                torch.tensor(carry, dtype=torch.bool),
                torch.arange(self.B, dtype=torch.long) + self.lane_offset,
                torch.tensor(win, dtype=torch.long), torch.tensor(ds, dtype=torch.long))


class FineWebEduLaneDataset(IterableDataset):
    """Yields whole [n_lanes, seq_len] lane batches (use with."""

    def __init__(self, tokenizer: GPT2TokenizerFast, seq_len: int, n_lanes: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.n_lanes   = n_lanes

    def _docs(self, ds) -> Iterator[List[int]]:
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text)
            if ids:
                yield ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        info = torch.utils.data.get_worker_info()
        wid = 0
        if info is not None:
            ds = ds.shard(num_shards=info.num_workers, index=info.id)
            wid = info.id
        stream = LaneStream(self._docs(ds), self.n_lanes, self.seq_len,
                            lane_offset=wid * self.n_lanes)
        while True:
            try:
                batch = stream.next_batch()
            except StopIteration:
                return
            yield batch


class WikiText103ValDataset(IterableDataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="test", streaming=True)
        buffer: List[int] = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            buffer.extend(self.tokenizer.encode(text))
            while len(buffer) >= self.seq_len + 1:
                x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                y = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                yield x, y
                buffer = buffer[self.seq_len:]


class WikiText103LaneValDataset(IterableDataset):
    """WikiText-103 validation set laid out as lanes, for the stateful
    (carried-state) validation protocol."""

    _TITLE = re.compile(r"^ = [^=].* = \n?$")

    def __init__(self, tokenizer: GPT2TokenizerFast, seq_len: int, n_lanes: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.n_lanes   = n_lanes
        self._cache: Tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def _rows():
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="test", streaming=True)
        for example in ds:
            yield example.get("text", "")

    def _stream(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None:
            toks: List[int] = []
            docs: List[int] = []
            art = -1
            for text in self._rows():
                if not text.strip():
                    continue
                if art < 0 or self._TITLE.match(text):
                    art += 1
                ids = self.tokenizer.encode(text)
                toks.extend(ids)
                docs.extend([art] * len(ids))
            self._cache = (torch.tensor(toks, dtype=torch.long),
                           torch.tensor(docs, dtype=torch.long))
        return self._cache

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        toks, docs = self._stream()
        T, L = self.seq_len, self.n_lanes
        n_win = (toks.numel() - 1) // T
        if n_win == 0:
            return
        L = min(L, n_win)
        base, rem = divmod(n_win, L)
        sizes = [base + (1 if b < rem else 0) for b in range(L)]
        starts = [sum(sizes[:b]) for b in range(L)]
        for j in range(max(sizes)):
            lanes = [b for b in range(L) if sizes[b] > j]
            w = torch.tensor([starts[b] + j for b in lanes], dtype=torch.long)  # window index
            off = w.unsqueeze(1) * T + torch.arange(T + 1).unsqueeze(0)  # [B,T+1]
            xy = toks[off]
            d = docs[off[:, :T]]
            if j == 0:
                carry = torch.zeros(len(lanes), dtype=torch.bool)
            else:
                carry = docs[w * T] == docs[(w - 1) * T]
            yield (xy[:, :T].contiguous(), xy[:, 1:].contiguous(), carry,
                   torch.tensor(lanes, dtype=torch.long),
                   torch.full((len(lanes),), j, dtype=torch.long), d.contiguous())


def build_loaders(cfg: Config, tokenizer: GPT2TokenizerFast):
    """Builds the train and validation DataLoaders."""
    train_dl = DataLoader(FineWebEduLaneDataset(tokenizer, cfg.seq_len, cfg.batch_size),
                          batch_size=None, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(WikiText103ValDataset(tokenizer, cfg.seq_len),
                          batch_size=cfg.batch_size, num_workers=0)
    val_carry_dl = DataLoader(WikiText103LaneValDataset(tokenizer, cfg.seq_len, cfg.batch_size),
                              batch_size=None, num_workers=0)
    return train_dl, [val_dl, val_carry_dl]


class _LaneStore:
    """Per-lane carried recurrent state between consecutive windows."""

    _KEYS = ("delta", "nuc", "col_sum")

    def __init__(self) -> None:
        self.t: Dict[str, torch.Tensor] = {}
        self.last_win: torch.Tensor | None = None
        self.age: torch.Tensor | None = None
        self.broken: torch.Tensor | int = 0

    def _grow(self, n: int, like: Dict[str, object], device) -> None:
        cur = 0 if self.last_win is None else self.last_win.numel()
        if n <= cur:
            return
        flat = {"delta": torch.stack(like["delta"], 1), "nuc": torch.stack(like["nuc"], 1),
                "col_sum": torch.stack(like["col_sum"], 1), "col_cnt": like["col_cnt"],
                "esn": like["esn"], "query": like["query"]}
        for k, v in flat.items():
            new = torch.zeros((n,) + tuple(v.shape[1:]), dtype=torch.float32, device=device)
            if k in self.t:
                new[:cur] = self.t[k]
            self.t[k] = new
        lw = torch.full((n,), -2, dtype=torch.long, device=device)
        ag = torch.zeros(n, dtype=torch.long, device=device)
        if self.last_win is not None:
            lw[:cur] = self.last_win
            ag[:cur] = self.age
        self.last_win, self.age = lw, ag

    def gather(self, lane: torch.Tensor, win: torch.Tensor, carry: torch.Tensor):
        """-> (state | None, carry_valid [B] bool)."""
        if self.last_win is None:
            self.broken = self.broken + carry.sum()
            return None, torch.zeros_like(carry)
        n = self.last_win.numel()
        inb = lane < n
        li = lane.clamp_max(n - 1)
        valid = carry & inb & (self.last_win[li] == win - 1)
        self.broken = self.broken + (carry & ~valid).sum()  # on-device, no sync
        st = {k: self.t[k][li] for k in self.t}
        state = {"delta": list(st["delta"].unbind(1)), "nuc": list(st["nuc"].unbind(1)),
                 "col_sum": list(st["col_sum"].unbind(1)), "col_cnt": st["col_cnt"],
                 "esn": st["esn"], "query": st["query"]}
        return state, valid

    def scatter(self, lane: torch.Tensor, win: torch.Tensor, carry_used: torch.Tensor,
                new_state: Dict[str, object]) -> None:
        n_need = int(lane.max().item()) + 1
        self._grow(n_need, new_state, lane.device)
        flat = {"delta": torch.stack(new_state["delta"], 1), "nuc": torch.stack(new_state["nuc"], 1),
                "col_sum": torch.stack(new_state["col_sum"], 1), "col_cnt": new_state["col_cnt"],
                "esn": new_state["esn"], "query": new_state["query"]}
        for k, v in flat.items():
            self.t[k][lane] = v.detach().float()
        self.age[lane] = torch.where(carry_used, self.age[lane] + 1, torch.zeros_like(lane))
        self.last_win[lane] = win


class Lit(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg      = cfg
        self.model    = SenceliumModel(cfg)
        self.best_ppl: float = float("inf")
        self.last_ppl: float = float("inf")
        self._val_losses: List[float] = []
        self._val_losses_no_mem: List[float] = []

        self.register_buffer('_global_slow', torch.zeros(N_GLOBAL))
        self.register_buffer('_colony_pheromone', torch.zeros(N_LOCAL))
        self.register_buffer('_colony_diversity_buf', torch.zeros(1))

        self.register_buffer('_mem_content', torch.zeros(0, cfg.d_model))
        self.register_buffer('_mem_U', torch.zeros(0))
        self.register_buffer('_mem_emotion', torch.zeros(0, cfg.n_emotion))
        self.register_buffer('_mem_idle_steps', torch.zeros(0))
        self.register_buffer('_join_sim_history',
                              torch.zeros(cfg.mem_novelty_history_size))
        self.register_buffer('_join_sim_history_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('_total_docs_seen', torch.zeros((), dtype=torch.long))
        self.register_buffer('_mem_commit_count', torch.zeros((), dtype=torch.long))
        self._pending_spawn_decisions: deque = deque(maxlen=4096)
        self.register_buffer('_cat_id', torch.zeros(0, dtype=torch.long))
        self.register_buffer('_cat_content', torch.zeros(0, cfg.d_model))
        self.register_buffer('_cat_U', torch.zeros(0))
        self.register_buffer('_cat_emotion', torch.zeros(0, cfg.n_emotion))
        self.register_buffer('_cat_member_shadow', torch.zeros(0))
        self.register_buffer('_cat_top_shadow', torch.zeros(()))
        self.register_buffer('_cat_join_hist', torch.zeros(cfg.mem_novelty_history_size))
        self.register_buffer('_cat_join_hist_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('_emo_mag_mean', torch.tensor(cfg.emo_tau))
        self.register_buffer('_infer_event_size_ref',
                              torch.tensor(float(cfg.batch_size * cfg.seq_len)))
        self.register_buffer('_docs_since_spawn', torch.zeros((), dtype=torch.long))
        self._train_lanes = _LaneStore()
        self._val_lanes: Dict[str, _LaneStore] = {}
        self._last_doc: torch.Tensor | None = None
        self._last_carry_frac: float = 0.0
        self._last_carry_data_frac: float = 0.0
        self._last_write_events: int = 0
        self._last_write_tokens: int = 0
        self._val_carry_acc: Dict[str, float] = {}
        self._last_u: torch.Tensor | None = None
        self._last_e: torch.Tensor | None = None
        self._last_gate_effort: torch.Tensor | None = None
        self._last_spawn_mass: float = 0.0
        self._spawn_events: int = 0
        self._merge_events: int = 0
        self._last_c_sub_mean: float = 0.0
        self._last_k_nov_mean: float = 0.0
        self._last_drought_boost_mean: float = 0.0
        self._last_novelty_margin_max: float = 0.0
        self._last_novelty_margin_mean: float = 0.0
        self._last_cat_load_bias_mean: float = 0.0
        self._last_cat_load_bias_spread: float = 0.0
        self._last_cat_transform_events: int = 0
        self._last_decisions_dropped_reassigned: int = 0
        self._cat_transform_events_total: int = 0
        self._cat_decisions_dropped_reassigned_total: int = 0

        self._sig_acc:  List[torch.Tensor] = []
        self._gate_acc: List[float] = []
        self._step_count: int = 0

        self._router_isolation_verified: bool = False

    @property
    def _colony_diversity(self) -> float:
        return self._colony_diversity_buf.item()

    @_colony_diversity.setter
    def _colony_diversity(self, value) -> None:
        if torch.is_tensor(value):
            self._colony_diversity_buf.copy_(value.detach().reshape(1))
        else:
            self._colony_diversity_buf.fill_(float(value))

    def _build_global_ctx(self, B: int, T: int, device, dtype) -> torch.Tensor:
        vals = self._global_slow.to(device=device, dtype=dtype)
        return vals.view(1, 1, N_GLOBAL).expand(B, T, N_GLOBAL).contiguous()

    def _build_colony_ctx(self, B: int, T: int, device, dtype) -> torch.Tensor:
        pheromone = self._colony_pheromone.to(device=device, dtype=dtype)
        diversity = self._colony_diversity_buf.to(device=device, dtype=dtype)
        colony = torch.cat([pheromone, diversity])
        return colony.view(1, 1, N_COLONY).expand(B, T, N_COLONY).contiguous()

    def _step(self, batch):
        lane_batch = len(batch) > 2
        x, y = batch[0], batch[1]
        B, T = x.shape
        global_ctx = self._build_global_ctx(B, T, x.device, torch.float32)
        colony_ctx = self._build_colony_ctx(B, T, x.device, torch.float32)
        if lane_batch and self.training:
            _, _, carry_data, lane, win, doc = batch
            state, carry = None, None
            if self.cfg.carry_state:
                state, carry = self._train_lanes.gather(lane, win, carry_data)
            if state is None:
                carry = torch.zeros_like(carry_data)
            logits, u, e_t, L_div, new_state = self.model(
                x, global_ctx, colony_ctx, self._mem_content, self._cat_id, self._cat_content,
                state=state, carry=carry if state is not None else None, return_state=True)
            self._train_lanes.scatter(lane, win, carry, new_state)
            self._last_doc = doc
            self._last_carry_frac = carry.float().mean()
            self._last_carry_data_frac = carry_data.float().mean()
        else:
            logits, u, e_t, L_div = self.model(x, global_ctx, colony_ctx, self._mem_content,
                                                self._cat_id, self._cat_content)
            if self.training:
                self._last_doc = None
        if self.training:
            self._last_u = u.detach()
            self._last_e = e_t.detach()
            self._last_gate_effort = self.model.voice_loop._last_gate_effort.detach()

        ce_per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction='none')
        ce_loss = ce_per_token.mean()

        L_emo_corr = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.training:
            surprise = ce_per_token.detach().reshape(B, T).float()
            emotion_mag = e_t.abs().sum(-1).float()  # [B,T], HAS gradient
            a = emotion_mag.flatten() - emotion_mag.mean()
            b = surprise.flatten() - surprise.mean()
            L_emo_corr = (a * b).sum() / (a.norm() * b.norm() + 1e-6)

        return ce_loss, logits, L_div, L_emo_corr

    def training_step(self, batch, batch_idx):
        log_step = (self._step_count + 1) % self.cfg.log_every == 0
        self.model.set_diagnostics(log_step)
        if self.cfg.use_router_distill:
            self.model._router_per_doc_diag = log_step
        ce_loss, _, L_div, L_emo_corr = self._step(batch)
        loss = ce_loss + self.cfg.lambda_div * L_div - self.cfg.lambda_emo * L_emo_corr
        if self.cfg.use_router_distill:
            L_router = self.model._last_router_loss
            if L_router is not None:
                if not self._router_isolation_verified:
                    self._verify_router_isolation(L_router)
                loss = loss + self.cfg.lambda_router * L_router
            if log_step:
                self.log("router/degenerate_skips_total",
                         float(self.model._router_degenerate_skips), on_step=True, on_epoch=False)
            if log_step and self.model._last_router_diag is not None:
                for name, val in self.model._last_router_diag.items():
                    self.log(name, val, on_step=True, on_epoch=False)
                self.log("router/isolation_verified", float(self._router_isolation_verified),
                         on_step=True, on_epoch=False)
            self.model._last_router_loss = None
        if log_step:
            self.log("esn/alpha", self.model.esn_alpha.detach().float().item(),
                     on_step=True, on_epoch=False)
            if self.model._last_esn_rel_norm is not None:
                self.log("esn/rel_norm", float(self.model._last_esn_rel_norm),
                         on_step=True, on_epoch=False)
            if self.model.esn_residual._last_diag is not None:
                for name, val in self.model.esn_residual._last_diag.items():
                    self.log(name, float(val), on_step=True, on_epoch=False)
            st = self._train_lanes
            self.log("carry/frac", float(self._last_carry_frac), on_step=True, on_epoch=False)
            self.log("carry/data_frac", float(self._last_carry_data_frac), on_step=True, on_epoch=False)
            self.log("carry/broken_total", float(st.broken), on_step=True, on_epoch=False)
            if st.age is not None:
                self.log("carry/age_mean", st.age.float().mean().item(), on_step=True, on_epoch=False)
                self.log("carry/age_max", float(st.age.max().item()), on_step=True, on_epoch=False)
                self.log("carry/delta_state_rms", st.t["delta"].pow(2).mean().sqrt().item(),
                         on_step=True, on_epoch=False)
                self.log("carry/nuc_state_rms", st.t["nuc"].pow(2).mean().sqrt().item(),
                         on_step=True, on_epoch=False)
                self.log("carry/esn_state_rms", st.t["esn"].pow(2).mean().sqrt().item(),
                         on_step=True, on_epoch=False)
        self.log("train_loss", ce_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("aux/diversity_loss", L_div, on_step=True, on_epoch=False)
        self.log("aux/emotion_surprise_corr", L_emo_corr, on_step=True, on_epoch=False)
        self._step_count += 1
        return loss

    def _verify_router_isolation(self, L_router: torch.Tensor) -> None:
        """Verifies the router's KL loss reaches only its own parameters."""
        router_ids = {id(p) for p in self.model.read_router.parameters()}
        others = [(n, p) for n, p in self.named_parameters()
                  if p.requires_grad and id(p) not in router_ids]
        g_others = torch.autograd.grad(L_router, [p for _, p in others],
                                       retain_graph=True, allow_unused=True)
        leaked = [n for (n, _), g in zip(others, g_others)
                  if g is not None and bool(g.abs().max() > 0)]
        if leaked:
            raise RuntimeError(f"ReadRouter isolation violated — router KL loss reaches "
                               f"shared parameters: {leaked[:10]}")
        g_own = torch.autograd.grad(L_router, [self.model.read_router.Wq_router.weight],
                                    retain_graph=True, allow_unused=True)[0]
        if g_own is None or not bool(g_own.abs().max() > 0):
            raise RuntimeError("ReadRouter isolation check is vacuous: Wq_router got no gradient")
        self._router_isolation_verified = True
        print(f"[router] isolation verified at step {self._step_count}: "
              f"{len(others)} non-router params get no gradient from the KL loss", flush=True)

    def on_before_optimizer_step(self, optimizer):
        """Logs router/total gradient norms; no behavior change."""
        if not self.cfg.use_router_distill or self._step_count % self.cfg.log_every != 0:
            return
        with torch.no_grad():
            r = [p.grad for p in self.model.read_router.parameters() if p.grad is not None]
            a = [p.grad for p in self.parameters() if p.grad is not None]
            rn = torch.sqrt(sum((g.float() ** 2).sum() for g in r)) if r else torch.zeros(())
            tn = torch.sqrt(sum((g.float() ** 2).sum() for g in a)) if a else torch.zeros(())
        self.log("router/grad_norm", float(rn), on_step=True, on_epoch=False)
        self.log("router/total_grad_norm", float(tn), on_step=True, on_epoch=False)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Clips the router's gradients separately from the main model's."""
        if not self.cfg.use_router_distill:
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm)
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return
        if gradient_clip_algorithm not in (None, "norm"):
            raise NotImplementedError("router arm implements norm clipping only "
                                      "(the only algorithm this file's Trainer uses)")
        r_ids = {id(p) for p in self.model.read_router.parameters()}
        shared = [p for p in self.parameters() if id(p) not in r_ids and p.grad is not None]
        own = [p for p in self.model.read_router.parameters() if p.grad is not None]
        if shared:
            torch.nn.utils.clip_grad_norm_(shared, gradient_clip_val)
        if own:
            torch.nn.utils.clip_grad_norm_(own, gradient_clip_val)


    def _update_memory_pool(self) -> None:
        """Sencelium: WRITE + dynamic slot growth for the PRETRAINING."""
        cfg = self.cfg
        device = self._mem_content.device
        h_tr = self.model._last_trunk_h
        sig_tr = self.model._last_trunk_sig
        if h_tr is None or sig_tr is None:
            return
        self.model._last_trunk_h = None
        self.model._last_trunk_sig = None
        B, T, _ = h_tr.shape
        with torch.no_grad():
            last_blk = self.model.blocks[self.model._trunk_k - 1]
            meta = last_blk.meta
            sig_n = ((sig_tr - meta.sig_mean) / (meta.sig_std + 1e-6)).clamp(-5.0, 5.0)
            with torch.autocast(device_type=h_tr.device.type, dtype=torch.bfloat16,
                                enabled=h_tr.device.type == "cuda"):
                e_tr = self.model.emotion_head(sig_n.to(h_tr.dtype))
        effort = self._last_gate_effort
        if effort is None or effort.shape != (B, T):
            effort = torch.zeros(B, T, device=h_tr.device)
        segments: List[Tuple[int, int, int]] = []
        if self._last_doc is not None and self._last_doc.shape == (B, T):
            cut = (self._last_doc[:, 1:] != self._last_doc[:, :-1]).nonzero().tolist()
            per_row: List[List[int]] = [[] for _ in range(B)]
            for r, c in cut:
                per_row[r].append(c + 1)
            for r in range(B):
                edges = [0] + per_row[r] + [T]
                segments += [(r, edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
        else:
            segments = [(r, 0, T) for r in range(B)]
        segments = [sg for sg in segments if sg[2] - sg[1] >= 2]
        self._last_write_events = len(segments)
        self._last_write_tokens = sum(e - s for _, s, e in segments)

        spawn_shares: List[float] = []
        c_sub_vals: List[int] = []
        k_nov_vals: List[int] = []
        drought_boost_vals: List[float] = []
        novelty_margin_vals: List[float] = []
        cat_load_bias_vals: List[float] = []
        cat_load_bias_spread_vals: List[float] = []
        step_transform_events = 0
        step_decisions_dropped = 0
        for r, s, e in segments:
            u_flat = h_tr[r, s:e].reshape(-1, cfg.d_model).float()
            e_flat = e_tr[r, s:e].reshape(-1, cfg.n_emotion).float()
            gate_effort_flat = effort[r, s:e].reshape(-1).float()
            cat = CategoryState(self._cat_content, self._cat_U, self._cat_member_shadow,
                                 self._cat_join_hist, self._cat_join_hist_count,
                                 self._cat_top_shadow, self._cat_emotion)
            new_content, new_U, new_emotion, new_idle_steps, new_cat_id, stats = apply_write_event(
                self.model.mem_pool, self._mem_content, self._mem_U, self._mem_emotion,
                self._mem_idle_steps,
                self._join_sim_history, self._join_sim_history_count,
                self.model.mem_pool.spawn_bias, self._pending_spawn_decisions, self._total_docs_seen,
                self._mem_commit_count, self._cat_id, cat,
                u_flat, e_flat, gate_effort_flat, cfg,
                lr_cap=cfg.mem_max_lr, emo_threshold=self._emo_mag_mean.item(),
                docs_since_spawn=self._docs_since_spawn, mass_floor=0.0)
            self._mem_content = new_content
            self._mem_U = new_U
            self._mem_emotion = new_emotion
            self._mem_idle_steps = new_idle_steps
            self._cat_id = new_cat_id
            self._cat_content, self._cat_U = cat.content, cat.U
            self._cat_member_shadow = cat.member_shadow
            self._cat_emotion = cat.emotion
            spawn_shares.append(stats.spawn_share)
            c_sub_vals.append(stats.c_sub)
            k_nov_vals.append(stats.k_nov)
            drought_boost_vals.append(stats.drought_boost)
            novelty_margin_vals.append(stats.novelty_margin)
            cat_load_bias_vals.append(stats.cat_load_bias)
            cat_load_bias_spread_vals.append(stats.cat_load_bias_spread)
            step_transform_events += stats.cat_transform_events
            step_decisions_dropped += stats.decisions_dropped_reassigned
            self._cat_transform_events_total += stats.cat_transform_events
            self._cat_decisions_dropped_reassigned_total += stats.decisions_dropped_reassigned
            if stats.spawned:
                self._spawn_events += 1
            if stats.merged:
                self._merge_events += 1
        self._last_spawn_mass = sum(spawn_shares) / max(len(spawn_shares), 1)
        self._last_c_sub_mean = sum(c_sub_vals) / max(len(c_sub_vals), 1)
        self._last_k_nov_mean = sum(k_nov_vals) / max(len(k_nov_vals), 1)
        self._last_drought_boost_mean = sum(drought_boost_vals) / max(len(drought_boost_vals), 1)
        if novelty_margin_vals:
            self._last_novelty_margin_max = max(novelty_margin_vals)
            self._last_novelty_margin_mean = sum(novelty_margin_vals) / len(novelty_margin_vals)
        self._last_cat_load_bias_mean = sum(cat_load_bias_vals) / max(len(cat_load_bias_vals), 1)
        self._last_cat_load_bias_spread = sum(cat_load_bias_spread_vals) / max(len(cat_load_bias_spread_vals), 1)
        self._last_cat_transform_events = step_transform_events
        self._last_decisions_dropped_reassigned = step_decisions_dropped

    def commit_experience(self, ids_full: torch.Tensor, span_start: int, span_end: int,
                           gate_effort: torch.Tensor) -> WriteStats:
        """Writes a span of live-generated text into the memory pool, outside
        the training loop."""
        cfg = self.cfg
        device = ids_full.device
        B = ids_full.shape[0]
        Cchunk = cfg.chunk_size
        pad = (-span_end) % Cchunk
        ids_p = F.pad(ids_full[:, :span_end], (0, pad), value=0) if pad else ids_full[:, :span_end]
        Tp = ids_p.shape[1]
        k_blocks = mem_trunk_k(cfg)
        with torch.no_grad():
            global_ctx = self._build_global_ctx(B, Tp, device, torch.float32)
            colony_ctx = self._build_colony_ctx(B, Tp, device, torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_trunk, e_trunk = self.model.forward_trunk(ids_p, global_ctx, colony_ctx, k_blocks)
        u_flat = h_trunk[:, span_start:span_end].reshape(-1, cfg.d_model).float()
        e_flat = e_trunk[:, span_start:span_end].reshape(-1, cfg.n_emotion).float()
        gate_effort_flat = gate_effort.reshape(-1).float()
        self.update_signal_state_inference(u_flat.shape[0], e_flat=e_flat)
        cat = CategoryState(self._cat_content, self._cat_U, self._cat_member_shadow,
                             self._cat_join_hist, self._cat_join_hist_count,
                             self._cat_top_shadow, self._cat_emotion)
        new_content, new_U, new_emotion, new_idle_steps, new_cat_id, stats = apply_write_event(
            self.model.mem_pool, self._mem_content, self._mem_U, self._mem_emotion,
            self._mem_idle_steps,
            self._join_sim_history, self._join_sim_history_count,
            self.model.mem_pool.spawn_bias, self._pending_spawn_decisions, self._total_docs_seen,
            self._mem_commit_count, self._cat_id, cat,
            u_flat, e_flat, gate_effort_flat, cfg,
            lr_cap=cfg.mem_max_lr_infer, emo_threshold=self._emo_mag_mean.item(),
            docs_since_spawn=self._docs_since_spawn,
            mass_floor=cfg.mem_write_mass_floor_infer)
        self._mem_content = new_content
        self._mem_U = new_U
        self._mem_emotion = new_emotion
        self._mem_idle_steps = new_idle_steps
        self._cat_id = new_cat_id
        self._cat_content, self._cat_U = cat.content, cat.U
        self._cat_member_shadow = cat.member_shadow
        self._cat_emotion = cat.emotion
        if stats.spawned:
            self._spawn_events += 1
        if stats.merged:
            self._merge_events += 1
        self._last_cat_load_bias_mean = stats.cat_load_bias
        self._last_cat_load_bias_spread = stats.cat_load_bias_spread
        self._last_cat_transform_events = stats.cat_transform_events
        self._last_decisions_dropped_reassigned = stats.decisions_dropped_reassigned
        self._cat_transform_events_total += stats.cat_transform_events
        self._cat_decisions_dropped_reassigned_total += stats.decisions_dropped_reassigned
        return stats

    def _signal_event_weight(self, n_event: int) -> float:
        """Scales an inference event's contribution relative to a running
        reference event size."""
        ratio_cap = self.cfg.mem_max_lr_infer / self.cfg.mem_max_lr
        n_ref = max(self._infer_event_size_ref.item(), 1.0)
        return ratio_cap * min(1.0, n_event / n_ref)

    def update_signal_state_inference(self, n_event: int,
                                       e_flat: torch.Tensor | None = None) -> None:
        """Updates the running signal mean/std/EMA state from a live inference
        event, the inference-time counterpart to the training-loop update."""
        all_sig = [blk.meta._last_signals for blk in self.model.blocks]
        if any(s is None for s in all_sig):
            return
        w = self._signal_event_weight(n_event)
        if w <= 0.0:
            return

        for blk, sig in zip(self.model.blocks, all_sig):
            local_now = sig[:N_LOCAL].to(blk.prev_local_sig.device)
            blk.prev_local_sig.mul_(1 - w).add_(w * local_now)

        all_prev_local = torch.stack([blk.prev_local_sig for blk in self.model.blocks])
        self._colony_pheromone = all_prev_local.mean(0).detach()
        self._colony_diversity = all_prev_local.std(0).mean().item()

        self._global_slow = (self.cfg.colony_slow_alpha * self._global_slow
                              + (1 - self.cfg.colony_slow_alpha) * self._colony_pheromone)

        n_alpha = 1.0 - (1.0 - self.cfg.ema_norm) * w
        for blk, sig in zip(self.model.blocks, all_sig):
            dev       = blk.meta.sig_mean.device
            s         = sig.to(dev)
            prev_mean = blk.meta.sig_mean.clone()
            blk.meta.sig_mean.mul_(n_alpha).add_((1 - n_alpha) * s)
            if blk.meta._last_signals_std is not None:
                s_std = blk.meta._last_signals_std.to(dev)
                blk.meta.sig_std.mul_(n_alpha).add_((1 - n_alpha) * s_std)
            delta        = s - prev_mean
            blk.meta.sig_temporal_var.mul_(n_alpha).add_((1 - n_alpha) * delta ** 2)
            temporal_std = blk.meta.sig_temporal_var.sqrt()
            bmask        = blk.meta.is_broadcast
            blk.meta.sig_std[bmask] = temporal_std[bmask]
            blk.meta.sig_std.clamp_(min=1e-2)

        if e_flat is not None:
            emo_mag_now = e_flat.abs().sum(-1).float().mean()
            self._emo_mag_mean.mul_(1 - w).add_(w * emo_mag_now)

        self._infer_event_size_ref.mul_(self.cfg.ema_norm).add_(
            (1 - self.cfg.ema_norm) * float(n_event))

    @torch.no_grad()
    def generate_and_learn(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float = 1.0, top_k: int | None = None,
                            commit_every: int | None = None
                            ) -> Tuple[torch.Tensor, List["WriteStats"]]:
        """Generates tokens autoregressively while committing experience into
        the memory pool as it goes. Single-stream (batch size 1) by design."""
        assert prompt_ids.shape[0] == 1, \
            "generate_and_learn is single-stream by design — see docstring"
        was_training = self.training
        self.eval()
        cfg = self.cfg
        device = prompt_ids.device
        ids = prompt_ids.clone()
        commit_start = ids.shape[1]
        write_stats: List[WriteStats] = []

        def _forward_full(seq: torch.Tensor):
            B, T = seq.shape
            Cchunk = cfg.chunk_size
            pad = (-T) % Cchunk
            seq_p = F.pad(seq, (0, pad), value=0) if pad else seq
            Tp = seq_p.shape[1]
            global_ctx = self._build_global_ctx(B, Tp, device, torch.float32)
            colony_ctx = self._build_colony_ctx(B, Tp, device, torch.float32)
            logits, _, _, _ = self.model(seq_p, global_ctx, colony_ctx, self._mem_content,
                                          self._cat_id, self._cat_content)
            gate_effort = self.model.voice_loop._last_gate_effort
            return logits[:, :T], gate_effort[:, :T]

        def _commit(span_end: int):
            nonlocal commit_start
            _, gate_effort_full = _forward_full(ids)
            ws = self.commit_experience(ids, commit_start, span_end,
                                         gate_effort_full[:, commit_start:span_end])
            write_stats.append(ws)
            commit_start = span_end

        for step in range(max_new_tokens):
            logits, _ = _forward_full(ids)
            next_logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

            if commit_every is not None and (step + 1) % commit_every == 0:
                _commit(ids.shape[1])

        if commit_start < ids.shape[1]:
            _commit(ids.shape[1])

        if was_training:
            self.train()
        return ids, write_stats

    def save_memory_pool(self, path: str) -> None:
        """Saves the memory pool's state to a standalone file for deployment."""
        torch.save({
            "mem_content": self._mem_content.detach().cpu(),
            "mem_U":       self._mem_U.detach().cpu(),
            "mem_emotion": self._mem_emotion.detach().cpu(),
            "spawn_events": self._spawn_events,
            "mem_idle_steps": self._mem_idle_steps.detach().cpu(),
            "merge_events":   self._merge_events,
            "join_sim_history":       self._join_sim_history.detach().cpu(),
            "join_sim_history_count": self._join_sim_history_count.detach().cpu(),
            "total_docs_seen":        self._total_docs_seen.detach().cpu(),
            "mem_commit_count":       self._mem_commit_count.detach().cpu(),
            "docs_since_spawn":       self._docs_since_spawn.detach().cpu(),
            "spawn_bias_state_dict":  self.model.mem_pool.spawn_bias.state_dict(),
            "cat_id":                 self._cat_id.detach().cpu(),
            "cat_content":            self._cat_content.detach().cpu(),
            "cat_U":                  self._cat_U.detach().cpu(),
            "cat_emotion":            self._cat_emotion.detach().cpu(),
            "cat_member_shadow":      self._cat_member_shadow.detach().cpu(),
            "cat_top_shadow":         self._cat_top_shadow.detach().cpu(),
            "cat_join_hist":          self._cat_join_hist.detach().cpu(),
            "cat_join_hist_count":    self._cat_join_hist_count.detach().cpu(),
            "merge_nn_sim_mean":      self.model.mem_pool._merge_nn_sim_mean.detach().cpu(),
            "merge_nn_sim_var":       self.model.mem_pool._merge_nn_sim_var.detach().cpu(),
            "merge_nn_sim_count":     self.model.mem_pool._merge_nn_sim_count.detach().cpu(),
            "emo_mag_mean":           self._emo_mag_mean.detach().cpu(),
            "infer_event_size_ref":   self._infer_event_size_ref.detach().cpu(),
            "colony_pheromone":       self._colony_pheromone.detach().cpu(),
            "colony_diversity_buf":   self._colony_diversity_buf.detach().cpu(),
            "global_slow":            self._global_slow.detach().cpu(),
            "prev_local_sig":  torch.stack([blk.prev_local_sig.detach().cpu()
                                            for blk in self.model.blocks]),
            "sig_mean":        torch.stack([blk.meta.sig_mean.detach().cpu()
                                            for blk in self.model.blocks]),
            "sig_std":         torch.stack([blk.meta.sig_std.detach().cpu()
                                            for blk in self.model.blocks]),
            "sig_temporal_var": torch.stack([blk.meta.sig_temporal_var.detach().cpu()
                                             for blk in self.model.blocks]),
            "fingerprint": {
                "d_model":       self.cfg.d_model,
                "mem_d_k_mult":  self.cfg.mem_d_k_mult,
                "mem_value":     "content",
                "n_blocks":      self.cfg.n_blocks,
                "n_emotion":     self.cfg.n_emotion,
            },
        }, path)

    def load_memory_pool(self, path: str, strict: bool = False) -> None:
        """Loads a previously-saved shared memory pool AND signal state."""
        data = torch.load(path, map_location="cpu")
        fp = data.get("fingerprint", {})
        mismatch = (fp.get("d_model")      != self.cfg.d_model
                    or fp.get("mem_d_k_mult") != self.cfg.mem_d_k_mult
                    or ("n_emotion" in fp and fp.get("n_emotion") != self.cfg.n_emotion))
        if mismatch:
            print(f"WARNING [Sencelium]: memory pool fingerprint mismatch loading "
                  f"{path}: saved={fp} vs current config "
                  f"d_model={self.cfg.d_model} mem_d_k_mult={self.cfg.mem_d_k_mult} "
                  f"mem_value=content")
            if strict:
                raise ValueError("memory pool fingerprint mismatch (strict=True)")
        device = self._mem_content.device
        self._mem_content = data["mem_content"].to(device)
        self._mem_U       = data["mem_U"].to(device)
        C = self._mem_content.shape[0]
        if "mem_emotion" in data:
            self._mem_emotion = data["mem_emotion"].to(device)
        elif C > 0:
            self._mem_emotion = torch.zeros(C, self.cfg.n_emotion, device=device)
        self._spawn_events = data.get("spawn_events", 0)
        if "mem_idle_steps" in data:
            self._mem_idle_steps = data["mem_idle_steps"].to(device)
        else:
            self._mem_idle_steps = torch.zeros(C, device=device)
        self._merge_events = data.get("merge_events", 0)

        if "join_sim_history" in data:
            self._join_sim_history.copy_(data["join_sim_history"].to(device))
            self._join_sim_history_count.copy_(data["join_sim_history_count"].to(device))
        if "total_docs_seen" in data:
            self._total_docs_seen.copy_(data["total_docs_seen"].to(device))
        if "mem_commit_count" in data:
            self._mem_commit_count.copy_(data["mem_commit_count"].to(device))
        if "docs_since_spawn" in data:
            self._docs_since_spawn.copy_(data["docs_since_spawn"].to(device))
        if "spawn_bias_state_dict" in data:
            self.model.mem_pool.spawn_bias.load_state_dict(data["spawn_bias_state_dict"], strict=False)
        if "cat_id" in data:
            self._cat_id = data["cat_id"].to(device)
            self._cat_content = data["cat_content"].to(device)
            self._cat_U = data["cat_U"].to(device)
            if "cat_emotion" in data:
                self._cat_emotion = data["cat_emotion"].to(device)
            self._cat_member_shadow = data["cat_member_shadow"].to(device)
            self._cat_top_shadow.copy_(data["cat_top_shadow"].to(device))
            self._cat_join_hist.copy_(data["cat_join_hist"].to(device))
            self._cat_join_hist_count.copy_(data["cat_join_hist_count"].to(device))
        elif C > 0:
            w = self._mem_U.clamp(min=1e-6)
            wsum = w.sum().clamp(min=1e-6)
            self._cat_id = torch.zeros(C, dtype=torch.long, device=device)
            self._cat_content = ((self._mem_content * w.unsqueeze(-1)).sum(0, keepdim=True)
                                  / wsum).to(self._mem_content.dtype)
            self._cat_U = self._mem_U.sum().view(1)
            self._cat_emotion = ((self._mem_emotion * w.unsqueeze(-1)).sum(0, keepdim=True)
                                  / wsum).to(self._mem_emotion.dtype)
            self._cat_member_shadow = torch.ones(1, device=device)
            self._cat_top_shadow.fill_(1.0)
        if "merge_nn_sim_mean" in data:
            self.model.mem_pool._merge_nn_sim_mean.copy_(data["merge_nn_sim_mean"].to(device))
            self.model.mem_pool._merge_nn_sim_var.copy_(data["merge_nn_sim_var"].to(device))
            self.model.mem_pool._merge_nn_sim_count.copy_(data["merge_nn_sim_count"].to(device))
        if "emo_mag_mean" in data:
            self._emo_mag_mean.copy_(data["emo_mag_mean"].to(device))
        if "infer_event_size_ref" in data:
            self._infer_event_size_ref.copy_(data["infer_event_size_ref"].to(device))
        if "colony_pheromone" in data:
            self._colony_pheromone.copy_(data["colony_pheromone"].to(device))
            self._colony_diversity_buf.copy_(data["colony_diversity_buf"].to(device))
            self._global_slow.copy_(data["global_slow"].to(device))
        n_blocks_match = fp.get("n_blocks") == len(self.model.blocks)
        if "prev_local_sig" in data and n_blocks_match:
            for i, blk in enumerate(self.model.blocks):
                blk.prev_local_sig.copy_(data["prev_local_sig"][i].to(device))
                blk.meta.sig_mean.copy_(data["sig_mean"][i].to(device))
                blk.meta.sig_std.copy_(data["sig_std"][i].to(device))
                blk.meta.sig_temporal_var.copy_(data["sig_temporal_var"][i].to(device))
        elif "prev_local_sig" in data and not n_blocks_match:
            print(f"WARNING [Sencelium]: n_blocks mismatch loading {path} "
                  f"(saved={fp.get('n_blocks')} vs current={len(self.model.blocks)}) — "
                  f"per-block signal state NOT restored, memory pool still was.")

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restores per-block signal state on checkpoint load, when it exists
        and the block count matches."""
        sd = checkpoint.get("state_dict", {})
        for name in ("_mem_content", "_mem_U", "_mem_emotion", "_mem_idle_steps", "_cat_id",
                     "_cat_content", "_cat_U", "_cat_member_shadow", "_cat_emotion"):
            if name in sd and sd[name].shape != getattr(self, name).shape:
                setattr(self, name, torch.zeros_like(sd[name]))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        α = self.cfg.ema_alpha

        gate_mags, all_sig_means = [], []
        for blk in self.model.blocks:
            sig = blk.meta._last_signals
            if sig is None:
                continue
            all_sig_means.append(sig)
            gate_mags.append(blk.meta._last_gate_mag)

        if not all_sig_means:
            return

        gate_mean = sum(gate_mags) / len(gate_mags)

        for blk, sig in zip(self.model.blocks, all_sig_means):
            blk.phi6_slow.mul_(α).add_((1 - α) * sig[2].to(blk.phi6_slow.dtype))

        for blk, sig in zip(self.model.blocks, all_sig_means):
            blk.prev_local_sig.copy_(sig[:N_LOCAL].detach())

        all_prev_local = torch.stack([
            blk.prev_local_sig for blk in self.model.blocks
        ])
        self._colony_pheromone = all_prev_local.mean(0).detach()
        self._colony_diversity = all_prev_local.std(0).mean()

        slow_α = self.cfg.colony_slow_alpha
        self._global_slow = (slow_α * self._global_slow
                             + (1 - slow_α) * self._colony_pheromone)

        if self._step_count > self.cfg.sig_norm_warmup:
            α_n = self.cfg.ema_norm
            for blk, sig in zip(self.model.blocks, all_sig_means):
                dev       = blk.meta.sig_mean.device
                s         = sig.to(dev)
                prev_mean = blk.meta.sig_mean.clone()
                blk.meta.sig_mean.mul_(α_n).add_((1 - α_n) * s)

                if blk.meta._last_signals_std is not None:
                    s_std = blk.meta._last_signals_std.to(dev)
                    blk.meta.sig_std.mul_(α_n).add_((1 - α_n) * s_std)

                delta        = s - prev_mean
                blk.meta.sig_temporal_var.mul_(α_n).add_((1 - α_n) * delta ** 2)
                temporal_std = blk.meta.sig_temporal_var.sqrt()
                bmask        = blk.meta.is_broadcast
                blk.meta.sig_std[bmask] = temporal_std[bmask]
                blk.meta.sig_std.clamp_(min=1e-2)

            if self._last_e is not None:
                emo_mag_now = self._last_e.abs().sum(-1).float().mean()
                self._emo_mag_mean.mul_(α_n).add_((1 - α_n) * emo_mag_now)

        self._block_diag_snapshot = [
            dict(gate_delta_mag=blk._last_gate_delta_mag,
                 gate_mag=blk.meta._last_gate_mag,
                 da_signed=blk._last_da_signed,
                 corr_signed=blk.meta._last_corr_signed,
                 gate_nuc_frac=blk._last_gate_nuc_frac,
                 gate_delta_frac=blk._last_gate_delta_frac,
                 nuc_half_life=blk._last_nuc_half_life,
                 nuc_half_life_per_head=blk._last_nuc_half_life_per_head,
                 wm_a_rms=blk._last_wm_a_rms,
                 wm_a_shift_rms_tracker=blk._last_wm_a_shift_rms_tracker)
            for blk in self.model.blocks
        ]

        for blk in self.model.blocks:
            blk.commit_trackers()
        self.model.set_diagnostics(False)
        self._update_memory_pool()

        self._sig_acc.append(torch.stack(all_sig_means).mean(0))
        self._gate_acc.append(gate_mean)

        if self._step_count % self.cfg.log_every == 0 and self._sig_acc:
            mean_sig = torch.stack(self._sig_acc).mean(0)
            for i, name in enumerate(SIGNAL_NAMES):
                self.log(f"sig/{name}", mean_sig[i].item(), on_step=True, on_epoch=False)
            self.log("meta/gate_magnitude",
                     float(sum(self._gate_acc) / len(self._gate_acc)),
                     on_step=True, on_epoch=False)
            self.log("global/slow_mean", self._global_slow.mean().item(),
                     on_step=True, on_epoch=False)

            ret_w = self.model.blocks[0].retention_mod[0].weight.abs().mean().item()
            self.log("meta/retention_mod_weight", ret_w, on_step=True, on_epoch=False)

            _snap = self._block_diag_snapshot
            _snap = [{k: (v if (v is None or (torch.is_tensor(v) and v.dim() > 0)) else float(v))
                      for k, v in s.items()} for s in _snap]
            ret_mags  = [s['gate_delta_mag'] for s in _snap]
            meta_mags = [s['gate_mag'] for s in _snap]
            self.log("retention/gate_delta_mag", sum(ret_mags) / len(ret_mags),
                     on_step=True, on_epoch=False)
            if len(ret_mags) > 1:
                r = torch.tensor(ret_mags)
                m = torch.tensor(meta_mags)
                corr = ((r - r.mean()) * (m - m.mean())).sum() / (
                    (r - r.mean()).norm() * (m - m.mean()).norm() + 1e-6)
                self.log("retention/meta_gate_corr_across_blocks", corr.item(),
                         on_step=True, on_epoch=False)

            dir_corrs = []
            for s in _snap:
                da_s, corr_s = s['da_signed'], s['corr_signed']
                if da_s is None or corr_s is None:
                    continue
                a = da_s.flatten() - da_s.mean()
                b = corr_s.flatten() - corr_s.mean()
                denom = a.norm() * b.norm()
                if denom.item() > 1e-9:
                    dir_corrs.append(((a * b).sum() / denom).item())
            if dir_corrs:
                self.log("retention/meta_directional_corr_within_block",
                         sum(dir_corrs) / len(dir_corrs), on_step=True, on_epoch=False)

            self.log("colony/diversity", self._colony_diversity,
                     on_step=True, on_epoch=False)
            all_sens = torch.stack([
                blk.layer_sens.detach().cpu() for blk in self.model.blocks
            ])
            self.log("layer_sens/abs_mean", all_sens.abs().mean().item(),
                     on_step=True, on_epoch=False)
            self.log("layer_sens/diversity", all_sens.std(0).mean().item(),
                     on_step=True, on_epoch=False)

            self.log("nucleus/gate_nuc_frac",
                     sum(s['gate_nuc_frac'] for s in _snap) / len(_snap),
                     on_step=True, on_epoch=False)
            self.log("nucleus/gate_delta_frac",
                     sum(s['gate_delta_frac'] for s in _snap) / len(_snap),
                     on_step=True, on_epoch=False)
            self.log("nucleus/half_life_mean",
                     sum(s['nuc_half_life'] for s in _snap) / len(_snap),
                     on_step=True, on_epoch=False)
            self.log("nucleus/wm_a_shift_std_mean",
                     sum(s['wm_a_rms'] for s in _snap) / len(_snap),
                     on_step=True, on_epoch=False)
            self.log("nucleus/wm_a_shift_rms_tracker_mean",
                     sum(s['wm_a_shift_rms_tracker'] for s in _snap) / len(_snap),
                     on_step=True, on_epoch=False)
            _hl_heads = [s['nuc_half_life_per_head'] for s in _snap
                         if s['nuc_half_life_per_head'] is not None]
            if _hl_heads:
                _hl_cat = torch.cat([t.float().cpu() for t in _hl_heads])
                collapse_frac = ((_hl_cat < 2.0) | (_hl_cat > 2000.0)).float().mean().item()
                self.log("nucleus/half_life_collapse_frac", collapse_frac,
                         on_step=True, on_epoch=False)

            n_slots = self._mem_content.shape[0]
            self.log("mem/n_slots", float(n_slots), on_step=True, on_epoch=False)
            self.log("mem/write_events", float(self._last_write_events), on_step=True, on_epoch=False)
            self.log("mem/write_tokens", float(self._last_write_tokens), on_step=True, on_epoch=False)
            self.log("mem/read_entropy", self.model._last_read_entropy.item(),
                     on_step=True, on_epoch=False)
            self.log("mem/read_entropy_norm", self.model._last_read_entropy_norm.item(),
                     on_step=True, on_epoch=False)
            _nm = self.model.mem_pool._last_null_mass
            if _nm is not None:
                self.log("mem/null_mass", float(_nm), on_step=True, on_epoch=False)
            self.log("mem/null_bias", float(self.model.mem_pool.b_null.detach()),
                     on_step=True, on_epoch=False)
            self.log("mem/spawn_rate", self._last_spawn_mass, on_step=True, on_epoch=False)
            self.log("mem/spawn_events_total", float(self._spawn_events), on_step=True, on_epoch=False)
            self.log("mem/merge_events_total", float(self._merge_events), on_step=True, on_epoch=False)
            self.log("mem/c_sub_mean", self._last_c_sub_mean, on_step=True, on_epoch=False)
            self.log("mem/k_nov_mean", self._last_k_nov_mean, on_step=True, on_epoch=False)
            self.log("mem/novelty_history_count",
                     float(self._join_sim_history_count.item()), on_step=True, on_epoch=False)
            self.log("mem/drought_boost_mean", self._last_drought_boost_mean,
                     on_step=True, on_epoch=False)
            self.log("mem/docs_since_spawn", float(self._docs_since_spawn.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/novelty_margin_max", self._last_novelty_margin_max,
                     on_step=True, on_epoch=False)
            self.log("mem/novelty_margin_mean", self._last_novelty_margin_mean,
                     on_step=True, on_epoch=False)
            sb = self.model.mem_pool.spawn_bias
            self.log("mem/spawn_bias_w_margin", sb.w_margin.item(), on_step=True, on_epoch=False)
            self.log("mem/spawn_bias_w_std", sb.w_std.item(), on_step=True, on_epoch=False)
            self.log("mem/spawn_bias_w_emo", sb.w_emo.item(), on_step=True, on_epoch=False)
            self.log("mem/spawn_bias_w_effort", sb.w_effort.item(), on_step=True, on_epoch=False)
            self.log("mem/spawn_bias_bias", sb.bias.item(), on_step=True, on_epoch=False)
            self.log("mem/sim_std_run_mean", sb._ss_mean.item(), on_step=True, on_epoch=False)
            self.log("mem/sim_std_run_std", sb._ss_var.item() ** 0.5, on_step=True, on_epoch=False)
            self.log("mem/cat_load_bias_mean", self._last_cat_load_bias_mean,
                     on_step=True, on_epoch=False)
            self.log("mem/cat_load_bias_spread", self._last_cat_load_bias_spread,
                     on_step=True, on_epoch=False)
            self.log("mem/cat_transform_events", float(self._last_cat_transform_events),
                     on_step=True, on_epoch=False)
            self.log("mem/cat_transform_events_total", float(self._cat_transform_events_total),
                     on_step=True, on_epoch=False)
            self.log("mem/cat_decisions_dropped_reassigned",
                     float(self._last_decisions_dropped_reassigned),
                     on_step=True, on_epoch=False)
            self.log("mem/spawn_bias_n_updates", float(sb.n_updates.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/sim_std_nonfinite", float(sb._ss_nonfinite.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/emo_mag_nonfinite", float(sb._emo_nonfinite.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/merge_nn_sim_nonfinite",
                     float(self.model.mem_pool._merge_nn_sim_nonfinite.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/cat_join_hist_count", float(self._cat_join_hist_count.item()),
                     on_step=True, on_epoch=False)
            self.log("mem/pending_decisions", float(len(self._pending_spawn_decisions)),
                     on_step=True, on_epoch=False)
            self.log("mem/total_docs_seen", float(self._total_docs_seen.item()),
                     on_step=True, on_epoch=False)
            n_cats = self._cat_content.shape[0]
            self.log("mem/n_categories", float(n_cats), on_step=True, on_epoch=False)
            if n_cats > 0:
                empty_cats = (torch.bincount(self._cat_id, minlength=n_cats) == 0).sum().item()
                self.log("mem/empty_categories", float(empty_cats), on_step=True, on_epoch=False)
                self.log("mem/empty_categories_frac", empty_cats / n_cats, on_step=True, on_epoch=False)
                self.log("mem/cat_member_shadow_mean", self._cat_member_shadow.mean().item(),
                         on_step=True, on_epoch=False)
                cat_sizes = torch.bincount(self._cat_id, minlength=n_cats).float()
                self.log("mem/cat_size_mean", cat_sizes.mean().item(), on_step=True, on_epoch=False)
                self.log("mem/cat_size_max", cat_sizes.max().item(), on_step=True, on_epoch=False)
                if n_cats > 1:
                    self.log("mem/cat_size_std", cat_sizes.std().item(), on_step=True, on_epoch=False)
                    self.log("mem/cat_emotion_std", self._cat_emotion.std().item(),
                             on_step=True, on_epoch=False)
                else:
                    self.log("mem/cat_size_std", 0.0, on_step=True, on_epoch=False)
                    self.log("mem/cat_emotion_std", 0.0, on_step=True, on_epoch=False)
            self.log("mem/cat_top_shadow", self._cat_top_shadow.item(), on_step=True, on_epoch=False)
            self.log("mem/commit_count", float(self._mem_commit_count.item()),
                     on_step=True, on_epoch=False)
            self.log("emotion/tau_effective", self._emo_mag_mean.item(),
                     on_step=True, on_epoch=False)
            if n_slots > 0:
                occupied_abs = (self._mem_U > self.cfg.mem_occupied_u_threshold).sum().item()
                self.log("mem/occupied_slots_u_gt_thresh", float(occupied_abs),
                         on_step=True, on_epoch=False)
                n_mass, u_eff = _pool_usage_concentration(self._mem_U,
                                                          self.cfg.mem_usage_mass_fraction)
                self.log("mem/u_mass_slots_p90", float(n_mass), on_step=True, on_epoch=False)
                self.log("mem/u_mass_slots_p90_frac", n_mass / max(n_slots, 1),
                         on_step=True, on_epoch=False)
                self.log("mem/u_eff_slots", u_eff, on_step=True, on_epoch=False)
                self.log("mem/usage_max", self._mem_U.max().item(), on_step=True, on_epoch=False)
                self.log("mem/usage_mean", self._mem_U.mean().item(), on_step=True, on_epoch=False)
                self.log("mem/usage_std", self._mem_U.std(unbiased=False).item(), on_step=True, on_epoch=False)
                self.log("mem/idle_max", self._mem_idle_steps.max().item(), on_step=True, on_epoch=False)
                self.log("mem/idle_mean", self._mem_idle_steps.mean().item(), on_step=True, on_epoch=False)
            if self._last_e is not None:
                self.log("emotion/abs_mean", self._last_e.abs().mean().item(),
                         on_step=True, on_epoch=False)
            self.log("voice/gate_mean", float(self.model.voice_loop._last_gate_mean),
                     on_step=True, on_epoch=False)
            self.log("voice/gate_std_channel", float(self.model.voice_loop._last_gate_std_ch),
                     on_step=True, on_epoch=False)
            self.log("voice/gate_std_position", float(self.model.voice_loop._last_gate_std_pos),
                     on_step=True, on_epoch=False)
            for i, rd in enumerate(self.model.voice_loop._last_round_deltas):
                self.log(f"voice/round_delta_{i}", float(rd), on_step=True, on_epoch=False)
            for i, gm in enumerate(self.model.voice_loop._last_round_gate_means):
                self.log(f"voice/round_gate_mean_{i}", float(gm), on_step=True, on_epoch=False)
            for i, gc in enumerate(self.model.voice_loop._last_round_gate_change):
                self.log(f"voice/round_gate_change_{i}", float(gc), on_step=True, on_epoch=False)

            self._sig_acc.clear()
            self._gate_acc.clear()

    def on_train_end(self) -> None:
        self.model.set_diagnostics(True)

    def on_validation_epoch_start(self) -> None:
        self._val_lanes = {"mem": _LaneStore(), "no_mem": _LaneStore()}
        self._val_carry_acc = {}

    def _validation_step_carry(self, batch) -> None:
        """Validation step for the stateful (carried-state) protocol."""
        x, y, carry_data, lane, win, _doc = batch
        B, T = x.shape
        gctx = self._build_global_ctx(B, T, x.device, torch.float32)
        cctx = self._build_colony_ctx(B, T, x.device, torch.float32)
        empty_content     = self._mem_content.new_zeros(0, self.cfg.d_model)
        empty_cat_id      = self._cat_id.new_zeros(0)
        empty_cat_content = self._cat_content.new_zeros(0, self.cfg.d_model)
        pools = {"mem": (self._mem_content, self._cat_id, self._cat_content),
                 "no_mem": (empty_content, empty_cat_id, empty_cat_content)}
        acc = self._val_carry_acc

        def _ce_sum(logits: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   reduction='none').float().view(B, T).sum(-1)  # [B]

        def _add(name: str, per_row: torch.Tensor, rows: torch.Tensor) -> None:
            acc[name + "_sum"] = acc.get(name + "_sum", 0.0) + float(per_row[rows].sum())
            acc[name + "_n"] = acc.get(name + "_n", 0.0) + float(rows.sum()) * T

        all_rows = torch.ones(B, dtype=torch.bool, device=x.device)
        valid_by_chain = {}
        for chain in ("mem", "no_mem"):
            store = self._val_lanes[chain]
            state, valid = store.gather(lane, win, carry_data)
            logits, _, _, _, new_state = self.model(
                x, gctx, cctx, *pools[chain], state=state,
                carry=valid if state is not None else None, return_state=True)
            store.scatter(lane, win, valid if state is not None else torch.zeros_like(valid),
                          new_state)
            valid_by_chain[chain] = valid
            ce = _ce_sum(logits)
            name = "carry" if chain == "mem" else "carry_no_mem"
            _add(name, ce, all_rows)
            _add("cw_" + name, ce, valid)
        logits_f, _, _, _ = self.model(x, gctx, cctx, *pools["mem"])
        ce_f = _ce_sum(logits_f)
        _add("fresh", ce_f, all_rows)
        _add("cw_fresh", ce_f, valid_by_chain["mem"])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.set_diagnostics(False)
        if dataloader_idx == 1:
            self._validation_step_carry(batch)
            return
        ce_loss, _, _, _ = self._step(batch)
        self._val_losses.append(ce_loss.detach().float().item())

        x, y = batch
        B, T = x.shape
        global_ctx = self._build_global_ctx(B, T, x.device, torch.float32)
        colony_ctx = self._build_colony_ctx(B, T, x.device, torch.float32)
        empty_content     = self._mem_content.new_zeros(0, self.cfg.d_model)
        empty_cat_id      = self._cat_id.new_zeros(0)
        empty_cat_content = self._cat_content.new_zeros(0, self.cfg.d_model)
        logits_no_mem, _, _, _ = self.model(x, global_ctx, colony_ctx,
                                             empty_content, empty_cat_id, empty_cat_content)
        ce_no_mem = F.cross_entropy(logits_no_mem.reshape(-1, logits_no_mem.size(-1)),
                                     y.reshape(-1))
        self._val_losses_no_mem.append(ce_no_mem.detach().float().item())

    def on_validation_epoch_end(self):
        acc = self._val_carry_acc
        if acc.get("carry_n", 0) > 0:
            def _ppl(name: str) -> float | None:
                n = acc.get(name + "_n", 0.0)
                return math.exp(min(acc[name + "_sum"] / n, 20.0)) if n > 0 else None
            vals = {k: _ppl(k) for k in ("carry", "fresh", "carry_no_mem",
                                          "cw_carry", "cw_fresh", "cw_carry_no_mem")}
            self.log("val_ppl_carry", vals["carry"], prog_bar=True)
            self.log("val_ppl_carry_fresh", vals["fresh"])
            self.log("val_ppl_carry_gain", vals["fresh"] - vals["carry"], prog_bar=True)
            self.log("val_ppl_carry_no_mem", vals["carry_no_mem"])
            self.log("val_ppl_carry_mem_delta", vals["carry_no_mem"] - vals["carry"])
            self.log("val_carry_window_frac", acc.get("cw_carry_n", 0.0) / acc["carry_n"])
            if vals["cw_carry"] is not None:
                self.log("val_ppl_cw_carry", vals["cw_carry"])
                self.log("val_ppl_cw_fresh", vals["cw_fresh"])
                self.log("val_ppl_cw_carry_no_mem", vals["cw_carry_no_mem"])
            self._val_carry_acc = {}
        if self._val_losses:
            ppl = math.exp(min(sum(self._val_losses) / len(self._val_losses), 20.0))
            self.last_ppl = ppl
            if ppl < self.best_ppl:
                self.best_ppl = ppl
            self.log("val_ppl_wikitext103", ppl, prog_bar=True)
            self._val_losses.clear()

            if self._val_losses_no_mem:
                ppl_no_mem = math.exp(min(sum(self._val_losses_no_mem)
                                           / len(self._val_losses_no_mem), 20.0))
                self.log("val_ppl_no_mem", ppl_no_mem, prog_bar=False)
                self.log("val_ppl_mem_delta", ppl_no_mem - ppl, prog_bar=True)
                self._val_losses_no_mem.clear()

            n_slots_now = self._mem_content.shape[0]
            if n_slots_now > 1:
                _, _, _, raw_mc, dm_mc = _pool_near_dup_rates(
                    self._mem_content, self.cfg.mem_neardup_cos_threshold,
                    self.cfg.mem_neardup_max_slots)
                self.log("mem/slot_cos_mean_raw", raw_mc, prog_bar=False)
                self.log("mem/slot_cos_mean_direction_removed", dm_mc, prog_bar=False)

    def configure_optimizers(self):
        if self.cfg.use_router_distill:
            r_ids = {id(p) for p in self.model.read_router.parameters()}
            groups = [
                {"params": [p for p in self.parameters() if id(p) not in r_ids]},
                {"params": list(self.model.read_router.parameters()),
                 "lr": self.cfg.lr * self.cfg.router_lr_mult},
            ]
            opt = torch.optim.AdamW(groups, lr=self.cfg.lr,
                                    betas=(0.9, 0.95), weight_decay=0.01)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr,
                                    betas=(0.9, 0.95), weight_decay=0.01)
        warmup, total = self.cfg.warmup_steps, self.cfg.max_steps
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main() -> None:
    pl.seed_everything(42, workers=True)
    cfg       = load_config(sys.argv[1:])
    import os
    cfg.use_router_distill = os.environ.get("SENC2_ROUTER", "1") == "1"
    cfg.carry_state = os.environ.get("SENC2_CARRY", "0") == "1"
    print(f"[{cfg.tag}] use_router_distill={cfg.use_router_distill}  "
          f"carry_state={cfg.carry_state}  esn_residual=always-on", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_dl, val_dl = build_loaders(cfg, tokenizer)
    lit       = Lit(cfg)
    n_params  = sum(p.numel() for p in lit.parameters())
    root      = Path(__file__).parent.parent
    tag       = cfg.tag

    blk_params   = sum(p.numel() for p in lit.model.blocks[0].parameters())
    emb_params   = sum(p.numel() for p in lit.model.emb.parameters())
    voice_params = sum(p.numel() for p in lit.model.voice_loop.parameters())
    mem_params   = sum(p.numel() for p in lit.model.mem_pool.parameters())
    emo_params   = sum(p.numel() for p in lit.model.emotion_head.parameters())
    sencelium_params = voice_params + mem_params + emo_params

    print(f"[{tag}] params={n_params/1e6:.3f}M  d_model={cfg.d_model}  "
          f"n_heads={cfg.n_heads}  n_blocks={cfg.n_blocks}  ffn_hidden={cfg.ffn_hidden}")
    print(f"  backbone: emb={emb_params/1e6:.3f}M  block={blk_params/1e6:.4f}M×{cfg.n_blocks}  "
          f"total={(emb_params + blk_params*cfg.n_blocks)/1e6:.3f}M")
    print(f"  Sencelium: voice_loop={voice_params}  mem_pool={mem_params}  emotion_head={emo_params}  "
          f"total={sencelium_params/1e3:.2f}K ({100*sencelium_params/n_params:.2f}% of model)")
    print(f"  Train: FineWebEdu, {cfg.batch_size} lanes (carry_state={cfg.carry_state})  "
          f"Val: WikiText-103")
    print(f"  tokens={cfg.max_steps * cfg.batch_size * cfg.seq_len / 1e9:.3f}B  "
          f"steps={cfg.max_steps}  batch={cfg.batch_size}")

    trainer = pl.Trainer(
        max_steps=cfg.max_steps, accelerator="gpu", devices=1, precision="bf16-mixed",
        gradient_clip_val=cfg.grad_clip, log_every_n_steps=cfg.log_every,
        val_check_interval=cfg.val_interval, limit_val_batches=50,
        logger=CSVLogger(str(root / "logs"), name=tag),
        callbacks=[
            ModelCheckpoint(dirpath=str(root / "checkpoints" / tag),
                            filename="last", every_n_train_steps=cfg.val_interval,
                            save_top_k=0, save_last=True),
            ModelCheckpoint(dirpath=str(root / "checkpoints" / tag),
                            filename="best", monitor="val_ppl_wikitext103",
                            mode="min", save_top_k=1, save_weights_only=True),
        ],
    )
    t0 = time.time()
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    elapsed = time.time() - t0

    print(f"\n[{tag}] DONE  params={n_params/1e6:.3f}M  "
          f"val_ppl={lit.last_ppl:.2f}  best_ppl={lit.best_ppl:.2f}  "
          f"elapsed={elapsed:.0f}s")

    n_slots = lit._mem_content.shape[0]
    print(f"\n  Memory Pool: grew to {n_slots} slots (started at 0, ceiling={cfg.mem_max_slots})  "
          f"spawn_events={lit._spawn_events}  merge_events={lit._merge_events}")
    if n_slots > 0:
        n_mass, u_eff = _pool_usage_concentration(lit._mem_U, cfg.mem_usage_mass_fraction)
        u_total = lit._mem_U.sum().item()
        occupied_abs = (lit._mem_U > cfg.mem_occupied_u_threshold).sum().item()
        print(f"  usage: {n_mass}/{n_slots} slots hold "
              f"{100*cfg.mem_usage_mass_fraction:.0f}% of total U ({100*n_mass/n_slots:.2f}% of pool)  "
              f"u_eff_slots={u_eff:.1f} ({100*u_eff/n_slots:.2f}%)  "
              f"U_total={u_total:.1f} (fixed budget ~{1.0/(1.0-cfg.mem_usage_decay):.0f})")
        print(f"  [legacy] {occupied_abs}/{n_slots} above absolute "
              f"mem_occupied_u_threshold={cfg.mem_occupied_u_threshold} "
              f"(capped by the fixed U budget — not a health measure at scale)  "
              f"usage_max={lit._mem_U.max().item():.4f}  usage_mean={lit._mem_U.mean().item():.4f}  "
              f"usage_std={lit._mem_U.std(unbiased=False).item():.4f}")
        print(f"  idle_max={lit._mem_idle_steps.max().item():.1f}  "
              f"idle_mean={lit._mem_idle_steps.mean().item():.4f}")
        raw_r, dm_r, n_used, raw_mc, dm_mc = _pool_near_dup_rates(
            lit._mem_content, cfg.mem_neardup_cos_threshold, cfg.mem_neardup_max_slots)
        sampled = "" if n_used >= n_slots else f" (random subsample of {n_used}/{n_slots} slots)"
        print(f"  near-dup (cos>{cfg.mem_neardup_cos_threshold}){sampled}:  "
              f"raw={100*raw_r:.2f}% (mean cos {raw_mc:.4f})  "
              f"mean-direction-removed={100*dm_r:.2f}% (mean cos {dm_mc:.4f})  "
              f"<- the second one is the real redundancy measure")
    sbh = lit.model.mem_pool.spawn_bias
    n_cats = int(lit._cat_content.shape[0])
    cat_sizes_end = torch.bincount(lit._cat_id, minlength=max(n_cats, 1))
    n_live_cats = int((cat_sizes_end[:n_cats] > 0).sum().item()) if n_cats > 0 else 0
    cat_sizes_end_f = cat_sizes_end.float()
    print(f"  categories: {n_cats} total, {n_live_cats} with >=1 member, "
          f"max size={int(cat_sizes_end_f.max().item()) if n_cats > 0 else 0}  "
          f"cat_load_bias(mean/spread)={lit._last_cat_load_bias_mean:.6f}/"
          f"{lit._last_cat_load_bias_spread:.6f}  "
          f"cat_transform_events_total={lit._cat_transform_events_total}  "
          f"decisions_dropped_reassigned_total={lit._cat_decisions_dropped_reassigned_total}")
    print(f"  non-finite observations refused (must be 0): "
          f"sim_std={int(sbh._ss_nonfinite.item())}  "
          f"emo_mag={int(sbh._emo_nonfinite.item())}  "
          f"merge_nn_sim={int(lit.model.mem_pool._merge_nn_sim_nonfinite.item())}")
    print(f"  {'✅ dynamic allocation happened' if lit._spawn_events > 0 else '⚠️  no slots were ever spawned'}")
    print(f"  novelty history: {lit._join_sim_history_count.item()} confirmed-join observations  "
          f"emo_tau_effective={lit._emo_mag_mean.item():.4f} (seeded {cfg.emo_tau}, shared by LM+trunk)  "
          f"docs_since_spawn={lit._docs_since_spawn.item()}")

    ret_w = lit.model.blocks[0].retention_mod[0].weight.abs().mean().item()
    print(f"  retention_mod input layer weight: {ret_w:.4f}"
          f"  {'✅ alive' if ret_w > 0.005 else '⚠️  dead'}")

    gate_mags = [blk.meta._last_gate_mag for blk in lit.model.blocks]
    gate_mean = sum(gate_mags) / len(gate_mags)
    print(f"  gate_magnitude: {gate_mean:.4f}"
          f"  {'✅ MetaCtrl active' if gate_mean > 0.05 else '⚠️  MetaCtrl weak'}")


if __name__ == "__main__":
    main()
