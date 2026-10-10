from __future__ import annotations

# ==============================================================================
# Transformer 125M baseline — RoPE, pre-norm, weight tying
#
# Based on train_transformer_65M.py.
# Architecture identical — only the scale differs.
#
# Changes relative to 65M:
#   d_model:  384 → 512   (d_head=64 unchanged, n_heads: 6 → 8)
#   n_layers: 26  → 32    (→ ~126.4M params total)
#
# Train:  FineWebEdu (HuggingFaceFW/fineweb-edu, streaming)
# Val:    WikiText-103 test (Salesforce/wikitext, wikitext-103-raw-v1)
# Metric: val_ppl_wikitext103
#
# VRAM: ~14-20GB (gradient_checkpoint enabled in TransformerBlock).
# ==============================================================================

import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from pathlib import Path

from datasets import load_dataset
from transformers import GPT2TokenizerFast


@dataclass
class Config:
    vocab_size: int   = 50257
    d_model:    int   = 512     # 384→512, d_head=64 (same), n_heads: 6→8
    n_heads:    int   = 8
    n_layers:   int   = 32      # 32 layers → ~126.4M total (with weight tying)

    # ckpt_interval = how often last.ckpt is written. Resume: set env
    # RESUME_CKPT=<path to last.ckpt>.
    max_steps:    int   = 152_588  # 152_588 × 32 × 512 = 2.5B tokens (Chinchilla 20× params)
    warmup_steps: int   = 1_000
    lr:           float = 1e-4
    grad_clip:    float = 0.5
    log_every:    int   = 500
    val_interval: int   = 25_000
    ckpt_interval: int  = 10_000

    seq_len:    int   = 512
    batch_size: int   = 32


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def _build_rope_cache(d_head: int, max_len: int, device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos   = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, theta)                        # [T, d_head//2]
    cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, d_head]
    sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, n_heads, T, d_head]  cos/sin: [T, d_head]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


# ─── Transformer blocks ───────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model,     bias=False)
        self.scale   = math.sqrt(self.d_head)

        cos, sin = _build_rope_cache(self.d_head, max_seq, torch.device("cpu"))
        self.register_buffer("rope_cos", cos)  # [max_seq, d_head]
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,d]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k = _apply_rope(q, k, self.rope_cos[:T], self.rope_sin[:T])

        mask   = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn   = torch.softmax(torch.einsum('bhid,bhjd->bhij', q, k) / self.scale + mask, dim=-1)
        out    = torch.einsum('bhij,bhjd->bhid', attn, v)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_checkpoint(self._forward, x, use_reentrant=False)


class TransformerModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks    = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.seq_len)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ─── Data (HuggingFaceFW/fineweb-edu) ────────────────────────────────────────

class FineWebEduDataset(IterableDataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
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


class WikiText103ValDataset(IterableDataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
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


def build_loaders(cfg: Config, tokenizer: GPT2TokenizerFast):
    train_dl  = DataLoader(FineWebEduDataset(tokenizer, cfg.seq_len),
                           batch_size=cfg.batch_size, num_workers=0)
    val_wt_dl = DataLoader(WikiText103ValDataset(tokenizer, cfg.seq_len),
                           batch_size=cfg.batch_size, num_workers=0)
    return train_dl, val_wt_dl


# ─── Lightning ────────────────────────────────────────────────────────────────

class TransformerLightning(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg      = cfg
        self.model    = TransformerModel(cfg)
        self.best_ppl: float = float("inf")
        self._val_losses_wt: List[float] = []
        self.last_ppl_wt: float = float("inf")

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch)
        self._val_losses_wt.append(loss.detach().float().item())

    def on_validation_epoch_end(self):
        if self._val_losses_wt:
            ppl_wt = math.exp(min(sum(self._val_losses_wt) / len(self._val_losses_wt), 20.0))
            self.last_ppl_wt = ppl_wt
            if ppl_wt < self.best_ppl:
                self.best_ppl = ppl_wt
            self.log("val_ppl_wikitext103", ppl_wt, prog_bar=True)
            self._val_losses_wt.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr,
                                betas=(0.9, 0.95), weight_decay=0.01)
        warmup = self.cfg.warmup_steps
        total  = self.cfg.max_steps
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main() -> None:
    pl.seed_everything(42, workers=True)
    cfg       = Config()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_dl, val_wt_dl = build_loaders(cfg, tokenizer)
    model    = TransformerLightning(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    root     = Path(__file__).parent.parent.parent

    tag = "transformer_125M"
    print(f"[{tag}] params={n_params/1e6:.2f}M  d={cfg.d_model}  heads={cfg.n_heads}  "
          f"layers={cfg.n_layers}  batch={cfg.batch_size}  steps={cfg.max_steps}  "
          f"tokens={cfg.max_steps * cfg.batch_size * cfg.seq_len / 1e9:.2f}B  "
          f"(Transformer + RoPE, pre-norm, weight tying, grad_checkpoint)")

    trainer = pl.Trainer(
        max_steps=cfg.max_steps, accelerator="gpu", devices=1, precision="bf16-mixed",
        gradient_clip_val=cfg.grad_clip, log_every_n_steps=cfg.log_every,
        val_check_interval=cfg.val_interval, limit_val_batches=50,
        logger=CSVLogger(str(root / "logs"), name=tag),
        callbacks=[
            ModelCheckpoint(dirpath=str(root / "checkpoints" / tag),
                            filename="last", every_n_train_steps=cfg.ckpt_interval,
                            save_top_k=0, save_last=True),
            ModelCheckpoint(dirpath=str(root / "checkpoints" / tag),
                            filename="best_wt", monitor="val_ppl_wikitext103",
                            mode="min", save_top_k=1, save_weights_only=True),
        ],
    )
    resume_path = os.environ.get("RESUME_CKPT") or None
    if resume_path:
        print(f"[{tag}] RESUMING from {resume_path}", flush=True)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=[val_wt_dl],
                ckpt_path=resume_path)
    elapsed = time.time() - t0
    print(f"\n[{tag}] params={n_params/1e6:.2f}M  val_ppl_wt={model.last_ppl_wt:.2f}  "
          f"best_ppl_wt={model.best_ppl:.2f}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
