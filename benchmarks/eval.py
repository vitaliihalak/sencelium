"""
Evaluation / benchmarking script for Sencelium and Transformer-baseline checkpoints.

Usage:
  python benchmarks/eval.py ppl --script scripts/train_sencelium.py \\
      --config configs/65M.yaml --checkpoint path/to/best.ckpt

  python benchmarks/eval.py ppl --script scripts/transformer/train_transformer_65M.py \\
      --checkpoint path/to/best_wt.ckpt

  python benchmarks/eval.py throughput --script scripts/train_sencelium.py \\
      --config configs/65M.yaml [--checkpoint path] [--steps 50]

  python benchmarks/eval.py longcontext --script scripts/train_sencelium.py \\
      --config configs/65M.yaml --checkpoint path/to/best.ckpt \\
      --lengths 512,1024,2048,4096

`--script` points at the training script to evaluate (Sencelium or either
Transformer baseline); `--config` is only needed (and only exists) for
Sencelium's YAML-driven Config. Works against any checkpoint produced by
that script, without reimplementing the model or data pipeline.
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch


def _load_module(script_path: str):
    path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _load_cfg(module, args):
    overrides = getattr(args, "overrides", [])
    if hasattr(module, "load_config"):
        argv = (["--config", args.config] if args.config else []) + overrides
        return module.load_config(argv)
    cfg = module.Config()
    for item in overrides:
        key, _, raw_value = item.lstrip("-").partition("=")
        from ast import literal_eval
        try:
            value = literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        setattr(cfg, key, value)
    return cfg


def _lit_class(module):
    return getattr(module, "Lit", None) or getattr(module, "TransformerLightning")


def _load_lit(module, cfg, checkpoint: str, device: torch.device):
    LitClass = _lit_class(module)
    lit = LitClass.load_from_checkpoint(checkpoint, cfg=cfg, map_location=device, strict=True)
    lit.to(device)
    lit.eval()
    return lit


def cmd_features(args) -> None:
    module = _load_module(args.script)
    if not hasattr(module, "SenceliumModel"):
        raise ValueError(f"{args.script} has no SenceliumModel -- 'features' only "
                          f"applies to Sencelium, not the Transformer baseline")
    cfg = _load_cfg(module, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = _load_lit(module, cfg, args.checkpoint, device)
    lit.model.set_diagnostics(True)

    content = lit._mem_content
    n_slots = content.shape[0]
    n_categories = lit._cat_content.shape[0]
    n_mass, u_eff = module._pool_usage_concentration(lit._mem_U, cfg.mem_usage_mass_fraction)
    raw_r, dm_r, n_used, _, _ = module._pool_near_dup_rates(
        content, cfg.mem_neardup_cos_threshold, cfg.mem_neardup_max_slots)
    esn_alpha = float(lit.model.esn_alpha.item())

    print(f"memory pool: n_slots={n_slots}  n_categories={n_categories}")
    print(f"  usage: {n_mass}/{n_slots} slots hold "
          f"{100*cfg.mem_usage_mass_fraction:.0f}% of total usage  "
          f"(effective slots={u_eff:.1f})")
    print(f"  near-duplicate rate: raw={100*raw_r:.2f}%  "
          f"mean-direction-removed={100*dm_r:.2f}%  (n={n_used})")
    print(f"esn_alpha (learned ESN-residual blend weight): {esn_alpha:.4f}")

    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    _, val_dl = module.build_loaders(cfg, tokenizer)
    val_dl0 = val_dl[0] if isinstance(val_dl, list) else val_dl
    it = iter(val_dl0)

    null_masses, gate_mags, round_deltas = [], [], []
    with torch.no_grad():
        for _ in range(args.batches):
            batch = next(it)
            batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
            _loss(lit, batch)
            null_masses.append(float(lit.model.mem_pool._last_null_mass))
            gate_mags.append(sum(float(b.meta._last_gate_mag) for b in lit.model.blocks)
                              / len(lit.model.blocks))
            round_deltas.append([float(d) for d in lit.model.voice_loop._last_round_deltas])

    avg_null_mass = sum(null_masses) / len(null_masses)
    avg_gate_mag = sum(gate_mags) / len(gate_mags)
    n_rounds = len(round_deltas[0])
    avg_round_deltas = [sum(rd[i] for rd in round_deltas) / len(round_deltas)
                         for i in range(n_rounds)]

    print(f"null_mass (fraction of reads that abstain from memory): {avg_null_mass:.4f}")
    print(f"gate_magnitude (MetaController signal-correction strength, mean over blocks): {avg_gate_mag:.4f}")
    print(f"voice_loop round deltas (relative change per round, round 1..{n_rounds}): "
          + ", ".join(f"{d:.4f}" for d in avg_round_deltas))


def cmd_ppl(args) -> None:
    import pytorch_lightning as pl

    module = _load_module(args.script)
    cfg = _load_cfg(module, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = _load_lit(module, cfg, args.checkpoint, device)

    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    _, val_dl = module.build_loaders(cfg, tokenizer)
    val_dls = val_dl if isinstance(val_dl, list) else [val_dl]

    trainer = pl.Trainer(accelerator=device.type, devices=1, logger=False,
                          enable_checkpointing=False, limit_val_batches=args.batches)
    results = trainer.validate(lit, dataloaders=val_dls)
    for k, v in results[0].items():
        print(f"{k}: {v:.4f}")


def _loss_bf16(lit, batch):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        return _loss(lit, batch)


def _loss(lit, batch):
    out = lit._step(batch)
    return out[0] if isinstance(out, tuple) else out


def cmd_throughput(args) -> None:
    module = _load_module(args.script)
    cfg = _load_cfg(module, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        lit = _load_lit(module, cfg, args.checkpoint, device)
    else:
        LitClass = _lit_class(module)
        lit = LitClass(cfg).to(device)
    lit.train()

    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    train_dl, _ = module.build_loaders(cfg, tokenizer)
    print("fetching batches...")
    it = iter(train_dl)
    batches = [next(it) for _ in range(args.warmup + args.steps)]

    def _to_device(batch):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)

    print("running warmup...")
    optimizer = torch.optim.AdamW(lit.parameters(), lr=1e-4)
    for batch in batches[:args.warmup]:
        loss = _loss_bf16(lit, _to_device(batch))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize() if device.type == "cuda" else None

    t0 = time.time()
    for batch in batches[args.warmup:]:
        loss = _loss_bf16(lit, _to_device(batch))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t0

    it_s = args.steps / elapsed
    tokens_per_step = cfg.batch_size * cfg.seq_len
    print(f"steps={args.steps}  elapsed={elapsed:.2f}s  it/s={it_s:.3f}  "
          f"tokens/s={it_s * tokens_per_step:.0f}")


def _wikitext_stream(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
    toks = []
    for example in ds:
        text = example.get("text", "")
        if text.strip():
            toks.extend(tokenizer.encode(text))
    return toks


def cmd_longcontext(args) -> None:
    module = _load_module(args.script)
    cfg = _load_cfg(module, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = _load_lit(module, cfg, args.checkpoint, device)

    lengths = [int(x) for x in args.lengths.split(",")]

    if hasattr(module, "_build_rope_cache"):
        n = 0
        max_len = max(lengths)
        for blk in lit.model.blocks:
            attn = blk.attn
            cos, sin = module._build_rope_cache(attn.d_head, max_len, device)
            attn.register_buffer("rope_cos", cos, persistent=False)
            attn.register_buffer("rope_sin", sin, persistent=False)
            n += 1
        print(f"extended RoPE cache to {max_len} tokens across {n} blocks")

    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    toks = _wikitext_stream(tokenizer)

    for length in lengths:
        windows = [toks[i:i + length + 1] for i in range(0, len(toks) - length - 1, length)]
        windows = windows[:args.windows]
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for w in windows:
                x = torch.tensor(w[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(w[1:], dtype=torch.long, device=device).unsqueeze(0)
                loss = _loss(lit, (x, y))
                total_loss += float(loss) * y.numel()
                total_tokens += y.numel()
        ppl = math.exp(min(total_loss / total_tokens, 20.0))
        print(f"length={length}  windows={len(windows)}  ppl={ppl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_ppl = sub.add_parser("ppl")
    p_ppl.add_argument("--script", required=True)
    p_ppl.add_argument("--config", default=None)
    p_ppl.add_argument("--checkpoint", required=True)
    p_ppl.add_argument("--batches", type=int, default=50)
    p_ppl.set_defaults(func=cmd_ppl)

    p_thr = sub.add_parser("throughput")
    p_thr.add_argument("--script", required=True)
    p_thr.add_argument("--config", default=None)
    p_thr.add_argument("--checkpoint", default=None)
    p_thr.add_argument("--steps", type=int, default=50)
    p_thr.add_argument("--warmup", type=int, default=10)
    p_thr.set_defaults(func=cmd_throughput)

    p_lc = sub.add_parser("longcontext")
    p_lc.add_argument("--script", required=True)
    p_lc.add_argument("--config", default=None)
    p_lc.add_argument("--checkpoint", required=True)
    p_lc.add_argument("--lengths", default="512,1024,2048,4096")
    p_lc.add_argument("--windows", type=int, default=20)
    p_lc.set_defaults(func=cmd_longcontext)

    p_feat = sub.add_parser("features")
    p_feat.add_argument("--script", required=True)
    p_feat.add_argument("--config", default=None)
    p_feat.add_argument("--checkpoint", required=True)
    p_feat.add_argument("--batches", type=int, default=10)
    p_feat.set_defaults(func=cmd_features)

    args, args.overrides = parser.parse_known_args()
    args.func(args)
    # Sidesteps a harmless-but-noisy crash on interpreter shutdown caused by
    # DataLoader worker processes (num_workers>0) not being torn down cleanly.
    import os
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
