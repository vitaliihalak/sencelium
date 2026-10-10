"""Perplexity versus context length on the SAME text at every length.

    python benchmarks/eval_long_context.py --script scripts/train_sencelium.py \
        --config configs/65M.yaml --checkpoint path/to/best.ckpt

    python benchmarks/eval_long_context.py --script scripts/transformer/train_transformer_65M.py \
        --checkpoint path/to/best_wt.ckpt

The first `--tokens` tokens of the WikiText-103 test split are cut into consecutive,
non-overlapping windows of each length, so every length scores exactly the same tokens
(a window of length L predicts its next L tokens from the L before them). For the
Transformer the rotary cache is extended to the longest length (no retraining), and an
extra row scores the same tokens with a sliding window: 512 tokens of context, stride 256,
only the newest 256 targets of each window counted (the first window counts all 512).
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent


def _eval_module():
    spec = importlib.util.spec_from_file_location("sencelium_eval", HERE / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sencelium_eval"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lengths", default="512,1024,2048,4096")
    p.add_argument("--tokens", type=int, default=122880)
    p.add_argument("--out", default=None)
    args, rest = p.parse_known_args()
    args.overrides = rest

    ev = _eval_module()
    module = ev._load_module(args.script)
    cfg = ev._load_cfg(module, args)
    device = torch.device("cuda")
    lit = ev._load_lit(module, cfg, args.checkpoint, device)
    lengths = [int(x) for x in args.lengths.split(",")]
    is_sencelium = hasattr(lit, "_mem_content")

    if hasattr(module, "_build_rope_cache"):
        for blk in lit.model.blocks:
            attn = blk.attn
            cos, sin = module._build_rope_cache(attn.d_head, max(lengths), device)
            attn.register_buffer("rope_cos", cos, persistent=False)
            attn.register_buffer("rope_sin", sin, persistent=False)
    if is_sencelium:
        lit.model.set_diagnostics(False)
        pool = (lit._mem_content, lit._cat_id, lit._cat_content)

    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    toks = ev._wikitext_stream(tokenizer)[:args.tokens + 1]

    def logits_of(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not is_sencelium:
                return lit.model(x)
            b, t = x.shape
            g = lit._build_global_ctx(b, t, device, torch.float32)
            c = lit._build_colony_ctx(b, t, device, torch.float32)
            return lit.model(x, g, c, *pool)[0]

    def ce_sum(logits, y):
        total = 0.0
        for j in range(0, logits.shape[1], 2048):
            total += F.cross_entropy(logits[0, j:j + 2048].float(), y[0, j:j + 2048], reduction="sum").item()
        return total

    result = {"checkpoint": args.checkpoint, "tokens": args.tokens, "ppl": {}}
    with torch.no_grad():
        for length in lengths:
            total, count = 0.0, 0
            for i in range(0, args.tokens - length + 1, length):
                w = toks[i:i + length + 1]
                x = torch.tensor(w[:-1], device=device).unsqueeze(0)
                y = torch.tensor(w[1:], device=device).unsqueeze(0)
                total += ce_sum(logits_of(x), y)
                count += y.numel()
            result["ppl"][str(length)] = math.exp(total / count)
            result.setdefault("scored_tokens", {})[str(length)] = count
        if not is_sencelium:
            window, stride, total, count = 512, 256, 0.0, 0
            for i in range(0, args.tokens - window + 1, stride):
                w = toks[i:i + window + 1]
                x = torch.tensor(w[:-1], device=device).unsqueeze(0)
                y = torch.tensor(w[1:], device=device).unsqueeze(0)
                lg = logits_of(x)
                keep = slice(0, window) if i == 0 else slice(window - stride, window)
                total += F.cross_entropy(lg[0, keep].float(), y[0, keep], reduction="sum").item()
                count += y[0, keep].numel()
            result["ppl"]["sliding_512_stride_256"] = math.exp(total / count)
            result["scored_tokens"]["sliding_512_stride_256"] = count

    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
