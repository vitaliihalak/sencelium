"""Perplexity on every WikiText-103 test window, weighted by tokens.

    python benchmarks/eval_all_windows.py --script scripts/train_sencelium.py \
        --config configs/65M.yaml --checkpoint path/to/best.ckpt

    python benchmarks/eval_all_windows.py --script scripts/transformer/train_transformer_65M.py \
        --checkpoint path/to/best_wt.ckpt

Every 512-token window of the test split is scored once (553 windows) and the mean token
cross-entropy is exponentiated, so two models are always compared on identical tokens.
For Sencelium it also reports the same windows with the memory pool forced empty
(`no_mem`) and, on the lane-ordered copy of the windows, with recurrent state carried
across a document's windows (`carry`) against the same lane with state reset (`carry_fresh`).
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


def _ce_sum(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", default=None)
    args, rest = p.parse_known_args()
    args.overrides = rest

    ev = _eval_module()
    module = ev._load_module(args.script)
    cfg = ev._load_cfg(module, args)
    device = torch.device("cuda")
    lit = ev._load_lit(module, cfg, args.checkpoint, device)
    tokenizer = module.GPT2TokenizerFast.from_pretrained("gpt2")
    windows = list(module.WikiText103ValDataset(tokenizer, cfg.seq_len))
    is_sencelium = hasattr(lit, "_mem_content")
    result = {"checkpoint": args.checkpoint, "windows": len(windows)}

    if is_sencelium:
        lit.model.set_diagnostics(False)
        pool = (lit._mem_content, lit._cat_id, lit._cat_content)
        empty = (pool[0].new_zeros(0, cfg.d_model), pool[1].new_zeros(0), pool[2].new_zeros(0, cfg.d_model))

    def forward(x, pools):
        if not is_sencelium:
            return lit.model(x)
        b, t = x.shape
        g = lit._build_global_ctx(b, t, device, torch.float32)
        c = lit._build_colony_ctx(b, t, device, torch.float32)
        return lit.model(x, g, c, *pools)[0]

    def score(pools):
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(windows), args.batch_size):
                chunk = windows[i:i + args.batch_size]
                x = torch.stack([w[0] for w in chunk]).to(device)
                y = torch.stack([w[1] for w in chunk]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = forward(x, pools)
                total += _ce_sum(logits, y)
                count += y.numel()
        return math.exp(total / count)

    result["val_ppl"] = score(pool if is_sencelium else None)
    if is_sencelium:
        result["val_ppl_no_mem"] = score(empty)
        result["mem_delta"] = result["val_ppl_no_mem"] - result["val_ppl"]
        lanes = list(module.WikiText103LaneValDataset(tokenizer, cfg.seq_len, cfg.batch_size))
        sums = {"carry": [0.0, 0], "carry_fresh": [0.0, 0]}
        store = module._LaneStore()
        with torch.no_grad():
            for x, y, carry_data, lane, win, _doc in lanes:
                x, y, carry_data, lane, win = (t.to(device) for t in (x, y, carry_data, lane, win))
                b, t = x.shape
                g = lit._build_global_ctx(b, t, device, torch.float32)
                c = lit._build_colony_ctx(b, t, device, torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fresh = lit.model(x, g, c, *pool)[0]
                for j in range(0, b, 4):
                    sums["carry_fresh"][0] += _ce_sum(fresh[j:j + 4], y[j:j + 4])
                sums["carry_fresh"][1] += y.numel()
                del fresh
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state, valid = store.gather(lane, win, carry_data)
                    carried, _, _, _, new_state = lit.model(
                        x, g, c, *pool, state=state, carry=valid if state is not None else None, return_state=True)
                store.scatter(lane, win, valid if state is not None else torch.zeros_like(valid), new_state)
                for j in range(0, b, 4):
                    sums["carry"][0] += _ce_sum(carried[j:j + 4], y[j:j + 4])
                sums["carry"][1] += y.numel()
                del carried
        for name, (total, count) in sums.items():
            result[f"val_ppl_{name}"] = math.exp(total / count)
        result["carry_gain"] = result["val_ppl_carry_fresh"] - result["val_ppl_carry"]
        result["carry_gain_pct"] = 100 * result["carry_gain"] / result["val_ppl_carry_fresh"]

    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
