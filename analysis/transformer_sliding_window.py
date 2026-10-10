"""Evaluation of an unmodified Transformer checkpoint on the same windows: token-weighted perplexity, a per-position split, and a
sliding-window evaluation (window 512, stride 256) as the reference for the carried-state numbers.

python analysis/transformer_sliding_window.py 65M [--checkpoint path/to/best_wt.ckpt]
"""
import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tt = _scale.load_transformer(SCALE)
Config, TransformerLightning, WikiText103ValDataset = tt.Config, tt.TransformerLightning, tt.WikiText103ValDataset
from datasets import load_dataset

DEV = "cuda"
cfg = Config()
m = TransformerLightning(cfg)
ck = torch.load(CKPT,
                map_location="cpu", weights_only=False)
print("global_step", ck.get("global_step"), m.load_state_dict(ck["state_dict"]))
m.eval().to(DEV)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
wins = list(WikiText103ValDataset(tok, cfg.seq_len))
print("windows", len(wins))
K = 128
s = [0.0, 0.0, 0.0]; n = [0, 0, 0]; batch_means = []
with torch.no_grad():
    for i in range(0, len(wins), 32):
        x = torch.stack([w[0] for w in wins[i:i+32]]).to(DEV); y = torch.stack([w[1] for w in wins[i:i+32]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.model(x)
        ce = torch.cat([F.cross_entropy(lg[j:j+4].float().reshape(-1, lg.size(-1)), y[j:j+4].reshape(-1),
                                        reduction="none").view(-1, x.shape[1]) for j in range(0, x.shape[0], 4)])
        batch_means.append(ce.mean().item())
        s[0] += ce.sum().item(); n[0] += ce.numel()
        s[1] += ce[:, :K].sum().item(); n[1] += ce[:, :K].numel()
        s[2] += ce[:, K:].sum().item(); n[2] += ce[:, K:].numel()
print(f"batch-mean convention (== val_ppl_wikitext103): {math.exp(sum(batch_means)/len(batch_means)):.3f}")
print(f"token-weighted: {math.exp(s[0]/n[0]):.3f}   t<{K}: {math.exp(s[1]/n[1]):.3f}   t>={K}: {math.exp(s[2]/n[2]):.3f}")

# sliding window over the same continuous token stream the windows are cut from
ids = []
for ex in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True):
    t = ex.get("text", "")
    if t.strip():
        ids.extend(tok.encode(t))
ids = torch.tensor(ids)
T, stride = 512, 256
tot = cnt = 0.0
with torch.no_grad():
    starts = list(range(0, len(ids) - 1, stride))
    for b0 in range(0, len(starts), 16):
        xs, ys, masks = [], [], []
        for st in starts[b0:b0+16]:
            end = min(st + T, len(ids) - 1)
            if end - st < 2: continue
            x = ids[st:end]; y = ids[st+1:end+1]
            msk = torch.zeros(end - st, dtype=torch.bool)
            msk[(0 if st == 0 else T - stride):] = True
            pad = T - (end - st)
            xs.append(F.pad(x, (0, pad))); ys.append(F.pad(y, (0, pad))); masks.append(F.pad(msk, (0, pad)))
        x = torch.stack(xs).to(DEV); y = torch.stack(ys).to(DEV); msk = torch.stack(masks).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.model(x)
        for j in range(0, x.shape[0], 4):
            ce = F.cross_entropy(lg[j:j+4].float().reshape(-1, lg.size(-1)), y[j:j+4].reshape(-1), reduction="none")
            mm = msk[j:j+4].reshape(-1)
            tot += ce[mm].sum().item(); cnt += mm.sum().item()
print(f"sliding window (512, stride 256) over the whole test stream: ppl={math.exp(tot/cnt):.3f}  tokens={int(cnt)}")
