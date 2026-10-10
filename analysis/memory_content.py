"""Memory-content ablations: is the benefit of the memory read in the stored content, or in the read path being present?
Scores the 552 test windows that form full batches of eight (rows interleaved so each batch mixes distant text) with the real pool,
no pool, a single slot holding the pool mean, random slots with matched moments, permuted slots, and slots replaced by their category
centroids, and repeats the comparison under the carried-state protocol.

python analysis/memory_content.py 65M [--checkpoint path/to/best.ckpt]
"""
import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tsm = _scale.load(SCALE)
Config, Lit, WikiText103ValDataset, WikiText103LaneValDataset, _LaneStore = tsm.Config, tsm.Lit, tsm.WikiText103ValDataset, tsm.WikiText103LaneValDataset, tsm._LaneStore

DEV = "cuda"; B = 8
cfg = Config(); lit = Lit(cfg)
ck = torch.load(CKPT,
                map_location="cpu", weights_only=False)
lit.on_load_checkpoint(ck); lit.load_state_dict(ck["state_dict"], strict=False)
lit.eval().to(DEV); model = lit.model; model.set_diagnostics(False)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
wins = list(WikiText103ValDataset(tok, cfg.seq_len))
N = (len(wins) // B) * B; nb = N // B
batches = [(torch.stack([wins[j + r*nb][0] for r in range(B)]), torch.stack([wins[j + r*nb][1] for r in range(B)]))
           for j in range(nb)]
C = lit._mem_content; cid = lit._cat_id; cc = lit._cat_content; U = lit._mem_U
d = cfg.d_model


def ppl(pools):
    s = n = 0.0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(DEV), y.to(DEV)
            g = lit._build_global_ctx(B, x.shape[1], DEV, torch.float32)
            c = lit._build_colony_ctx(B, x.shape[1], DEV, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, *_ = model(x, g, c, *pools)
            s += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    return math.exp(s / n)


def one_slot(v):
    v = v.view(1, d).to(C.dtype)
    return (v, cid.new_zeros(1), v.clone())

res = {}
res["real pool"] = ppl((C, cid, cc))
res["no pool (C=0)"] = ppl((C.new_zeros(0, d), cid.new_zeros(0), cc.new_zeros(0, d)))
w = U.clamp_min(1e-6); res["1 slot = U-weighted mean"] = ppl(one_slot((C.float() * w[:, None]).sum(0) / w.sum()))
res["1 slot = plain mean"] = ppl(one_slot(C.float().mean(0)))
for seed in (3, 4, 5):
    gg = torch.Generator(device=DEV).manual_seed(seed)
    fake = (torch.randn(C.shape, generator=gg, device=DEV) * C.float().std(0, keepdim=True)
            + C.float().mean(0, keepdim=True)).to(C.dtype)
    res[f"random slots seed{seed} (real cat_content)"] = ppl((fake, cid, cc))
gg = torch.Generator(device=DEV).manual_seed(6)
fake = (torch.randn(C.shape, generator=gg, device=DEV) * C.float().std(0, keepdim=True)
        + C.float().mean(0, keepdim=True)).to(C.dtype)
# categories recomputed from the fake slots (plain mean per category) -> routing also random-content
fc = torch.zeros_like(cc, dtype=torch.float32).index_add_(0, cid, fake.float())
fc = (fc / torch.bincount(cid, minlength=cc.shape[0]).clamp_min(1).float()[:, None]).to(cc.dtype)
res["random slots + their own centroids"] = ppl((fake, cid, fc))
perm = torch.randperm(C.shape[0], generator=torch.Generator().manual_seed(2)).to(DEV)
res["slots permuted"] = ppl((C[perm], cid, cc))
res["each slot -> its category centroid"] = ppl((cc[cid], cid, cc))
for k, v in res.items():
    print(f"  {k:45s} ppl={v:8.3f}   delta vs real={v - res['real pool']:+7.3f}", flush=True)

# stateful carry protocol, real vs random pool
lanes = list(WikiText103LaneValDataset(tok, cfg.seq_len, cfg.batch_size))[:50]
def carry_ppl(pools):
    st = _LaneStore(); s = n = 0.0
    with torch.no_grad():
        for x, y, cd, lane, win, _ in lanes:
            x, y, cd, lane, win = (t.to(DEV) for t in (x, y, cd, lane, win))
            Bb, T = x.shape
            g = lit._build_global_ctx(Bb, T, DEV, torch.float32); c = lit._build_colony_ctx(Bb, T, DEV, torch.float32)
            state, valid = st.gather(lane, win, cd)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _, _, ns = model(x, g, c, *pools, state=state,
                                        carry=valid if state is not None else None, return_state=True)
            st.scatter(lane, win, valid if state is not None else torch.zeros_like(valid), ns)
            s += sum(F.cross_entropy(lg[i:i+4].float().reshape(-1, lg.size(-1)), y[i:i+4].reshape(-1),
                                     reduction="sum").item() for i in range(0, Bb, 4)); n += y.numel()
            del lg
    return math.exp(s / n)
print(f"  carry protocol: real pool={carry_ppl((C, cid, cc)):.3f}  random slots(seed6, own centroids)={carry_ppl((fake, cid, fc)):.3f}  "
      f"no pool={carry_ppl((C.new_zeros(0, d), cid.new_zeros(0), cc.new_zeros(0, d))):.3f}")
