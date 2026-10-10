"""Decomposition of the carried-state gain: per-position split (first 128 tokens against the rest), per episode age, and
leave-one-component-out (the delta, nucleus, ESN or colony state restarts from its initial value at every window).

python analysis/state_carry.py 65M [--checkpoint path/to/best.ckpt]
"""
import sys, math, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tsm = _scale.load(SCALE)
Config, Lit, WikiText103LaneValDataset, _LaneStore = tsm.Config, tsm.Lit, tsm.WikiText103LaneValDataset, tsm._LaneStore

DEV = "cuda"

cfg = Config()
lit = Lit(cfg)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
lit.on_load_checkpoint(ck)
lit.load_state_dict(ck["state_dict"], strict=False)
lit.eval().to(DEV)
model = lit.model
model.set_diagnostics(False)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
ds = WikiText103LaneValDataset(tok, cfg.seq_len, cfg.batch_size)
batches = list(ds)[:50]   # limit_val_batches=50
print(f"lane batches={len(batches)}  windows={sum(b[0].shape[0] for b in batches)}", flush=True)
POOLS = (lit._mem_content, lit._cat_id, lit._cat_content)
K = min(cfg.chunk_size, cfg.seq_len)


def reset_component(state, comp, B):
    st = dict(state)
    if comp == "delta":
        st["delta"] = [blk.S_init.unsqueeze(0).expand(B, *blk.S_init.shape).float()
                       for blk in model.blocks]
    elif comp == "nuc":
        st["nuc"] = [blk.S0_n.unsqueeze(0).expand(B, *blk.S0_n.shape).float() for blk in model.blocks]
    elif comp == "esn":
        st["esn"] = model.esn_residual.essence_init.unsqueeze(0).expand(B, -1, -1).float()
    elif comp == "colony":
        st["col_sum"] = [torch.zeros_like(t) for t in st["col_sum"]]
        st["col_cnt"] = torch.zeros_like(st["col_cnt"])
    return st


variants = ["fresh", "carry_all", "no_delta", "no_nuc", "no_esn", "no_colony"]
acc = {v: {"all": [0.0, 0], "lt": [0.0, 0], "ge": [0.0, 0], "cw": [0.0, 0],
           "cw_lt": [0.0, 0], "cw_ge": [0.0, 0]} for v in variants}
age_acc = {}   # age -> [fresh_sum, carry_sum, n]
stores = {v: _LaneStore() for v in variants if v != "fresh"}
ages = None
t0 = time.time()
with torch.no_grad():
    for bi, (x, y, carry_data, lane, win, _d) in enumerate(batches):
        x, y, carry_data, lane, win = (t.to(DEV) for t in (x, y, carry_data, lane, win))
        B, T = x.shape
        g = lit._build_global_ctx(B, T, x.device, torch.float32)
        c = lit._build_colony_ctx(B, T, x.device, torch.float32)
        if ages is None:
            ages = torch.zeros(1024, dtype=torch.long, device=DEV)
        ages[lane] = torch.where(carry_data, ages[lane] + 1, torch.zeros_like(lane))
        ce_by = {}
        valid_ref = None
        for v in variants:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if v == "fresh":
                    logits, *_ = model(x, g, c, *POOLS)
                else:
                    store = stores[v]
                    state, valid = store.gather(lane, win, carry_data)
                    if state is not None and v != "carry_all":
                        state = reset_component(state, v[3:], B)
                    logits, _, _, _, ns = model(x, g, c, *POOLS, state=state,
                                                carry=valid if state is not None else None,
                                                return_state=True)
                    store.scatter(lane, win, valid if state is not None else torch.zeros_like(valid), ns)
                    if v == "carry_all":
                        valid_ref = valid
            ce = torch.cat([F.cross_entropy(logits[i:i+4].float().reshape(-1, logits.size(-1)),
                                            y[i:i+4].reshape(-1), reduction="none").view(-1, T)
                            for i in range(0, B, 4)])
            del logits
            ce_by[v] = ce
        vr = valid_ref if valid_ref is not None else torch.zeros(B, dtype=torch.bool, device=DEV)
        for v, ce in ce_by.items():
            a = acc[v]
            a["all"][0] += ce.sum().item(); a["all"][1] += ce.numel()
            a["lt"][0] += ce[:, :K].sum().item(); a["lt"][1] += ce[:, :K].numel()
            a["ge"][0] += ce[:, K:].sum().item(); a["ge"][1] += ce[:, K:].numel()
            if vr.any():
                a["cw"][0] += ce[vr].sum().item(); a["cw"][1] += ce[vr].numel()
                a["cw_lt"][0] += ce[vr][:, :K].sum().item(); a["cw_lt"][1] += ce[vr][:, :K].numel()
                a["cw_ge"][0] += ce[vr][:, K:].sum().item(); a["cw_ge"][1] += ce[vr][:, K:].numel()
        ag = ages[lane]
        for b in range(B):
            k_ = min(int(ag[b]), 8)
            e = age_acc.setdefault(k_, [0.0, 0.0, 0])
            e[0] += ce_by["fresh"][b].sum().item(); e[1] += ce_by["carry_all"][b].sum().item(); e[2] += T
print(f"({time.time()-t0:.0f}s)")
ppl = lambda s: math.exp(s[0] / s[1]) if s[1] else float("nan")
print(f"\n{'variant':12s} {'all':>9s} {'t<128':>9s} {'t>=128':>9s} | carried windows only: {'all':>9s} {'t<128':>9s} {'t>=128':>9s}")
for v in variants:
    a = acc[v]
    print(f"{v:12s} {ppl(a['all']):9.3f} {ppl(a['lt']):9.3f} {ppl(a['ge']):9.3f} | "
          f"{'':22s}{ppl(a['cw']):9.3f} {ppl(a['cw_lt']):9.3f} {ppl(a['cw_ge']):9.3f}")

print("\nby episode age (0 = fresh window; 8 = age>=8): fresh_ppl -> carry_ppl  (n windows)")
for k_ in sorted(age_acc):
    f, cr, n = age_acc[k_]
    print(f"  age {k_}: {math.exp(f/n):8.2f} -> {math.exp(cr/n):8.2f}   rel {100*(1-math.exp((cr-f)/n)):5.1f}%   n={n//cfg.seq_len}")
