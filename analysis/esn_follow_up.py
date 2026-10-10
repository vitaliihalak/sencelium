"""Follow-up measurements on the ESN residual: perplexity as the read temperature moves below its clip, the distribution of the slot decays,
and the logit shift the residual adds to tokens that already occurred in the window.

python analysis/esn_follow_up.py 65M [--checkpoint path/to/best.ckpt]
"""
import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tsm = _scale.load(SCALE)
Config, Lit, WikiText103ValDataset = tsm.Config, tsm.Lit, tsm.WikiText103ValDataset

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
POOLS = (lit._mem_content, lit._cat_id, lit._cat_content)
esn = model.esn_residual
cap = {}
orig = esn.forward
tau_override = [None]

def fwd(h_in, s0=None, carry=None):
    with torch.autocast(device_type="cuda", enabled=False):
        h = h_in.float(); Bq, T, _ = h.shape
        write = torch.sigmoid(esn.Ww(h)); v_all = esn.Wv(h).view(Bq, T, esn.n_slots, esn.d_model)
        b = write.unsqueeze(-1) * v_all
        log_a = F.logsigmoid(esn.Wd(h) + 3.0)
        S0 = esn.essence_init.unsqueeze(0).expand(Bq, -1, -1)
        init_slice = esn.essence_init.view(1, 1, esn.n_slots, esn.d_model).expand(Bq, 1, -1, -1)
        er = tsm._decay_scan_chunked(b.permute(0, 2, 1, 3), log_a.transpose(1, 2), S0, esn.chunk).permute(0, 2, 1, 3)
        ea = esn.ln_e(er); ep = torch.cat([init_slice, ea[:, :-1]], 1)
        tau = tau_override[0] if tau_override[0] is not None else esn.tau.clamp(0.1, 10.0)
        attn = torch.softmax(tau * torch.einsum('btd,btkd->btk', F.normalize(h, dim=-1), ep), -1)
        out = esn.ln_ctx(torch.einsum('btk,btkd->btd', attn, ep))
        cap["decay"] = log_a.exp(); cap["write"] = write; cap["out"] = out
    return out.to(h_in.dtype)
esn.forward = fwd

def run(max_batches=None):
    s = n = 0.0
    with torch.no_grad():
        for x, y in batches[:max_batches]:
            x, y = x.to(DEV), y.to(DEV)
            g = lit._build_global_ctx(B, x.shape[1], DEV, torch.float32); c = lit._build_colony_ctx(B, x.shape[1], DEV, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, *_ = model(x, g, c, *POOLS)
            s += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item(); n += y.numel()
    return math.exp(s / n)

for tv in [None, 0.0, -0.05, -0.1, -0.2, -0.5]:
    tau_override[0] = None if tv is None else torch.tensor(tv, device=DEV)
    print(f"  tau={'trained(0.1)' if tv is None else tv}: ppl={run():.3f}", flush=True)
tau_override[0] = None

# decay distribution + logit effect, on 20 batches
dec_all, wr_all = [], []
stats = {k: [0.0, 0] for k in ["x_t", "x_t-1", "x_t-2", "seen_earlier_not_last2", "true_next", "true_next_seen", "true_next_unseen", "vocab_mean"]}
E = model.head.weight.float()
alpha = model.esn_alpha.float()
with torch.no_grad():
    for x, y in batches[:20]:
        x, y = x.to(DEV), y.to(DEV)
        g = lit._build_global_ctx(B, x.shape[1], DEV, torch.float32); c = lit._build_colony_ctx(B, x.shape[1], DEV, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, g, c, *POOLS)
        dec_all.append(cap["decay"].flatten().cpu()); wr_all.append(cap["write"].flatten().cpu())
        d = (alpha * cap["out"]) @ E.T                    # [B,T,V] logit delta from ESN
        d = d - d.mean(-1, keepdim=True)                  # softmax-invariant: center per position
        T = x.shape[1]
        def add(k, v): stats[k][0] += v.sum().item(); stats[k][1] += v.numel()
        add("x_t", d.gather(-1, x.unsqueeze(-1)).squeeze(-1))
        add("x_t-1", d[:, 1:].gather(-1, x[:, :-1].unsqueeze(-1)).squeeze(-1))
        add("x_t-2", d[:, 2:].gather(-1, x[:, :-2].unsqueeze(-1)).squeeze(-1))
        add("true_next", d.gather(-1, y.unsqueeze(-1)).squeeze(-1))
        # seen earlier in window (positions <= t-3), and true-next seen/unseen split
        for bb in range(B):
            seen = torch.zeros(d.shape[-1], dtype=torch.bool, device=DEV)
            xs = x[bb]
            for t in range(T):
                if t >= 3:
                    seen[xs[t - 3]] = True
                if t % 16 == 0 and t >= 3:
                    m = seen.clone(); m[xs[t]] = False; m[xs[t - 1]] = False; m[xs[t - 2]] = False
                    if m.any(): add("seen_earlier_not_last2", d[bb, t][m])
                nxt = y[bb, t]
                st = seen[nxt] | (nxt == xs[t]) | (t >= 1 and nxt == xs[t - 1]) | (t >= 2 and nxt == xs[t - 2])
                add("true_next_seen" if bool(st) else "true_next_unseen", d[bb, t, nxt].view(1))
        add("vocab_mean", d.mean(-1))
dec = torch.cat(dec_all); wr = torch.cat(wr_all)
qs = torch.tensor([0.1, 0.5, 0.9, 0.99, 0.999])
print(f"\n  ESN decay quantiles {qs.tolist()}: {torch.quantile(dec[torch.randperm(len(dec))[:2_000_000]], qs).numpy().round(4)}")
print(f"  frac decay>0.9: {(dec>0.9).float().mean():.4f}  >0.99: {(dec>0.99).float().mean():.5f}  "
      f"(horizon>10 / >100 tokens)   mean horizon={(1/(1-dec).clamp_min(1e-6)).mean():.1f}  median={(1/(1-dec).clamp_min(1e-6)).median():.2f}")
print(f"  write gate mean={wr.mean():.3f}")
print("\n  centered logit delta from the ESN term (alpha*head(esn_out)), mean per category:")
for k, (s, n) in stats.items():
    print(f"    {k:26s} {s/max(n,1):+8.3f}   (n={n})")
