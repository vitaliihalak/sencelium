"""Read-routing conditions (learned router, incumbent rule, random categories, full pool, no memory) and ESN conditions (hard zero,
dataset mean, another document, temperature sweep, gradient of the loss with respect to the temperature).

python analysis/routing_and_esn_ablations.py 65M [max_windows] [parts] [--checkpoint path/to/best.ckpt]
"""
import sys, math, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tsm = _scale.load(SCALE)
Config, Lit, WikiText103ValDataset = tsm.Config, tsm.Lit, tsm.WikiText103ValDataset

torch.manual_seed(0)
DEV = "cuda"

B = 8
MAX_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
PARTS = sys.argv[2] if len(sys.argv) > 2 else "AB"

cfg = Config()
lit = Lit(cfg)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
lit.on_load_checkpoint(ck)
missing, unexpected = lit.load_state_dict(ck["state_dict"], strict=False)
print(f"global_step={ck['global_step']} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
lit.eval().to(DEV)
model = lit.model
model.set_diagnostics(False)
for p in lit.parameters():
    p.requires_grad_(False)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
wins = []
for x, y in WikiText103ValDataset(tok, cfg.seq_len):
    wins.append((x, y))
    if len(wins) >= MAX_WIN:
        break
N = (len(wins) // B) * B
wins = wins[:N]
nb = N // B
batches = []
for j in range(nb):
    idx = [j + r * nb for r in range(B)]
    batches.append((torch.stack([wins[i][0] for i in idx]), torch.stack([wins[i][1] for i in idx])))
print(f"windows={N} (of the val_dl stream), batches={nb} x B={B}", flush=True)

POOLS = (lit._mem_content, lit._cat_id, lit._cat_content)
EMPTY = (lit._mem_content.new_zeros(0, cfg.d_model), lit._cat_id.new_zeros(0),
         lit._cat_content.new_zeros(0, cfg.d_model))
K = min(cfg.chunk_size, cfg.seq_len)


def run(name, pools=POOLS, keep_pred=False, ref_pred=None):
    ce_all = ce_pre = ce_post = 0.0
    n_all = n_pre = n_post = 0
    agree = 0
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for bi, (x, y) in enumerate(batches):
            x, y = x.to(DEV), y.to(DEV)
            Bb, T = x.shape
            g = lit._build_global_ctx(Bb, T, x.device, torch.float32)
            c = lit._build_colony_ctx(Bb, T, x.device, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, *_ = model(x, g, c, *pools)
            lf = logits.float()
            ce = F.cross_entropy(lf.reshape(-1, lf.size(-1)), y.reshape(-1), reduction="none").view(Bb, T)
            ce_all += ce.sum().item(); n_all += ce.numel()
            ce_pre += ce[:, :K].sum().item(); n_pre += ce[:, :K].numel()
            ce_post += ce[:, K:].sum().item(); n_post += ce[:, K:].numel()
            pr = lf.argmax(-1).cpu()
            if keep_pred:
                preds.append(pr)
            if ref_pred is not None:
                agree += (pr == ref_pred[bi]).sum().item()
    r = dict(name=name, ppl=math.exp(ce_all / n_all), ppl_pos_lt_k=math.exp(ce_pre / n_pre),
             ppl_pos_ge_k=math.exp(ce_post / n_post), nll=ce_all / n_all, nll_ge_k=ce_post / n_post)
    if ref_pred is not None:
        r["top1_agree_vs_default"] = agree / n_all
    print(f"  {name:48s} ppl={r['ppl']:8.3f}  ppl[t<{K}]={r['ppl_pos_lt_k']:8.3f}  "
          f"ppl[t>={K}]={r['ppl_pos_ge_k']:8.3f}"
          + (f"  top1_agree={100*r['top1_agree_vs_default']:.2f}%" if ref_pred is not None else "")
          + f"  ({time.time()-t0:.0f}s)", flush=True)
    return r, preds


results = {}
print("\n=== default (router in control, real pool, trained ESN) ===", flush=True)
r, ref = run("default", keep_pred=True)
results["default"] = r

router = model.read_router
print(f"router._in_control={bool(router._in_control.item())}  lift_ema(inc,router)={router._lift_ema.tolist()}")

if "A" in PARTS:
    print("\n=== PART A: routing / memory ===", flush=True)
    # A1 incumbent routing
    router._in_control.fill_(False)
    results["incumbent_routing"], _ = run("incumbent routing (router out of control)", ref_pred=ref)
    router._in_control.fill_(True)

    # A2 random routing: router in control, but its category scores are
    # column-permuted per batch -> same per-row margin-based selection COUNT
    # distribution, random WHICH categories.
    orig_cos = router.cos_scores
    gen = torch.Generator().manual_seed(1)
    def rand_cos(q, kn):
        s = orig_cos(q, kn)
        perm = torch.randperm(s.shape[1], generator=gen).to(s.device)
        return s[:, perm]
    router.cos_scores = rand_cos
    results["random_routing"], _ = run("random category routing (matched count)", ref_pred=ref)
    router.cos_scores = orig_cos

    # A3 full pool for every position (no routing at all)
    orig_sel = tsm._select_read_slots
    def full_sel(h, cat_id, cat_content, mem_content, *a, return_row_mask=False, **kw):
        full = torch.arange(mem_content.shape[0], device=mem_content.device)
        return (full, None, None) if return_row_mask else full
    tsm._select_read_slots = full_sel
    model.read_router = None
    results["full_pool_no_routing"], _ = run("full pool, no routing (all positions)", ref_pred=ref)
    model.read_router = router
    tsm._select_read_slots = orig_sel

    # A4 no memory
    results["no_mem"], _ = run("no memory (empty pool)", pools=EMPTY, ref_pred=ref)

    # A5 slot content permuted (cat_id/cat_content intact -> routing picks the
    # same categories, but the slots it reads hold other categories' content)
    g2 = torch.Generator().manual_seed(2)
    perm = torch.randperm(lit._mem_content.shape[0], generator=g2).to(DEV)
    results["slot_content_permuted"], _ = run(
        "slot content permuted (routing intact)",
        pools=(lit._mem_content[perm], lit._cat_id, lit._cat_content), ref_pred=ref)

    # A6 pool replaced by random gaussian slots with matched per-dim mean/std
    mc = lit._mem_content.float()
    g3 = torch.Generator(device=DEV).manual_seed(3)
    fake = (torch.randn(mc.shape, generator=g3, device=DEV) * mc.std(0, keepdim=True)
            + mc.mean(0, keepdim=True)).to(lit._mem_content.dtype)
    results["slot_content_random_matched"], _ = run(
        "slot content = random, per-dim mean/std matched",
        pools=(fake, lit._cat_id, lit._cat_content), ref_pred=ref)

if "B" in PARTS:
    print("\n=== PART B: ESN residual ===", flush=True)
    esn = model.esn_residual
    orig_fwd = esn.forward
    state = {"tau": None, "transform": None, "mean_vec": None, "collect": None}

    def esn_forward(h_in, s0=None, carry=None):
        # copy of ScanESNContext.forward (fresh path only) with tau override
        # (NO clamp) and an optional output transform.
        out_dtype = h_in.dtype
        with torch.autocast(device_type=h_in.device.type, enabled=False):
            h = h_in.float()
            Bq, T, _ = h.shape
            write = torch.sigmoid(esn.Ww(h))
            v_all = esn.Wv(h).view(Bq, T, esn.n_slots, esn.d_model)
            b = write.unsqueeze(-1) * v_all
            log_a = F.logsigmoid(esn.Wd(h) + 3.0)
            h_init = esn.essence_init.view(1, 1, esn.n_slots, esn.d_model)
            S0 = esn.essence_init.unsqueeze(0).expand(Bq, -1, -1)
            init_slice = h_init.expand(Bq, 1, -1, -1)
            essence_raw = tsm._decay_scan_chunked(b.permute(0, 2, 1, 3), log_a.transpose(1, 2),
                                                  S0, esn.chunk).permute(0, 2, 1, 3)
            essence_all = esn.ln_e(essence_raw)
            essence_prev = torch.cat([init_slice, essence_all[:, :-1]], dim=1)
            tau = state["tau"] if state["tau"] is not None else esn.tau.clamp(0.1, 10.0)
            h_norm = F.normalize(h, dim=-1)
            raw_scores = torch.einsum('btd,btkd->btk', h_norm, essence_prev)
            attn = torch.softmax(tau * raw_scores, dim=-1)
            ctx = torch.einsum('btk,btkd->btd', attn, essence_prev)
            out = esn.ln_ctx(ctx)
            if state["collect"] is not None:
                state["collect"].append((out.sum((0, 1)).detach(), out.shape[0] * out.shape[1],
                                         attn.detach(), raw_scores.detach(), log_a.detach()))
            if state["transform"] == "mean":
                out = state["mean_vec"].view(1, 1, -1).expand_as(out)
            elif state["transform"] == "shuffle":
                out = out.roll(1, dims=0)
        return out.to(out_dtype)

    esn.forward = esn_forward

    # sanity: patched forward with no override must reproduce default exactly
    results["esn_patched_identity"], _ = run("ESN patched fwd, no override (== default?)", ref_pred=ref)

    # collect mean output + attention stats on the first 1/4 of batches only
    state["collect"] = []
    with torch.no_grad():
        for (x, y) in batches[: max(1, nb // 4)]:
            x = x.to(DEV)
            g = lit._build_global_ctx(B, x.shape[1], x.device, torch.float32)
            c = lit._build_colony_ctx(B, x.shape[1], x.device, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, g, c, *POOLS)
    tot = sum(s for s, _, _, _, _ in state["collect"]); cnt = sum(n for _, n, _, _, _ in state["collect"])
    state["mean_vec"] = tot / cnt
    attn_all = torch.cat([a.reshape(-1, a.shape[-1]) for _, _, a, _, _ in state["collect"]])
    rs_all = torch.cat([r_.reshape(-1, r_.shape[-1]) for _, _, _, r_, _ in state["collect"]])
    la_all = torch.cat([l.reshape(-1, l.shape[-1]) for _, _, _, _, l in state["collect"]])
    state["collect"] = None
    decay = la_all.exp()
    print(f"  ESN attention: max-weight mean={attn_all.max(-1).values.mean():.4f} "
          f"(uniform=0.125)  min-weight mean={attn_all.min(-1).values.mean():.4f}")
    spread = rs_all.max(-1).values - rs_all.min(-1).values
    print(f"  raw score (h_norm . ln_e(slot)) per-token spread across 8 slots: mean={spread.mean():.3f} "
          f"p50={spread.median():.3f} p99={spread.quantile(0.99):.3f}  (x tau=0.1 -> logit spread)")
    print(f"  raw score mean per slot: {rs_all.mean(0).cpu().numpy().round(3)}")
    print(f"  per-slot mean decay: {decay.mean(0).cpu().numpy().round(3)}  "
          f"per-slot median horizon 1/(1-d): {(1/(1-decay).clamp_min(1e-6)).median(0).values.cpu().numpy().round(1)}")
    print(f"  ||mean ESN out||={state['mean_vec'].norm():.2f}  vs typical ||out|| ~ {esn.ln_ctx.weight.norm():.2f}")

    orig_alpha = model.esn_alpha.data.clone()
    model.esn_alpha.data.zero_()
    results["esn_alpha_zero"], _ = run("ESN alpha=0 (hard zero)", ref_pred=ref)
    model.esn_alpha.data.copy_(orig_alpha)

    state["transform"] = "mean"
    results["esn_mean_ablation"], _ = run("ESN out -> its dataset-mean vector", ref_pred=ref)
    state["transform"] = "shuffle"
    results["esn_crossdoc_shuffle"], _ = run("ESN out from ANOTHER doc (row roll)", ref_pred=ref)
    state["transform"] = None

    for tv in [0.0, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0]:
        state["tau"] = torch.tensor(tv, device=DEV)
        results[f"esn_tau_{tv}"], _ = run(f"ESN tau={tv} (unclamped override)", ref_pred=ref)
    state["tau"] = None

    # dCE/dtau with the clamp bypassed, per batch -> mean, std, sign consistency
    print("\n  dCE/dtau (clamp bypassed), token-mean CE, per batch:", flush=True)
    for tv in [0.1, 0.05, 0.02]:
        gs = []
        for (x, y) in batches:
            x, y = x.to(DEV), y.to(DEV)
            tau_t = torch.tensor(float(tv), device=DEV, requires_grad=True)
            state["tau"] = tau_t
            g = lit._build_global_ctx(B, x.shape[1], x.device, torch.float32)
            c = lit._build_colony_ctx(B, x.shape[1], x.device, torch.float32)
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, *_ = model(x, g, c, *POOLS)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                (gt,) = torch.autograd.grad(loss, tau_t)
            gs.append(gt.item())
        gs_t = torch.tensor(gs)
        print(f"    tau={tv}: mean dCE/dtau={gs_t.mean():+.5f}  std={gs_t.std():.5f}  "
              f"frac>0={(gs_t > 0).float().mean():.2f}  (n={len(gs)})  "
              f"SNR(mean/std)={gs_t.mean()/gs_t.std():+.2f}", flush=True)
        results[f"dCE_dtau_{tv}"] = dict(mean=gs_t.mean().item(), std=gs_t.std().item(),
                                         frac_pos=(gs_t > 0).float().mean().item())
    state["tau"] = None
    esn.forward = orig_fwd

print("\nDone (read-only).")
