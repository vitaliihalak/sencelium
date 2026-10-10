"""Category-level structure of the memory pool: sizes, redundancy between category centroids, the tokens each category matches best,
and the contexts of the largest categories.

python analysis/categories.py 65M [documents] [--checkpoint path/to/best.ckpt]
"""
import sys, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
tsm = _scale.load(SCALE)
Config, Lit, WikiText103ValDataset, mem_trunk_k = tsm.Config, tsm.Lit, tsm.WikiText103ValDataset, tsm.mem_trunk_k

DEV = "cuda"
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cfg = Config(); lit = Lit(cfg)
ck = torch.load(CKPT,
                map_location="cpu", weights_only=False)
lit.on_load_checkpoint(ck); lit.load_state_dict(ck["state_dict"], strict=False)
lit.eval().to(DEV); model = lit.model; model.set_diagnostics(False)

C = lit._mem_content.float(); cid = lit._cat_id; cc = lit._cat_content.float()
M = cc.shape[0]
sizes = torch.bincount(cid, minlength=M)
print(f"slots={C.shape[0]} cats={M} populated={(sizes>0).sum().item()} max={sizes.max().item()} "
      f"({100*sizes.max().item()/C.shape[0]:.2f}%) mean={sizes.float().mean():.2f} "
      f"median={sizes.float().median():.0f}  sizes<=3: {(sizes<=3).sum().item()}  size==1: {(sizes==1).sum().item()}")
top_sizes = torch.sort(sizes, descending=True).values[:10].tolist()
print(f"top-10 sizes: {top_sizes}  top-10 share of slots: {100*sum(top_sizes)/C.shape[0]:.1f}%")

pool_mean = C.mean(0, keepdim=True)

def pair_stats(X, label):
    Xn = F.normalize(X, dim=-1)
    S = Xn @ Xn.T
    S.fill_diagonal_(-2)
    nn_sim = S.max(1).values
    off = S[~torch.eye(len(X), dtype=torch.bool, device=X.device)]
    print(f"  [{label}] mean pairwise cos={off.mean():.4f}  NN cos: median={nn_sim.median():.3f} "
          f"p90={nn_sim.quantile(0.9):.3f} max={nn_sim.max():.3f}  "
          f"cats with NN cos>0.9: {(nn_sim>0.9).sum().item()}  >0.8: {(nn_sim>0.8).sum().item()}  "
          f">0.7: {(nn_sim>0.7).sum().item()}")
    return nn_sim, S

print("\n1. centroid redundancy")
nn_raw, _ = pair_stats(cc, "content raw")
nn_dm, S_dm = pair_stats(cc - pool_mean, "content mean-removed (pool mean)")
with torch.no_grad():
    kk = model.mem_pool.Wk(lit._cat_content).float()
nn_k, _ = pair_stats(kk, "routing key Wk(cat) raw")
nn_kdm, _ = pair_stats(kk - kk.mean(0, keepdim=True), "routing key mean-removed")
# does NN similarity depend on category size?
small = sizes <= 6; big = sizes >= 20
print(f"  mean-removed NN cos: small cats (<=6, n={small.sum().item()}) median={nn_dm[small].median():.3f}  "
      f"big cats (>=20, n={big.sum().item()}) median={nn_dm[big].median():.3f}")
# member slots: within-category vs between-category slot cosine (mean-removed)
Cn = F.normalize(C - pool_mean, dim=-1)
same = [];
for m in range(M):
    idx = (cid == m).nonzero(as_tuple=True)[0]
    if idx.numel() >= 2:
        s = Cn[idx] @ Cn[idx].T
        same.append(s[~torch.eye(len(idx), dtype=torch.bool, device=s.device)].mean().item())
g = torch.Generator(device=DEV).manual_seed(0)
ri = torch.randint(0, C.shape[0], (20000,), device=DEV, generator=g); rj = torch.randint(0, C.shape[0], (20000,), device=DEV, generator=g)
rand = (Cn[ri] * Cn[rj]).sum(-1)
print(f"  slot cos (mean-removed): within-category mean={sum(same)/len(same):.3f}  random pairs mean={rand.mean():.3f}")

print(f"\n2. token-level NN labels over {N_DOCS} WikiText-103 test windows")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
k_blocks = mem_trunk_k(cfg)
vecs, toks, ctxs = [], [], []
it = iter(WikiText103ValDataset(tok, cfg.seq_len))
with torch.no_grad():
    for _ in range(N_DOCS):
        x, _y = next(it)
        xb = x.unsqueeze(0).to(DEV)
        gctx = lit._build_global_ctx(1, xb.shape[1], DEV, torch.float32)
        cctx = lit._build_colony_ctx(1, xb.shape[1], DEV, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model.forward_trunk(xb, gctx, cctx, k_blocks)
        vecs.append(h[0].float()); toks.append(x)
V = torch.cat(vecs)                     # [N*T, d]
allt = torch.cat(toks)
print(f"  token vectors: {V.shape[0]}  mem_trunk_k={k_blocks}")

def labels(Vx, Cx, tag):
    Vn = F.normalize(Vx, dim=-1); Cn_ = F.normalize(Cx, dim=-1)
    top = []
    for s in range(0, M, 100):
        sims = Vn @ Cn_[s:s+100].T               # [N*T, 100]
        top.append(sims.topk(10, dim=0).indices.T.cpu())   # [100,10]
    top = torch.cat(top)                          # [M,10]
    top1_tok = [tok.decode([int(allt[top[m, 0]])]) for m in range(M)]
    # majority token among top-10 matches, and its share
    maj = []
    for m in range(M):
        cnt = collections.Counter(tok.decode([int(allt[i])]) for i in top[m])
        t, n = cnt.most_common(1)[0]; maj.append((t, n))
    c1 = collections.Counter(top1_tok)
    cm = collections.Counter(t for t, _ in maj)
    n_distinct = len(c1)
    print(f"  [{tag}] distinct top-1 tokens across {M} cats: {n_distinct}; "
          f"most shared top-1 tokens: {c1.most_common(8)}")
    print(f"  [{tag}] majority-of-top10 token: distinct={len(cm)}; most shared: {cm.most_common(8)}")
    return top, top1_tok

top_raw, t1_raw = labels(V, cc, "raw")
top_dm, t1_dm = labels(V - pool_mean, cc - pool_mean, "mean-removed")

print("\n3. the 6 biggest categories + 6 random small ones, top-5 contexts (mean-removed)")
def show(m, top):
    print(f"  cat {m} (size {sizes[m].item()}), NN cat cos(dm)={nn_dm[m]:.3f}:")
    for i in top[m, :5].tolist():
        lo = max(0, i - 10); d0 = (i // cfg.seq_len) * cfg.seq_len; lo = max(lo, d0)
        s = tok.decode(allt[lo:i].tolist()).replace("\n", " ")
        print(f"      ...{s[-60:]} >>>{tok.decode([int(allt[i])])}<<<")
big_ids = torch.argsort(sizes, descending=True)[:6].tolist()
g2 = torch.Generator().manual_seed(0)
small_pool = (sizes <= 6).nonzero(as_tuple=True)[0].cpu()
small_ids = small_pool[torch.randperm(len(small_pool), generator=g2)[:6]].tolist()
for m in big_ids + small_ids:
    show(m, top_dm)

# Near-duplicate clusters: groups of categories whose mean-removed centroid NN cos > 0.8
print("\n4. category pairs with mean-removed centroid cos > 0.8 (up to 15), with their top-1 tokens")
iu = torch.triu_indices(M, M, 1, device=S_dm.device)
vals = S_dm[iu[0], iu[1]]
order = torch.argsort(vals, descending=True)
for o in order[:15].tolist():
    a, b = iu[0, o].item(), iu[1, o].item()
    if vals[o] < 0.8: break
    print(f"  cats {a}(n={sizes[a].item()}) & {b}(n={sizes[b].item()}): cos={vals[o]:.3f}  "
          f"top1: {t1_dm[a]!r} / {t1_dm[b]!r}")
print(f"  total pairs >0.8: {(vals>0.8).sum().item()}  >0.9: {(vals>0.9).sum().item()}")
