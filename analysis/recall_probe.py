"""Inference-time recall probe: commits an invented fact about an invented subject through commit_experience and scores the log-probability
of the target before and after, against a control that commits an unrelated fact (paired design, 20 trials).

python analysis/recall_probe.py 65M [--checkpoint path/to/best.ckpt]
"""
import sys, copy, math, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import _scale
SCALE, CKPT = _scale.args()
tsm = _scale.load(SCALE)

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

Config, Lit = tsm.Config, tsm.Lit
args = type('Args', (), {'scale': SCALE})()

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== inference-recall probe: {args.scale} ===")
print(f"device={DEVICE}  ckpt={CKPT}")

cfg = Config()
lit = Lit(cfg)
checkpoint = torch.load(CKPT, map_location="cpu")
lit.on_load_checkpoint(checkpoint)
missing, unexpected = lit.load_state_dict(checkpoint["state_dict"], strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
lit.eval()
lit.to(DEVICE)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# (invented subject, category phrase, common target word)
TRIALS = [
    ("Quendrim",  "favorite color",  "red"),
    ("Brelkon",   "favorite color",  "blue"),
    ("Zalvorn",   "favorite day",    "Monday"),
    ("Thindral",  "favorite day",    "Friday"),
    ("Morquez",   "favorite number", "three"),
    ("Velkarin",  "favorite number", "seven"),
    ("Xendros",   "favorite animal", "dog"),
    ("Ravmoth",   "favorite animal", "cat"),
    ("Plendor",   "favorite color",  "green"),
    ("Yarnavix",  "favorite day",    "Sunday"),
    ("Ostrevil",  "favorite number", "five"),
    ("Kelwyndra", "favorite animal", "bird"),
    ("Drenvale",  "favorite color",  "yellow"),
    ("Solmira",   "favorite day",    "Wednesday"),
    ("Ferqual",   "favorite number", "nine"),
    ("Alborix",   "favorite animal", "horse"),
    ("Benwick",   "favorite color",  "purple"),
    ("Tovarnel",  "favorite day",    "Tuesday"),
    ("Dravoth",   "favorite number", "two"),
    ("Elkanis",   "favorite animal", "fish"),
]
N = len(TRIALS)


def fact_text(subj, cat, tgt):
    return f"{subj}'s {cat} is {tgt}."


def prompt_text(subj, cat):
    return f"{subj}'s {cat} is"


CHUNK = cfg.chunk_size


def pad_to_chunk(ids):
    pad = (-len(ids)) % CHUNK
    return ids + [0] * pad


def commit_fact(lit_copy, subj, cat, tgt, times=1):
    ids = tokenizer.encode(fact_text(subj, cat, tgt))
    x = torch.tensor([ids], device=DEVICE)
    gate_effort = torch.zeros(1, len(ids), device=DEVICE)
    stats = None
    for _ in range(times):
        stats = lit_copy.commit_experience(x, 0, len(ids), gate_effort)
    return stats


def score_target(lit_copy, subj, cat, tgt):
    prompt_ids = tokenizer.encode(prompt_text(subj, cat))
    tgt_id = tokenizer.encode(" " + tgt)[0]
    full = prompt_ids + [tgt_id]
    padded = pad_to_chunk(full)
    x = torch.tensor([padded], device=DEVICE)
    B, T = x.shape
    with torch.no_grad():
        global_ctx = lit_copy._build_global_ctx(B, T, DEVICE, torch.float32)
        colony_ctx = lit_copy._build_colony_ctx(B, T, DEVICE, torch.float32)
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, u, e_t, L_div = lit_copy.model(
                    x, global_ctx, colony_ctx,
                    lit_copy._mem_content, lit_copy._cat_id, lit_copy._cat_content)
        else:
            logits, u, e_t, L_div = lit_copy.model(
                x, global_ctx, colony_ctx,
                lit_copy._mem_content, lit_copy._cat_id, lit_copy._cat_content)
    pos = len(prompt_ids) - 1
    logp = F.log_softmax(logits[0, pos].float(), dim=-1)[tgt_id]
    return -logp.item(), math.exp(logp.item())


results = []
for i, (subj, cat, tgt) in enumerate(TRIALS):
    other_subj, other_cat, other_tgt = TRIALS[(i + 1) % N]

    loss_before, p_before = score_target(lit, subj, cat, tgt)

    matched = copy.deepcopy(lit)
    st_m = commit_fact(matched, subj, cat, tgt, times=1)
    loss_matched, p_matched = score_target(matched, subj, cat, tgt)

    mismatch = copy.deepcopy(lit)
    st_x = commit_fact(mismatch, other_subj, other_cat, other_tgt, times=1)
    loss_mismatch, p_mismatch = score_target(mismatch, subj, cat, tgt)

    repeated = copy.deepcopy(lit)
    st_r = commit_fact(repeated, subj, cat, tgt, times=3)
    loss_repeated, p_repeated = score_target(repeated, subj, cat, tgt)

    row = dict(
        subj=subj, cat=cat, tgt=tgt,
        loss_before=loss_before, p_before=p_before,
        loss_matched=loss_matched, p_matched=p_matched,
        loss_mismatch=loss_mismatch, p_mismatch=p_mismatch,
        loss_repeated=loss_repeated, p_repeated=p_repeated,
        delta_matched=loss_before - loss_matched,
        delta_mismatch=loss_before - loss_mismatch,
        delta_repeated=loss_before - loss_repeated,
        spawned_matched=bool(st_m.spawned), merged_matched=bool(st_m.merged),
        mass_matched=float(st_m.total_write_mass), mass_mismatch=float(st_x.total_write_mass),
        mass_repeated=float(st_r.total_write_mass),
    )
    results.append(row)
    print(f"[{i:2d}] {subj:10s} {tgt:8s}  "
          f"p_before={p_before:.4f}  p_matched={p_matched:.4f}  "
          f"p_mismatch={p_mismatch:.4f}  p_repeated={p_repeated:.4f}  "
          f"d_match={row['delta_matched']:+.4f}  d_mismatch={row['delta_mismatch']:+.4f}  "
          f"d_repeat={row['delta_repeated']:+.4f}  mass={st_m.total_write_mass:.4f} "
          f"(floor={cfg.mem_write_mass_floor_infer})  spawn={st_m.spawned} merge={st_m.merged}")


def summarize(key):
    vals = [r[key] for r in results]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    se = math.sqrt(var / len(vals))
    wins = sum(1 for v in vals if v > 0)
    return mean, se, wins


print("\n=== summary ===")
for key, label in [("delta_matched", "matched (1x)"),
                    ("delta_mismatch", "mismatch (control)"),
                    ("delta_repeated", "matched (3x)")]:
    mean, se, wins = summarize(key)
    print(f"{label:20s}  mean_delta_nll={mean:+.4f} +/- {se:.4f}  "
          f"({wins}/{N} trials improved)")

n_spawn = sum(1 for r in results if r["spawned_matched"])
n_merge = sum(1 for r in results if r["merged_matched"])
print(f"\nspawn/merge on matched commits: spawn={n_spawn}/{N}  merge={n_merge}/{N}")
