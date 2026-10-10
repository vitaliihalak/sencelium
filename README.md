# Sencelium

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23134120.svg)](https://doi.org/10.5281/zenodo.23134120)

Sencelium is an independent research project building a non-Transformer
language model architecture: gated-delta-rule blocks with a content-blind
"nucleus" channel, a memory pool that keeps spawning, merging and
reorganizing itself during training and at inference, live self-monitoring
signals, and an ESN residual path.

The project doesn't lead with a perplexity win over Transformers — it isn't
one, at these scales (the gap is +18.6%, +11.5% and +9.3% at 30M, 65M and
125M, and narrows with size). What it reports, including the parts that did
not turn out as hoped: the memory pool is load-bearing but its benefit is
structural rather than content-specific; perplexity keeps falling as the
context grows well past the trained length, where a plain Transformer's
rises; and every claim below comes with the control that tests it.

## Key findings

Full methodology and all numbers: [`benchmarks/results.md`](benchmarks/results.md) · [sencelium.com/results](https://sencelium.com/results/).

**Memory structures itself during training, unsupervised — but most of it is idle.**
Starting from zero slots, it spawns new ones and merges similar ones into
categories as training goes: no labels, no fixed taxonomy, and no category
hub (the largest category holds about 2% of the pool). The pool is much
larger than the part that is used, though: 85 to 91% of the slots have almost
no usage, and most slots have a near-duplicate.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| categories formed | 317 | 500 | 708 |
| spawn → merge → reassignments | 5,354 → 1,659 → 8,492 | 5,137 → 778 → 6,850 | 8,452 → 1,789 → 15,814 |
| slots holding 90% of usage | 76 (2.1%) | 154 (3.5%) | 286 (4.3%) |
| slots with usage below 0.01 | 91% | 86% | 85% |
| slots with a near-duplicate (cosine > 0.9) | 44% | 71% | 72% |

**The memory's benefit is structural, not content-specific.** Emptying the
pool at evaluation time (`no_mem`) costs perplexity at every scale, and the
cost is not monotonic in size. But replacing the stored vectors with random
ones of matched statistics changes perplexity by less than 0.7 points, and a
single slot holding the pool's mean restores 57%, 95% and 97% of the benefit:
what the loss rewards is a non-empty, in-distribution read, not what is
stored. A fact written at inference is not used better than an unrelated one.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| val_ppl / no_mem | 270.80 / 275.01 | 124.36 / 148.32 | 79.98 / 86.11 |
| mem_delta | 4.21 (1.6%) | 23.96 (19.3%) | 6.13 (7.7%) |

**Carrying state across a document helps, for free, but not beyond a Transformer's
sliding window.** No retraining, no extra parameters: just not resetting
recurrent state between a document's own windows at evaluation time. Most of
the gain (71 to 74%) is in the first 128 tokens of a window, the removal of a
cold start. A vanilla Transformer scored on the same text with a sliding window
does better still (194.9, 92.8, 60.1 against Sencelium's carried 245.4, 107.0,
66.4).

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| carry_gain | 9.4% | 13.9% | 17.0% |

**The ESN residual acts as a recency prior; the inner voice loop is not converged.**
The ESN branch raises the logits of tokens that already occurred in the window.
Its learned scalar `esn_alpha` grows with scale, but it follows the integrated
learning rate (alpha divided by that integral: −0.95, −0.90, −0.67), so the growth
mostly reflects the length of training, not an architectural signal. The voice
loop's per-round updates shrink but stay nonzero through the third round.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| esn_alpha | −1.74 | −3.61 | −5.14 |
| voice-loop deltas (round 1→3) | 0.299 → 0.140 → 0.110 | 0.467 → 0.251 → 0.126 | 0.643 → 0.369 → 0.123 |

A fair ablation of the ESN branch — replacing its output with the dataset
mean, or with another document's ESN output, rather than zeroing it (zeroing
mostly measures distribution shock) — shows a real, content-specific contribution
of roughly a third:

| val_ppl with the ESN output | 30M | 65M | 125M |
|---|---|---|---|
| real | 270.66 | 124.30 | 79.94 |
| replaced by the dataset mean | 343.53 | 166.30 | 107.95 |
| taken from another document | 404.29 | 194.27 | 124.13 |

## Results

Full numbers, methodology, and reproduction commands: [`benchmarks/results.md`](benchmarks/results.md).
All perplexities are on every WikiText-103 test window, weighted by tokens
(`benchmarks/eval_all_windows.py`); long-context perplexities score the same
tokens at every length (`benchmarks/eval_long_context.py`).

| | Sencelium 30M | Transformer 30M | Sencelium 65M | Transformer 65M | Sencelium 125M | Transformer 125M |
|---|---|---|---|---|---|---|
| val_ppl | 270.80 | 228.33 | 124.36 | 111.56 | 79.98 | 73.17 |
| val_ppl (4096 tokens) | 238.10 | 640.80 | 103.05 | 655.28 | 64.30 | 551.84 |

At the trained context length the Transformer is ahead. Past it, Sencelium's
perplexity keeps falling up to 16 times the trained length while the Transformer
baseline's rises sharply (its rotary cache is extended, with no retraining and no
length-extension method). Two fairer references: against the Transformer's own score
at its trained length, Sencelium is clearly ahead only at 65M and 125M, by a few
percent at 65M and up to 10% at 125M; and against a Transformer evaluated with a
sliding window, Sencelium does not win at any length or scale.

Training a step takes about three times as long as the Transformer's on the same GPU
(1.35 against 4.00 steps per second at 125M, batch of 4, A100 40 GB), even though the
Transformer runs with gradient checkpointing and Sencelium does not.

## Quickstart

```
pip install -r requirements.txt

python scripts/train_sencelium.py --config configs/30M.yaml
python scripts/train_sencelium.py --config configs/65M.yaml --lr=2e-4
python scripts/train_sencelium.py --config configs/125M.yaml

python benchmarks/eval_all_windows.py --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/checkpoint.ckpt
```

`--config` points at a YAML file overriding `Config`'s defaults (see
`configs/`); any trailing `--key=value` overrides the config file. `tag`
(the checkpoint/log directory name) defaults to `sencelium_<config filename>`.

Transformer baselines (for comparison, not part of Sencelium itself):

```
python scripts/transformer/train_transformer_30M.py
python scripts/transformer/train_transformer_65M.py
python scripts/transformer/train_transformer_125M.py
```

## Layout

```
scripts/train_sencelium.py      architecture + training, config-driven
scripts/transformer/            Transformer baselines (30M, 65M, 125M)
configs/                        per-scale YAML configs
benchmarks/                     evaluation scripts (perplexity, long context, throughput, features) and results.md
analysis/                       read-only measurements on the checkpoints (memory, carry, ESN, routing, recall probe)
logs/                           training metrics of the six canonical runs
```

## Architecture

Full write-up, diagrams and design rationale: [sencelium.com/architecture](https://sencelium.com/architecture).

## License

Code and model weights: [Apache License 2.0](LICENSE).
Text, diagrams and the paper: [CC BY 4.0](NOTICE).

## Citation

See [`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository" button.

## Links

[sencelium.com](https://sencelium.com) · [Hugging Face](https://huggingface.co/Sencelium) · [contact](https://sencelium.com/contact/)
