# Benchmarks

Every number on this page was measured on one NVIDIA A100 40 GB (via Modal) with the scripts in
`benchmarks/` and `analysis/`, against the released checkpoints. WikiText-103 test split, GPT-2
tokenizer, bf16 autocast (the precision the models were trained in). The Transformer baselines are the
vanilla ones in `scripts/transformer/`.

> **Correction (2026-10).** An earlier version of this page compared the models with `eval.py ppl`
> under different batch sizes: Sencelium 65M was scored on the first 400 test windows and the
> Transformer 65M on all 553, which overstated the gap at 65M (15.4% instead of 11.5%). It also
> scored a fixed number of windows per context length, so each length covered different text. Both
> are fixed below: `eval_all_windows.py` scores every window once, weighted by tokens, and
> `eval_long_context.py` scores the same tokens at every length.

## Perplexity

`benchmarks/eval_all_windows.py`: all 553 test windows of 512 tokens, mean token cross-entropy,
exponentiated. `val_ppl_no_mem` scores the same windows with the memory pool forced empty;
`val_ppl_carry` carries recurrent state across the windows of a document (lane-ordered copy of the
windows) and `val_ppl_carry_fresh` is the same lane with the state reset.

| Model            | Params   | val_ppl | val_ppl_no_mem | mem_delta     | val_ppl_carry | carry_gain | carry_gain_% |
|------------------|----------|---------|----------------|---------------|---------------|------------|--------------|
| Sencelium 30M    | 29.638M  | 270.80  | 275.01         | 4.21 (1.6%)   | 245.40        | 25.40      | 9.4%         |
| Transformer 30M  | 29.403M  | 228.33  | —              | —             | —             | —          | —            |
| Sencelium 65M    | 65.885M  | 124.36  | 148.32         | 23.96 (19.3%) | 107.05        | 17.31      | 13.9%        |
| Transformer 65M  | 65.346M  | 111.56  | —              | —             | —             | —          | —            |
| Sencelium 125M   | 126.485M | 79.98   | 86.11          | 6.13 (7.7%)   | 66.38         | 13.61      | 17.0%        |
| Transformer 125M | 126.461M | 73.17   | —              | —             | —             | —          | —            |

The gap of Sencelium to the Transformer is +18.6%, +11.5% and +9.3% at 30M, 65M and 125M. The
values logged at the end of training (bf16, mean of per-batch means) are 275.61 / 232.20, 126.57 /
113.51 and 81.35 / 74.49, the same gaps (+18.7%, +11.5%, +9.2%).

`mem_delta` (the cost of emptying the memory) is **not** monotonic in scale: 1.6%, 19.3%, 7.7% of
`val_ppl`. Substituting the memory's content shows that it measures the presence of a read, not what
is stored (see below).

`carry_gain` is `val_ppl_carry_fresh - val_ppl_carry`; `carry_gain_%` is that as a fraction of
`val_ppl_carry_fresh`. Carrying state is free at inference (no retraining, no extra parameters). It is
**not** better than what a Transformer gets from the same kind of context: scored on the same text with
a sliding window of 512 tokens and stride 256 (`analysis/transformer_sliding_window.py`, no retraining),
the Transformers reach 194.94 (30M), 92.84 (65M) and 60.08 (125M), lower than Sencelium's carried
perplexity at every scale.

## Long context

`benchmarks/eval_long_context.py`: the first 122,880 tokens of the test split are cut into
consecutive, non-overlapping windows of each length, so **every length scores the same tokens**. The
Transformer's rotary cache is extended to the longest length; nothing is retrained and no
length-extension method is used. The last row scores the same tokens with the Transformer and a
sliding window (512 tokens, stride 256).

| length | Sencelium 30M | Transformer 30M | Sencelium 65M | Transformer 65M | Sencelium 125M | Transformer 125M |
|--------|---------------|-----------------|---------------|-----------------|----------------|------------------|
| 512    | 264.98        | 218.26          | 121.44        | 107.34          | 78.80          | 70.71            |
| 1024   | 249.29        | 242.50          | 110.96        | 153.79          | 70.38          | 108.01           |
| 2048   | 241.73        | 400.39          | 105.65        | 337.17          | 66.24          | 262.82           |
| 4096   | 238.10        | 640.80          | 103.05        | 655.28          | 64.30          | 551.84           |
| 8192   | 236.64        | 900.83          | 101.93        | 1059.33         | 63.50          | 967.50           |
| Transformer, sliding window |  | 184.11 |   | 88.33 |   | 57.66 |

Sencelium's perplexity keeps falling as the context grows, to 16 times the trained length; the
Transformer's rises as soon as the window exceeds 512 tokens. At the same length Sencelium is ahead
from 2048 tokens at 30M and from 1024 at 65M and 125M, with raw ratios at 4096 tokens of 2.7, 6.4 and
8.6. Two fairer references: against the Transformer's own score at its trained length (the 512 row),
Sencelium never reaches it at 30M, is ahead from 2048 tokens at 65M (by 1.6%, 4.0%, 5.0% at 2048,
4096, 8192) and from 1024 at 125M (by 0.5%, 6.3%, 9.1%, 10.2% at 1024 to 8192); and against the
Transformer evaluated with a sliding window, Sencelium does not win at any length or scale (its best
values are 29%, 15% and 10% above 184.11, 88.33 and 57.66).

## Throughput

`benchmarks/eval.py throughput`: 50 timed training steps after 10 warm-up steps, sequence length 512,
bf16 autocast (the precision the models were trained in), mean of two repetitions, one NVIDIA RTX PRO 6000
(via Modal), at the training batch of 32 and at a batch of 4.

| Model             | batch | 30M   | 65M  | 125M |
|-------------------|-------|-------|------|------|
| Sencelium, it/s   | 32    | 4.38  | 3.37 | 2.33 |
| Transformer, it/s | 32    | 8.42  | 4.98 | 3.16 |
| Ratio             | 32    | 1.9   | 1.5  | 1.4  |
| Sencelium, it/s   | 4     | 4.66  | 3.86 | 3.11 |
| Transformer, it/s | 4     | 14.04 | 9.45 | 8.43 |
| Ratio             | 4     | 3.0   | 2.4  | 2.7  |

Repetitions differ by at most 7% at batch 32 and by up to 53% at batch 4 (Transformer 65M: 6.95 and 11.96),
so the batch-4 ratios cannot rank the scales. Forward and backward only, no `torch.compile`. Both
Transformer scripts run with gradient checkpointing, Sencelium does not, and Sencelium's per-step
memory-pool bookkeeping is not included; both favour Sencelium, so the real gap is larger. The Transformer
gains much more than Sencelium from the smaller batch, consistent with an architecture that is
latency-bound (many small sequential operations) rather than compute-bound.

> An earlier version of this table ran the benchmark in fp32 on an A100 at batch 4 (ratios 2.9, 3.6, 3.0),
> although both models train in bf16; that overstated the gap because the Transformer gains more from bf16.

## What the memory and the other components do

`analysis/memory_content.py`, 552 test windows. Change in perplexity relative to the real pool:

| Condition | 30M | 65M | 125M |
|---|---|---|---|
| Real pool (perplexity) | 270.66 | 124.30 | 79.94 |
| No pool (empty read) | +4.20 | +23.96 | +6.13 |
| One slot holding the usage-weighted pool mean | +1.79 | +1.13 | +0.18 |
| Random slots, per-dimension mean and spread matched (3 seeds) | -0.64 to -0.27 | -0.09 to +0.03 | +0.43 to +0.48 |
| Slot contents permuted | +0.05 | +0.15 | +0.04 |
| Each slot replaced by its category centroid | -0.04 | +0.27 | +0.08 |

The benefit of the memory is structural: what the loss rewards is a non-empty, in-distribution read,
not the stored vectors. A single slot holding the pool mean restores 57%, 95% and 97% of the benefit.
A fact written at inference through `commit_experience` is not used better than an unrelated one
(`analysis/recall_probe.py`, 20 trials per scale; paired difference +0.0018 ± 0.0028, +0.0007 ± 0.0035,
+0.0009 ± 0.0046 nats at 30M, 65M, 125M).

Carried state (`analysis/state_carry.py`): 71 to 74% of the gain is in the first 128 tokens of a
window (removal of a cold start); the delta state carries nearly all of it (restarting it removes 75%,
91% and 95% of the gain); the gain does not grow with the age of the episode.

The ESN residual (`analysis/routing_and_esn_ablations.py`, `analysis/esn_follow_up.py`) raises the
logits of tokens that already occurred in the window (a recency and copy prior). Replacing its output
by its dataset mean raises perplexity by 27%, 34%, 35%; zeroing it by 50%, 91%, 74%, which mixes
value with distribution shock. Its learned scalar follows the integrated learning rate (alpha divided
by the integral of the learning rate: -0.95, -0.90, -0.67), so its growth with scale mostly reflects the
length of training.

## Sencelium features

Whether the architecture's non-Transformer pieces are doing something. Memory-growth counters are from
the training logs (`logs/`); the rest is reproduced from the checkpoint with `eval.py features` and
`analysis/`.

**Memory pool growth (training logs):**

|                        | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|------------------------|---------------|---------------|----------------|
| documents processed    | 1,753,468     | 3,841,322     | 7,321,083      |
| spawn events           | 5,354         | 5,137         | 8,452          |
| merge events           | 1,659         | 778           | 1,789          |
| category reassignments | 8,492         | 6,850         | 15,814         |

**Final state and component activity (checkpoint):**

|                                                      | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|------------------------------------------------------|---------------|---------------|----------------|
| slots                                                | 3,677         | 4,359         | 6,625          |
| categories (populated)                               | 317 (313)     | 500 (500)     | 708 (707)      |
| largest category                                     | 81 (2.2%)     | 93 (2.1%)     | 133 (2.0%)     |
| slots holding 90% of usage                           | 76 (2.1%)     | 154 (3.5%)    | 286 (4.3%)     |
| slots with usage below 0.01                          | 91%           | 86%           | 85%            |
| slot pairs with cosine above 0.9 (direction removed) | 0.53%         | 1.53%         | 3.46%          |
| slots with at least one such near-duplicate          | 44%           | 71%           | 72%            |
| null_mass (reads that abstain from memory)           | 0.64%         | 0.72%         | 17.16%         |
| esn_alpha (learned ESN-residual scalar)              | -1.74         | -3.61         | -5.14          |
| gate_magnitude (signal-correction strength)          | 0.039         | 0.060         | 0.074          |
| voice-loop round deltas (round 1 -> 3)               | 0.299 -> 0.140 -> 0.110 | 0.467 -> 0.251 -> 0.126 | 0.643 -> 0.369 -> 0.123 |

The category hub is gone (the largest category holds about 2% of the pool), but the pool is mostly
idle and redundant: 85 to 91% of the slots have almost no usage and most slots have a near-duplicate.
The pairwise rate (the quantity logged during training) counts all pairs, so it understates this. The
abstaining part of the read grows during 125M training for a reason we do not know. The inner voice
loop has not converged by its third round at any scale.

## Reproducing

```
python benchmarks/eval_all_windows.py --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt

python benchmarks/eval_all_windows.py --script scripts/transformer/train_transformer_65M.py \
    --checkpoint path/to/best_wt.ckpt

python benchmarks/eval_long_context.py --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt --lengths 512,1024,2048,4096,8192

python benchmarks/eval.py throughput --script scripts/train_sencelium.py \
    --config configs/65M.yaml --batch_size=4 --steps=50 --warmup=10

python benchmarks/eval.py features --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt --batch_size=8

python analysis/memory_content.py 65M --checkpoint path/to/best.ckpt
```

The analysis scripts take the scale as their first argument and are described in
[`analysis/README.md`](../analysis/README.md). Use `configs/30M.yaml` or `configs/125M.yaml` for the
other scales, and `scripts/transformer/train_transformer_<scale>.py` (no `--config`) for the baselines.
