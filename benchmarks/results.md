# Benchmarks

Measured locally on a single RTX 4060 (8GB), via `benchmarks/eval.py` against
the canonical checkpoints. WikiText-103 test split, GPT-2 tokenizer. Training
used bf16; these eval runs use PyTorch Lightning's default precision (fp32,
no autocast).

- `ppl`: 50 validation batches, batch_size=16 (Sencelium 65M and both 125M
  models used 8, to fit 8GB)
- `throughput`: 50 timed steps after 10 warmup steps, batch_size=4, seq_len=512
- `longcontext`: 30 non-overlapping windows per length, batch_size=1, drawn
  from the start of WikiText-103 test -- so each length row scores a
  different amount of underlying text (~15k tokens at 512, ~123k at 4096).
  Transformer's RoPE cache is extended to the max tested length before
  evaluation (no retraining, no sliding-window re-scoring).

## Perplexity

| Model           | Params  | val_ppl | val_ppl_no_mem | mem_delta | val_ppl_carry | carry_gain | carry_gain_% |
|-----------------|---------|---------|----------------|-----------|---------------|------------|--------------|
| Sencelium 30M   | 29.638M | 272.46  | 276.68         | 4.22      | 244.97        | 25.99      | 9.6%         |
| Transformer 30M | 29.403M | 229.49  | —              | —         | —             | —          | —            |
| Sencelium 65M   | 65.885M | 129.39  | 154.36         | 24.97     | 105.40        | 17.72      | 14.4%        |
| Transformer 65M | 65.346M | 112.14  | —              | —         | —             | —          | —            |
| Sencelium 125M  | 126.485M | 83.44  | 89.90          | 6.46      | 65.33         | 14.09      | 17.7%        |
| Transformer 125M | 126.461M | 76.09 | —              | —         | —             | —          | —            |

`val_ppl_no_mem` re-evaluates with the memory pool forced empty (same
checkpoint, same data). `mem_delta` = `val_ppl_no_mem - val_ppl`: how much
worse the model gets without its memory.

`val_ppl_carry` evaluates with recurrent state carried across a document's
windows instead of resetting at each one, using a second validation
dataloader (lane 1) that serves documents in carry-friendly order.
`val_ppl_fresh` is that same lane's own no-carry baseline -- 270.96 (30M) /
123.12 (65M) / 79.42 (125M) -- not the headline `val_ppl` column above, which comes from a
different dataloader (lane 0) and isn't batch-for-batch comparable to it.
`carry_gain = val_ppl_fresh - val_ppl_carry`, `carry_gain_%` = that as a
fraction of `val_ppl_fresh`. This is a separate claim from memory ablation:
carrying state is free at inference (no retraining, no extra parameters)
and measurably improves prediction quality on both scales -- the model
keeps using what it has already read within the same document, not just
what is stored permanently in the memory pool. The relative gain is larger
at 65M (14.4%) than at 30M (9.6%), and larger again at 125M (17.7%), even
though the absolute PPL delta shrinks -- the effect does not weaken with
scale, it is just measured against a lower baseline PPL.

`mem_delta`, by contrast, does not grow steadily: 4.22 (1.5% of val_ppl) at
30M, 24.97 (19.3%) at 65M, 6.46 (7.7%) at 125M. Memory still helps at every
scale, but the size of the effect is not monotonic in model size.

## Throughput

| Model           | it/s | tokens/s |
|-----------------|------|----------|
| Sencelium 30M   | 3.59 | 7,360    |
| Transformer 30M | 7.12 | 14,580   |
| Sencelium 65M   | 2.68 | 5,480    |
| Transformer 65M | 3.64 | 7,451    |

Training step (forward + backward only, via `eval.py`'s own `_step` call --
not the full `training_step`, so Sencelium's per-step memory-pool upkeep
isn't included here). No `torch.compile`. Both Transformer scripts always
run with gradient checkpointing; Sencelium runs with `use_checkpoint=False`.
Both of these cut in the same direction -- the real training-time gap is
larger than this table shows, not smaller.

Throughput was measured at 30M and 65M only.

## Long context

val_ppl at increasing input length, same checkpoints as above.

| length | Sencelium 30M | Transformer 30M | Sencelium 65M | Transformer 65M | Sencelium 125M | Transformer 125M |
|--------|---------------|-----------------|---------------|-----------------|----------------|------------------|
| 512    | 305.29        | 247.07          | 138.05        | 120.88          | 86.26          | 78.32            |
| 1024   | 269.58        | 260.86          | 117.51        | 160.72          | 74.47          | 112.20           |
| 2048   | 232.58        | 391.22          | 101.45        | 323.69          | 63.82          | 253.81           |
| 4096   | 238.19        | 640.93          | 102.99        | 654.76          | 64.27          | 551.81           |

Sencelium's perplexity falls as context grows; the Transformer baseline's
rises sharply past its training length (512) despite the extended RoPE
cache. At 4096 tokens the raw gap is 2.7x at 30M, 6.4x at 65M and 8.6x at 125M -- but
that compares against a Transformer pushed past the only length it was
ever trained on, with no sliding-window re-scoring to fall back on. A
fairer floor for the Transformer is simply what it already scores at its
native length -- the `512` row above (247.07 at 30M, 120.88 at 65M, 78.32
at 125M), which
a sliding 512-token window would reproduce at every row since no
retraining happens either way. Against that floor, Sencelium 65M is a
clear win from 2048 tokens on (101-103 vs ~121); Sencelium 125M is ahead of
its floor from 1024 tokens on (74.47 vs 78.32) and by about 18% from 2048
(63.8-64.3 vs 78.32); Sencelium 30M is more mixed -- it loses to the floor
at 512-1024 and only edges ahead at 2048-4096 (232-238 vs ~247).

## Sencelium features

Whether the architecture's non-Transformer pieces are actually doing
something, not just present in the code. Memory-growth counters below are
from the original training run's own logs (cumulative counters, not part
of the saved checkpoint, so not independently re-verifiable without
re-training); everything else is reproduced live from the checkpoint via
`eval.py features`. The two don't line up exactly slot-for-slot (spawn - merge
!= final slots for 30M, by 18) because `best.ckpt` is the lowest-val_ppl step,
not necessarily the last step the log totals above are read from.

**Memory pool growth (from training logs):**

|                        | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|------------------------|---------------|---------------|----------------|
| documents processed    | 1,753,468     | 3,841,322     | 7,321,083      |
| spawn events           | 5,354         | 5,137         | 8,452          |
| merge events           | 1,659         | 778           | 1,789          |
| category reassignments | 8,492         | 6,850         | 15,814         |

**Final state and component activity (reproduced from checkpoint):**

|                                             | Sencelium 30M           | Sencelium 65M           | Sencelium 125M          |
|---------------------------------------------|-------------------------|-------------------------|-------------------------|
| slots                                       | 3,677                   | 4,359                   | 6,625                   |
| categories                                  | 317                     | 500                     | 708                     |
| slots holding 90% of usage                  | 76 (2.1%)               | 154 (3.5%)              | 286 (4.3%)              |
| near-dup rate (mean-direction-removed)      | 0.53%                   | 1.54%                   | 3.38%                   |
| null_mass (reads that abstain from memory)  | 0.64%                   | 0.72%                   | 17.16%                  |
| esn_alpha (learned ESN-residual weight)     | -1.74                   | -3.61                   | -5.14                   |
| gate_magnitude (signal-correction strength) | 0.039                   | 0.060                   | 0.074                   |
| voice-loop round deltas (round 1 -> 3)      | 0.299 -> 0.140 -> 0.110 | 0.467 -> 0.251 -> 0.126 | 0.643 -> 0.369 -> 0.123 |

Memory is actively used, not decorative: usage concentrates in a small
fraction of slots (2-4%) and the near-duplicate rate stays under 4%,
though it rises with scale (0.53% to 3.38%). At 30M and 65M the model
reads from memory almost every time (null_mass under 1%); at 125M about one
read in six abstains (17.16%), a shift this evaluation doesn't explain.
`esn_alpha`'s magnitude grows with scale (-1.74, -3.61, -5.14); since the ESN branch has its own learnable input weights, the sign
alone doesn't establish whether the blend is additive or inverted -- what's
measurable is that the branch's contribution grows with scale, not its
direction. The inner voice loop hasn't converged by its last round (deltas
are still meaningfully nonzero at round 3), consistent with
`voice_rounds=3` being a real iterative refinement, not padding.

## Reproducing

```
python benchmarks/eval.py ppl --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt --batches 50 --batch_size=8

python benchmarks/eval.py throughput --script scripts/train_sencelium.py \
    --config configs/65M.yaml --batch_size=4 --steps=50 --warmup=10

python benchmarks/eval.py longcontext --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt \
    --lengths 512,1024,2048,4096 --windows 30

python benchmarks/eval.py features --script scripts/train_sencelium.py \
    --config configs/65M.yaml --checkpoint path/to/best.ckpt --batch_size=8
```

Use `configs/30M.yaml` / `configs/125M.yaml` for the other scales. Replace
`--script`/`--config` with `scripts/transformer/train_transformer_65M.py`
(no `--config`) to run `ppl`/`throughput`/`longcontext` against the Transformer
baseline. `features` is Sencelium-only -- it refuses to run against a script
without a `SenceliumModel`.
