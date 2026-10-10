# Sencelium

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23134121.svg)](https://doi.org/10.5281/zenodo.23134121)

Sencelium is an independent research project building a non-Transformer
language model architecture: gated-delta-rule blocks with a content-blind
"nucleus" channel, a memory pool that keeps spawning, merging and
reorganizing itself during training and at inference, live self-monitoring
signals, and an ESN residual path.

The project doesn't lead with a PPL win over Transformers at short context —
it isn't one, at these scales. What it does claim, honestly: the memory is
measurably load-bearing (ablating it hurts at every scale tested), and
perplexity scales very differently with context length.

## Key findings

Full methodology and all numbers: [`benchmarks/results.md`](benchmarks/results.md) · [sencelium.com/results](https://sencelium.com/results/).

**Memory structures itself during training, unsupervised.** Starting from
zero slots, it spawns new ones and merges similar ones into categories as
training goes — no labels, no fixed taxonomy. Usage stays concentrated in a
small fraction of slots and near-duplicate content stays low (under 4%,
though it rises with scale), at all three scales.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| categories formed | 317 | 500 | 708 |
| spawn → merge → reassignments | 5,354 → 1,659 → 8,492 | 5,137 → 778 → 6,850 | 8,452 → 1,789 → 15,814 |
| slots holding 90% of usage | 76 (2.1%) | 154 (3.5%) | 286 (4.3%) |
| near-dup rate | 0.53% | 1.54% | 3.38% |

**Memory is load-bearing at every scale.** Forcing the pool empty at eval
time (`no_mem`) costs real perplexity on the same checkpoint, same data.
The size of the cost is not monotonic in model size: it jumps from 30M to
65M and is smaller again at 125M.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| val_ppl / no_mem | 272.46 / 276.68 | 129.39 / 154.36 | 83.44 / 89.90 |
| mem_delta | 4.22 (1.5%) | 24.97 (19.3%) | 6.46 (7.7%) |

**Carrying state across a document helps, for free.** No retraining, no
extra parameters — just not resetting recurrent state between a document's
own windows at eval time.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| carry_gain | 9.6% | 14.4% | 17.7% |

**The ESN residual and the inner voice loop are both doing real work.**
`esn_alpha`'s magnitude grows with scale instead of decaying toward zero;
the voice loop's per-round updates shrink but stay nonzero through the
third round — iterative refinement, not a no-op pass.

| | Sencelium 30M | Sencelium 65M | Sencelium 125M |
|---|---|---|---|
| esn_alpha | −1.74 | −3.61 | −5.14 |
| voice-loop deltas (round 1→3) | 0.299 → 0.140 → 0.110 | 0.467 → 0.251 → 0.126 | 0.643 → 0.369 → 0.123 |

A fair ablation of the ESN branch at 65M -- replacing its output with the
dataset mean, or with another document's ESN output, rather than zeroing
it (zeroing mostly measures distribution shock, not content value) --
shows a real, measurable content-specific contribution:

| ESN ablation (65M) | val_ppl |
|---|---|
| trained (real) | 124.30 |
| output → dataset-mean | 166.29 |
| output → another document's ESN output | 194.27 |

## Results

Full numbers, methodology, and reproduction commands: [`benchmarks/results.md`](benchmarks/results.md).

| | Sencelium 30M | Transformer 30M | Sencelium 65M | Transformer 65M | Sencelium 125M | Transformer 125M |
|---|---|---|---|---|---|---|
| val_ppl | 272.46 | 229.49 | 129.39 | 112.14 | 83.44 | 76.09 |
| val_ppl (4096) | 238.19 | 640.93 | 102.99 | 654.76 | 64.27 | 551.81 |

At short context Transformer is ahead on raw perplexity. Past ~1024 tokens
that reverses: Sencelium's perplexity keeps falling with context, the
Transformer baseline's rises sharply past the only length it was ever
trained on (512), even with its RoPE cache extended and no retraining.
Measured against that floor (its own 512 score) rather than a raw ratio,
Sencelium 65M is a clear win from 2048 tokens on and 125M from 1024; 30M is
more mixed.

## Quickstart

```
pip install -r requirements.txt

python scripts/train_sencelium.py --config configs/30M.yaml
python scripts/train_sencelium.py --config configs/65M.yaml --lr=2e-4
python scripts/train_sencelium.py --config configs/125M.yaml

python benchmarks/eval.py ppl --script scripts/train_sencelium.py \
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
benchmarks/eval.py              ppl / throughput / long-context evaluation
benchmarks/results.md           full results + methodology
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
