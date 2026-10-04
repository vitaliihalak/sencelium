# Sencelium

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23134121.svg)](https://doi.org/10.5281/zenodo.23134121)

Sencelium is an independent research project building a non-Transformer
language model architecture: gated-delta-rule blocks with a content-blind
"nucleus" channel, a memory pool that keeps spawning, merging and
reorganizing itself during training and at inference, live self-monitoring
signals, and an ESN residual path.

The project doesn't lead with a PPL win over Transformers at short context —
it isn't one, at these scales. What it does claim, honestly: the memory is
measurably load-bearing (ablating it hurts, more at 65M than 30M), and
perplexity scales very differently with context length.

## Results

Full numbers, methodology, and reproduction commands: [`benchmarks/results.md`](benchmarks/results.md).

| | Sencelium 30M | Transformer 30M | Sencelium 65M | Transformer 65M |
|---|---|---|---|---|
| val_ppl | 272.46 | 229.49 | 129.39 | 112.14 |
| val_ppl (4096) | 238.19 | 640.93 | 102.99 | 654.76 |

At short context Transformer is ahead on raw perplexity. Past ~1024 tokens
that reverses: Sencelium's perplexity keeps falling with context, the
Transformer baseline's rises sharply past the only length it was ever
trained on (512), even with its RoPE cache extended and no retraining.
Measured against that floor (its own 512 score) rather than a raw ratio,
Sencelium 65M is a clear win from 2048 tokens on; 30M is more mixed.

## Quickstart

```
pip install -r requirements.txt

python scripts/train_sencelium.py --config configs/30M.yaml
python scripts/train_sencelium.py --config configs/65M.yaml --lr=2e-4

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
```

## Layout

```
scripts/train_sencelium.py      architecture + training, config-driven
scripts/transformer/            Transformer baselines (30M, 65M)
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
