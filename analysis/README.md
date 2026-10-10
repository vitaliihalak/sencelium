# Analysis scripts

Read-only measurements on the released checkpoints. Each script takes the scale as its first argument
and loads the training script and configuration of that scale itself (`_scale.py`), so nothing needs
editing:

```
python analysis/<script>.py 65M [--checkpoint path/to/best.ckpt] [script-specific arguments]
```

| script | what it measures |
|---|---|
| `memory_content.py` | memory-content ablations: real pool, no pool, one slot with the pool mean, random slots with matched moments, permuted slots, slots replaced by their category centroid; also under carried state |
| `state_carry.py` | decomposition of the carried-state gain: first 128 tokens against the rest, per episode age, and leave-one-component-out |
| `categories.py` | category sizes, redundancy between category centroids, the tokens each category matches best |
| `esn_follow_up.py` | ESN residual: perplexity as the read temperature moves below its clip, slot decays, logit shift on tokens already seen |
| `routing_and_esn_ablations.py` | read-routing conditions (router, incumbent rule, random categories, full pool, no memory) and ESN conditions (zero, dataset mean, another document, temperature sweep) |
| `recall_probe.py` | inference-time recall probe: commit an invented fact through `commit_experience`, score the target before and after, against an unrelated fact as control |
| `transformer_sliding_window.py` | the Transformer on the same tokens with a sliding window (512, stride 256), the reference for the carried-state numbers (`--checkpoint` is a `best_wt.ckpt`) |

The results on this repository's checkpoints are summarised in
[`benchmarks/results.md`](../benchmarks/results.md). The scripts use a GPU; the 65M and 125M
checkpoints need more than 8 GB for the carried-state and long-window evaluations, so run those on a
larger card.
