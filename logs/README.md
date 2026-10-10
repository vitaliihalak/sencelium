# Training logs

`metrics.csv` of the six canonical runs (one per model and scale), written by the training scripts'
CSV logger: training loss every 500 steps, validation perplexity at each validation step, and the
diagnostic quantities discussed in the paper (columns named `mem/...`, `esn/...`, `meta/...`,
`nucleus/...`). The Transformer files contain only loss and validation perplexity.
