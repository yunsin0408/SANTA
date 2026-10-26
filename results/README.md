# Results (XML track)

Each experiment's launchers (`experiments/NN_*/`), results (`results/NN_*/`) and write-up live on its own branch,
branched from `master`. `master` holds the shared code, the XML data, the demo and the plotting scripts.

| Branch | Experiment |
| --- | --- |
| `exp/01-baselines` | T5 vs. CodeT5, `p_max_len` 256 vs. 1024, all 12 pairs |
| `exp/02-pooling-fix` | Mean pooling / cosine fix |
| `exp/03-query-leakage` | Identifier-free (anonymized) queries vs. leaky originals |
| `exp/04-pool-size` | Widened 332-document candidate pool |
| `exp/05-ablation-study` | Ablation study: which pretraining loss (SDA / MEP), pretrain epoch sweep |
| `exp/06-same-domain` | Pretrain and finetune on two halves of S1000D_spec_sample |
| `exp/07-anonymized` | Anonymized-query version of every S1000D experiment |
| `exp/08-pooled` | Pooled 508-doc split, Poincare similarity, tau sweep, ss25 / q160 variants |
| `exp/09-advtest` | CodeSearchNet AdvTest with the Poincare recipe; zero-shot from S1000D checkpoints |

On `master`:

| Folder | What |
| --- | --- |
| `demo/` | Demo model (`experiments/demo/run_demo_model.sh`) and its gradient-checkpointing rebuild |
| `figures/` | Plots from `viz/` |
