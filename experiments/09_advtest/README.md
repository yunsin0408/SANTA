### AdvTest with the Poincare recipe

Launcher: [`run.sh`](run.sh) ([`run_advtest_poincare.sh`](run_advtest_poincare.sh)). Results: [`results/09_advtest/`](../../results/09_advtest/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move the test MRR slightly.

Does the recipe that works best on S1000D (exp 08: mean pooling, Poincare similarity, `--poincare_scale 0.25`, temperature 0.05) also
work on code? This runs SANTA's Python code track (Table 6 in this repo: CodeSearchNet Python pretraining with SDA+MEP, then finetuning on
AdvTest train) with that recipe instead of the paper's first-token pooling and dot product. Everything else follows `shell/pretrain-code.sh` /
`shell/finetune-code.sh`: the same pre-tokenized data (`data/processed/code/python/`), batch 16 x gradient accumulation 8,
q/p/l max length 50/240/35, pretrain 10 epochs at lr 5e-5, finetune 12 epochs at lr 1e-5, seed 42. Both stages pick their best checkpoint by
AdvTest valid MRR (`data/xml/retrieval_advtest/`, built by `processing/XML/datasets/build_advtest_retrieval.py`); test is AdvTest test,
19,210 queries x 19,210 functions.

Needs the code track's data first (gitignored): `data/AdvTest/` (download in `PIPELINE.md`) and `data/processed/code/python/`
(`shell/process-pretrain-code.sh python`, `shell/process-finetune-code.sh python`).

| Model (AdvTest test MRR) | Pretrain only | Pretrain + finetune |
| --- | --- | --- |
| Poincare recipe (`advtest_poincare_s42`) | 0.3275 | 0.3415 |
| Paper recipe, this repo's Table 6 run (zero-shot / in-batch / hard-negative evals; logs are local, `runs/` is not in git) | 0.420 | 0.428 (in-batch) / 0.419 (hard negatives) |

Best AdvTest valid MRR: 0.3947 after pretraining (`checkpoint-31000`), 0.4099 after finetuning (`checkpoint-22000`).

- **The S1000D recipe does not transfer to code here.** It trails the paper's own recipe by ~0.09 at both stages, so Poincare similarity's
  gain on the pooled S1000D split (exp 08) doesn't carry over to this track (single seed).
- The scale matters: without `--poincare_scale` (mean-pooled CodeT5 reps have norm ~3-4, so `tanh` puts every point on the ball's
  boundary), an earlier Poincare run on this track scored 0.0.
