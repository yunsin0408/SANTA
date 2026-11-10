### AdvTest Poincare ablation: exp 09 vs exp 11, one variable at a time

Launcher: [`run.sh`](run.sh) ([`run_advtest_ablation.sh`](run_advtest_ablation.sh)). Results: [`results/12_advtest_ablation/`](../../results/12_advtest_ablation/).

Exp 09 (mean pooling, Poincare) reaches 0.3415 AdvTest test MRR; exp 11 (the paper's first-token pooling, Poincare) only 0.098.
The two differ in five settings at once, so the gap can't be attributed. Both run the same track: CodeSearchNet python
pretraining (SDA+MEP, 10 epochs, lr 5e-5) -> AdvTest finetuning (12 epochs, lr 1e-5), Poincare similarity, seed 42, best
checkpoint per stage by AdvTest valid MRR, test on AdvTest test (19,210 x 19,210).

| | Pooling | `poincare_scale` | Temperature | Batch | `p_max_len` | Data |
| --- | --- | --- | --- | --- | --- | --- |
| exp 09 | mean | 0.25 | 0.05 | 16 x 8 | 240 | 240 / 256 |
| exp 11 | first | 0.006 | 1.0 | 128 | 256 | 240 / 256 |
| `ft_temp005` | first | 0.006 | **0.05** | 128 | 256 | 240 / 256 |
| `mean_p256_b128` | **mean** | **0.25** | 0.05 | 128 | 256 | 240 / 256 |
| `ft_b16x8` | first | 0.006 | (from round 1) | **16 x 8** | 256 | 240 / 256 |
| `mean_p1024_b128` | mean | 0.25 | 0.05 | 128 | **1024** | **1024** |

"Data" is the code truncation baked into the pre-tokenized pretrain / finetune files (`processing/Code/build_code_entity.py`,
`build_code.py`). `p_max_len` above that does nothing for training, so the 1024 run uses data rebuilt with `--max_len 1024`
(`run.sh build1024` -> `data/processed/code/python_1024/`; same raw inputs and seed, only the truncation differs).
The scales 0.25 (mean) and 0.006 (first) put the reps at the same hyperbolic radius (~1.75), so `mean_p256_b128` vs
`ft_temp005` is effectively pooling. 16 x 8 is not just a memory setting: it leaves 15 in-batch negatives instead of 127.

Comparisons:
- exp 11 vs `ft_temp005`: temperature
- `ft_temp005` vs `mean_p256_b128`: pooling (+ matched scale)
- exp 09 vs `mean_p256_b128`: batch + `p_max_len` 240 -> 256
- `mean_p256_b128` vs `mean_p1024_b128`: length
- exp 11 (or `ft_temp005`) vs `ft_b16x8`: batch

Results: pending.
