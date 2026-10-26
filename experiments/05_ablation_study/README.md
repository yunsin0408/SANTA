### Ablation Study (S1000D_spec_sample, mean pooling + cosine tau 0.05, p_max_len 1024, seed 42)

Launcher: [`run.sh`](run.sh). Results: [`results/05_ablation_study/`](../../results/05_ablation_study/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets).

**Ablation 1: which pretraining** (`mepablate_s42`: SDA only, `--use_generate False`; `meponly_s42`: MEP only,
`--use_contrastive False`; SDA+MEP is exp 02's `fixA_meancos_p1024_s42`, same config and seed).

| Pretrain source | SDA-only (MEP off) | MEP-only (SDA off) | SDA+MEP |
| --- | --- | --- | --- |
| BIKE_7 | 0.8333 | 0.8878 | 0.8984 |
| FOSSIG | 0.9286 | 0.8929 | 0.9286 |
| s1kd-tools-doc | 0.8968 | 0.9524 | 0.9524 |
| **mean** | **0.8862** | **0.9110** | **0.9265** |

SDA+MEP is 0.040 above SDA-only (MEP helps BIKE_7 and s1kd-tools-doc, ties on FOSSIG) and 0.016 above MEP-only, all single-seed
and inside config A's seed spread (sd 0.059, exp 02). None of the three beats exp 02's no-pretraining control (0.940), so
**neither loss on its own is the reason pretraining doesn't help.**

**Ablation 2: pretrain epoch** (`longpretrain{5,10,30,50,100}_s42`):

| Pretrain epochs | Test MRR BIKE_7 | FOSSIG | s1kd-tools-doc | **mean** | Pretrain-dev MRR BIKE_7 / FOSSIG / s1kd-tools-doc |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.8651 | 0.8968 | 0.8825 | **0.8815** | 0.6111 / 0.7500 / 0.8750 |
| 10 | 0.8984 | 0.8968 | 0.9524 | **0.9159** | 0.6111 / 0.7500 / 0.8750 |
| 30 | 0.8571 | 0.9286 | 0.9524 | **0.9127** | 0.6111 / 0.7500 / 0.7500 |
| 50 | 0.7873 | 0.8730 | 0.9524 | **0.8709** | 0.4444 / 0.5000 / 0.8750 |
| 100 | 0.8730 | 0.8571 | 0.9444 | **0.8915** | 0.6667 / 0.5000 / 0.8750 |

Pretraining losses (SDA + MEP, from each run's `pretrain_losses.jsonl`), per source BIKE_7 / FOSSIG / s1kd-tools-doc:

| Pretrain epochs | Final train loss | Final dev loss | Lowest dev loss (at epoch) |
| --- | --- | --- | --- |
| 5 | 3.02 / 6.40 / 3.68 | 1.48 / 4.80 / 3.53 | 1.48 (5) / 4.80 (5) / 3.53 (5) |
| 10 | 1.87 / 4.60 / 1.73 | 1.48 / 3.39 / 2.74 | 1.36 (7) / 3.39 (10) / 2.74 (10) |
| 30 | 0.186 / 1.22 / 0.072 | 3.16 / 1.80 / 2.04 | 1.29 (6) / 1.80 (25) / 2.04 (28) |
| 50 | 0.013 / 0.141 / 0.005 | 5.13 / 2.56 / 2.13 | 1.33 (6) / 1.87 (20) / 1.87 (23) |
| 100 | 0.002 / 0.011 / 0.003 | 2.98 / 3.29 / 2.19 | 1.27 (14) / 1.94 (20) / 1.81 (32) |

- **Train loss keeps falling** Past ~30 epochs the train loss is near zero while the dev loss sits above its minimum
  (BIKE_7's dev loss bottoms out around epoch 6-14 and is 2-4x higher by the end of the 30-100 epoch runs).
- **The model memorizes its pretrain set.** By 100 epochs the train loss is ~0.01 or below for every source (FOSSIG has only 8
  documents), while FOSSIG's pretrain-dev MRR falls from 0.75 to 0.50 from 50 epochs on.
- **Test MRR peaks at 10-30 epochs** (0.916 / 0.913) and drops at 50 and 100 (0.871 / 0.892). 5 epochs is lower too (0.882), and no
  setting beats no pretraining (0.940, exp 02). **More pretraining points toward overfitting**
- Each run's per-step train loss, dev loss every 5 steps and the run's mean train loss are in
  `results/05_ablation_study/<run>/<pair>/pretrain_losses.jsonl` (rows `kind` = `train` / `eval` / `summary`).
  `viz/plot_pretrain_overfitting.py` draws the curves (`results/figures/pretrain_overfitting.png`).

