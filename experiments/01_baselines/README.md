# 01 Baselines: T5 vs. CodeT5, `p_max_len` 256 vs. 1024

Launcher: [`run.sh`](run.sh). Results: [`results/01_baselines/`](../../results/01_baselines/). Embedding diagnostics
(collapse, BM25, centering): [`results/01_baselines/t5_vs_codet5/README.md`](../../results/01_baselines/t5_vs_codet5/README.md).

**Results may vary by run.** GPU training is not bit-exact (T5's `relative_attention_bias` gradients are
nondeterministic), and with near-collapsed embeddings that is enough to move a cell's test MRR by up to ~0.05 between
runs (e.g. T5 FOSSIG -> S1000D_spec_sample 0.1498 vs. 0.2025). 

Each target's test corpus contains only its own test-split documents (S1000D_spec_sample 21, BIKE_7 6, s1kd-tools-doc 3, FOSSIG 2), so a random
ordering already scores MRR = H(n)/n: 0.1736, 0.4083, 0.6111, 0.7500.

Three experiments below, all seed 42, original XML, dot product, batch 16, in-batch negatives:

| Name here | Folder | Model | `p_max_len` | Data | Pairs run |
| --- | --- | --- | --- | --- | --- |
| T5 | `01_baselines/t5_s42/` | `t5-base` | 256 | `data/xml/tokenized/data_t5/` | 12 / 12 |
| CodeT5 | `01_baselines/codet5_s42/` | `Salesforce/codet5-base` | 256 | `data/xml/tokenized/data_codet5/` | 12 / 12 |
| CodeT5 p1024 (codet5_1024) | `01_baselines/cnt_p1024_s42/` | `Salesforce/codet5-base` | 1024 | `data/xml/tokenized/data_cnt/` | 12 / 12 |

`01_baselines/cnt_p256_s42/` is CodeT5 at `p_max_len` 256 on `data_cnt`, the 3 S1000D_spec_sample pairs only (table at the end).

## Fine-tuning test evaluation

**T5** (`01_baselines/t5_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.6111 | 0.7500 | 0.1671 |
| s1kd-tools-doc | 0.4917 | — | 0.7500 | 0.1666 |
| fossig | 0.4361 | 0.6111 | — | 0.2025 |
| s1000d-spec | 0.4306 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

**CodeT5** (`01_baselines/codet5_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.6111 | 0.7500 | 0.1777 |
| s1kd-tools-doc | 0.4083 | — | 0.7500 | 0.1696 |
| fossig | 0.4083 | 0.6111 | — | 0.1631 |
| s1000d-spec | 0.4306 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

**CodeT5 p1024 / codet5_1024** (`01_baselines/cnt_p1024_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.8333 | 0.7500 | 0.1538 |
| s1kd-tools-doc | 0.4556 | — | 0.7500 | 0.1756 |
| fossig | 0.4083 | 0.6111 | — | 0.1976 |
| s1000d-spec | 0.4222 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

## Observation

- **Everything is at or near chance.** 20 of the 27 test cells on the three small targets (all three experiments) equal chance exactly (0.6111, 0.7500,
  0.4083), which is what a model scoring every document the same produces. Of the 7 exceptions, 6 are on the bike target (6 test docs, 0.4222–0.4917) and
  one is BIKE_7 -> s1kd-tools-doc in CodeT5 p1024 (0.8333 vs. chance 0.6111, a 3-doc test set), i.e. one or two queries moving.

`p_max_len` 256 vs. 1024 (test MRR, chance 0.1736):

| Pretrain source -> S1000D_spec_sample | p256 (`cnt_p256_s42`) | p1024 (`cnt_p1024_s42`) |
| --- | --- | --- |
| BIKE_7 | 0.1777 | 0.1538 |
| FOSSIG | 0.1631 | 0.1976 |
| s1kd-tools-doc | 0.1696 | 0.1756 |

Neither length is consistently better; all six cells are within run-to-run noise of chance.

The fix (mean pooling + cosine) is experiment 02, branch `exp/02-pooling-fix`.
