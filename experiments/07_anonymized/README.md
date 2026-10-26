### Anonymized queries: every S1000D_spec_sample experiment rerun on identifier-free queries

Launcher: [`run.sh`](run.sh). Results: [`results/07_anonymized/`](../../results/07_anonymized/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets).

Exps 01-06 on the anonymized queries of exp 03 (`data/xml/retrieval_anonq/`, no identifiers copied from the document). Training data is
tokenized with a 1024-token document cap (`data_anonq1024`, as `data_cnt` for the original queries); exp 03's own anonymized runs used
`data_anonq` with a 512-token cap, so its numbers are not directly comparable with these. T5 uses `data_anonq_t5` and same-domain uses
`data_samedomain{,_alt}_anonq` (512-token cap, like their original-query counterparts). The "original" column is each run's original-query
counterpart from exps 01-06. Mean test MRR over the 3 S1000D_spec_sample pairs (chance 0.1736), seed 42 unless noted.

| Setting | Folder | Anonymized | Original |
| --- | --- | --- | --- |
| Untrained CodeT5 (zero-shot) | `zeroshot_untrained_retrieval{,_anonq}/codet5-base` | 0.1736 | 0.1739 |
| Untrained T5 (zero-shot) | `zeroshot_untrained_retrieval{,_anonq}/t5-base` | 0.1736 | 0.1733 |
| First-token, T5, p256 | `anonq1024_t5_s42` | 0.1717 | 0.1787 |
| First-token, CodeT5, p256 | `anonq1024_codet5_s42` | 0.1794 | 0.1701 |
| First-token, CodeT5, p1024 | `anonq1024_cnt_p1024_s42` | 0.1977 | 0.1756 |
| Mean pooling + dot, p1024 | `anonq1024_meandot_p1024_s42` | 0.8014 | 0.8748 |
| Mean pooling + cosine, p256 | `anonq1024_meancos_p256_s42` | 0.2999 | 0.5597 |
| **Config A** (mean + cosine, p1024) | `anonq1024_meancos_p1024_s42` | **0.8649** | **0.9265** |
| SDA-only pretraining | `anonq1024_mepablate_s42` | 0.8272 | 0.8862 |
| MEP-only pretraining | `anonq1024_meponly_s42` | 0.8347 | 0.9110 |

- **Exp 01 and 02 replicate.** Untrained and first-token models stay at chance; mean pooling lifts the score to 0.80-0.86. The
  shorter window hurts much more here: p256 drops to 0.30 (0.56 on the original queries), so on paraphrased queries the model needs
  more of each document.
- **No pretraining benefit, as before.** Config A over 3 seeds (`anonq1024_meancos_p1024_s{42,1,2}`) averages 0.842 (sd 0.072) against
  0.859 (sd 0.045) for no pretraining over 5 seeds (`anonq1024_nopre_meancos_p1024_s{42,1,2,3,4}`: 0.897 / 0.893 / 0.857 / 0.785 / 0.865).
  SDA+MEP is above SDA-only and MEP-only, as in exp 05, but none beats no pretraining.

**Pretrain epoch sweep** (`anonq1024_longpretrain{5,10,30,50,100}_s42`, finetuning fixed at 12 epochs):

| Pretrain epochs | Anonymized | Original (exp 05) |
| --- | --- | --- |
| 5 | 0.8735 | 0.8815 |
| 10 | 0.8701 | 0.9159 |
| 30 | 0.7835 | 0.9127 |
| 50 | 0.7787 | 0.8709 |
| 100 | 0.7717 | 0.8915 |

Overfitting sets in earlier and harder than on the original queries: the best setting is 5-10 epochs, and 30+ epochs lose ~0.09.

Pretraining losses (from each run's `pretrain_losses.jsonl`), per source BIKE_7 / FOSSIG / s1kd-tools-doc:

| Pretrain epochs | Final train loss | Final dev loss | Lowest dev loss (at epoch) |
| --- | --- | --- | --- |
| 5 | 3.02 / 6.40 / 3.68 | 1.48 / 4.80 / 3.53 | 1.48 (5) / 4.80 (5) / 3.53 (5) |
| 10 | 1.91 / 4.60 / 1.73 | 1.46 / 3.39 / 2.73 | 1.34 (7) / 3.39 (10) / 2.73 (10) |
| 30 | 0.194 / 1.22 / 0.073 | 3.28 / 1.80 / 2.08 | 1.29 (6) / 1.80 (25) / 2.08 (28) |
| 50 | 0.016 / 0.138 / 0.004 | 6.12 / 2.76 / 2.15 | 1.32 (8) / 1.87 (20) / 1.87 (23) |
| 100 | 0.001 / 0.010 / 0.001 | 4.24 / 3.10 / 2.37 | 1.32 (9) / 1.94 (20) / 1.88 (27) |

Only the finetuning and test queries are anonymized; pretraining uses the same SDA/MEP data as exp 05, so these losses track exp 05's
closely (identical at 5 epochs, GPU noise after that). The test-MRR drop at 30+ epochs therefore comes from finetuning an overfitted
checkpoint on the harder queries, not from different pretraining. Per-step losses for the sweep, `anonq1024_meponly_s42` and
`anonq1024_mepablate_s42` are in `results/07_anonymized/<run>/<pair>/pretrain_losses.jsonl` (rows `kind` = `train` / `eval` / `summary`).

**332-document pool** (`eval_test_all332.json`, exp 04's wider search pool): pretrained 0.4691 (3 sources, seed 42) vs. no pretraining
0.4517. Exp 04's anonymized runs, trained with the 512-token cap, scored 0.4951 / 0.4993.

**Same-domain** (`anonq1024_samedomain_s{42,1,2}`, `anonq1024_samedomain_alt_s42`, as exp 06): pretraining gain = pretrained - finetune-only.

| | orig split, s42 | orig split, s1 | orig split, s2 | alt split, s42 |
| --- | --- | --- | --- | --- |
| A->B gain | -0.1003 | +0.1049 | -0.1677 | +0.0906 |
| B->A gain | +0.0331 | -0.1432 | -0.1635 | -0.0007 |

**Exp 06's result does not replicate on anonymized queries.** There, pretraining consistently helped the harder finetune half
(+0.06 to +0.13 in all 4 runs). Here the gains swing between -0.17 and +0.10 with no consistent direction, and every arm scores much lower
(0.48-0.73 vs. 0.79-0.94), so the same-domain benefit seen in exp 06 may have depended on the identifiers the original queries copy from their documents.
