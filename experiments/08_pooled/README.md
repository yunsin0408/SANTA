### Pooled split with all four folders merged

Launcher: [`run.sh`](run.sh). Results: `results/08_pooled*/` (one folder per variant below). Tables are mean ± sd over seeds of
each run's `eval_test.json`.

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths.

Exps 01-07 trained on one folder at a time, and the small folders give test sets of 2-21 documents. Here all four folders are pooled into one
split (`processing/XML/datasets/build_pooled_split.py`, `data/xml/{pretrain,retrieval}_pooled{,_anonq}/`): 388 train / 20 dev / 100 test
documents (test: 65 S1000D_spec_sample, 23 BIKE_7, 9 s1kd-tools-doc, 3 FOSSIG), so one test set of 100 documents (chance 0.052). The
training documents are split into halves A / B for a same-domain pretrain -> finetune swap as in exp 06. CodeT5, mean pooling, p_max_len
1024, query budget 50 tokens, seeds 42 / 1 / 2.

| Arm | Pretrain on | Finetune on |
| --- | --- | --- |
| `full` | -- | all 388 |
| `pretrainA_finetuneB` / `pretrainB_finetuneA` | half A / half B (SDA+MEP) | the other half |
| `finetuneB_only` / `finetuneA_only` | -- | half B / half A |
| `pretrainonly_sdamep` | all 388 (SDA+MEP) | -- (final pretraining checkpoint, `run_pooled_pretrainonly.sh`) |
| `pretrainonly_mep` | all 388 (MEP only) | -- |

**Main result, cosine** (`08_pooled`, test MRR, mean ± sd over 3 seeds; BM25 0.3900 original / 0.2891 anonymized, from
`BM25_DIR=data/xml/retrieval_pooled{,_anonq}/pooled .venv/bin/python diagnostics/bm25_baseline.py`):

| Arm | Original queries | Anonymized queries |
| --- | --- | --- |
| `full` | 0.550 ± 0.057 | 0.529 ± 0.082 |
| `pretrainA_finetuneB` | 0.586 ± 0.060 | 0.513 ± 0.038 |
| `finetuneB_only` | 0.290 ± 0.012 | 0.195 ± 0.010 |
| `pretrainB_finetuneA` | 0.621 ± 0.037 | 0.535 ± 0.040 |
| `finetuneA_only` | 0.294 ± 0.023 | 0.222 ± 0.021 |
| `pretrainonly_sdamep` | 0.721 ± 0.012 | 0.614 ± 0.021 |
| `pretrainonly_mep` | 0.127 ± 0.008 | 0.127 ± 0.019 |

- **Same-domain pretraining helps a lot on the pooled split.** Pretraining on one half before finetuning on the other gains +0.24 to
  +0.38 over finetuning on that half alone, in every seed, both directions and both query versions (per seed, original:
  A->B +0.286 / +0.367 / +0.236, B->A +0.353 / +0.286 / +0.341; anonymized: A->B +0.370 / +0.303 / +0.278, B->A +0.380 / +0.265 / +0.291).
  Unlike exps 06-07, the direction doesn't matter and the gain is far outside the seed spread.
- **Pretrain on one half + finetune on the other matches finetuning on everything** (0.59-0.62 vs. `full` 0.550 on original queries,
  0.51-0.54 vs. 0.529 anonymized).
- **SDA+MEP pretraining alone beats finetuning** (0.721 vs. 0.550 original, 0.614 vs. 0.529 anonymized). Caveat: the SDA descriptions are
  the finetuning queries, so `pretrainonly_sdamep` sees the same pairs as `full`; the difference is MEP plus the pretraining schedule
  (lr 5e-5 x 10 epochs vs. 2e-5 x 12). MEP alone (0.127) learns almost nothing usable for retrieval.
- **`full`, the pretrained arms and SDA+MEP pretrain-only all beat BM25** (0.39 / 0.29), unlike the per-folder test sets of exps 01-04
  where BM25 scored ~1.0. The half-size finetune-only controls (0.29 / 0.20) do not.

**Variants** (test MRR of the main arms, original / anonymized queries, mean over 3 seeds):

| Variant | Folder | `full` | `pretrainB_finetuneA` | `pretrainonly_sdamep` |
| --- | --- | --- | --- | --- |
| Cosine (above) | `08_pooled` | 0.550 / 0.529 | 0.621 / 0.535 | 0.721 / 0.614 |
| Poincare similarity | `08_pooled_poincare` | 0.671 / 0.637 | 0.738 / 0.685 | 0.762 / 0.720 |
| Query budget 160 tokens | `08_pooled_q160` | 0.552 / 0.507 | 0.647 / 0.497 | -- |
| Checkpoint every 25 steps | `08_pooled_ss25` | 0.560 / 0.538 | 0.461 / 0.486 | -- |
| Both | `08_pooled_ss25_poincare` | 0.661 / 0.629 | 0.614 / 0.507 | -- |

- **Poincare similarity is the best setting**: +0.04 to +0.15 over cosine on `full`, the pretrained arms and SDA+MEP pretrain-only, for
  both query versions, and the pretraining gains grow (half-split gains +0.27 to +0.57). It helps the finetune-only controls less
  (+0.005 to +0.09) and makes MEP-only worse (0.082 / 0.070). Temperature 0.05 was chosen by dev MRR on the anonymized `full` arm
  (`08_pooled_poincare_tau/`: dev 0.724 / 0.771 / **0.900** / 0.762 / 0.639 for tau 0.01 / 0.02 / 0.05 / 0.1 / 0.2).
- **A longer query budget doesn't help.** The LLM descriptions average ~90 tokens and are truncated at 50, but 160 tokens changes `full` by
  +0.00 / -0.02.
