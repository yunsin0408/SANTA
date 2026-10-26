### New-query experiment: removing identifier leakage from the queries (S1000D_spec_sample only)

Launcher: [`run.sh`](run.sh). Results: [`results/03_query_leakage/`](../../results/03_query_leakage/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets).

Motivation: BM25 scores 1.0 on the original S1000D test queries because they quote identifiers/dates from their own document
(`processing/XML/datasets/README.md`, "Should queries be regenerated..."). This experiment builds a second query set that can't be solved that way, to
see whether exp 02's pretrained-vs-no-pretrain result still holds once string matching is taken away. 

**Pipeline** :
- `processing/XML/datasets/generate_anon_queries.py` -- new prompt (`anon_prompt()`) that forbids copying any code/number/date verbatim from the
  document (uses "the latest issue" rather than an issue number). Reads the existing
  `data/xml/pretrain/S1000D_spec_sample/sda_pairs.*.jsonl` `structured` field (same documents, same train/dev/test split) and writes new
  descriptions to `data/xml/pretrain_anonq/S1000D_spec_sample/sda_pairs.*.jsonl`. 

**BM25 on the new queries** (`diagnostics/bm25_baseline.py`, pointed at `data/xml/retrieval_anonq`): **MRR = 0.9762** (20/21 queries still rank
their document 1st).
With only 21 test documents, each on a
genuinely distinct narrow S1000D topic (e.g. "front matter copyright notice" vs. "crew procedures chapter"), ordinary topic vocabulary is
enough to separate them. 

**Pretrained vs. no-pretraining on the new queries** (mean pooling + cosine tau 0.05, p_max_len 1024, launchers `experiments/common/run_pairs.sh` /
`run_nopretrain.sh` with `DATA_ROOT=data_anonq`, folders `anonq_meancos_p1024_s{42,1,2}/`, `anonq_nopre_meancos_p1024_s{42,1,2}/`):

| | Pretrained (3 sources) | No pretraining | BM25 |
| --- | --- | --- | --- |
| Original queries (exp 02) | 0.917 (sd 0.059) | 0.940 (sd 0.041, 5 seeds) | 1.0000 |
| New (anonymized) queries | 0.866 (sd 0.048) | 0.845 (sd 0.052) | 0.9762 |

- **The pretraining result doesn't change, still no clear benefit.** Pretrained is 0.02 above no-pretraining here (0.02 below on the original
  queries), well inside the spread of either setting.
- **Both dense settings dropped versus the original queries** (0.92/0.94 -> 0.87/0.85), consistent with the new queries being harder, but the dense model still trails BM25 by about 0.11-0.13, a wider gap than before
  (0.06-0.08 originally). Removing identifier leakage did not let the dense model close the gap to BM25 or reveal a pretraining benefit;
  it mainly confirmed the benchmark was already lexically easy for reasons beyond identifiers.
- Per-seed pretrained means: 0.897 / 0.856 / 0.846 (seeds 42/1/2). Per-pair breakdown and raw JSON in
  `results/03_query_leakage/anonq_meancos_p1024_s{42,1,2}/` and `results/03_query_leakage/anonq_nopre_meancos_p1024_s{42,1,2}/`.
