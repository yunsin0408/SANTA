### Widened candidate pool experiment (S1000D_spec_sample)

Launcher: [`run.sh`](run.sh). Results: [`results/04_pool_size/`](../../results/04_pool_size/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets).

This widens the *search pool* to all 332 S1000D documents
(train+dev+test) without retraining on a new split or changing the test queries, to check how much of the earlier result was the
small pool rather than the query wording.

**BM25 collapses on the wider pool** (`processing/XML/datasets/build_full_corpus.py` output, `corpus.all332.jsonl`; scored with
`BM25_POOL=all332 BM25_DIR=data/xml/retrieval{,_anonq}/S1000D_spec_sample .venv/bin/python diagnostics/bm25_baseline.py`, idf over the 332 documents):

| Candidates | Chance | BM25, original queries | BM25, anonymized queries |
| --- | --- | --- | --- |
| 21 (old, test-split only) | 0.1736 | 1.0000 | 0.9762 |
| 332 (train+dev+test) | 0.0192 | **0.1250** (0/21 at rank 1) | **0.1147** (0/21 at rank 1) |

| | BM25 | Dense, pretrained | Dense, no pretraining (1 seed) |
| --- | --- | --- | --- |
| Original queries | 0.1250 | 0.6577 | 0.6378 |
| Anonymized queries | 0.1147 | 0.4951 | 0.4993 |

- **Still no visible pretraining benefit**
  Pretrained vs. no-pretrain: +0.02 on original queries, -0.004 on
  anonymized queries
- Raw numbers: `results/04_pool_size/pool332/*_eval_test_all332.json`.
