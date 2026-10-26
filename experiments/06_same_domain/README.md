### Same-domain training: pretrain and finetune on S1000D_spec_sample's own data

Launcher: [`run.sh`](run.sh). Results: [`results/06_same_domain/`](../../results/06_same_domain/).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets).

**Design** 
`processing/XML/datasets/build_samedomain_split.py` splits the 297 train documents into two disjoint halves by a deterministic hash on `source_file` (halfA=150, halfB=147 docs), leaving the existing dev (14) and test (21) splits untouched. `processing/XML/prepare_samedomain.py` tokenizes four combinations with the codet5 tokenizer:

| Arm | Pretrain on | Finetune on | Purpose |
| --- | --- | --- | --- |
| `pretrainA_finetuneB` | halfA (SDA+MEP) | halfB | same-domain pretrain -> finetune |
| `finetuneB_only` | -- | halfB | matched-size no-pretrain control |
| `pretrainB_finetuneA` | halfB (SDA+MEP) | halfA | same-domain pretrain -> finetune, swapped |
| `finetuneA_only` | -- | halfA | matched-size no-pretrain control |

Two seeds (42, 1), mean pooling + cosine, p_max_len 1024. Launcher `experiments/common/run_samedomain.sh` (via `run.sh`). 

**Result**

| Seed | A->B: pretrained | A->B: no-pretrain | diff | B->A: pretrained | B->A: no-pretrain | diff |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.8988 | 0.8631 | +0.0357 | 0.9365 | 0.8057 | **+0.1308** |
| 1 | 0.8202 | 0.8378 | -0.0175 | 0.9116 | 0.8005 | **+0.1111** |

- **B->A (pretrain on halfB, finetune on halfA) shows a large, seed-consistent gain: +0.13 and +0.11.** This is the first pretraining comparison in these experiments where the effect is both large and reproduces across seeds in the same direction.
- **A->B  does not reproduce: +0.036 then -0.018.** The two directions are not symmetric, so something about which half is used for pretraining vs. finetuning matters. The hash-based split is not stratified, so the two halves aren't guaranteed equally easy, and halfA (the harder finetune target) is the direction where pretraining shows a large gain, while halfB (an easier target) shows little room for pretraining to add.
- **Pretrain-dev SDA MRR is above chance**, unlike almost every cross-domain source tried earlier: halfA pretrain-dev 0.3955, halfB pretrain-dev 0.4917, against a chance of 0.2323 (14 dev docs), consistent with the domain-content-gap diagnosis (SDA's contrastive task learns better when pretrain and finetune content are topically related).
- **Caveats:** the asymmetry between directions is itself unexplained and could be either a genuine effect (pretraining helps more on a harder finetune target) or a property of this specific 150/147 split. A stratified or multiply-resampled split, and more seeds on the B->A direction, would settle whether the asymmetry is real.


#### Follow-up: a 3rd seed and an independent split resolve the asymmetry

Added seed 2 on the original 150A/147B split, plus a second, independently-hashed split (`--salt`, `data/xml/pretrain_samedomain_alt/`, `data/xml/retrieval_samedomain_alt/`, `SAMEDOMAIN_SUFFIX=_alt`): 143A/154B, seed 42 only. Same config throughout.

| | orig split, s42 | orig split, s1 | orig split, s2 | alt split, s42 |
| --- | --- | --- | --- | --- |
| A->B diff (finetune on the *easier* half) | +0.0357 | -0.0175 | +0.0414 | +0.0132 |
| B->A diff (finetune on the *harder* half) | +0.1308 | +0.1111 | +0.0594 | +0.0634 |


**The asymmetry replicates on a second, independently-partitioned split, and holds up under a 3rd seed.**

| | orig split (n=3 seeds) | alt split (n=1 seed) |
| --- | --- | --- |
| No-pretrain gap (easier - harder) | 0.0316 | 0.0574 |
| Pretraining gain on the **harder** half | **+0.100** (0.131 / 0.111 / 0.059, all positive) | **+0.063** |
| Pretraining gain on the **easier** half | +0.020 (0.036 / -0.018 / 0.041, straddles zero) | +0.013 |

Across all 4 split/seed combinations, the harder half always gains more from pretraining than the easier half from the same split, and the harder-half gain is positive in every single instance (4/4), while the easier-half gain is small and inconsistent in sign (2/3 positive on the original split).

**Conclusion: same-domain pretraining's benefit is concentrated on the harder finetune target.**
