### Lorentz (hyperboloid) similarity on the pooled split

Launcher: [`run.sh`](run.sh). Results: `results/10_lorentz*/`. Tables are test MRR of each run's `eval_test.json`, mean ± sd over seeds
where there are several. [`make_xlsx.py`](make_xlsx.py) collects them, with exp 08's Poincare runs alongside, into
`results/10_lorentz/lorentz_results.xlsx` (gitignored).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths.

Exp 08 found Poincare similarity to be the best setting on the pooled split. This experiment scores with negative Lorentz distance
instead, in pretraining, finetuning and evaluation. Everything else matches exp 08: 388 train / 20 dev / 100 test documents, CodeT5, mean
pooling, `p_max_len` 1024, query budget 50 tokens, temperature 0.05, seeds 42 / 1 / 2.

Each encoder output `v` is mapped onto the hyperboloid with `x = (cosh r, sinh r * v / |v|)`, where `r = |v| * lorentz_scale`. The score is
`-acosh(-<q, p>_L)` with curvature -1, computed in float64 outside bf16 autocast (`lorentz_scores` in `model.py`).

**What this can and can't show.** The Poincare ball and the hyperboloid are two coordinate systems for the same space. `lorentz_scale 0.5`
gives the same distances as exp 08's `poincare_scale 0.25`, because `to_ball`'s tanh map places a point at hyperbolic radius
`2 * poincare_scale * |v|`. At that scale, any gap from exp 08 comes from numerics and seed noise rather than from the geometry. Other scales
change the radius, but exp 08 could reach the same radii with a different `poincare_scale`, until the ball's boundary clamp takes effect.

**Scale check** (`10_lorentz_scale`, full arm, anonymized queries, seed 42; picked by dev MRR):

| `lorentz_scale` | Dev MRR | Test MRR |
| --- | --- | --- |
| 0.25 | 0.768 | 0.529 |
| **0.5** | **0.892** | **0.675** |
| 1.0 | 0.597 | 0.262 |
| 2.0 | 0.281 | 0.081 |
| Poincare 0.25, exp 08 (`08_pooled_poincare_tau/poincare_tau0.05`, same radii as 0.5) | 0.900 | 0.645 |

- **0.5 is best**, and matches exp 08's Poincare run at the same radii (dev 0.892 vs. 0.900, test 0.675 vs. 0.645). The main runs use 0.5.
- **Larger radii hurt badly.** At 1.0 and 2.0 the training loss stays above 1.6 (0.007 at 0.5) and test MRR falls to 0.26 and 0.08.

**Near the boundary** (same setup). `lorentz_scale 2.0` and `poincare_scale 1.0` place points at the same radii, about 7 for a rep of norm
3.5, or 0.998 of the way to the ball's boundary. Any gap between these two runs comes from numerics: the ball's limited precision and boundary
clamp near the edge.

| Similarity | Scale | Dev MRR | Test MRR |
| --- | --- | --- | --- |
| Lorentz | 2.0 | 0.281 | 0.081 |
| Poincare | 1.0 | 0.365 | 0.170 |

- **Lorentz's extra precision doesn't help here.** Both runs fail, from the same starting loss (9.23), and Poincare scores a little higher.
  With one seed each, the gap between two failed runs means little. The likely cause of both failures is the geometry and temperature
  (distances this large make the scores extreme at tau 0.05), not the ball's numerics; a larger temperature at this scale would test that.

**Main result** (`10_lorentz`, test MRR, mean ± sd over 3 seeds; Poincare from exp 08's `08_pooled_poincare`):

| Arm | Original: Poincare | Original: Lorentz | Diff | Anonymized: Poincare | Anonymized: Lorentz | Diff |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 0.671 ± 0.047 | 0.662 ± 0.022 | -0.008 | 0.637 ± 0.102 | 0.669 ± 0.041 | +0.033 |
| `pretrainA_finetuneB` | 0.696 ± 0.039 | 0.699 ± 0.039 | +0.003 | 0.658 ± 0.077 | 0.690 ± 0.032 | +0.031 |
| `finetuneB_only` | 0.376 ± 0.041 | 0.367 ± 0.033 | -0.009 | 0.200 ± 0.037 | 0.216 ± 0.009 | +0.016 |
| `pretrainB_finetuneA` | 0.738 ± 0.068 | 0.728 ± 0.018 | -0.010 | 0.685 ± 0.038 | 0.596 ± 0.114 | -0.088 |
| `finetuneA_only` | 0.356 ± 0.079 | 0.351 ± 0.029 | -0.005 | 0.257 ± 0.017 | 0.258 ± 0.007 | +0.001 |
| `pretrainonly_sdamep` | 0.762 ± 0.051 | 0.735 ± 0.026 | -0.027 | 0.720 ± 0.014 | 0.725 ± 0.055 | +0.005 |

**Lorentz, per seed** (test MRR):

| Arm | Original s42 | Original s1 | Original s2 | Anonymized s42 | Anonymized s1 | Anonymized s2 |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 0.685 | 0.640 | 0.663 | 0.716 | 0.651 | 0.641 |
| `pretrainA_finetuneB` | 0.681 | 0.744 | 0.671 | 0.652 | 0.711 | 0.705 |
| `finetuneB_only` | 0.352 | 0.345 | 0.406 | 0.226 | 0.209 | 0.214 |
| `pretrainB_finetuneA` | 0.713 | 0.748 | 0.725 | 0.472 | 0.620 | 0.697 |
| `finetuneA_only` | 0.350 | 0.380 | 0.323 | 0.251 | 0.264 | 0.259 |
| `pretrainonly_sdamep` | 0.723 | 0.765 | 0.717 | 0.742 | 0.769 | 0.663 |

- **Lorentz matches Poincare within seed noise.** 11 of 12 cells differ by at most 0.033, less than the seed sd, and neither similarity wins
  consistently (Lorentz slightly lower on original queries, slightly higher on anonymized). This is what the shared geometry predicts.
- **The one larger gap is a single bad run.** Anonymized `pretrainB_finetuneA` is -0.088, driven by seed 42 (0.472, vs. 0.620 / 0.697 for
  the other seeds); its dev MRR was also low (0.625), so the run went badly from the start.
- **Exp 08's findings hold.** Same-domain pretraining on one half still gains +0.33 to +0.47 over finetuning on that half alone, and SDA+MEP
  pretrain-only is still the best arm.

So Lorentz is a working drop-in replacement for Poincare, but on this split it doesn't improve retrieval.

**Note on `pretrainonly_sdamep`.** A bug in the first version of `run_pooled_pretrainonly.sh` (fixed) wrote all six runs' scores to one
folder, each overwriting the last. The training and rankings were unaffected and each run's TREC file was kept, so the six `eval_test.json`
files were rebuilt from those with `shell/evaluate_xml.sh`; the rebuilt score of the last run matches the one it wrote itself. The
pretraining logs of the first five runs were overwritten, and their `embeddings.npz` were lost.
