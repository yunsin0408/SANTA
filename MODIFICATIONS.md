# Modifications vs. upstream

Local `master` vs. `origin/master` (`yunsin0408/SANTA`, a fork of `OpenMatch/SANTA`
).

**25 files changed** (23 modified, 1 deleted,
1 added — the delete+add is the `finetune-product .sh` rename). Total **90 hunks** .

---

## `code_best_dev/evaluate_code_finetune.py` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 12 | `from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,` → same without `AdamW` | Newer `transformers` stopped re-exporting `AdamW` from the top-level package; the unmodified import raises `ImportError` before the script can even start (`AdamW` isn't used anywhere in the file, so dropping it is safe). |
| 220 | `best_path = args.model_name_or_path + "best_dev"` → `best_path = os.path.join(args.model_name_or_path, "best_dev")` | String concatenation silently produces a malformed path when `model_name_or_path` doesn't already end in `/` (e.g. `checkpoint-1000best_dev` instead of `checkpoint-1000/best_dev`). |

## `code_best_dev/evaluate_code_pretrain.py` (2 hunks — identical shape to above)
| Lines | Change | Why |
|---|---|---|
| 12 | `AdamW` removed from transformers import | Same broken-import fix as the finetune script above. |
| 220 | `best_path = args.model_name_or_path + "best_dev"` → `os.path.join(...)` | Same path-concatenation bug as the finetune script above. |

## `evaluate_product/evaluate_product.py` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 60–64 | + pandas ≥2.2 fix: comment (3 lines) + `if isinstance(query_id, tuple): query_id = query_id[0]` | Newer pandas' `groupby(by=[single_col])` yields a 1-tuple key during iteration where older pandas yielded the bare scalar; left unhandled, the written `query_id` no longer matches `test.qrels`'s plain (non-tuple) `query_id`, breaking downstream scoring. |
| 129–132 | `score = torch.diagonal(torch.mm(query_vec, doc_vec.transpose(0, 1)).to('cpu')).tolist()` → 3-line comment + `score = (query_vec * doc_vec).sum(-1).tolist()` | The original builds the full N×N score matrix just to read its diagonal — O(n²) compute and, at this dataset's size, a ~120GB intermediate — when only the paired `(query_i, doc_i)` scores are ever used. The elementwise-product-and-sum form is O(n) and numerically identical. |

## `model.py` (1 hunk)
| Lines | Change | Why |
|---|---|---|
| 37–41 | + `_keys_to_ignore_on_save = None` with 3-line explanatory comment + blank line | HF `Trainer`'s checkpoint-resume path reads this attribute unconditionally, assuming `self.model` is a `transformers.PreTrainedModel`. `SModel` is a plain `nn.Module` wrapper, so without a stand-in attribute, resuming from checkpoint (see `train_santa.py` below) fails with an `AttributeError`. |

## `processing/Code/build_code_entity.py` (4 hunks)
| Lines | Change | Why |
|---|---|---|
| 9–14 | `import tree_sitter` → `from tree_sitter import Language, Parser` + 5 per-language grammar imports | The old workflow loaded every grammar from one merged `tree-sitter-languages.so` built against a pre-0.20 tree-sitter ABI; that `.so` is incompatible with current tree-sitter Python bindings, so grammars are now loaded from their individual `tree_sitter_<lang>` packages instead. |
| 17–28 | + `LANGUAGE_GETTERS` dict mapping `code_type` → grammar getter, with explanatory comment | Provides the lookup the new per-package grammar loading (below) needs, replacing the old `.so`-based dispatch. |
| 212 | `parser = ArgumentP` → `parser = ArgumentParser()` | Fixes a typo that made this a `NameError` on every invocation — the script couldn't run at all before this. |
| 230–231 | Old 3-line `tree_sitter.Language(...)`/`tree_sitter.Parser()`/`.set_language(...)` → 2-line `Language(LANGUAGE_GETTERS[args.code_type]())` / `Parser(lang)` | Matches the current tree-sitter API (`Parser(lang)` at construction) now that grammars come from `LANGUAGE_GETTERS` instead of the old merged `.so` + `set_language`. |

## `product_best_dev/evaluate_shop_finetune.py` (1 hunk)
| Lines | Change | Why |
|---|---|---|
| 171 | `best_path = checkpoint_path + "best_dev"` → `best_path = os.path.join(checkpoint_path, "best_dev")` | Same path-concatenation bug as the code-side evaluate scripts above. |

*(`score = (product_vec * query_vec).sum(-1)` at line 146 is untouched from upstream — already an O(n) paired computation, nothing to fix there.)*

## `product_best_dev/evaluate_shop_pretrain.py` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | + `import os` | Needed by the `os.path.join` fix below — upstream used string concatenation and never imported `os`. |
| 138 | `best_path=checkpoint_path+"best_dev"` → `best_path=os.path.join(checkpoint_path, "best_dev")` | Same path-concatenation bug as the other evaluate scripts. |

*(`scores = np.matmul(query_vecs, positive_vecs.T)` is untouched from upstream — this matrix is needed in full for MRR ranking, not just a diagonal, so there's no O(n²)-avoidance angle here.)*

## `train_santa.py` (1 hunk)
| Lines | Change | Why |
|---|---|---|
| 120 | `trainer.train()` → `trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)` | Without this, a killed/restarted training run has no way to resume from its last checkpoint — HF `Trainer` supports it natively via this argument, which the script simply never wired up. Pairs with the `model.py` fix above (resuming needs `_keys_to_ignore_on_save` to exist). |

---

## `shell/*.sh` (path-portability, parameterization, and training-correctness fixes)

These 16 files (17 counting the `finetune-product .sh`→`finetune-product.sh`
rename as two) turn scripts that were hardcoded to one author's local
filesystem and, in several cases, to a single hardcoded language (`ruby`) into
generic, reusable, per-language scripts rooted at this repo's own
`/workspace/SANTA_v2/...` tree (`data/` included, not just `runs/`). This repo
is fully independent of `SANTA/`: it does not read from or write to that
repo's directories at all. (An earlier iteration of this repo shared
`SANTA/`'s already-downloaded `data/` to avoid duplicating 41G; that was
reversed in favor of full independence — see "First-time setup" in
`PIPELINE.md` for what that means in practice: the full data
download/processing pipeline has to run once for this repo too.)

### `shell/build-code-test.sh` (7 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–7 | Old `TEST_PATH`/`test_data` exports → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Converts a hardcoded, single-run script into a reusable one driven by a `language` argument; `set -e` makes it fail loudly instead of silently continuing past an error. |
| 9–16 | + `DATA_ROOT` + `if/else` branching AdvTest (python) vs. CodeSearch (other 5 languages) query/doc input paths | python evaluates on AdvTest, a single paired file used for both query and doc; the other 5 languages evaluate on CodeSearch, where the query set has code stripped and the doc set has docstrings stripped — genuinely different files, which the old script had no way to express. |
| 18–20 | `OUTPUT_PATH`/`output_query`/`output_doc` hardcoded to the original author's local path → derived from `DATA_ROOT`/`language` | The hardcoded path doesn't exist on this machine; deriving it makes the script portable and per-language. |
| 25–26 | + `mkdir -p ${OUTPUT_PATH}` | The output directory doesn't exist on a first run; without this the script fails immediately. |
| 29 | `--input ${TEST_PATH}/${test_data}` → `--input ${QUERY_INPUT}` | Uses the new per-language query path introduced above. |
| 34 | `--input ${TEST_PATH}/${test_data}` → `--input ${DOC_INPUT}` | Uses the new per-language doc path introduced above. |
| 36 (EOF) | Removed trailing blank line at end of file | Cosmetic. |

### `shell/build-product-test.sh` (4 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/ESCI` | Path portability — the original path doesn't exist here. |
| 5 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/test` | Same. |
| 12–13 | + `mkdir -p ${OUTPUT_PATH}` | Output directory doesn't exist on first run. |
| 21 | `--input ${TEST_PATH}/${test_data}` → `--input ${TEST_PATH}` | Bug fix — `test_data` was never defined anywhere in this script, so the original line referenced an empty/unbound variable. |

### `shell/dev_code_finetune.sh` (3 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–4 | Hardcoded `CHECK_POINT`/`EVAL_PATH`/`eval_data_file`/`codebase_file` → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Same reusable-per-language conversion as above. |
| 6–23 | + `RUN_TAG`, derived `CHECK_POINT`, `if/else` on `python` vs. other languages for `EVAL_PATH`/`eval_data_file`/`codebase_file` | Fixes a silent near-zero-MRR bug: for non-python (CodeSearch) languages, `valid.jsonl` has code blanked out and is query-only, while `codebase.jsonl` has docstrings blanked out and is doc-only. The original script used `valid.jsonl` for both eval and codebase, which degenerates the codebase into near-duplicate queries and tanks MRR. `RUN_TAG` lets separate model variants write to their own checkpoint path instead of colliding. |
| 32 | Removed 2 trailing blank lines at EOF | Cosmetic. |

### `shell/dev_code_pretrain.sh` (3 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–4 | Hardcoded ruby-specific paths → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Same reusable-per-language conversion — the original only worked for ruby. |
| 6–12 | + `RUN_TAG`, `CHECK_POINT`/`EVAL_PATH`/`eval_data_file`/`codebase_file` derived from `language` instead of hardcoded to ruby | Generalizes to all 6 languages instead of only ruby; `RUN_TAG` for multi-run isolation. |
| 21 | Removed 2 trailing blank lines at EOF | Cosmetic. |

### `shell/dev_product_finetune.sh` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/finetune` | Path portability. |
| 3–5 | Hardcoded `CHECK_POINT` → `RUN_TAG` + derived `CHECK_POINT`; comment typo `checkponits`→`checkpoints` | `RUN_TAG` lets separate model variants avoid overwriting each other's checkpoints. |

### `shell/dev_product_pretrain.sh` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/pretrain` | Path portability. |
| 3–5 | Hardcoded `CHECK_POINT` → `RUN_TAG` + derived `CHECK_POINT`; typo fix | Same multi-run isolation as above. |

### `shell/evaluate_code.sh` (3 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–4 | Hardcoded `TREC_PATH` → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Reusable-per-language conversion. |
| 6–7 | + `RUN_TAG` + derived `TREC_PATH` | Multi-run isolation, consistent with the other scripts. |
| 11 | `--trec_save_path ${TREC_PATH}` (had trailing whitespace) + 2 trailing blank lines removed | Cosmetic cleanup. |

### `shell/finetune-code.sh` (11 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–5 | Hardcoded ruby `FINETUNE_PATH`/`finetune_data`/`Pretrain_checkpoint` → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Reusable-per-language conversion — the original only worked for ruby. |
| 7–11 | + `DATA_ROOT`, derived `FINETUNE_PATH`/`finetune_data`/`RUN_TAG`/`Pretrain_checkpoint` | Generalizes paths across languages and run variants. |
| 13–15 | `OUPUT`/`output_dir`/`logging_dir` hardcoded to ruby paths → derived, generic (`checkpoints`/`logs`) | Same generalization. |
| 17 | Comment typo `checkponit`→`checkpoint` | Cosmetic. |
| 20–37 | + explanatory comments and: `FINETUNE_EPOCHS=12`, per-language `FINETUNE_LR` (`1e-5` python / `2e-5` others), `N`/`MAX_STEPS` computation | Matches the paper's Appendix A.4 CodeSearch benchmark protocol (lr=2e-5 for the 5 CodeSearch languages, lr=1e-5 for python/Adv, 12 epochs for both) — the prior script used a flat 10 epochs and a literal, non-evaluating `--learning_rate 1e-5 or 2e-5` string for every language, a real gap against the paper's own protocol. `MAX_STEPS` is computed explicitly rather than trusted from `num_train_epochs` auto-derivation (see `pretrain-code.sh` below for why). |
| 44 | + `--overwrite_output_dir` | Lets a re-run reuse the same output dir instead of erroring on existing files. |
| 46 | + `--save_total_limit 3` | Bounds disk usage — without a limit, every `save_steps` checkpoint accumulates indefinitely. |
| 48–50 | `--per_device_train_batch_size 128` → `16` + `--gradient_accumulation_steps 8` + `--bf16 True` | Fits the model/batch in available GPU memory on this hardware (same effective batch size via gradient accumulation) and speeds up training with bf16. |
| 52 | `--learning_rate 1e-5 or 2e-5` (invalid literal) → `--learning_rate ${FINETUNE_LR}` | The original passed a literal string `"1e-5 or 2e-5"` as the learning rate — not a valid float, so the flag was broken as written. Now resolves to the per-language rate computed above. |
| 56–57 | `--num_train_epochs 10` → `${FINETUNE_EPOCHS}` + `--max_steps ${MAX_STEPS}` | Uses the paper-matching epoch count and the explicitly-computed step count from above. |
| 60–63 | + final-checkpoint materialization as `checkpoint-${MAX_STEPS}/` | `evaluate_code_finetune.py`'s dev-checkpoint selection only globs `checkpoint-*` subdirectories; the final/most-converged save otherwise lands directly in the output root and gets skipped by that comparison. |

### `shell/finetune-product .sh` → `shell/finetune-product.sh` (rename, 2 hunks: 1 delete + 1 add)
- Old file (29 lines, trailing space in filename) deleted entirely — the space in the filename was itself a footgun (breaks naive `bash finetune-product.sh` invocation).
- New file (41 lines): same non-language-specific shape as `finetune-code.sh` above —
  `/workspace/SANTA_v2/...` paths, `RUN_TAG`, `save_total_limit 3`, `bf16 True`,
  explicit `MAX_STEPS` computation (6 epochs), final-checkpoint materialization —
  for the same reasons as that file. No filename space.

### `shell/index-code.sh` (3 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–12 | 5 hardcoded exports → `#!/bin/bash`, `set -e`, usage comment, `language=$1`, `DATA_ROOT`, all paths derived | Reusable-per-language conversion; `MODEL_PATH` now points at this repo's actual finetune-checkpoint location instead of a hardcoded external path that doesn't exist here. |
| 15 | Comment typo `checkpint`→`checkpoint` | Cosmetic. |
| 36 (EOF) | Removed 2 trailing blank lines at end of file | Cosmetic. |

### `shell/pretrain-code.sh` (10 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–4 | Hardcoded ruby `PRETRAIN_PATH`/`pretrain_data` → `#!/bin/bash`, `set -e`, usage comment, `language=$1` | Reusable-per-language conversion — the original only worked for ruby. |
| 6–9 | + `DATA_ROOT`, derived `PRETRAIN_PATH`/`pretrain_data` | Same generalization. |
| 10–15 | `OUPUT`/`output_dir`/`logging_dir` ruby-hardcoded → `RUN_TAG` (with explanatory comment) + derived generic paths | `RUN_TAG` lets separate model variants (e.g. a different pretraining recipe) write to their own output tree instead of overwriting an existing run's checkpoints. |
| 17 | Comment typo fix | Cosmetic. |
| 22–29 | + comment + `N`/`MAX_STEPS` computation | `num_train_epochs`-based step auto-computation was observed to be unreliable when multiple `accelerate`-backed training processes start concurrently on this box — one run silently got `max_steps` far below the correct value despite `num_train_epochs=10` being honored correctly in isolation. Computing `max_steps` explicitly from the known example count sidesteps that. |
| 33 | Original author's local model path → `Salesforce/codet5-base` | The hardcoded local path doesn't exist here; the public HF hub id is portable and reproducible from a clean environment. |
| 36 | + `--save_total_limit 3` | Bounds disk usage from accumulating `save_steps` checkpoints. |
| 38–40 | `--per_device_train_batch_size 128` → `16` + `--gradient_accumulation_steps 8` + `--bf16 True` | Fits available GPU memory while preserving effective batch size; bf16 speeds up training. |
| 47 | + `--max_steps ${MAX_STEPS}` | Uses the explicitly-computed step count from above instead of relying on the unreliable auto-derivation. |
| 50–56 | + comment + final-checkpoint materialization as `checkpoint-${MAX_STEPS}/` | The final save lands directly in the output root, not a `checkpoint-N` subdir, but `evaluate_code_pretrain.py`'s dev-checkpoint selection only globs `checkpoint-*` subdirs — without this, the most-converged checkpoint is silently excluded from best-dev comparison. |

### `shell/pretrain-product.sh` (10 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/pretrain` | Path portability. |
| 5 | Trailing-whitespace-only cleanup on comment line | Cosmetic. |
| 7–10 | `OUPUT`/`output_dir`/`logging_dir` hardcoded → `RUN_TAG` + derived generic paths | Same multi-run isolation as `pretrain-code.sh`. |
| 12 | Comment typo fix | Cosmetic. |
| 16–20 | + comment (references `pretrain-code.sh`) + `N`/`MAX_STEPS` computation | Same `max_steps` auto-derivation unreliability fix as `pretrain-code.sh`. |
| 24 | Original author's local model path → `t5-base` | Public HF hub id in place of a hardcoded local path that doesn't exist here. |
| 27 | + `--save_total_limit 3` | Bounds disk usage. |
| 30 | + `--bf16 True` | Training speedup. |
| 37 | + `--max_steps ${MAX_STEPS}` | Uses the explicitly-computed step count. |
| 40–43 | + final-checkpoint materialization as `checkpoint-${MAX_STEPS}/` | Same dev-checkpoint-glob compatibility fix as `pretrain-code.sh`. |

### `shell/process-finetune-code.sh` (2 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–19 | Hardcoded `FINETUNE_RAW_PATH`/`finetune_raw_data`/`FINETUNE_PATH`/`finetune_data` (AdvTest only) → `#!/bin/bash`, `set -e`, usage comment, `language=$1`, `DATA_ROOT`, `if/else` python (AdvTest) vs. other languages (CodeSearch) raw path, `mkdir -p ${FINETUNE_PATH}` | Generalizes a script that only handled AdvTest/python to all 6 languages, each sourced from the right raw dataset; `mkdir -p` avoids a first-run failure. |
| 24 | Fixed missing trailing newline at EOF | Some tools/shells mishandle a file with no trailing newline; cosmetic robustness fix. |

### `shell/process-finetune-product.sh` (4 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/ESCI` | Path portability. |
| 6 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/finetune` | Same. |
| 10 | Trailing-whitespace-only cleanup on comment line | Cosmetic. |
| 13–14 | + `mkdir -p ${FINETUNE_PATH}` | Output directory doesn't exist on first run. |

### `shell/process-pretrain-code.sh` (5 hunks)
| Lines | Change | Why |
|---|---|---|
| 1–5 | Hardcoded ruby raw path → `#!/bin/bash`, `set -e`, usage comment (lists all 6 languages), `language=$1` | Reusable-per-language conversion — the original only worked for ruby. |
| 7–10 | + `DATA_ROOT`, derived `PRETRAIN_RAW_PATH`/`PRETRAIN_PATH`/`pretrain_data` | Same generalization. |
| 12–22 | Hardcoded ruby paths (3 lines) → `mkdir -p`, comment, `if/else` merging CodeSearchNet train+test for python vs. train-only for other languages, derived `pretrain_raw_data` (11 lines) | Per the README, python's pretraining data is CodeSearchNet train+test merged, while every other language uses train only — this branching logic didn't exist before (the script was hardcoded to ruby with no merge step at all). |
| 23 | Removed now-redundant comment | The `language` variable's meaning is explained inline at its point of use instead. |
| 26 | `--input ${PRETRAIN_RAW_PATH}/${pretrain_raw_data}` → `--input ${PRETRAIN_PATH}/${pretrain_raw_data}` | Bug fix — the merged/copied raw file from the branch above is written under `PRETRAIN_PATH`, not `PRETRAIN_RAW_PATH`; the original line read from the wrong (nonexistent) location. |

### `shell/process-pretrain-product.sh` (4 hunks)
| Lines | Change | Why |
|---|---|---|
| 1 | Original author's local path → `/workspace/SANTA_v2/data/ESCI` | Path portability. |
| 6 | Original author's local path → `/workspace/SANTA_v2/data/processed/product/pretrain` | Same. |
| 10 | Trailing-whitespace-only cleanup on comment line | Cosmetic. |
| 13–14 | + `mkdir -p ${PRETRAIN_PATH}` | Output directory doesn't exist on first run. |

---

## New files added beyond the tracked diff (filling a gap the README documents but upstream never shipped)

`README.md`'s "Evaluating Product Retrieval" section (unmodified from upstream —
`README.md` isn't part of the diff above) instructs the reader to run
`bash index-product.sh` and `bash evaluate_product.sh` to encode test
embeddings and compute the final hypothesis/qrels files, and
`shell/PIPELINE.md`'s reference table (in `SANTA/`, not carried into this repo)
notes a further NDCG-computation step. None of `shell/index-product.sh`,
`shell/evaluate_product.sh`, or a script to compute NDCG from the resulting
files exist in upstream (`git ls-tree origin/master -- shell/` lists 16
scripts, and neither of the two is among them) — the README's documented
pipeline was incomplete in the base repo. These four files fill that gap:

- `shell/index-product.sh` — encodes test query/doc jsonl into embedding
  pickles via the new `evaluate_product/encode_product.py`.
- `shell/evaluate_product.sh` — runs `evaluate_product.py` (already
  present upstream) against those pickles to write `hypothesis.results` /
  `test.qrels`.
- `evaluate_product/encode_product.py` — new. Mirrors the encode/pooling
  logic already used for dev-checkpoint selection in
  `product_best_dev/evaluate_shop_pretrain.py` (T5, zero decoder input,
  last hidden state at position 0).
- `evaluate_product/compute_ndcg.py` — new. Computes NDCG@100 from
  `hypothesis.results` / `test.qrels` in both the paper's Two-C (binary,
  Exact-only) and Four-C (graded) settings — the step
  `evaluate_product.sh` itself stops short of.

These are identical to `SANTA/`'s versions of the same files, with the same
full path substitution as the rest of `shell/` (`index-product.sh`,
`evaluate_product.sh`, etc. all point at `/workspace/SANTA_v2/...` for both
data and runs).

## Table 6 extras: Zero-Shot and SANTA (Hard Negative)

Also carried over, same substitution rule:

- `shell/zeroshot-eval-code.sh` — evaluates the **pretrain** (not finetuned)
  `best_dev` checkpoint directly on the test set, reproducing Table 6's
  "Zero-Shot" row.
- `shell/mine-hard-negatives-code.sh` — retrieves top-100 candidates per
  finetune-train query using the pretrain checkpoint, samples one wrong one
  per query as a hard negative.
- `shell/finetune-code-hardneg.sh` — finetunes from the pretrain `best_dev`
  with `train_n_passages=2` (1 positive + 1 hard negative) and
  `warmup_ratio 0.1`, reproducing Table 6's "SANTA (Hard Negative)" row.
- `shell/index-code-hardneg.sh` — same as `index-code.sh` but reads the
  hard-negative finetune checkpoint.
- `processing/Code/build_code_hard_negative.py` — merges mined negatives back
  onto the raw train pairs; called by `mine-hard-negatives-code.sh`.

These reproduce rows of the paper's Table 6 that go beyond the base
README-documented pipeline, and involve several judgment calls not explicitly
stated in the paper (mining corpus choice, static-vs-iterative mining,
retrieval depth, random seed) — see the comments at the top of
`mine-hard-negatives-code.sh` for the full reasoning on each, since they
affect how closely the reproduced numbers should be expected to match the
paper's reported ones.
