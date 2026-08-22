
## First-time setup

1. **Python environment. (DONE)** 
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install transformers==4.35.2 nltk==3.7 "numpy>=1.26.4" "datasets>=2.4.0" \
       "tree-sitter>=0.20.4" tree-sitter-javascript tree-sitter-php tree-sitter-go \
       tree-sitter-java tree-sitter-ruby "faiss-cpu>=1.7.4" "scikit-learn>=1.1.2" \
       "pandas>=2.1.1" pytrec-eval==0.5 tensorboard beautifulsoup4 torch accelerate==0.25.0
   ```
   (This repo has no dependency changes vs. upstream) Needs CUDA-enabled `torch`
   and Python 3.12+ for compatibility with the rest of the list.

   The `.venv` must be **activated in every new shell/terminal** before running
   any `shell/*.sh` script — they all invoke a bare `python` (not `python3`),
   which only resolves once the venv is active. Ubuntu 24.04 ships no `python`
   binary at all, only `python3`, so forgetting this step fails with `python:
   command not found`.

2. **OpenMatch. (DONE)** `shell/index-code.sh`, `shell/index-code-hardneg.sh`, and
   `shell/zeroshot-eval-code.sh` all do `cd ../OpenMatch/src` and run
   `python -m openmatch.driver.{build_index,retrieve}`, so they expect an
   `OpenMatch/` checkout as a sibling of `shell/` *inside this repo* (a
   relative `cd`, so it can't point outside `SANTA_v2/`):
   ```
   git clone https://github.com/OpenMatch/OpenMatch.git /workspace/SANTA_v2/OpenMatch
   ```

3. **Data. (DONE)** From
   `README.md`'s "Data Download" section:
   ```
   # Code: CodeSearchNet
   git clone https://github.com/github/CodeSearchNet.git /workspace/SANTA_v2/data/CodeSearchNet
   cd /workspace/SANTA_v2/data/CodeSearchNet && script/setup

   # Code: AdvTest
   mkdir -p /workspace/SANTA_v2/data/AdvTest && cd /workspace/SANTA_v2/data
   wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
   unzip dataset.zip && rm dataset.zip && mv dataset/* AdvTest/ && rmdir dataset && cd AdvTest
   wget https://zenodo.org/record/7857872/files/python.zip
   unzip python.zip && python preprocess.py && rm -r python *.pkl python.zip

   # Code: CodeSearch
   mkdir -p /workspace/SANTA_v2/data/CSN_CodeSearch && cd /workspace/SANTA_v2/data/CSN_CodeSearch
   wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
   unzip dataset.zip && rm dataset.zip && mv dataset CSN && cd CSN && bash run.sh

   # Product: ESCI
   # Download shopping_queries_dataset_examples.parquet and
   # shopping_queries_dataset_products.parquet from
   # https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset
   # into /workspace/SANTA_v2/data/ESCI/
   ```
   Then run the `process-pretrain-*.sh` / `process-finetune-*.sh` /
   `build-*-test.sh` scripts per the tables below — these are the ones that
   turn the raw downloads above into the `data/processed/...` files the rest
   of the pipeline reads. Budget real time and disk for this: `SANTA/`'s
   equivalent processed+raw tree is 41G, and CodeSearchNet/CodeSearch/AdvTest
   downloads are large.

4. **GPU.** `train_santa.py` uses HF `TrainingArguments`, which picks CUDA
   automatically whenever available — none of these scripts pass `--no_cuda`.
   Set `CUDA_VISIBLE_DEVICES` before launching to control which GPU is used
   (this box has 2× RTX 6000 Ada). The pretrain/finetune scripts pass
   `--bf16 True`; if your GPU doesn't support bf16 (pre-Ampere), you'll need
   to edit that flag out before running.

5. **Nothing in `runs/` exists yet.** This repo's `runs/` paths
   (`/workspace/SANTA_v2/runs/...`) start empty — every script's `mkdir -p`
   / `save_steps` will create it as needed. Don't expect `best_dev`
   checkpoints to exist before you've run the corresponding `dev_*` script.

All commands below run from inside `shell/` (each script `cd`s to its target
directory relative to there). Code-track scripts take a language argument
(`ruby`, `python`, `javascript`, `go`, `java`, `php`); product-track scripts
take no argument.

One useful env var, settable before any script in either track:
- `RUN_TAG` — suffix appended to run directories (e.g. `RUN_TAG=-v2`) so an
  experiment variant doesn't overwrite a baseline run's checkpoints.

---

## Code track (repeat per language)

| Order | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 1 | `process-pretrain-code.sh <lang>` | Build pretraining data from raw CodeSearchNet | `data/CodeSearchNet/resources/data/<lang>/final/jsonl/train/` (+ `test/` merged in for `python` only) | `data/processed/code/<lang>/pretrain/<lang>.jsonl` |
| 2 | `pretrain-code.sh <lang>` | Continue-pretrain CodeT5-base on that data (`train_santa.py`) | `data/processed/code/<lang>/pretrain/<lang>.jsonl` | `runs/code/<lang>/pretrain<RUN_TAG>/checkpoints/checkpoint-*` |
| 3 | `dev_code_pretrain.sh <lang>` | Score every pretrain checkpoint on CodeSearchNet valid split, keep the best | `runs/code/<lang>/pretrain<RUN_TAG>/checkpoints/checkpoint-*`, `data/CodeSearchNet/.../valid/<lang>_valid.jsonl` | `runs/code/<lang>/pretrain<RUN_TAG>/checkpoints/best_dev/` |
| 4 | `process-finetune-code.sh <lang>` | Build finetuning data (AdvTest for `python`, CodeSearch for the rest) | `data/AdvTest/train.jsonl` or `data/CSN_CodeSearch/CSN/dataset/<lang>/train.jsonl` | `data/processed/code/<lang>/finetune/<lang>.jsonl` |
| 5 | `finetune-code.sh <lang>` | Finetune from step 3's `best_dev` checkpoint | `runs/code/<lang>/pretrain<RUN_TAG>/checkpoints/best_dev`, `data/processed/code/<lang>/finetune/<lang>.jsonl` | `runs/code/<lang>/finetune<RUN_TAG>/checkpoints/checkpoint-*` |
| 6 | `dev_code_finetune.sh <lang>` | Score every finetune checkpoint on the task's valid split, keep the best | `runs/code/<lang>/finetune<RUN_TAG>/checkpoints/checkpoint-*`, valid split (AdvTest `valid.jsonl`, or CodeSearch `valid.jsonl` query + `codebase.jsonl` docs) | `runs/code/<lang>/finetune<RUN_TAG>/checkpoints/best_dev/` |
| 7 | `build-code-test.sh <lang>` | Build held-out test query/doc files | AdvTest `test.jsonl`, or CodeSearch `test.jsonl` (query) + `codebase.jsonl` (doc) | `data/processed/code/<lang>/test/<lang>_test_query.jsonl`, `<lang>_test_doc.jsonl` |
| 8 | `index-code.sh <lang>` | Encode docs, build FAISS index, retrieve (inner-product) | step 6's `best_dev`, step 7's query/doc files, OpenMatch checkout | `data/processed/code/<lang>/test/<lang>_inference.trec` |
| 9 | `evaluate_code.sh <lang>` | Compute final MRR from the `.trec` file | `..._inference<RUN_TAG>.trec` | MRR printed to stdout |

## Product track (no per-run argument)

| Order | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 1 | `process-pretrain-product.sh` | Build pretraining data + pretrain dev split from raw ESCI | `data/ESCI/shopping_queries_dataset_{examples,products}.parquet` | `data/processed/product/pretrain/product_train.jsonl`, `product_eval.jsonl` |
| 2 | `pretrain-product.sh` | Continue-pretrain T5-base (`train_santa.py`) | `product_train.jsonl` | `runs/product/pretrain<RUN_TAG>/checkpoints/checkpoint-*` |
| 3 | `dev_product_pretrain.sh` | Score every pretrain checkpoint on `product_eval.jsonl`, keep the best | checkpoints + `product_eval.jsonl` | `runs/product/pretrain<RUN_TAG>/checkpoints/best_dev/` |
| 4 | `process-finetune-product.sh` | Build finetuning data + finetune dev split from raw ESCI | `data/ESCI/shopping_queries_dataset_{examples,products}.parquet` | `data/processed/product/finetune/product_train.jsonl`, `product_eval.jsonl` |
| 5 | `finetune-product.sh` | Finetune from step 3's `best_dev` checkpoint | pretrain `best_dev`, finetune `product_train.jsonl` | `runs/product/finetune<RUN_TAG>/checkpoints/checkpoint-*` |
| 6 | `dev_product_finetune.sh` | Score every finetune checkpoint (NDCG) on finetune `product_eval.jsonl`, keep the best | checkpoints + `product_eval.jsonl` | `runs/product/finetune<RUN_TAG>/checkpoints/best_dev/` |
| 7 | `build-product-test.sh` | Build held-out test query/doc files | `data/ESCI/` | `data/processed/product/test/product_test_query.jsonl`, `product_test_doc.jsonl` |
| 8 | `index-product.sh` | Encode test query/doc into embedding pickles using step 6's `best_dev` | step 6's `best_dev`, step 7's query/doc files | `data/processed/product/test/product_test_query<RUN_TAG>.pkl`, `product_test_doc<RUN_TAG>.pkl` |
| 9 | `evaluate_product.sh` | Retrieve + score, write hypothesis/qrels files | step 8's pickles, `data/ESCI/` | `runs/product/eval<RUN_TAG>/hypothesis.csv`, `hypothesis.results`, `test.qrels` |
| 10 | `python ../evaluate_product/compute_ndcg.py --qrels <path> --results <path>` (no shell wrapper; run directly) | Compute the final NDCG@100 (both Two-C and Four-C, per the paper) | step 9's `test.qrels` + `hypothesis.results` | NDCG@100 printed to stdout |

Steps 8–10 of the product track (`index-product.sh`, `evaluate_product.sh`,
`compute_ndcg.py`) are the "missing scripts" — see MODIFICATIONS.md's "New
files added beyond the tracked diff" section. Without them the product track
has no way to reach a final NDCG number at all, matching neither a Poincaré
nor a baseline concern — they were simply never written upstream.

### Table 6 extras: Zero-Shot and SANTA (Hard Negative)

Not part of the base README pipeline — reproduces the other two rows of the
paper's Table 6. Requires step 3 (`dev_code_pretrain.sh <lang>`) and step 7
(`build-code-test.sh <lang>`) from the code track above to already be done.

| Order | Script | Purpose | Input | Output |
|---|---|---|---|---|
| Z1 | `zeroshot-eval-code.sh <lang>` | Evaluate the **pretrain** (not finetuned) `best_dev` checkpoint directly on the test set — reproduces the "Zero-Shot" row | code track step 3's `best_dev`, step 7's query/doc files | `data/processed/code/<lang>/test/<lang>_inference-zeroshot.trec` |
| — | `RUN_TAG=-zeroshot bash evaluate_code.sh <lang>` | Score it | `..._inference-zeroshot.trec` | MRR printed to stdout |
| H1 | `mine-hard-negatives-code.sh <lang>` | Retrieve top-100 candidates for every finetune-train query using the **pretrain** checkpoint, sample 1 wrong one per query as a hard negative | code track step 3's `best_dev`, raw `train.jsonl`, raw `codebase.jsonl` | `data/processed/code/<lang>/finetune/<lang>_hardneg.jsonl` |
| H2 | `finetune-code-hardneg.sh <lang>` | Finetune from step 3's `best_dev` with `train_n_passages=2` (1 positive + 1 hard negative) | code track step 3's `best_dev`, H1's output | `runs/code/<lang>/finetune-hardneg/checkpoints/checkpoint-*` |
| H3 | `RUN_TAG=-hardneg bash dev_code_finetune.sh <lang>` | Pick the best hard-negative finetune checkpoint (script is already generic via `RUN_TAG`) | H2's checkpoints | `runs/code/<lang>/finetune-hardneg/checkpoints/best_dev/` |
| H4 | `index-code-hardneg.sh <lang>` | Encode/retrieve on the test set using the hard-negative finetune checkpoint — reproduces "SANTA (Hard Negative)" | H3's `best_dev`, step 7's query/doc files | `data/processed/code/<lang>/test/<lang>_inference-hardneg.trec` |
| — | `RUN_TAG=-hardneg bash evaluate_code.sh <lang>` | Score it | `..._inference-hardneg.trec` | MRR printed to stdout |

**Known gaps and assumptions** (from the comments at the top of
`mine-hard-negatives-code.sh` — read those directly for the full reasoning):
the paper (Appendix A.4) states batch 128 / lr 2e-5 (CodeSearch) or 1e-5
(Adv) / epoch 12 / warmup ratio 0.1, and "one hard negative... randomly
sampled from the top 100 retrieved negative codes by pretrained SANTA."
Beyond that quote, several things are this pipeline's own judgment call, not
things the paper states: the mining corpus (assumed to be the same
per-language `codebase.jsonl` used at test time, not just the training
split's own positives — inferred from Table 7's document counts matching
`codebase.jsonl`'s size, not stated outright); static vs. iterative mining
(this pipeline mines once from the zero-shot checkpoint, per Appendix A.4's
reading, even though the paper's main body separately describes an
ANCE-style iterative scheme that doesn't fully reconcile with the Appendix);
retrieval depth 101 not 100 (to still leave 100 true negatives after
dropping a query's own positive if retrieved); and the random sampling seed
(pinned to 42, matching `build_code_entity.py --random_seed 42`'s existing
convention, since the paper doesn't specify one). Expect the reproduced
Zero-Shot/Hard-Negative numbers to track the paper's reported ones but not
necessarily match exactly, given these gaps.

---

## Hyperparameters and why

Everything below is either quoted directly from the paper (Appendix A.4) or a
judgment call made explicit in code comments — see MODIFICATIONS.md for the
exact `shell/*.sh` lines.

| Stage | Batch size | LR | Epochs | Notes |
|---|---|---|---|---|
| Code pretrain | 16 × grad-accum 8 (=128 effective) | 5e-5 | 10 | `--bf16 True` |
| Code finetune | 16 × grad-accum 8 (=128 effective) | **1e-5 for python, 2e-5 for the other 5 languages** | 12 | Paper's Appendix A.4 split by language; the pre-existing script used a flat 2e-5/epoch=10 for every language, which doesn't match the paper for python |
| Product pretrain | 16 | 1e-4 | 6 | `--bf16 True` |
| Product finetune | 16 | 1e-4 | 6 | Same batch/LR/epoch as pretrain, per README |
| Code finetune (hard negative) | 16 × grad-accum 8 (=128 effective) | 1e-5 / 2e-5, same split as above | 12 | Same as code finetune, plus `train_n_passages 2` (1 positive + 1 hard negative) and `warmup_ratio 0.1`, which the paper states explicitly for this recipe but the baseline in-batch run doesn't set |

**`max_steps` is computed explicitly**, not left to `num_train_epochs`
auto-derivation — `pretrain-code.sh`/`pretrain-product.sh`/`finetune-code.sh`/
`finetune-product.sh` all compute `N` (line count of the training file) and
derive `MAX_STEPS` from it before calling `train_santa.py`. This was done
because `num_train_epochs`-based step counting was observed to be unreliable
when multiple `accelerate`-backed training runs start concurrently on the
same box — one run silently trained for far fewer steps than
`num_train_epochs` implied. If you're running one language/track at a time
with nothing else training concurrently, this is extra safety, not a
correctness requirement — but it's cheap, so it stays on unconditionally.

**The final checkpoint is copied into a `checkpoint-<N>/` subdirectory** after
each pretrain/finetune run. The `dev_*` scripts' best-checkpoint selection
only globs `checkpoint-*` subdirectories, but the trainer's true final/most-
converged save lands directly in the output root — without this copy, the
last (often best) checkpoint would silently never be considered for
`best_dev`.

## First-time-run gotchas

- **Order is strict.** Each stage hardcodes the path written by the previous
  one (e.g. `finetune-code.sh` reads `.../pretrain<RUN_TAG>/checkpoints/best_dev`
  as its starting model). Skipping a `dev_*` step means `best_dev` is never
  created and the next script fails to find its model directory.
- **`best_dev` isn't a fixed checkpoint number.** The `dev_*` scripts
  evaluate *every* saved `checkpoint-*` and copy whichever scored highest
  into `best_dev/`. Re-running a `dev_*` script overwrites `best_dev/` with
  whatever's currently best among saved checkpoints at that point.
- **CodeSearch languages need `codebase.jsonl` as the doc/codebase, not
  `valid.jsonl`/`test.jsonl` twice.** Only `python` (AdvTest) has real code +
  docstring in the same file; the other five languages split queries
  (code-stripped) from the codebase (docstring-stripped) into separate
  files. `dev_code_finetune.sh` and `build-code-test.sh` already branch on
  this — but if you ever call the underlying `evaluate_code_finetune.py` /
  `build_test_code.py` directly instead of through the shell scripts, using
  the same file for both query and codebase silently degenerates to a
  near-empty codebase and near-zero MRR, with no error.
- **`save_total_limit 3` caps checkpoint disk usage**, but combined with the
  `checkpoint-<N>/` final-copy step, expect roughly 4 checkpoint directories'
  worth of disk per run (3 rolling + 1 final copy) — for CodeT5-base/T5-base
  at these batch sizes, budget a few GB per checkpoint directory times 6
  languages × 2 stages (pretrain+finetune) if running the full code track.
- **`bf16` requires Ampere or newer.** If training fails immediately with a
  dtype/capability error, your GPU predates bf16 support — remove
  `--bf16 True` from the relevant `python train_santa.py` invocation in the
  shell script (not a flag this pipeline exposes via env var).
- **Product finetuning with hard negatives** (mentioned in `README.md`) means
  setting `train_n_passages` to `n+1` in `finetune-product.sh`'s
  `train_santa.py` call — this repo's `finetune-product.sh` runs the
  in-batch-negatives-only baseline (`train_n_passages 1`) by default, matching
  upstream. (The *code*-track hard-negative pipeline — `mine-hard-negatives-code.sh`
  → `finetune-code-hardneg.sh` — is included; see the Table 6 extras table
  above. There's no product-track equivalent script in this repo, only the
  code-track one the paper's Table 6 covers.)
