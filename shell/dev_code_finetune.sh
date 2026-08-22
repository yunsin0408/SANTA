#!/bin/bash
set -e
# Usage: bash dev_code_finetune.sh <language>
export language=$1

export RUN_TAG=${RUN_TAG:-}
export CHECK_POINT=/workspace/SANTA_v2/runs/code/${language}/finetune${RUN_TAG}/checkpoints
if [ "${language}" = "python" ]; then
  # AdvTest's valid.jsonl has real code + docstring in the same rows.
  export EVAL_PATH=/workspace/SANTA_v2/data/AdvTest
  export eval_data_file=valid.jsonl
  export codebase_file=valid.jsonl
else
  # CodeSearch's valid.jsonl has code blanked out (query-only); codebase.jsonl has
  # docstrings blanked out (code-only) -- see shell/build-code-test.sh for the same
  # split. Using valid.jsonl for both (as the original script did) degenerates to a
  # near-empty codebase and near-zero MRR.
  export EVAL_PATH=/workspace/SANTA_v2/data/CSN_CodeSearch/CSN/dataset/${language}
  export eval_data_file=valid.jsonl
  export codebase_file=codebase.jsonl
fi

# CHECK_POINT is the directory which includes checkpoints

cd ../code_best_dev
python evaluate_code_finetune.py \
    --model_name_or_path ${CHECK_POINT} \
    --eval_data_file ${EVAL_PATH}/${eval_data_file}  \
    --codebase_file ${EVAL_PATH}/${codebase_file}  \
    --code_length 240 \
    --nl_length 50 \
    --eval_batch_size 64
