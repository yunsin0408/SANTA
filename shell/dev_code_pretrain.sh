#!/bin/bash
set -e
# Usage: bash dev_code_pretrain.sh <language>
export language=$1

export RUN_TAG=${RUN_TAG:-}
export CHECK_POINT=/workspace/SANTA_v2/runs/code/${language}/pretrain${RUN_TAG}/checkpoints
export EVAL_PATH=/workspace/SANTA_v2/data/CodeSearchNet/resources/data/${language}/final/jsonl/valid
export eval_data_file=${language}_valid.jsonl
export codebase_file=${language}_valid.jsonl

# CHECK_POINT is the directory which includes checkpoints

cd ../code_best_dev
python evaluate_code_pretrain.py \
    --model_name_or_path ${CHECK_POINT} \
    --eval_data_file ${EVAL_PATH}/${eval_data_file}  \
    --codebase_file ${EVAL_PATH}/${codebase_file}  \
    --code_length 240 \
    --nl_length 50 \
    --eval_batch_size 64
