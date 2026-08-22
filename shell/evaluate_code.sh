#!/bin/bash
set -e
# Usage: bash evaluate_code.sh <language>
export language=$1

export RUN_TAG=${RUN_TAG:-}
export TREC_PATH=/workspace/SANTA_v2/data/processed/code/${language}/test/${language}_inference${RUN_TAG}.trec

cd ../evaluate_code
python evaluate_code.py \
    --trec_save_path ${TREC_PATH}
