#!/bin/bash
set -e
# Usage: bash process-pretrain-code.sh <language>
# <language> is one of: ruby javascript go python java php
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
export PRETRAIN_RAW_PATH=${DATA_ROOT}/CodeSearchNet/resources/data/${language}/final/jsonl/train
export PRETRAIN_PATH=${DATA_ROOT}/processed/code/${language}/pretrain
export pretrain_data=${language}.jsonl

mkdir -p ${PRETRAIN_PATH}

# Per the README: Python pretraining data merges CodeSearchNet's train + test;
# every other language uses train only.
if [ "${language}" = "python" ]; then
  export test_path=${DATA_ROOT}/CodeSearchNet/resources/data/${language}/final/jsonl/test/${language}_test.jsonl
  cat ${PRETRAIN_RAW_PATH}/${language}_train.jsonl ${test_path} > ${PRETRAIN_PATH}/${language}_pretrain_raw.jsonl
else
  cp ${PRETRAIN_RAW_PATH}/${language}_train.jsonl ${PRETRAIN_PATH}/${language}_pretrain_raw.jsonl
fi
export pretrain_raw_data=${language}_pretrain_raw.jsonl

cd ../processing/Code
python build_code_entity.py \
--input ${PRETRAIN_PATH}/${pretrain_raw_data} \
--tree ./tree-sitter-languages.so \
--output ${PRETRAIN_PATH}/${pretrain_data} \
--code_type ${language}
