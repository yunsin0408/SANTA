#!/bin/bash
set -e
# Usage: bash process-finetune-code.sh <language>
# python finetunes on AdvTest; the other 5 languages finetune on CodeSearch.
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
if [ "${language}" = "python" ]; then
  export FINETUNE_RAW_PATH=${DATA_ROOT}/AdvTest
  export finetune_raw_data=train.jsonl
else
  export FINETUNE_RAW_PATH=${DATA_ROOT}/CSN_CodeSearch/CSN/dataset/${language}
  export finetune_raw_data=train.jsonl
fi

export FINETUNE_PATH=${DATA_ROOT}/processed/code/${language}/finetune
export finetune_data=${language}.jsonl

mkdir -p ${FINETUNE_PATH}

cd ../processing/Code
python build_code.py \
--input ${FINETUNE_RAW_PATH}/${finetune_raw_data} \
--output ${FINETUNE_PATH}/${finetune_data}
