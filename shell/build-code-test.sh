#!/bin/bash
set -e
# Usage: bash build-code-test.sh <language>
# python evaluates on AdvTest (query+doc from the same paired test.jsonl); the
# other 5 languages evaluate on CodeSearch, where the query set (test.jsonl) has
# code stripped and the doc/codebase set (codebase.jsonl) has docstrings stripped.
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
if [ "${language}" = "python" ]; then
  export QUERY_INPUT=${DATA_ROOT}/AdvTest/test.jsonl
  export DOC_INPUT=${DATA_ROOT}/AdvTest/test.jsonl
else
  export QUERY_INPUT=${DATA_ROOT}/CSN_CodeSearch/CSN/dataset/${language}/test.jsonl
  export DOC_INPUT=${DATA_ROOT}/CSN_CodeSearch/CSN/dataset/${language}/codebase.jsonl
fi

export OUTPUT_PATH=${DATA_ROOT}/processed/code/${language}/test
export output_query=${language}_test_query.jsonl
export output_doc=${language}_test_doc.jsonl

# OUTPUT_PATH is the path where the processed test data is saved.
# data_type chooses whether to process the query or code.

mkdir -p ${OUTPUT_PATH}

cd ../evaluate_code
python build_test_code.py \
     --input ${QUERY_INPUT}  \
     --output ${OUTPUT_PATH}/${output_query}  \
     --data_type query  \

python build_test_code.py \
     --input ${DOC_INPUT}  \
     --output ${OUTPUT_PATH}/${output_doc}  \
     --data_type doc  \
