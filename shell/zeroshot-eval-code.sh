#!/bin/bash
set -e
# Usage: bash zeroshot-eval-code.sh <language>
# Reproduces the "Zero-Shot" row of Table 6: evaluate the continuously-pretrained
# SANTA checkpoint directly on the CodeSearch/Adv test set, with no finetuning at
# all. Run build-code-test.sh <language> first.
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
export TEST_PATH=${DATA_ROOT}/processed/code/${language}/test
export OUTPUT_PATH=${TEST_PATH}/zeroshot/
export MODEL_PATH=/workspace/SANTA_v2/runs/code/${language}/pretrain/checkpoints/best_dev
export DOC_PATH=${TEST_PATH}/${language}_test_doc.jsonl
export QUERY_PATH=${TEST_PATH}/${language}_test_query.jsonl
export TREC_PATH=${TEST_PATH}/${language}_inference-zeroshot.trec

mkdir -p ${OUTPUT_PATH}

cd ../OpenMatch/src
python -m openmatch.driver.build_index  \
    --output_dir ${OUTPUT_PATH} \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --corpus_path ${DOC_PATH}  \
    --doc_template "<code>"  \
    --q_max_len 50  \
    --p_max_len 240  \
    --dataloader_num_workers 1

python -m openmatch.driver.retrieve  \
    --output_dir ${OUTPUT_PATH}  \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --query_path ${QUERY_PATH} \
    --query_template "<query>"  \
    --q_max_len 50  \
    --trec_save_path ${TREC_PATH}

# Score with: RUN_TAG=-zeroshot bash evaluate_code.sh ${language}
