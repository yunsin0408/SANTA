#!/bin/bash
set -e
# Usage: bash mine-hard-negatives-code.sh <language>
#
# Mines one hard negative per finetuning-train example, per SANTA paper Appendix
# A.4: "the hard negative is randomly sampled from the top 100 retrieved negative
# codes by pretrained SANTA" -- retrieval uses the continuously-pretrained
# (zero-shot, NOT finetuned) checkpoint, run once before hard-negative finetuning
# starts (not an iterative ANCE-style re-mining loop -- see shell/PIPELINE.md for
# why we read it this way despite the paper's main text describing something
# closer to iterative ANCE elsewhere).
#
# ASSUMPTION (not stated explicitly in the paper): the mining corpus is the same
# per-language `codebase.jsonl` used for test-time evaluation, not just the
# training split's own positives -- CodeSearch's Table 7 "Document" counts match
# codebase.jsonl's size exactly (e.g. javascript: 13,981), far smaller than the
# training query count (58,025), so the paper very likely reuses one fixed corpus
# throughout rather than mining only within the train split.
#
# Prerequisite: runs/code/<language>/pretrain/checkpoints/best_dev must already
# exist (i.e. dev_code_pretrain.sh has been run).
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
if [ "${language}" = "python" ]; then
  # AdvTest has no separate codebase file; fall back to the train split's own
  # positives as the mining corpus -- the same fallback process-finetune-code.sh
  # and build-code-test.sh already make for AdvTest.
  export TRAIN_RAW=${DATA_ROOT}/AdvTest/train.jsonl
  export CODEBASE_RAW=${DATA_ROOT}/AdvTest/train.jsonl
else
  export TRAIN_RAW=${DATA_ROOT}/CSN_CodeSearch/CSN/dataset/${language}/train.jsonl
  export CODEBASE_RAW=${DATA_ROOT}/CSN_CodeSearch/CSN/dataset/${language}/codebase.jsonl
fi

export MINE_PATH=${DATA_ROOT}/processed/code/${language}/mine
export train_query=${language}_train_query.jsonl
export pool_doc=${language}_pool_doc.jsonl
export TREC_PATH=${MINE_PATH}/${language}_mine.trec

mkdir -p ${MINE_PATH}

# Reuse build_test_code.py (already generic query/doc extraction) to build the
# mining query set (train queries) and mining corpus (codebase docs).
cd ../evaluate_code
python build_test_code.py --input ${TRAIN_RAW} --output ${MINE_PATH}/${train_query} --data_type query
python build_test_code.py --input ${CODEBASE_RAW} --output ${MINE_PATH}/${pool_doc} --data_type doc

# Zero-shot checkpoint: continuously-pretrained, NOT finetuned -- same model used
# by zeroshot-eval-code.sh for the "Zero-Shot" row.
export MODEL_PATH=/workspace/SANTA_v2/runs/code/${language}/pretrain/checkpoints/best_dev

cd ../OpenMatch/src
python -m openmatch.driver.build_index  \
    --output_dir ${MINE_PATH}/  \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --corpus_path ${MINE_PATH}/${pool_doc}  \
    --doc_template "<code>"  \
    --q_max_len 50  \
    --p_max_len 240  \
    --dataloader_num_workers 1

# retrieve_depth 101, not 100: build_code_hard_negative.py drops the query's own
# positive if the retriever happens to surface it, so we ask for one extra to
# still guarantee 100 true negative candidates to sample from.
python -m openmatch.driver.retrieve  \
    --output_dir ${MINE_PATH}/  \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --query_path ${MINE_PATH}/${train_query}  \
    --query_template "<query>"  \
    --q_max_len 50  \
    --retrieve_depth 101  \
    --trec_save_path ${TREC_PATH}

# Merge the mined negatives back onto the raw train pairs.
cd ../../processing/Code
export FINETUNE_HARDNEG_PATH=${DATA_ROOT}/processed/code/${language}/finetune
mkdir -p ${FINETUNE_HARDNEG_PATH}
python build_code_hard_negative.py  \
    --input ${TRAIN_RAW}  \
    --trec ${TREC_PATH}  \
    --doc_pool ${MINE_PATH}/${pool_doc}  \
    --output ${FINETUNE_HARDNEG_PATH}/${language}_hardneg.jsonl  \
    --num_hard_negatives 1  \
    --top_k 100
