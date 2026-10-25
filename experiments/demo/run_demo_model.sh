#!/bin/bash
# Usage: GPU=<n> SEED=<s> bash experiments/demo/run_demo_model.sh
# Rebuilds the model served by demo/ (search demo for outside testers): the pretrainonly_sdamep arm of
# experiments/09_pooled/run_pooled_pretrainonly.sh with SIMILARITY=poincare on anonymized queries (pretrain on all 388 train docs,
# SDA+MEP, no finetuning, final checkpoint), seed 42 by default. Same settings as that launcher, but the checkpoint
# is KEPT at runs/xml/pretrain/demo/<tag>/checkpoints. Re-evaluates on the 100-doc test split so the result can be
# compared with results/09_pooled_poincare/pooled_anonq_poincare_s<seed>/pretrainonly_sdamep/eval_test.json.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
seed=${SEED:-42}; tag=anonq_poincare_sdamep_s$seed
export P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=${GPU:-1} MODEL_NAME=pooled_to_pooled
export RUN_TAG=/demo/$tag DATA_DIR=$SANTA/data/xml/tokenized/data_pooled_anonq/pretrain_full PRETRAIN_EXTRA_ARGS=""
export EXTRA_TRAIN_ARGS="--pooling mean --similarity poincare --poincare_scale 0.25 --temperature 0.05 --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed"
L=runs/xml/demo_logs; mkdir -p $L
R=$SANTA/results/demo/$tag; mkdir -p $R
bash shell/pretrain-xml.sh > $L/$tag.pretrain.log 2>&1 || { echo "FAIL pretrain (see $L/$tag.pretrain.log)"; exit 1; }
export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled_anonq MODEL_PATH=$SANTA/runs/xml/pretrain/demo/$tag/checkpoints
export TREC_PATH=$SANTA/runs/xml/retrieve/demo/$tag/test_inference.trec
EMB_SAVE_PATH=$R/embeddings.npz bash shell/index-xml.sh > $L/$tag.index.log 2>&1 || { echo "FAIL index"; exit 1; }
RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$tag.eval.log 2>&1 || { echo "FAIL eval"; exit 1; }
echo "DONE: model at $MODEL_PATH"; cat $R/eval_test.json
