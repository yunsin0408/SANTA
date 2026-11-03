#!/bin/bash
# Usage: GPU=<n> bash experiments/10_lorentz/run_pooled_pretrainonly.sh
# SDA+MEP pretrain-only arm of experiments/08_pooled/run_pooled_pretrainonly.sh (exp/08-pooled branch) with Lorentz
# similarity: pretrain on all 388 train docs, no finetuning, evaluate the final pretraining checkpoint (no dev-based
# checkpoint selection). One model per query version, seeds 42, 1, 2.
# Optional env LORENTZ_SCALE (default 0.5), LORENTZ_TEMP (default 0.05), SEEDS.
# Results -> results/10_lorentz/pooled{,_anonq}_lorentz_s<seed>/pretrainonly_sdamep/.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
SS=_lorentz
SIM_ARGS="--similarity lorentz --lorentz_scale ${LORENTZ_SCALE:-0.5} --temperature ${LORENTZ_TEMP:-0.05}"
export P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=${GPU:-1} MODEL_NAME=pooled_to_pooled
L=runs/xml/pooled_pretrainonly${SS}_logs; mkdir -p $L
say() { echo "[$(date +%T)] $*" >> $L/progress.log; }

run_one() {  # <query suffix: ''|_anonq> <seed>
  local q=$1 seed=$2 arm=pretrainonly_sdamep
  local tag=pooled${q}${SS}_s$seed
  local n=${tag}_$arm R=$SANTA/results/10_lorentz/$tag/$arm; mkdir -p $R  # separate line: tag must be set first
  export RUN_TAG=/$tag/$arm DATA_DIR=$SANTA/data/xml/tokenized/data_pooled$q/pretrain_full PRETRAIN_EXTRA_ARGS=""
  export EXTRA_TRAIN_ARGS="--pooling mean $SIM_ARGS --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed"
  say "START $n"
  bash shell/pretrain-xml.sh > $L/$n.pretrain.log 2>&1 || { say "FAIL pretrain $n"; return; }
  export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled$q MODEL_PATH=$SANTA/runs/xml/pretrain/$tag/$arm/checkpoints
  export TREC_PATH=$SANTA/runs/xml/retrieve/$tag/$arm/test_inference.trec
  EMB_SAVE_PATH=$R/embeddings.npz bash shell/index-xml.sh > $L/$n.index.log 2>&1 || { say "FAIL index $n"; return; }
  RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$n.eval.log 2>&1 || { say "FAIL eval $n"; return; }
  say "DONE  $n"
  rm -rf "$SANTA/runs/xml/pretrain/$tag/$arm"  # keep results/TREC/embeddings, drop checkpoints
}

for seed in ${SEEDS:-42 1 2}; do
  run_one "" $seed
  run_one _anonq $seed
done
say "QUEUE-DONE"
