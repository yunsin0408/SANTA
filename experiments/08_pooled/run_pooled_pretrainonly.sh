#!/bin/bash
# Usage: SIMILARITY=<cos|poincare> GPU=<n> bash experiments/08_pooled/run_pooled_pretrainonly.sh
# Pretrain-only arms on the pooled split: pretrain on all 388 train docs, NO finetuning, evaluate the final
# pretraining checkpoint (no dev-based checkpoint selection). Compare with the full arm (finetune on the same 388
# pairs) of experiments/08_pooled/run_pooled.sh: the SDA descriptions are the finetuning queries, so SDA+MEP vs. full isolates MEP
# plus the pretraining schedule (lr 5e-5 x 10 ep vs. 2e-5 x 12 ep).
#   pretrainonly_sdamep  SDA+MEP; one model per query version (SDA reads the descriptions)
#   pretrainonly_mep     MEP only (--use_contrastive False); MEP never reads the descriptions, so one model per seed
#                        (trained on the original-query data), evaluated on both query versions
# Seeds 42, 1, 2. Results -> results/08_pooled{,_poincare}/pooled{,_anonq}{,_poincare}_s<seed>/<arm>/.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
SS=""; SIM_ARGS="--normalize True --temperature 0.05"
if [[ "$SIMILARITY" == poincare ]]; then
  SS=_poincare
  SIM_ARGS="--similarity poincare --poincare_scale ${POINCARE_SCALE:-0.25} --temperature ${POINCARE_TEMP:-0.05}"
fi
export P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=${GPU:-1} MODEL_NAME=pooled_to_pooled
L=runs/xml/pooled_pretrainonly${SS}_logs; mkdir -p $L
say() { echo "[$(date +%T)] $*" >> $L/progress.log; }

run_one() {  # <train data suffix: ''|_anonq> <arm> <seed> <extra pretrain args> <eval query suffixes...>
  local dsuf=$1 arm=$2 seed=$3 pargs=$4; shift 4
  local tag=pooled${dsuf}${SS}_s$seed n=pooled${dsuf}${SS}_s${seed}_$arm
  export RUN_TAG=/$tag/$arm DATA_DIR=$SANTA/data/xml/tokenized/data_pooled$dsuf/pretrain_full PRETRAIN_EXTRA_ARGS="$pargs"
  export EXTRA_TRAIN_ARGS="--pooling mean $SIM_ARGS --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed"
  say "START $n"
  bash shell/pretrain-xml.sh > $L/$n.pretrain.log 2>&1 || { say "FAIL pretrain $n"; return; }
  for q in "$@"; do
    local etag=pooled${q}${SS}_s$seed R=$SANTA/results/08_pooled$SS/pooled${q}${SS}_s$seed/$arm; mkdir -p $R
    export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled$q MODEL_PATH=$SANTA/runs/xml/pretrain/$tag/$arm/checkpoints
    export TREC_PATH=$SANTA/runs/xml/retrieve/$etag/$arm/test_inference.trec
    EMB_SAVE_PATH=$R/embeddings.npz bash shell/index-xml.sh > $L/${etag}_$arm.index.log 2>&1 || { say "FAIL index ${etag}_$arm"; continue; }
    RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/${etag}_$arm.eval.log 2>&1 || { say "FAIL eval ${etag}_$arm"; continue; }
    say "DONE  ${etag}_$arm"
  done
  rm -rf "$SANTA/runs/xml/pretrain/$tag/$arm"  # keep results/TREC/embeddings, drop checkpoints
}

for seed in ${SEEDS:-42 1 2}; do  # SEEDS / ARMS (sdamep, mep) env vars rerun a subset
  if [[ " ${ARMS:-sdamep mep} " == *" sdamep "* ]]; then
    run_one "" pretrainonly_sdamep $seed "" ""
    run_one _anonq pretrainonly_sdamep $seed "" _anonq
  fi
  [[ " ${ARMS:-sdamep mep} " == *" mep "* ]] && run_one "" pretrainonly_mep $seed "--use_contrastive False" "" _anonq
done
say "QUEUE-DONE"
