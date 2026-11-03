#!/bin/bash
# Usage: bash experiments/10_lorentz/run_pooled.sh <queue: 0|1>
# experiments/08_pooled/run_pooled.sh (exp/08-pooled branch) with Lorentz similarity, in pretraining, finetuning and
# evaluation. Queue 0 = original queries, queue 1 = anonymized queries.
# Per seed (42, 1, 2), arms:
#   full                  no pretraining, finetune on the whole train pool
#   pretrainA_finetuneB   SDA+MEP pretrain on halfA, finetune on halfB  (control: finetuneB_only)
#   pretrainB_finetuneA   SDA+MEP pretrain on halfB, finetune on halfA  (control: finetuneA_only)
# Optional env LORENTZ_SCALE (default 0.5), LORENTZ_TEMP (default 0.05).
# Results -> results/10_lorentz/pooled{,_anonq}_lorentz_s<seed>/<arm>/.
Q=$1
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
SUF=$([[ "$Q" == 1 ]] && echo _anonq)
SS=_lorentz
SIM_ARGS="--similarity lorentz --lorentz_scale ${LORENTZ_SCALE:-0.5} --temperature ${LORENTZ_TEMP:-0.05}"
export Q_MAX_LEN=50
OUT=$SANTA/results/10_lorentz
export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled$SUF PRETRAIN_ROOT=$SANTA/data/xml/pretrain_pooled$SUF
export P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=${GPU:-$Q} MODEL_NAME=pooled_to_pooled  # GPU overrides the queue's default GPU
L=runs/xml/pooled${SUF}${SS}_logs; mkdir -p $L
say() { echo "[$(date +%T)] $*" >> $L/progress.log; }

run_one() {  # <tag> <arm> <data subdir> <yes|no: pretrain> <seed>
  local tag=$1 arm=$2 data=$3 do_pretrain=$4 seed=$5
  local n=${tag}_$arm R=$OUT/$tag/$arm; mkdir -p $R
  export RUN_TAG=/$tag/$arm DATA_DIR=$SANTA/data/xml/tokenized/data_pooled$SUF/$data
  export EXTRA_TRAIN_ARGS="--pooling mean $SIM_ARGS --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed"
  say "START $n"
  if [ "$do_pretrain" = yes ]; then
    unset PRETRAIN_CHECKPOINT
    bash shell/pretrain-xml.sh > $L/$n.pretrain.log 2>&1 || { say "FAIL pretrain $n"; return; }
    RESULTS_PATH=$R/best_dev_pretrain.json bash shell/dev_xml_pretrain.sh > $L/$n.bdp.log 2>&1 || { say "FAIL bdp $n"; return; }
  else
    export PRETRAIN_CHECKPOINT=Salesforce/codet5-base
  fi
  bash shell/finetune-xml.sh > $L/$n.finetune.log 2>&1 || { say "FAIL finetune $n"; return; }
  RESULTS_PATH=$R/best_dev_finetune.json bash shell/dev_xml_finetune.sh > $L/$n.bdf.log 2>&1 || { say "FAIL bdf $n"; return; }
  EMB_SAVE_PATH=$R/embeddings.npz bash shell/index-xml.sh > $L/$n.index.log 2>&1 || { say "FAIL index $n"; return; }
  RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$n.eval.log 2>&1 || { say "FAIL eval $n"; return; }
  rm -rf "$SANTA/runs/xml/pretrain/$tag/$arm" "$SANTA/runs/xml/finetune/$tag/$arm"  # keep results/TREC/embeddings, drop checkpoints
  say "DONE  $n"
}

mkdir -p $OUT
for seed in 42 1 2; do
  tag=pooled${SUF}${SS}_s$seed
  run_one $tag full full no $seed
  run_one $tag pretrainA_finetuneB pretrainA_finetuneB yes $seed
  run_one $tag finetuneB_only finetuneB_only no $seed
  run_one $tag pretrainB_finetuneA pretrainB_finetuneA yes $seed
  run_one $tag finetuneA_only finetuneA_only no $seed
done
say "QUEUE-DONE"
