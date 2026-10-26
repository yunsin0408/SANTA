#!/bin/bash
# Usage: bash experiments/08_pooled/run_pooled.sh <queue: 0|1>
# Pooled split (all four folders merged, 388 train / 20 dev / 100 test, datasets/build_pooled_split.py,
# tokenized by prepare_pooled.py). Queue 0 = original queries (data_pooled/, data/retrieval_pooled/), queue 1 =
# anonymized queries (data_pooled_anonq/, data/retrieval_pooled_anonq/); run both in parallel on GPU 0 and GPU 1.
# Best config so far: CodeT5-base, mean pooling + cosine tau 0.05, p_max_len 1024, finetune lr 2e-5 x 12 ep.
# Per seed (42, 1, 2), arms:
#   full                  no pretraining, finetune on the whole train pool
#   pretrainA_finetuneB   SDA+MEP pretrain on halfA, finetune on halfB  (control: finetuneB_only)
#   pretrainB_finetuneA   SDA+MEP pretrain on halfB, finetune on halfA  (control: finetuneA_only)
# Results -> results/08_pooled/pooled{,_anonq}_s<seed>/<arm>/.
# Optional env SAVE_STEPS=N: save a checkpoint every N steps (no save_total_limit cap) so best_dev really selects
# among checkpoints; tags get suffix _ss<N> and results go to results/08_pooled_ss<N>/.
# Optional env SIMILARITY=poincare: score with negative Poincare-ball distance instead of cosine, in pretraining,
# finetuning and evaluation (--normalize off, reps x POINCARE_SCALE (default 0.25) before the map into the ball,
# temperature POINCARE_TEMP (default 0.05, best dev MRR of 0.01-0.2 in experiments/08_pooled/run_poincare_tau.sh)); suffix _poincare.
# Optional env Q_MAX_LEN=N: query token budget in pretraining, finetuning and evaluation (default 50, which truncates
# every LLM description: mean ~90 tokens, max 148); suffix _q<N>.
Q=$1
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
SUF=$([[ "$Q" == 1 ]] && echo _anonq)
SS=${SAVE_STEPS:+_ss$SAVE_STEPS}
SIM_ARGS="--normalize True --temperature 0.05"
if [[ "$SIMILARITY" == poincare ]]; then
  SS=${SS}_poincare
  SIM_ARGS="--similarity poincare --poincare_scale ${POINCARE_SCALE:-0.25} --temperature ${POINCARE_TEMP:-0.05}"
fi
[[ -n "$Q_MAX_LEN" && "$Q_MAX_LEN" != 50 ]] && SS=${SS}_q$Q_MAX_LEN
export Q_MAX_LEN=${Q_MAX_LEN:-50}
OUT=$SANTA/results/08_pooled$SS
SAVE_ARGS=${SAVE_STEPS:+--save_steps $SAVE_STEPS --save_total_limit 1000}
export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled$SUF PRETRAIN_ROOT=$SANTA/data/xml/pretrain_pooled$SUF
export P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=${GPU:-$Q} MODEL_NAME=pooled_to_pooled  # GPU overrides the queue's default GPU
L=runs/xml/pooled${SUF}${SS}_logs; mkdir -p $L
say() { echo "[$(date +%T)] $*" >> $L/progress.log; }

run_one() {  # <tag> <arm> <data subdir> <yes|no: pretrain> <seed>
  local tag=$1 arm=$2 data=$3 do_pretrain=$4 seed=$5
  local n=${tag}_$arm R=$OUT/$tag/$arm; mkdir -p $R
  export RUN_TAG=/$tag/$arm DATA_DIR=$SANTA/data/xml/tokenized/data_pooled$SUF/$data
  export EXTRA_TRAIN_ARGS="--pooling mean $SIM_ARGS --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed $SAVE_ARGS"
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
BM25_DIR=$RETRIEVAL_ROOT/pooled .venv/bin/python diagnostics/bm25_baseline.py 2>/dev/null | grep BM25 > $OUT/bm25_pooled$SUF.txt
for seed in 42 1 2; do
  tag=pooled${SUF}${SS}_s$seed
  run_one $tag full full no $seed
  run_one $tag pretrainA_finetuneB pretrainA_finetuneB yes $seed
  run_one $tag finetuneB_only finetuneB_only no $seed
  run_one $tag pretrainB_finetuneA pretrainB_finetuneA yes $seed
  run_one $tag finetuneA_only finetuneA_only no $seed
done
say "QUEUE-DONE"
