#!/bin/bash
# Usage: GPU=<n> SEED=<s> VARIANTS="0 1" bash experiments/demo/run_demo_repro.sh
# Reproducibility check for the demo model (experiments/demo/run_demo_model.sh, original test MRR 0.7022 on 2026-10-07).
# Same training/eval settings as run_demo_model.sh, but every output is isolated so nothing of the demo, its
# backup, or another concurrent run is touched or shared:
#   checkpoints -> runs/xml/pretrain/demo_repro/<tag>_ckpt<V>/checkpoints
#   logs        -> runs/xml/demo_repro_logs/<tag>_ckpt<V>.*.log
#   results     -> results/demo_repro/<tag>_ckpt<V>/eval_test.json
# V = GRAD_CKPT value: 0 = the original setup (no activation checkpointing, needs ~30 GB GPU memory),
# 1 = with activation checkpointing (how the 2026-10-19 rebuild that scored 0.656 was trained).
# Waits until GPU <n> has MIN_FREE_MB free before each variant, and refuses to reuse an existing output dir.
set -e
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
seed=${SEED:-42}; base=anonq_poincare_sdamep_s$seed; gpu=${GPU:-1}
DATA_DIR_SRC=$SANTA/data/xml/tokenized/data_pooled_anonq/pretrain_full
RET=$SANTA/data/xml/retrieval_pooled_anonq
[[ -d $RET ]] || RET=$SANTA/data/xml/retrieval_pooled_anonq
L=$SANTA/runs/xml/demo_repro_logs; mkdir -p $L
for v in ${VARIANTS:-0 1}; do
  tag=${base}_ckpt$v; R=$SANTA/results/demo_repro/$tag
  if [[ -e $SANTA/runs/xml/pretrain/demo_repro/$tag ]]; then echo "SKIP $tag: output dir exists"; continue; fi
  need=${MIN_FREE_MB:-$([[ $v == 0 ]] && echo 32000 || echo 20000)}
  until free=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits) && (( free >= need )); do sleep 60; done
  echo "[$(date)] $tag on GPU $gpu (free ${free} MiB)"
  mkdir -p $R
  (
    export GRAD_CKPT=$v P_MAX_LEN=1024 CUDA_VISIBLE_DEVICES=$gpu MODEL_NAME=pooled_to_pooled
    export RUN_TAG=/demo_repro/$tag DATA_DIR=$DATA_DIR_SRC PRETRAIN_EXTRA_ARGS=""
    export EXTRA_TRAIN_ARGS="--pooling mean --similarity poincare --poincare_scale 0.25 --temperature 0.05 --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $seed"
    bash shell/pretrain-xml.sh > $L/$tag.pretrain.log 2>&1 || { echo "FAIL pretrain $tag (see $L/$tag.pretrain.log)"; exit 1; }
    export RETRIEVAL_ROOT=$RET FINETUNE_SOURCE=pooled MODEL_PATH=$SANTA/runs/xml/pretrain/demo_repro/$tag/checkpoints
    export TREC_PATH=$SANTA/runs/xml/retrieve/demo_repro/$tag/test_inference.trec
    EMB_SAVE_PATH=$R/embeddings.npz bash shell/index-xml.sh > $L/$tag.index.log 2>&1 || { echo "FAIL index $tag"; exit 1; }
    RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$tag.eval.log 2>&1 || { echo "FAIL eval $tag"; exit 1; }
  ) || continue
  echo "$tag: $(grep -o '"eval_mrr": [0-9.]*' $R/eval_test.json)  train_loss: $(tr '\r' '\n' < $L/$tag.pretrain.log | grep -aoE "train_loss': [0-9.]+")"
done
echo "DONE (original demo 2026-10-07: eval_mrr 0.7022, train_loss 1.6939, final eval_loss 0.5193)"
