#!/bin/bash
# Scale check for Lorentz similarity: full arm, anonymized queries, seed 42, tau 0.05, lorentz_scale in
# {0.25, 0.5, 1.0, 2.0} (override with SCALES="..."). 0.5 is the Poincare recipe's geometry (see --lorentz_scale). Pick
# the scale by dev MRR (final checkpoint), not test. Results -> results/10_lorentz_scale/.
# Also runs Poincare at poincare_scale 1.0 (override with POINCARE_SCALES="..."), the same radii as lorentz_scale 2.0,
# near the ball's boundary: any gap between the two comes from the numerics, not the geometry.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
export CUDA_VISIBLE_DEVICES=${GPU:-1} P_MAX_LEN=1024 MODEL_NAME=pooled_to_pooled PRETRAIN_CHECKPOINT=Salesforce/codet5-base
export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled_anonq DATA_DIR=$SANTA/data/xml/tokenized/data_pooled_anonq/full
L=runs/xml/lorentz_scale_logs; mkdir -p $L

run_one() {  # <lorentz|poincare> <scale>
  local sim=$1 scale=$2 tag=$1_scale$2 R=$SANTA/results/10_lorentz_scale/$1_scale$2; mkdir -p $R
  export RUN_TAG=/$tag/full EXTRA_TRAIN_ARGS="--pooling mean --similarity $sim --${sim}_scale $scale --temperature ${LORENTZ_TEMP:-0.05} --gradient_accumulation_steps 1 --logging_steps 1 --report_to none --seed 42"
  echo "[$(date +%T)] START $tag" >> $L/progress.log
  bash shell/finetune-xml.sh > $L/$tag.finetune.log 2>&1 \
    && RESULTS_PATH=$R/best_dev_finetune.json bash shell/dev_xml_finetune.sh > $L/$tag.bdf.log 2>&1 \
    && EMB_SAVE_PATH= bash shell/index-xml.sh > $L/$tag.index.log 2>&1 \
    && RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$tag.eval.log 2>&1 \
    && echo "[$(date +%T)] DONE  $tag" >> $L/progress.log || echo "[$(date +%T)] FAIL  $tag" >> $L/progress.log
  rm -rf "$SANTA/runs/xml/finetune/$tag"
}

for scale in ${SCALES:-0.25 0.5 1.0 2.0}; do run_one lorentz $scale; done
for scale in ${POINCARE_SCALES:-1.0}; do run_one poincare $scale; done
echo "[$(date +%T)] QUEUE-DONE" >> $L/progress.log
