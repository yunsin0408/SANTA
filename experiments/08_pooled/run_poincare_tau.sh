#!/bin/bash
# Temperature check for SIMILARITY=poincare before the main run: full arm, anonymized queries, seed 42,
# tau in {0.05, 0.1, 0.2} (override with TAUS="..."; 0.02 and 0.01 were added after 0.05 came out best). Pick tau by dev MRR (final checkpoint), not test. Results -> results/08_pooled_poincare_tau/.
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
export CUDA_VISIBLE_DEVICES=${GPU:-1} P_MAX_LEN=1024 MODEL_NAME=pooled_to_pooled PRETRAIN_CHECKPOINT=Salesforce/codet5-base
export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_pooled_anonq DATA_DIR=$SANTA/data/xml/tokenized/data_pooled_anonq/full
L=runs/xml/poincare_tau_logs; mkdir -p $L
for tau in ${TAUS:-0.05 0.1 0.2}; do
  tag=poincare_tau$tau; R=$SANTA/results/08_pooled_poincare_tau/$tag; mkdir -p $R
  export RUN_TAG=/$tag/full EXTRA_TRAIN_ARGS="--pooling mean --similarity poincare --poincare_scale 0.25 --temperature $tau --gradient_accumulation_steps 1 --logging_steps 1 --report_to none --seed 42"
  echo "[$(date +%T)] START $tag" >> $L/progress.log
  bash shell/finetune-xml.sh > $L/$tag.finetune.log 2>&1 \
    && RESULTS_PATH=$R/best_dev_finetune.json bash shell/dev_xml_finetune.sh > $L/$tag.bdf.log 2>&1 \
    && EMB_SAVE_PATH= bash shell/index-xml.sh > $L/$tag.index.log 2>&1 \
    && RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh > $L/$tag.eval.log 2>&1 \
    && echo "[$(date +%T)] DONE  $tag" >> $L/progress.log || echo "[$(date +%T)] FAIL  $tag" >> $L/progress.log
  rm -rf "$SANTA/runs/xml/finetune/$tag"
done
echo "[$(date +%T)] QUEUE-DONE" >> $L/progress.log
