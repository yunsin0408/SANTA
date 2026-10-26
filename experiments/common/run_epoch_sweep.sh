#!/bin/bash
# Usage: [QUERIES=orig|anon] bash experiments/common/run_epoch_sweep.sh <GPU> [EPOCHS...]   (default epochs: 5 30 50 10)
# Pretrain-epoch sweep extending Ablation 2 (10 vs. 100 epochs, results/README.md): runs run_longpretrain.sh once per
# epoch count, sequentially, writing results/05_pretrain_diagnosis/longpretrain<N>_s42/. Finetuning stays at 12 epochs.
# 10 is included to give a baseline from this exact script. Checkpoints are deleted after each epoch count finishes.
# GRAD_CKPT=1 (default here) turns on activation checkpointing (model.py): BIKE_7 peaks at ~32 GB without it, more than
# the shared GPUs have free. Same gradients, just slower; set GRAD_CKPT=0 to disable.
# QUERIES=anon runs the anonymized-query version (data_anonq1024/, data/retrieval_anonq/, as run_anonq1024.sh does) and
# writes results/08_anonymized/anonq1024_longpretrain<N>_s42/.
GPU=${1:?usage: run_epoch_sweep.sh <GPU> [EPOCHS...]}; shift
EPOCHS=${@:-5 30 50 10}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$SANTA"
export GRAD_CKPT=${GRAD_CKPT:-1}
if [ "${QUERIES:-orig}" = anon ]; then
  export DATA_ROOT=data_anonq1024 RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_anonq RESULTS_ROOT=$SANTA/results/08_anonymized; PFX=anonq1024_
else
  export RESULTS_ROOT=$SANTA/results/05_pretrain_diagnosis; PFX=
fi
L=runs/xml/${PFX}epoch_sweep_logs; mkdir -p $L
say() { echo "[$(date '+%F %T')] $*" >> $L/progress.log; }
for ep in $EPOCHS; do
  TAG=${PFX}longpretrain${ep}_s42
  say "START $TAG"
  bash experiments/common/run_longpretrain.sh $TAG $GPU $ep
  rm -rf "$SANTA/runs/xml/pretrain/$TAG" "$SANTA/runs/xml/finetune/$TAG"
  say "DONE  $TAG ($(grep -c '] DONE  ' runs/xml/${TAG}_logs/progress.log)/3 pairs ok)"
done
say "SWEEP-DONE"
