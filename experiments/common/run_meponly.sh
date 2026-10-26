#!/bin/bash
# Usage: bash experiments/common/run_meponly.sh <orig|anon> <GPU>
# MEP-only pretraining ablation (SANTA Table 3 row "MEP only"): SDA contrastive loss off during pretraining
# (--use_contrastive False, pretrain stage only via PRETRAIN_EXTRA_ARGS), MEP on. Everything else as mepablate_s42:
# mean pooling + cosine tau 0.05, p_max_len 1024, seed 42, 10 pretrain epochs, finetuning unchanged,
# BIKE_7 / FOSSIG / s1kd-tools-doc -> S1000D_spec_sample. Checkpoints are deleted after each pair's test evaluation.
MODE=$1; GPU=$2
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$SANTA"
COS="--pooling mean --normalize True --temperature 0.05"
if [ "$MODE" = orig ]; then
  TAG=meponly_s42; DATA_ROOT=data_cnt; export RESULTS_ROOT=$SANTA/results/05_pretrain_diagnosis
  unset RETRIEVAL_ROOT
else
  TAG=anonq1024_meponly_s42; DATA_ROOT=data_anonq1024; export RESULTS_ROOT=$SANTA/results/08_anonymized
  export RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_anonq
fi
export PRETRAIN_EXTRA_ARGS="--use_contrastive False"
for src in BIKE_7 FOSSIG s1kd-tools-doc; do
  pair=${src}_to_S1000D_spec_sample
  bash experiments/common/run_pairs.sh $TAG $GPU 1024 "$COS" 42 $DATA_ROOT $pair
  rm -rf runs/xml/pretrain/$TAG/$pair runs/xml/finetune/$TAG/$pair
  echo "[$(date +%T)] cleaned checkpoints of $TAG/$pair" >> runs/xml/${TAG}_logs/progress.log
done
echo "[$(date +%T)] MEPONLY-DONE $TAG" >> runs/xml/${TAG}_logs/progress.log
