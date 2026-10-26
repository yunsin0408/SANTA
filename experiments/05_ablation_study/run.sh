#!/bin/bash
# Usage: bash experiments/05_ablation_study/run.sh [GPU]
# Rebuilds results/05_ablation_study/ (see experiments/05_ablation_study/README.md). CodeT5, config A, p_max_len 1024,
# seed 42, the 3 S1000D_spec_sample pairs, original queries:
#   mepablate_s42                 SDA-only pretraining (--use_generate False)
#   meponly_s42                   MEP-only pretraining (--use_contrastive False)
#   longpretrain{5,10,30,50,100}_s42   pretrain epoch sweep, finetuning fixed at 12 epochs
# then writes each sweep and MEP-only run's pretrain_losses.jsonl from its pretrain logs.
# The sweep runs with gradient checkpointing (GRAD_CKPT=1, same gradients, less memory) and deletes its checkpoints as it goes.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
S1000D_PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
[[ -d $TOK/data_cnt ]] || $PY processing/XML/prepare_data.py --output-dir $TOK/data_cnt --safety-max-len 1024

A="--pooling mean --normalize True --temperature 0.05"
RESULTS_ROOT=$SANTA/results/05_ablation_study \
  bash experiments/common/run_pairs.sh mepablate_s42 $GPU 1024 "$A --use_generate False" 42 data_cnt $S1000D_PAIRS
bash experiments/common/run_meponly.sh orig $GPU
bash experiments/common/run_epoch_sweep.sh $GPU 5 10 30 50 100

$PY experiments/common/extract_losses.py results/05_ablation_study meponly_s42 longpretrain5_s42 longpretrain10_s42 longpretrain30_s42 \
  longpretrain50_s42 longpretrain100_s42
