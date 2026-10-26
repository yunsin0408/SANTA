#!/bin/bash
# Usage: bash experiments/07_anonymized/run.sh [GPU]
# Rebuilds results/07_anonymized/ (see experiments/07_anonymized/README.md): exps 01-06 on the anonymized queries
# (data/xml/retrieval_anonq/). run_anonq1024.sh queues 0 and 1 were originally run in parallel on GPUs 0 and 1; this runs
# them one after another on one GPU, then the anonymized pretrain epoch sweep (5/10/30/50; 100 is in queue 1) and MEP-only,
# then writes the pretraining runs' pretrain_losses.jsonl.
# The 30-100 epoch and same-domain pretraining runs may need GRAD_CKPT=1 to fit on a shared GPU.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
S1000D_PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
[[ -d $TOK/data_anonq1024 ]] || $PY processing/XML/prepare_data.py --retrieval-root data/xml/retrieval_anonq --output-dir $TOK/data_anonq1024 --safety-max-len 1024 --models $S1000D_PAIRS
[[ -d $TOK/data_anonq_t5 ]]  || $PY processing/XML/prepare_data.py --model_name_or_path t5-base --retrieval-root data/xml/retrieval_anonq --output-dir $TOK/data_anonq_t5 --models $S1000D_PAIRS
[[ -d $TOK/data_samedomain_anonq ]]     || SAMEDOMAIN_SUFFIX=_anonq $PY processing/XML/prepare_samedomain.py
[[ -d $TOK/data_samedomain_alt_anonq ]] || SAMEDOMAIN_SUFFIX=_alt_anonq $PY processing/XML/prepare_samedomain.py

# run_anonq1024.sh defaults to its queue number as the GPU index; GPU= overrides it
for q in 0 1; do
  GPU=$GPU bash experiments/07_anonymized/run_anonq1024.sh $q
done
QUERIES=anon bash experiments/common/run_epoch_sweep.sh $GPU 5 30 50 10
bash experiments/common/run_meponly.sh anon $GPU

$PY experiments/common/extract_losses.py results/07_anonymized anonq1024_mepablate_s42 anonq1024_meponly_s42 \
  anonq1024_longpretrain5_s42 anonq1024_longpretrain10_s42 anonq1024_longpretrain30_s42 anonq1024_longpretrain50_s42 anonq1024_longpretrain100_s42
