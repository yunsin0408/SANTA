#!/bin/bash
# Usage: bash experiments/03_query_leakage/run.sh [GPU]
# Rebuilds results/03_query_leakage/ (see experiments/03_query_leakage/README.md). CodeT5, config A, p_max_len 1024, on the
# anonymized-query benchmark (data/xml/retrieval_anonq/), S1000D_spec_sample target only:
#   anonq_meancos_p1024_s{42,1,2}        pretrained, 3 S1000D_spec_sample pairs
#   anonq_nopre_meancos_p1024_s{42,1,2}  no pretraining
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
S1000D_PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
[[ -d $TOK/data_anonq ]] || $PY processing/XML/prepare_data.py --retrieval-root data/xml/retrieval_anonq --output-dir $TOK/data_anonq --models $S1000D_PAIRS

A="--pooling mean --normalize True --temperature 0.05"
export RESULTS_ROOT=$SANTA/results/03_query_leakage RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_anonq
for s in 42 1 2; do
  bash experiments/common/run_pairs.sh anonq_meancos_p1024_s$s $GPU 1024 "$A" $s data_anonq $S1000D_PAIRS
done
for s in 42 1 2; do
  TARGET=S1000D_spec_sample DATA_PAIR=BIKE_7_to_S1000D_spec_sample DATA_ROOT=data_anonq \
    bash experiments/common/run_nopretrain.sh anonq_nopre_meancos_p1024_s$s $GPU 1024 "$A" $s
done
