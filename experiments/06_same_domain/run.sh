#!/bin/bash
# Usage: bash experiments/06_same_domain/run.sh [GPU]
# Rebuilds results/06_same_domain/ (see experiments/06_same_domain/README.md). CodeT5, mean pooling + cosine tau 0.05,
# p_max_len 1024, pretrain and finetune on two halves of S1000D_spec_sample's training documents:
#   samedomain_s{42,1,2}   original 150A/147B split, both directions (ab, ba) and their finetune-only controls
#   samedomain_alt_s42     independently hashed 143A/154B split (data/xml/{pretrain,retrieval}_samedomain_alt/)
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
[[ -d $TOK/data_samedomain ]]     || $PY processing/XML/prepare_samedomain.py
[[ -d $TOK/data_samedomain_alt ]] || SAMEDOMAIN_SUFFIX=_alt $PY processing/XML/prepare_samedomain.py

export RESULTS_ROOT=$SANTA/results/06_same_domain
for s in 42 1 2; do
  for dir in ab ba; do
    bash experiments/common/run_samedomain.sh samedomain_s$s $GPU $s $dir
  done
done
for dir in ab ba; do
  bash experiments/common/run_samedomain.sh samedomain_alt_s42 $GPU 42 $dir _alt
done
