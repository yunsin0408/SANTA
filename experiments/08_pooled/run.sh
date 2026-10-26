#!/bin/bash
# Usage: bash experiments/08_pooled/run.sh [GPU]
# Rebuilds results/08_pooled*/ (see experiments/08_pooled/README.md): the pooled split (all four folders merged,
# 388 train / 20 dev / 100 test), original and anonymized queries, seeds 42 / 1 / 2.
#   08_pooled                 cosine; half-split arms, full arm, pretrain-only arms
#   08_pooled_poincare        the same with Poincare similarity (tau 0.05, picked by 08_pooled_poincare_tau)
#   08_pooled_poincare_tau    Poincare temperature sweep, full arm, anonymized queries, seed 42
#   08_pooled_q160            query budget 160 tokens instead of 50
#   08_pooled_ss25            checkpoint every 25 steps, so best-dev selection picks among checkpoints
#   08_pooled_ss25_poincare   both of the above
# The original runs used two GPUs; this runs everything on one. Pretraining arms may need GRAD_CKPT=1 on a shared GPU.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
[[ -d $TOK/data_pooled && -d $TOK/data_pooled_anonq ]] || $PY processing/XML/prepare_pooled.py
export GPU

TAUS="0.05 0.1 0.2 0.02 0.01" bash experiments/08_pooled/run_poincare_tau.sh
for q in 0 1; do bash experiments/08_pooled/run_pooled.sh $q; done
SIMILARITY=cos bash experiments/08_pooled/run_pooled_pretrainonly.sh
for q in 0 1; do SIMILARITY=poincare bash experiments/08_pooled/run_pooled.sh $q; done
SIMILARITY=poincare bash experiments/08_pooled/run_pooled_pretrainonly.sh
for q in 0 1; do Q_MAX_LEN=160 bash experiments/08_pooled/run_pooled.sh $q; done
for q in 0 1; do SAVE_STEPS=25 bash experiments/08_pooled/run_pooled.sh $q; done
for q in 0 1; do SAVE_STEPS=25 SIMILARITY=poincare bash experiments/08_pooled/run_pooled.sh $q; done
