#!/bin/bash
# Usage: bash experiments/10_lorentz/run.sh [GPU]
# Rebuilds results/10_lorentz*/ (see experiments/10_lorentz/README.md): the exp 08 pooled split (388 train / 20 dev /
# 100 test) with Lorentz (hyperboloid) similarity instead of Poincare, original and anonymized queries, seeds 42 / 1 / 2.
#   10_lorentz_scale   scale sweep, full arm, anonymized queries, seed 42
#   10_lorentz         half-split arms, full arm, SDA+MEP pretrain-only arm (LORENTZ_SCALE, default 0.5)
# Pretraining arms may need GRAD_CKPT=1 on a shared GPU.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
[[ -d $TOK/data_pooled && -d $TOK/data_pooled_anonq ]] || $PY processing/XML/prepare_pooled.py
export GPU

bash experiments/10_lorentz/run_lorentz_scale.sh
for q in 0 1; do bash experiments/10_lorentz/run_pooled.sh $q; done
bash experiments/10_lorentz/run_pooled_pretrainonly.sh
