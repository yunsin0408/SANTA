#!/bin/bash
# Usage: bash experiments/01_baselines/run.sh [GPU]
# Rebuilds results/01_baselines/ (see experiments/01_baselines/README.md). Seed 42, original XML, first-token
# pooling, dot product, batch 16, in-batch negatives:
#   t5_s42         t5-base,                p_max_len 256,  data_t5,     all 12 pairs
#   codet5_s42     Salesforce/codet5-base, p_max_len 256,  data_codet5, all 12 pairs
#   cnt_p1024_s42  Salesforce/codet5-base, p_max_len 1024, data_cnt,    all 12 pairs
#   cnt_p256_s42   Salesforce/codet5-base, p_max_len 256,  data_cnt,    the 3 S1000D_spec_sample pairs
# The original runs split the 12 pairs across two GPUs; this runs them one after another with the same seeds.
# Retraining is not bit-exact (T5 relative_attention_bias gradients are nondeterministic on GPU), and with
# collapsed embeddings that is enough to move test MRR by ~0.002.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
[[ -d $TOK/data_t5 ]]     || $PY processing/XML/prepare_data.py --model_name_or_path t5-base --output-dir $TOK/data_t5
[[ -d $TOK/data_codet5 ]] || $PY processing/XML/prepare_data.py --output-dir $TOK/data_codet5
[[ -d $TOK/data_cnt ]]    || $PY processing/XML/prepare_data.py --output-dir $TOK/data_cnt --safety-max-len 1024

export RESULTS_ROOT=$SANTA/results/01_baselines
bash experiments/common/run_models.sh t5_s42 t5-base data_t5 $GPU 42
bash experiments/common/run_models.sh codet5_s42 Salesforce/codet5-base data_codet5 $GPU 42
P_MAX_LEN=1024 bash experiments/common/run_models.sh cnt_p1024_s42 Salesforce/codet5-base data_cnt $GPU 42
PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample" \
  bash experiments/common/run_models.sh cnt_p256_s42 Salesforce/codet5-base data_cnt $GPU 42
