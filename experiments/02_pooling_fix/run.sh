#!/bin/bash
# Usage: bash experiments/02_pooling_fix/run.sh [GPU]
# Rebuilds results/02_pooling_fix/ (see experiments/02_pooling_fix/README.md). CodeT5, original XML, batch 16, in-batch negatives.
#   fixA_meancos_p1024_s{42,1,2}   config A, p_max_len 1024, data_cnt,  3 S1000D_spec_sample pairs
#   fixB_meandot_p1024_s42         mean pooling, dot product (no normalize, tau 1)
#   fixC_meancos_p256_s42          config A, p_max_len 256
#   nopre_meancos_p1024_s{42,1,2,3,4}   no pretraining, finetune on S1000D_spec_sample only
#   allA_meancos_p{1024,2048}_s42  config A on the 9 pairs targeting BIKE_7 / FOSSIG / s1kd-tools-doc, data_long
#   nopreAll_meancos_p1024_s{42,1,2}    no pretraining on those three targets

GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=.venv/bin/python
TOK=data/xml/tokenized
S1000D_PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
OTHER_PAIRS="BIKE_7_to_FOSSIG S1000D_spec_sample_to_FOSSIG s1kd-tools-doc_to_FOSSIG BIKE_7_to_s1kd-tools-doc FOSSIG_to_s1kd-tools-doc
  S1000D_spec_sample_to_s1kd-tools-doc FOSSIG_to_BIKE_7 S1000D_spec_sample_to_BIKE_7 s1kd-tools-doc_to_BIKE_7"
[[ -d $TOK/data_cnt ]]  || $PY processing/XML/prepare_data.py --output-dir $TOK/data_cnt --safety-max-len 1024
[[ -d $TOK/data_long ]] || $PY processing/XML/prepare_data.py --output-dir $TOK/data_long --safety-max-len 4096 --models $OTHER_PAIRS

A="--pooling mean --normalize True --temperature 0.05"
export RESULTS_ROOT=$SANTA/results/02_pooling_fix

# S1000D_spec_sample target
for s in 42 1 2; do
  bash experiments/common/run_pairs.sh fixA_meancos_p1024_s$s $GPU 1024 "$A" $s data_cnt $S1000D_PAIRS
done
bash experiments/common/run_pairs.sh fixB_meandot_p1024_s42 $GPU 1024 "--pooling mean" 42 data_cnt $S1000D_PAIRS
bash experiments/common/run_pairs.sh fixC_meancos_p256_s42 $GPU 256 "$A" 42 data_cnt $S1000D_PAIRS
for s in 42 1 2 3 4; do
  bash experiments/common/run_nopretrain.sh nopre_meancos_p1024_s$s $GPU 1024 "$A" $s
done

# the other three targets; p_max_len 2048 needs gradient checkpointing to fit
bash experiments/common/run_pairs.sh allA_meancos_p1024_s42 $GPU 1024 "$A" 42 data_long $OTHER_PAIRS
GRAD_CKPT=1 bash experiments/common/run_pairs.sh allA_meancos_p2048_s42 $GPU 2048 "$A" 42 data_long $OTHER_PAIRS
for s in 42 1 2; do
  for t in FOSSIG:BIKE_7_to_FOSSIG s1kd-tools-doc:BIKE_7_to_s1kd-tools-doc BIKE_7:FOSSIG_to_BIKE_7; do
    TARGET=${t%%:*} DATA_PAIR=${t##*:} DATA_ROOT=data_long \
      bash experiments/common/run_nopretrain.sh nopreAll_meancos_p1024_s$s $GPU 1024 "$A" $s
  done
done
