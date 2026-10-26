#!/bin/bash
# Usage: bash experiments/04_pool_size/run.sh [GPU]
# Rebuilds results/04_pool_size/ (see experiments/04_pool_size/README.md). CodeT5, config A, p_max_len 1024, seed 42,
# S1000D_spec_sample target, original and anonymized queries:
#   pool332_{orig,anon}_s42         pretrained, 3 S1000D_spec_sample pairs (eval_test.json: the usual 21-document test pool)
#   pool332_{orig,anon}_nopre_s42   no pretraining
#   pool332/{orig,anon}_*_eval_test_all332.json   the same checkpoints searched over all 332 S1000D documents
# The original 332-document scores came from a one-off script; this uses the same recipe as exp 07 (index_xml.py on
# corpus.all332.jsonl, p_max_len 1024, top 100).
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
PY=$SANTA/.venv/bin/python
TOK=data/xml/tokenized
SOURCES="BIKE_7 FOSSIG s1kd-tools-doc"
S1000D_PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
[[ -d $TOK/data_cnt ]]   || $PY processing/XML/prepare_data.py --output-dir $TOK/data_cnt --safety-max-len 1024
[[ -d $TOK/data_anonq ]] || $PY processing/XML/prepare_data.py --retrieval-root data/xml/retrieval_anonq --output-dir $TOK/data_anonq --models $S1000D_PAIRS

A="--pooling mean --normalize True --temperature 0.05"
export RESULTS_ROOT=$SANTA/results/04_pool_size
OUT332=$RESULTS_ROOT/pool332; mkdir -p $OUT332

# eval332 <checkpoint> <retrieval root> <results json>: search all 332 S1000D documents with the test queries
eval332() {
  local trec=$SANTA/runs/xml/retrieve/pool332/$(basename $3 .json).trec
  (cd evaluate_xml && CUDA_VISIBLE_DEVICES=$GPU $PY index_xml.py --model_name_or_path $1 \
      --corpus_path $2/S1000D_spec_sample/corpus.all332.jsonl --query_path $2/S1000D_spec_sample/queries.test.jsonl \
      --trec_save_path $trec --per_device_eval_batch_size 64 --q_max_len 50 --p_max_len 1024 --topk 100 \
    && $PY evaluate_xml.py --trec_save_path $trec --qrels_path $2/S1000D_spec_sample/qrels.test.all332.tsv --results_path $3)
}

for q in orig anon; do
  if [[ $q == orig ]]; then DATA=data_cnt; RET=$SANTA/data/xml/retrieval; else DATA=data_anonq; RET=$SANTA/data/xml/retrieval_anonq; fi
  export RETRIEVAL_ROOT=$RET
  bash experiments/common/run_pairs.sh pool332_${q}_s42 $GPU 1024 "$A" 42 $DATA $S1000D_PAIRS
  TARGET=S1000D_spec_sample DATA_PAIR=BIKE_7_to_S1000D_spec_sample DATA_ROOT=$DATA \
    bash experiments/common/run_nopretrain.sh pool332_${q}_nopre_s42 $GPU 1024 "$A" 42
  for src in $SOURCES; do
    eval332 $SANTA/runs/xml/finetune/pool332_${q}_s42/${src}_to_S1000D_spec_sample/checkpoints/best_dev $RET $OUT332/${q}_${src}_eval_test_all332.json
  done
  eval332 $SANTA/runs/xml/finetune/pool332_${q}_nopre_s42/S1000D_spec_sample/checkpoints/best_dev $RET $OUT332/${q}_nopre_eval_test_all332.json
done
