#!/bin/bash
# Usage: bash experiments/07_anonymized/run_anonq1024.sh <queue: 0|1>
# Anonymized-query versions of every S1000D_spec_sample experiment, on data_anonq1024/ (codet5 tokenizer,
# document cap 1024 -- the same cap as data_cnt/, which all original-query runs used; data_anonq/ had 512).
# T5 uses data_anonq_t5/ (t5 tokenizer, cap 512, like data_t5/); same-domain uses data_samedomain{,_alt}_anonq/
# (cap 512, like data_samedomain{,_alt}/). All evaluation reads data/retrieval_anonq/. Results -> results/07_anonymized/.
# Queue 0 and queue 1 are meant to run in parallel on GPU 0 and GPU 1 (GPU=<n> overrides the GPU index).
Q=$1; G=${GPU:-$Q}  # G: GPU index, defaults to the queue number
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SANTA"
export RESULTS_ROOT=$SANTA/results/07_anonymized RETRIEVAL_ROOT=$SANTA/data/xml/retrieval_anonq
LOG=runs/xml/anonq1024_logs; mkdir -p $LOG
PAIRS="BIKE_7_to_S1000D_spec_sample FOSSIG_to_S1000D_spec_sample s1kd-tools-doc_to_S1000D_spec_sample"
COS="--pooling mean --normalize True --temperature 0.05"
say() { echo "[$(date +%T)] Q$Q $*" >> $LOG/progress.log; }

# Remove one finished experiment's checkpoints (results, TREC runs and embeddings are kept).
cleanup() {
  local tag=$1
  [[ -n "$tag" ]] || return
  rm -rf "$SANTA/runs/xml/pretrain/$tag" "$SANTA/runs/xml/finetune/$tag"
  say "cleaned checkpoints of $tag"
}

# Evaluate finetuned checkpoints of <tag> against all 332 S1000D documents (same queries).
eval332() {
  local tag=$1; shift
  for name in "$@"; do
    local out=$RESULTS_ROOT/$tag/$name trec=$SANTA/runs/xml/retrieve/$tag/$name/test_inference_all332.trec
    (cd evaluate_xml && CUDA_VISIBLE_DEVICES=$G ../.venv/bin/python index_xml.py \
        --model_name_or_path $SANTA/runs/xml/finetune/$tag/$name/checkpoints/best_dev \
        --corpus_path $RETRIEVAL_ROOT/S1000D_spec_sample/corpus.all332.jsonl \
        --query_path $RETRIEVAL_ROOT/S1000D_spec_sample/queries.test.jsonl \
        --trec_save_path $trec --per_device_eval_batch_size 64 --q_max_len 50 --p_max_len 1024 --topk 100 \
      && ../.venv/bin/python evaluate_xml.py --trec_save_path $trec \
        --qrels_path $RETRIEVAL_ROOT/S1000D_spec_sample/qrels.test.all332.tsv \
        --results_path $out/eval_test_all332.json) > $LOG/${tag}_${name}.eval332.log 2>&1 \
      && say "eval332 $tag/$name" || say "FAIL eval332 $tag/$name"
  done
}

# Untrained models (first-token pooling), original and anonymized queries.
zeroshot() {
  for m in Salesforce/codet5-base t5-base; do
    for q in retrieval retrieval_anonq; do
      local tag=zeroshot_untrained_${q}; local n=$(basename $m); local out=$RESULTS_ROOT/$tag/$n
      mkdir -p $out
      CUDA_VISIBLE_DEVICES=$G MODEL_NAME=x_to_S1000D_spec_sample RUN_TAG=/$tag/$n MODEL_PATH=$m P_MAX_LEN=256 \
        RETRIEVAL_ROOT=$SANTA/data/xml/$q EMB_SAVE_PATH=$out/embeddings.npz bash shell/index-xml.sh > $LOG/${tag}_$n.index.log 2>&1 \
      && MODEL_NAME=x_to_S1000D_spec_sample RUN_TAG=/$tag/$n RETRIEVAL_ROOT=$SANTA/data/xml/$q RESULTS_PATH=$out/eval_test.json \
        bash shell/evaluate_xml.sh > $LOG/${tag}_$n.eval.log 2>&1 && say "zeroshot $m $q" || say "FAIL zeroshot $m $q"
    done
  done
}

# First-token baselines (run_models.sh = original ga1 config: dot product, first-token pooling).
first_token() {  # <tag> <model> <data_root> <p_max_len>
  say "START $1"
  PAIRS="$PAIRS" P_MAX_LEN=$4 bash experiments/common/run_models.sh $1 $2 $3 $G 42
  cleanup $1
}

pairs() {  # <tag> <p_max_len> <extra> <seed> [332]
  say "START $1"
  bash experiments/common/run_pairs.sh $1 $G $2 "$3" $4 data_anonq1024 $PAIRS
  [[ "$5" == 332 ]] && eval332 $1 $PAIRS
  cleanup $1
}

nopre() {  # <tag> <seed> [332]
  say "START $1"
  DATA_ROOT=data_anonq1024 bash experiments/common/run_nopretrain.sh $1 $G 1024 "$COS" $2
  [[ "$3" == 332 ]] && eval332 $1 S1000D_spec_sample
  cleanup $1
}

samedomain() {  # <tag> <seed> <data suffix>
  say "START $1"
  PRETRAIN_ROOT=$SANTA/data/xml/pretrain_anonq bash experiments/common/run_samedomain.sh $1 $G $2 ab $3
  PRETRAIN_ROOT=$SANTA/data/xml/pretrain_anonq bash experiments/common/run_samedomain.sh $1 $G $2 ba $3
  cleanup $1
}

if [[ "$Q" == 0 ]]; then
  zeroshot
  first_token anonq1024_cnt_p1024_s42 Salesforce/codet5-base data_anonq1024 1024
  first_token anonq1024_codet5_s42 Salesforce/codet5-base data_anonq1024 256
  first_token anonq1024_t5_s42 t5-base data_anonq_t5 256
  pairs anonq1024_meancos_p1024_s42 1024 "$COS" 42 332
  pairs anonq1024_meancos_p1024_s1 1024 "$COS" 1
  pairs anonq1024_meancos_p1024_s2 1024 "$COS" 2
  nopre anonq1024_nopre_meancos_p1024_s42 42 332
  nopre anonq1024_nopre_meancos_p1024_s1 1
  samedomain anonq1024_samedomain_s2 2 _anonq
else
  nopre anonq1024_nopre_meancos_p1024_s2 2
  nopre anonq1024_nopre_meancos_p1024_s3 3
  nopre anonq1024_nopre_meancos_p1024_s4 4
  pairs anonq1024_meandot_p1024_s42 1024 "--pooling mean" 42
  pairs anonq1024_meancos_p256_s42 256 "$COS" 42
  pairs anonq1024_mepablate_s42 1024 "$COS --use_generate False" 42
  say "START anonq1024_longpretrain100_s42"
  DATA_ROOT=data_anonq1024 bash experiments/common/run_longpretrain.sh anonq1024_longpretrain100_s42 $G 100
  cleanup anonq1024_longpretrain100_s42
  samedomain anonq1024_samedomain_s42 42 _anonq
  samedomain anonq1024_samedomain_s1 1 _anonq
  samedomain anonq1024_samedomain_alt_s42 42 _alt_anonq
fi
say "QUEUE-DONE"
