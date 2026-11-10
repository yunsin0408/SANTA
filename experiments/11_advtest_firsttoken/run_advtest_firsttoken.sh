#!/bin/bash
# Usage: GPU=<n> SEED=<s> bash experiments/11_advtest_firsttoken/run_advtest_firsttoken.sh
# CodeSearchNet python -> AdvTest with exp 01's model settings (the paper's first-token pooling, no temperature) and
# Poincare similarity instead of the dot product. Exp 09 ran the same track with exp 08's mean-pooling recipe.
# First-token CodeT5 reps have norm ~143 on CodeSearchNet (mean-pooled: ~5-7), so --poincare_scale 0.006 puts them at
# the same hyperbolic radius (~1.75) as exp 08 / 09's poincare_scale 0.25; 0.25 would put every point on the boundary.
# Training follows the paper: batch 128 (one real batch, so 127 in-batch negatives; exp 09 used 16 x grad-accum 8),
# pretrain 10 epochs lr 5e-5 with SDA+MEP, finetune 12 epochs lr 1e-5 on AdvTest train, q/p/l max len 50/256/35.
# The pre-tokenized pretraining code is already cut at 240 tokens, so p_max_len 256 matters for finetuning and eval.
# Optional env: POINCARE_SCALE (0.006), POINCARE_TEMP (1.0), BATCH (128), GRAD_ACCUM (1), GRAD_CKPT=1 (activation
# checkpointing, if batch 128 doesn't fit on a shared GPU).
# Checkpoint selection: best AdvTest valid MRR among the saved checkpoints, for both stages, as in exp 09.
# Stages are skipped when their output already exists, so a rerun resumes after the last finished stage; an
# interrupted training stage resumes from its latest checkpoint-N.
# Launch detached so closing the terminal / Claude Code session doesn't kill it:
#   setsid nohup env GPU=1 bash experiments/11_advtest_firsttoken/run_advtest_firsttoken.sh > runs/xml/advtest_firsttoken_logs/driver.log 2>&1 &
# Results -> results/11_advtest_firsttoken/<tag>/{best_dev_pretrain,best_dev_finetune,eval_test_pretrain,eval_test}.json
set -e
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=$SANTA/.venv/bin/python
cd "$SANTA"
seed=${SEED:-42}; tag=advtest_firsttoken_poincare_s$seed
export CUDA_VISIBLE_DEVICES=${GPU:-1}
# transformers 4.35's resume path torch.load()s our own checkpoint-N/rng_state.pth (numpy RNG state), which torch
# 2.6+'s weights_only default rejects. These files are written by this pipeline, so allow the full load.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CSN=$SANTA/data/processed/code/python
ADV=$SANTA/data/xml/retrieval_advtest/advtest
[[ -f $ADV/qrels.test.tsv ]] || python3 -I $SANTA/processing/XML/datasets/build_advtest_retrieval.py
OUT=$SANTA/runs/xml/advtest/$tag; R=$SANTA/results/11_advtest_firsttoken/$tag; L=$SANTA/runs/xml/advtest_firsttoken_logs
mkdir -p $R $L
SIM_ARGS="--pooling first --similarity poincare --poincare_scale ${POINCARE_SCALE:-0.006} --temperature ${POINCARE_TEMP:-1.0}"
TRAIN_ARGS="--do_train --save_steps 1000 --save_total_limit 3 --per_device_train_batch_size ${BATCH:-128}
  --gradient_accumulation_steps ${GRAD_ACCUM:-1} --bf16 True --train_n_passages 1 --q_max_len 50 --p_max_len 256
  --l_max_len 35 --logging_steps 50 --report_to none --seed $seed $SIM_ARGS"
LEN_ARGS="--per_device_eval_batch_size 64 --q_max_len 50 --p_max_len 256 --topk 100"

best_dev() {  # <checkpoints dir> <results json>
  $PY xml_best_dev/evaluate_xml_finetune.py --model_path $1 --corpus_path $ADV/corpus.dev.jsonl \
    --query_path $ADV/queries.dev.jsonl --qrels_path $ADV/qrels.dev.tsv $LEN_ARGS --results_path $2
}
resume_arg() {  # <checkpoints dir> -> --resume_from_checkpoint <latest checkpoint-N>, if an interrupted run left one
  local last=$(ls -d $1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  # --overwrite_output_dir only skips train_santa.py's non-empty-dir check; it deletes nothing
  [[ -n $last ]] && echo "--overwrite_output_dir --resume_from_checkpoint $last"
}
test_eval() {  # <model> <trec> <results json>
  (cd evaluate_xml && $PY index_xml.py --model_name_or_path $1 --corpus_path $ADV/corpus.test.jsonl \
    --query_path $ADV/queries.test.jsonl --trec_save_path $2 $LEN_ARGS \
    && $PY evaluate_xml.py --trec_save_path $2 --qrels_path $ADV/qrels.test.tsv --results_path $3)
}

P=$OUT/pretrain/checkpoints
if [[ ! -f $P/openmatch_config.json ]]; then
  echo "[$(date)] pretrain $(resume_arg $P)"
  $PY train_santa.py --output_dir $P --model_name_or_path Salesforce/codet5-base --train_path $CSN/pretrain/python.jsonl \
    --learning_rate 5e-5 --num_train_epochs 10 --use_generate True --logging_dir $OUT/pretrain/logs $TRAIN_ARGS \
    $(resume_arg $P) >> $L/$tag.pretrain.log 2>&1
fi
if [[ ! -f $R/best_dev_pretrain.json ]]; then
  echo "[$(date)] best-dev pretrain"; best_dev $P $R/best_dev_pretrain.json > $L/$tag.bestdev_pretrain.log 2>&1
fi
if [[ ! -f $R/eval_test_pretrain.json ]]; then
  echo "[$(date)] test (pretrain only)"
  test_eval $P/best_dev $OUT/retrieve/test_inference_pretrain.trec $R/eval_test_pretrain.json > $L/$tag.test_pretrain.log 2>&1
fi

F=$OUT/finetune/checkpoints
if [[ ! -f $F/openmatch_config.json ]]; then
  echo "[$(date)] finetune $(resume_arg $F)"
  $PY train_santa.py --output_dir $F --model_name_or_path $P/best_dev --overwrite_output_dir \
    --train_path $CSN/finetune/python.jsonl --learning_rate 1e-5 --num_train_epochs 12 --use_generate False \
    --logging_dir $OUT/finetune/logs $TRAIN_ARGS $(resume_arg $F) >> $L/$tag.finetune.log 2>&1
fi
if [[ ! -f $R/best_dev_finetune.json ]]; then
  echo "[$(date)] best-dev finetune"; best_dev $F $R/best_dev_finetune.json > $L/$tag.bestdev_finetune.log 2>&1
fi
echo "[$(date)] test"
test_eval $F/best_dev $OUT/retrieve/test_inference.trec $R/eval_test.json > $L/$tag.test.log 2>&1
echo "DONE"; cat $R/eval_test_pretrain.json $R/eval_test.json
