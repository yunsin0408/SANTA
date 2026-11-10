#!/bin/bash
# Usage: GPU=<n> SEED=<s> bash experiments/11_advtest_firsttoken/run_advtest_hardneg.sh
# Hard-negative finetuning for exp 11, after run_advtest_firsttoken.sh: starts from the same pretrained best_dev
# checkpoint and finetunes on AdvTest train with one mined hard negative per query (python_hardneg.jsonl,
# --train_n_passages 2), as in this repo's Table 6 hard-negative run. Everything else matches the in-batch finetune
# of run_advtest_firsttoken.sh (first-token pooling, Poincare, batch 128, lr 1e-5 x 12 epochs, q/p/l 50/256/35).
# The three results of exp 11 are then:
#   zero-shot     eval_test_pretrain.json   (pretrained checkpoint, no finetuning)
#   in-batch neg  eval_test.json            (run_advtest_firsttoken.sh)
#   hard neg      eval_test_hardneg.json    (this script)
# Optional env as in run_advtest_firsttoken.sh. Resumable the same way.
set -e
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=$SANTA/.venv/bin/python
cd "$SANTA"
seed=${SEED:-42}; tag=advtest_firsttoken_poincare_s$seed
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CSN=$SANTA/data/processed/code/python
ADV=$SANTA/data/xml/retrieval_advtest/advtest
OUT=$SANTA/runs/xml/advtest/$tag; R=$SANTA/results/11_advtest_firsttoken/$tag; L=$SANTA/runs/xml/advtest_firsttoken_logs
P=$OUT/pretrain/checkpoints
[[ -d $P/best_dev ]] || { echo "no pretrained best_dev at $P; run run_advtest_firsttoken.sh first"; exit 1; }
SIM_ARGS="--pooling first --similarity poincare --poincare_scale ${POINCARE_SCALE:-0.006} --temperature ${POINCARE_TEMP:-1.0}"
TRAIN_ARGS="--do_train --save_steps 1000 --save_total_limit 3 --per_device_train_batch_size ${BATCH:-128}
  --gradient_accumulation_steps ${GRAD_ACCUM:-1} --bf16 True --train_n_passages 2 --q_max_len 50 --p_max_len 256
  --l_max_len 35 --logging_steps 50 --report_to none --seed $seed $SIM_ARGS"
LEN_ARGS="--per_device_eval_batch_size 64 --q_max_len 50 --p_max_len 256 --topk 100"
resume_arg() {
  local last=$(ls -d $1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [[ -n $last ]] && echo "--overwrite_output_dir --resume_from_checkpoint $last"
}

F=$OUT/finetune_hardneg/checkpoints
if [[ ! -f $F/openmatch_config.json ]]; then
  echo "[$(date)] finetune (hard negatives) $(resume_arg $F)"
  $PY train_santa.py --output_dir $F --model_name_or_path $P/best_dev --overwrite_output_dir \
    --train_path $CSN/finetune/python_hardneg.jsonl --learning_rate 1e-5 --num_train_epochs 12 --use_generate False \
    --logging_dir $OUT/finetune_hardneg/logs $TRAIN_ARGS $(resume_arg $F) >> $L/$tag.finetune_hardneg.log 2>&1
fi
if [[ ! -f $R/best_dev_finetune_hardneg.json ]]; then
  echo "[$(date)] best-dev finetune (hard negatives)"
  $PY xml_best_dev/evaluate_xml_finetune.py --model_path $F --corpus_path $ADV/corpus.dev.jsonl \
    --query_path $ADV/queries.dev.jsonl --qrels_path $ADV/qrels.dev.tsv $LEN_ARGS \
    --results_path $R/best_dev_finetune_hardneg.json > $L/$tag.bestdev_finetune_hardneg.log 2>&1
fi
echo "[$(date)] test (hard negatives)"
(cd evaluate_xml && $PY index_xml.py --model_name_or_path $F/best_dev --corpus_path $ADV/corpus.test.jsonl \
  --query_path $ADV/queries.test.jsonl --trec_save_path $OUT/retrieve/test_inference_hardneg.trec $LEN_ARGS \
  && $PY evaluate_xml.py --trec_save_path $OUT/retrieve/test_inference_hardneg.trec --qrels_path $ADV/qrels.test.tsv \
  --results_path $R/eval_test_hardneg.json) > $L/$tag.test_hardneg.log 2>&1
echo "DONE"; cat $R/eval_test_hardneg.json
