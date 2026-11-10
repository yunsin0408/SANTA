#!/bin/bash
# Usage: TAG=<name> GPU=<n> [env below] bash experiments/12_advtest_ablation/run_advtest_ablation.sh
# One-variable-at-a-time ablation between exp 09 (mean pooling, poincare_scale 0.25, temperature 0.05, batch 16 x 8,
# p_max_len 240) and exp 11 (first-token pooling, poincare_scale 0.006, temperature 1.0, batch 128, p_max_len 256).
# Same track as both: CodeSearchNet python pretraining (SDA+MEP, 10 epochs, lr 5e-5) -> AdvTest finetuning
# (12 epochs, lr 1e-5), Poincare similarity, seed 42; best checkpoint per stage by AdvTest valid MRR; test on AdvTest test.
# Env (defaults = exp 11):
#   SIMILARITY (poincare; dot ignores POINCARE_SCALE)
#   POOLING (first) | POINCARE_SCALE (0.006) | POINCARE_TEMP (1.0) | BATCH (128) | GRAD_ACCUM (1)
#   PMAX (256): p_max_len for training, dev and test. The pre-tokenized training data is cut at build time, so
#     PMAX > 256 only matters with a dataset built with a larger --max_len (see run.sh: data/processed/code/python_1024/).
#   CSN (data/processed/code/python): pre-tokenized pretrain/ and finetune/ dir.
#   GRAD_CKPT=1: activation checkpointing (read by model.py), for long passages / big batches.
#   SEED (42)
# Resumable: finished stages are skipped, an interrupted training stage resumes from its latest checkpoint-N.
# Results -> results/12_advtest_ablation/$TAG/{best_dev_pretrain,best_dev_finetune,eval_test_pretrain,eval_test}.json
set -e
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=$SANTA/.venv/bin/python
cd "$SANTA"
: "${TAG:?set TAG}"
seed=${SEED:-42}; PMAX=${PMAX:-256}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# transformers 4.35's resume path torch.load()s our own checkpoint-N/rng_state.pth, which torch 2.6+ rejects by default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CSN=${CSN:-$SANTA/data/processed/code/python}
ADV=$SANTA/data/xml/retrieval_advtest/advtest
[[ -f $ADV/qrels.test.tsv ]] || python3 -I $SANTA/processing/XML/datasets/build_advtest_retrieval.py
OUT=$SANTA/runs/xml/advtest/$TAG; R=$SANTA/results/12_advtest_ablation/$TAG; L=$SANTA/runs/xml/advtest_ablation_logs
mkdir -p $R $L
SIM_ARGS="--pooling ${POOLING:-first} --similarity ${SIMILARITY:-poincare} --poincare_scale ${POINCARE_SCALE:-0.006}
  --temperature ${POINCARE_TEMP:-1.0}"
TRAIN_ARGS="--do_train --save_steps 1000 --save_total_limit 3 --per_device_train_batch_size ${BATCH:-128}
  --gradient_accumulation_steps ${GRAD_ACCUM:-1} --bf16 True --q_max_len 50 --p_max_len $PMAX
  --l_max_len 35 --logging_steps 50 --report_to none --seed $seed $SIM_ARGS"
LEN_ARGS="--per_device_eval_batch_size 64 --q_max_len 50 --p_max_len $PMAX --topk 100"
# HARDNEG=1: finetune with one mined hard negative per query (python_hardneg.jsonl, as in the Table 6 hard-neg run)
if [[ ${HARDNEG:-0} == 1 ]]; then FT_DATA=$CSN/finetune/python_hardneg.jsonl; FT_NP=2
else FT_DATA=$CSN/finetune/python.jsonl; FT_NP=1; fi
{ echo "TAG=$TAG CSN=$CSN GRAD_CKPT=${GRAD_CKPT:-0} PRETRAIN_FROM=${PRETRAIN_FROM:-} FT_DATA=$FT_DATA FT_NP=$FT_NP"
  echo "SIM_ARGS=$SIM_ARGS"; echo "TRAIN_ARGS=$TRAIN_ARGS"; } \
  | tr -s ' \n' ' ' > $R/config.txt; echo >> $R/config.txt

best_dev() {  # <checkpoints dir> <results json>
  $PY xml_best_dev/evaluate_xml_finetune.py --model_path $1 --corpus_path $ADV/corpus.dev.jsonl \
    --query_path $ADV/queries.dev.jsonl --qrels_path $ADV/qrels.dev.tsv $LEN_ARGS --results_path $2
}
resume_arg() {  # <checkpoints dir> -> --resume_from_checkpoint <latest checkpoint-N>, if an interrupted run left one
  local last=$(ls -d $1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [[ -n $last ]] && echo "--overwrite_output_dir --resume_from_checkpoint $last"
}
test_eval() {  # <model> <trec> <results json>
  (cd evaluate_xml && $PY index_xml.py --model_name_or_path $1 --corpus_path $ADV/corpus.test.jsonl \
    --query_path $ADV/queries.test.jsonl --trec_save_path $2 $LEN_ARGS \
    && $PY evaluate_xml.py --trec_save_path $2 --qrels_path $ADV/qrels.test.tsv --results_path $3)
}

if [[ -n ${PRETRAIN_FROM:-} ]]; then
  # reuse another run's pretraining (same settings); only the finetune stage is new
  P=$SANTA/runs/xml/advtest/$PRETRAIN_FROM/pretrain/checkpoints
  [[ -d $P/best_dev ]] || { echo "no pretrained best_dev at $P"; exit 1; }
  for f in best_dev_pretrain eval_test_pretrain; do
    [[ -f $R/$f.json ]] || cp $SANTA/results/12_advtest_ablation/$PRETRAIN_FROM/$f.json $R/$f.json; done
else
  P=$OUT/pretrain/checkpoints
fi
if [[ ! -f $P/openmatch_config.json ]]; then
  echo "[$(date)] pretrain $(resume_arg $P)"
  $PY train_santa.py --output_dir $P --model_name_or_path Salesforce/codet5-base --train_path $CSN/pretrain/python.jsonl \
    --learning_rate 5e-5 --num_train_epochs 10 --use_generate True --logging_dir $OUT/pretrain/logs $TRAIN_ARGS \
    --train_n_passages 1 $(resume_arg $P) >> $L/$TAG.pretrain.log 2>&1
fi
if [[ ! -f $R/best_dev_pretrain.json ]]; then
  echo "[$(date)] best-dev pretrain"; best_dev $P $R/best_dev_pretrain.json > $L/$TAG.bestdev_pretrain.log 2>&1
fi
if [[ ! -f $R/eval_test_pretrain.json ]]; then
  echo "[$(date)] test (pretrain only)"
  test_eval $P/best_dev $OUT/retrieve/test_inference_pretrain.trec $R/eval_test_pretrain.json > $L/$TAG.test_pretrain.log 2>&1
fi

F=$OUT/finetune/checkpoints
if [[ ! -f $F/openmatch_config.json ]]; then
  echo "[$(date)] finetune $(resume_arg $F)"
  $PY train_santa.py --output_dir $F --model_name_or_path $P/best_dev --overwrite_output_dir \
    --train_path $FT_DATA --learning_rate 1e-5 --num_train_epochs 12 --use_generate False \
    --logging_dir $OUT/finetune/logs $TRAIN_ARGS --train_n_passages $FT_NP $(resume_arg $F) >> $L/$TAG.finetune.log 2>&1
fi
if [[ ! -f $R/best_dev_finetune.json ]]; then
  echo "[$(date)] best-dev finetune"; best_dev $F $R/best_dev_finetune.json > $L/$TAG.bestdev_finetune.log 2>&1
fi
if [[ ! -f $R/eval_test.json ]]; then
  echo "[$(date)] test"
  test_eval $F/best_dev $OUT/retrieve/test_inference.trec $R/eval_test.json > $L/$TAG.test.log 2>&1
fi
echo "[$(date)] DONE"; cat $R/eval_test_pretrain.json $R/eval_test.json
