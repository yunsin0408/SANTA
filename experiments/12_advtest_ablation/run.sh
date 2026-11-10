#!/bin/bash
# Usage: bash experiments/12_advtest_ablation/run.sh <run> [GPU]
#        bash experiments/12_advtest_ablation/run.sh build1024
# Runs (each changes one or two things relative to exp 09 / exp 11; see the table in README.md):
#   ft_temp005       exp 11 with temperature 0.05                              (vs exp 11: temperature)
#   mean_p256_b128   exp 09 recipe, p_max_len 256, batch 128                   (vs ft_temp005: pooling + scale)
#   ft_dot           ft_temp005 with dot-product similarity                    (vs ft_temp005: similarity)
#   ft_temp005_hardneg  ft_temp005's pretraining, finetuned with mined hard negatives (vs ft_temp005: negatives)
#   ft_dot_hardneg      ft_dot's pretraining, finetuned with mined hard negatives     (vs ft_dot: negatives)
#   ft_b16x8       exp 11, batch 16 x grad-accum 8, temperature $FT_TEMP     (vs exp 11 / ft_temp005: batch)
#   mean_p1024_b128  exp 09 recipe, p_max_len 1024, batch 128, 1024-token data (vs mean_p256_b128: length)
#   build1024        CPU: rebuild the python pretrain/finetune data with code cut at 1024 tokens instead of 240/256
# Launch detached:
#   setsid nohup bash experiments/12_advtest_ablation/run.sh ft_temp005 0 > runs/xml/advtest_ablation_logs/ft_temp005.driver.log 2>&1 &
set -e
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN=${1:?run name}; export GPU=${2:-0}
A=$SANTA/experiments/12_advtest_ablation/run_advtest_ablation.sh
D1024=$SANTA/data/processed/code/python_1024
MEAN="POOLING=mean POINCARE_SCALE=0.25 POINCARE_TEMP=0.05"
# The GPUs are shared with other users' jobs, which can push a run out of memory mid-training (exp 11's hard-neg
# finetune died this way at 77%). Every stage is resumable, so on failure wait until the GPU has room and rerun.
A_RETRY=$A
A=$SANTA/experiments/12_advtest_ablation/.retry.sh
cat > $A <<'EOF'
#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for try in 1 2 3 4 5; do
  bash "$A_RETRY" && exit 0
  echo "[$(date)] attempt $try failed; waiting for >= ${NEED_MIB:-30000} MiB free on GPU $GPU"
  until free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits) && (( free >= ${NEED_MIB:-30000} )); do
    sleep 120; done
done
exit 1
EOF
export A_RETRY
case $RUN in
  ft_temp005)      env TAG=$RUN POINCARE_TEMP=0.05 bash $A ;;
  ft_dot)          env TAG=$RUN SIMILARITY=dot POINCARE_TEMP=0.05 bash $A ;;
  # 128 queries + 256 passages need ~47 GB without activation checkpointing (OOM on a 48 GB card)
  ft_temp005_hardneg) env TAG=$RUN POINCARE_TEMP=0.05 PRETRAIN_FROM=ft_temp005 HARDNEG=1 GRAD_CKPT=1 bash $A ;;
  ft_dot_hardneg)  env TAG=$RUN SIMILARITY=dot POINCARE_TEMP=0.05 PRETRAIN_FROM=ft_dot HARDNEG=1 GRAD_CKPT=1 bash $A ;;
  mean_p256_b128)  env TAG=$RUN $MEAN PMAX=256 BATCH=128 GRAD_ACCUM=1 bash $A ;;
  ft_b16x8)        env TAG=$RUN POINCARE_TEMP=${FT_TEMP:?set FT_TEMP from ft_temp005 vs exp 11} BATCH=16 GRAD_ACCUM=8 bash $A ;;
  mean_p1024_b128) env TAG=$RUN $MEAN PMAX=1024 BATCH=128 GRAD_ACCUM=1 GRAD_CKPT=1 CSN=$D1024 bash $A ;;
  build1024)
    PY=$SANTA/.venv/bin/python; SRC=$SANTA/data/processed/code/python
    mkdir -p $D1024/pretrain $D1024/finetune; cd $SANTA/processing/Code
    # same raw inputs and seed as shell/process-{pretrain,finetune}-code.sh python; only the truncation length differs
    [[ -s $D1024/finetune/python.jsonl ]] || { $PY build_code.py --input $SANTA/data/AdvTest/train.jsonl \
      --output $D1024/finetune/python.jsonl.tmp --max_len 1024 \
      && mv $D1024/finetune/python.jsonl.tmp $D1024/finetune/python.jsonl; }
    [[ -s $D1024/pretrain/python.jsonl ]] || { $PY build_code_entity.py --input $SRC/pretrain/python_pretrain_raw.jsonl \
      --tree ./tree-sitter-languages.so --output $D1024/pretrain/python.jsonl.tmp --code_type python --max_len 1024 \
      && mv $D1024/pretrain/python.jsonl.tmp $D1024/pretrain/python.jsonl; }
    wc -l $D1024/*/python.jsonl ;;
  *) echo "unknown run $RUN"; exit 1 ;;
esac
