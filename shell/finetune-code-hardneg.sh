#!/bin/bash
set -e
# Usage: bash finetune-code-hardneg.sh <language>
# Reproduces the "SANTA (Hard Negative)" row of Table 6. Run
# mine-hard-negatives-code.sh <language> first -- this script reads the
# hard-negative-augmented training file it produces.
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
export FINETUNE_PATH=${DATA_ROOT}/processed/code/${language}/finetune
export finetune_data=${language}_hardneg.jsonl
export RUN_TAG=-hardneg
export Pretrain_checkpoint=/workspace/SANTA_v2/runs/code/${language}/pretrain/checkpoints/best_dev

export OUPUT=/workspace/SANTA_v2/runs/code/${language}/finetune${RUN_TAG}
export output_dir=checkpoints
export logging_dir=logs

# Same finetune hyperparameters as finetune-code.sh (Appendix A.4), plus:
#  - train_n_passages=2 (1 positive + 1 hard negative per example, on top of
#    in-batch negatives from the rest of the batch -- trainer.py/model.py already
#    handle this generically, no code changes needed there).
#  - warmup_ratio=0.1, which the paper states explicitly for this recipe but
#    finetune-code.sh's baseline (Inbatch) run doesn't set at all.
export FINETUNE_EPOCHS=12
if [ "${language}" = "python" ]; then
  export FINETUNE_LR=1e-5
else
  export FINETUNE_LR=2e-5
fi
export N=$(wc -l < ${FINETUNE_PATH}/${finetune_data})
export MAX_STEPS=$(python3 -c "import math; n=${N}; micro=math.ceil(n/16); spe=micro//8; print(math.ceil(${FINETUNE_EPOCHS}*spe))")

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path ${Pretrain_checkpoint}  \
     --do_train  \
     --overwrite_output_dir  \
     --save_steps 1000  \
     --save_total_limit 3  \
     --train_path ${FINETUNE_PATH}/${finetune_data} \
     --per_device_train_batch_size 16  \
     --gradient_accumulation_steps 8  \
     --bf16 True  \
     --train_n_passages 2  \
     --learning_rate ${FINETUNE_LR}  \
     --warmup_ratio 0.1  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs ${FINETUNE_EPOCHS}  \
     --max_steps ${MAX_STEPS}  \
     --use_generate False \
     --logging_dir ${OUPUT}/${logging_dir}

# See pretrain-code.sh for why the final save is also materialized as a checkpoint-N dir.
mkdir -p ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}
find ${OUPUT}/${output_dir} -maxdepth 1 -type f -exec cp {} ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}/ \;

# dev_code_finetune.sh already reads RUN_TAG generically -- select this run's best
# checkpoint with: RUN_TAG=-hardneg bash dev_code_finetune.sh ${language}
