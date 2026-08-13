#!/bin/bash
set -e
# Usage: bash finetune-code.sh <language>
# python finetunes on AdvTest; the other 5 languages finetune on CodeSearch.
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
export FINETUNE_PATH=${DATA_ROOT}/processed/code/${language}/finetune
export finetune_data=${language}.jsonl
export RUN_TAG=${RUN_TAG:-}
export Pretrain_checkpoint=/workspace/SANTA_v2/runs/code/${language}/pretrain${RUN_TAG}/checkpoints/best_dev

export OUPUT=/workspace/SANTA_v2/runs/code/${language}/finetune${RUN_TAG}
export output_dir=checkpoints
export logging_dir=logs

# OUPUT is the path to save checkpoint and log.
# output_dir is checkpoint path.
# logging_dir is logging path.
# train_batch_size 16 for adv and 128 for CodeSearch (README's original note --
# using 128 for both here since that's what the original scripts set as default).

# See pretrain-code.sh for why max_steps is computed explicitly rather than relying
# on num_train_epochs auto-derivation.
# Epoch count (12) and per-language learning rate follow the paper's Appendix A.4
# ("Additional Evaluation Results", the CodeSearch benchmark protocol): lr=2e-5 for
# the 5 CodeSearch languages, lr=1e-5 for python/Adv; epoch=12 for both. The earlier
# version of this script used epoch=10 and a flat 2e-5 for every language, including
# python -- a real, quantifiable gap against the paper's own protocol.
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
     --train_n_passages 1  \
     --learning_rate ${FINETUNE_LR}  \
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
