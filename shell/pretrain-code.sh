#!/bin/bash
set -e
# Usage: bash pretrain-code.sh <language>
export language=$1

export DATA_ROOT=/workspace/SANTA_v2/data
export PRETRAIN_PATH=${DATA_ROOT}/processed/code/${language}/pretrain
export pretrain_data=${language}.jsonl

# RUN_TAG lets separate model variants (e.g. RUN_TAG=-v2) write to their own
# output tree instead of overwriting an existing run's checkpoints.
export RUN_TAG=${RUN_TAG:-}
export OUPUT=/workspace/SANTA_v2/runs/code/${language}/pretrain${RUN_TAG}
export output_dir=checkpoints
export logging_dir=logs

# OUPUT is the path to save checkpoint and log.
# output_dir is checkpoint path.
# logging_dir is logging path.
# train_batch_size 16 for adv and 128 for CodeSearch

# num_train_epochs-based step auto-computation was observed to be unreliable when
# multiple accelerate-backed training processes start concurrently on this box (one
# run silently got max_steps << the correct value despite num_train_epochs=10 being
# honored correctly in isolation). Compute max_steps explicitly from the known
# example count instead of trusting the auto-derivation.
export N=$(wc -l < ${PRETRAIN_PATH}/${pretrain_data})
export MAX_STEPS=$(python3 -c "import math; n=${N}; micro=math.ceil(n/16); spe=micro//8; print(math.ceil(10*spe))")

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path Salesforce/codet5-base  \
     --do_train  \
     --save_steps 1000  \
     --save_total_limit 3  \
     --train_path ${PRETRAIN_PATH}/${pretrain_data} \
     --per_device_train_batch_size 16  \
     --gradient_accumulation_steps 8  \
     --bf16 True  \
     --train_n_passages 1  \
     --learning_rate 5e-5  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs 10  \
     --max_steps ${MAX_STEPS}  \
     --use_generate True \
     --logging_dir ${OUPUT}/${logging_dir}

# The final (most-converged) model lands directly in ${OUPUT}/${output_dir} root,
# not in a checkpoint-N subdir -- but evaluate_code_pretrain.py's dev-checkpoint
# selection only globs checkpoint-* subdirs. Materialize the final save as one so
# it's included in that comparison.
mkdir -p ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}
find ${OUPUT}/${output_dir} -maxdepth 1 -type f -exec cp {} ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}/ \;
