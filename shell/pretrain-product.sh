export PRETRAIN_PATH=/workspace/SANTA_v2/data/processed/product/pretrain
export pretrain_data=product_train.jsonl

# PRETRAIN_PATH is the path where the pretraining data is located.
# pretrain_data is product pretraining dataset.

export RUN_TAG=${RUN_TAG:-}
export OUPUT=/workspace/SANTA_v2/runs/product/pretrain${RUN_TAG}
export output_dir=checkpoints
export logging_dir=logs

# OUPUT is the path to save checkpoint and log.
# output_dir is checkpoint path.
# logging_dir is logging path.

# See pretrain-code.sh for why max_steps is computed explicitly rather than relying
# on num_train_epochs auto-derivation.
export N=$(wc -l < ${PRETRAIN_PATH}/${pretrain_data})
export MAX_STEPS=$(python3 -c "import math; n=${N}; spe=math.ceil(n/16); print(math.ceil(6*spe))")

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path t5-base  \
     --do_train  \
     --save_steps 1000  \
     --save_total_limit 3  \
     --train_path ${PRETRAIN_PATH}/${pretrain_data} \
     --per_device_train_batch_size 16  \
     --bf16 True  \
     --train_n_passages 1  \
     --learning_rate 1e-4  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs 6  \
     --max_steps ${MAX_STEPS}  \
     --use_generate True \
     --logging_dir ${OUPUT}/${logging_dir}

# See pretrain-code.sh for why the final save is also materialized as a checkpoint-N dir.
mkdir -p ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}
find ${OUPUT}/${output_dir} -maxdepth 1 -type f -exec cp {} ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}/ \;
