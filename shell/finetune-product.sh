export FINETUNE_PATH=/workspace/SANTA_v2/data/processed/product/finetune
export finetune_data=product_train.jsonl
export RUN_TAG=${RUN_TAG:-}
export Pretrain_checkpoint=/workspace/SANTA_v2/runs/product/pretrain${RUN_TAG}/checkpoints/best_dev

export OUPUT=/workspace/SANTA_v2/runs/product/finetune${RUN_TAG}
export output_dir=checkpoints
export logging_dir=logs

# OUPUT is the path to save checkponit and log.
# output_dir is checkpoint path.
# logging_dir is logging path.

# See pretrain-code.sh for why max_steps is computed explicitly rather than relying
# on num_train_epochs auto-derivation.
export N=$(wc -l < ${FINETUNE_PATH}/${finetune_data})
export MAX_STEPS=$(python3 -c "import math; n=${N}; spe=math.ceil(n/16); print(math.ceil(6*spe))")

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path ${Pretrain_checkpoint}  \
     --do_train  \
     --save_steps 1000  \
     --save_total_limit 3  \
     --train_path ${FINETUNE_PATH}/${finetune_data} \
     --per_device_train_batch_size 16  \
     --bf16 True  \
     --train_n_passages 1  \
     --learning_rate 1e-4  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs 6  \
     --max_steps ${MAX_STEPS}  \
     --use_generate False \
     --logging_dir ${OUPUT}/${logging_dir}

# See pretrain-code.sh for why the final save is also materialized as a checkpoint-N dir.
mkdir -p ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}
find ${OUPUT}/${output_dir} -maxdepth 1 -type f -exec cp {} ${OUPUT}/${output_dir}/checkpoint-${MAX_STEPS}/ \;
