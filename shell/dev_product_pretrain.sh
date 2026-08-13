export PRETRAIN_PATH=/workspace/SANTA_v2/data/processed/product/pretrain
export pretrain_eval_data=product_eval.jsonl
export RUN_TAG=${RUN_TAG:-}
export CHECK_POINT=/workspace/SANTA_v2/runs/product/pretrain${RUN_TAG}/checkpoints
# CHECK_POINT is the directory which includes checkpoints

cd ../product_best_dev
python evaluate_shop_pretrain.py \
      --per_gpu_batch_size 64 \
      --test_data ${PRETRAIN_PATH}/${pretrain_eval_data} \
      --block_size 240 \
      --query_block_size 50 \
      --model_path ${CHECK_POINT} \
      --tensorboard_path ${CHECK_POINT}  


