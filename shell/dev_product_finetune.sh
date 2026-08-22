export FINETUNE_PATH=/workspace/SANTA_v2/data/processed/product/finetune
export finetune_eval_data=product_eval.jsonl
export RUN_TAG=${RUN_TAG:-}
export CHECK_POINT=/workspace/SANTA_v2/runs/product/finetune${RUN_TAG}/checkpoints
# CHECK_POINT is the directory which includes checkpoints
# batch_size is must 1
cd ../product_best_dev
python evaluate_shop_finetune.py \
      --per_gpu_batch_size 1 \
      --test_data ${FINETUNE_PATH}/${finetune_eval_data} \
      --shop_product 256 \
      --shop_query 20 \
      --model_path ${CHECK_POINT} \
      --tensorboard_path ${CHECK_POINT}  


