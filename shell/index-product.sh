export RUN_TAG=${RUN_TAG:-}
export TEST_PATH=/workspace/SANTA_v2/data/processed/product/test
export MODEL_PATH=/workspace/SANTA_v2/runs/product/finetune${RUN_TAG}/checkpoints/best_dev

export OUTPUT_PATH=/workspace/SANTA_v2/data/processed/product/test

# TEST_PATH/OUTPUT_PATH are the paths where the processed test data (build-product-test.sh
# output) is located / where the encoded pickles are saved.
# MODEL_PATH is the finetuned checkpoint to encode with.

# shell/index-product.sh didn't exist in the original repo. It reproduces the same
# "encode -> pickle" step evaluate_product.py expects (--pickle_query_data /
# --pickle_doc_data), using the pooling already implemented for dev-checkpoint
# selection in product_best_dev/evaluate_shop_pretrain.py.

cd ../evaluate_product
python encode_product.py \
    --input ${TEST_PATH}/product_test_query.jsonl \
    --data_type query \
    --model_path ${MODEL_PATH} \
    --output ${OUTPUT_PATH}/product_test_query${RUN_TAG}.pkl

python encode_product.py \
    --input ${TEST_PATH}/product_test_doc.jsonl \
    --data_type doc \
    --model_path ${MODEL_PATH} \
    --output ${OUTPUT_PATH}/product_test_doc${RUN_TAG}.pkl
