export RUN_TAG=${RUN_TAG:-}
export ESCI_PATH=/workspace/SANTA_v2/data/ESCI
export PICKLE_PATH=/workspace/SANTA_v2/data/processed/product/test
export OUTPUT_PATH=/workspace/SANTA_v2/runs/product/eval${RUN_TAG}

# shell/evaluate_product.sh didn't exist in the original repo. It runs
# evaluate_product.py (which was already present) with the pickles produced by
# shell/index-product.sh, writing hypothesis.results and test.qrels -- per the
# README/script comments, the actual NDCG score is computed from those two files
# with an external trec_eval-compatible tool (e.g. amazon-science/esci-data's
# ranking eval scripts), which is outside this repo/environment.

mkdir -p ${OUTPUT_PATH}

cd ../evaluate_product
python evaluate_product.py \
    --pickle_query_data ${PICKLE_PATH}/product_test_query${RUN_TAG}.pkl \
    --pickle_doc_data ${PICKLE_PATH}/product_test_doc${RUN_TAG}.pkl \
    --input ${ESCI_PATH} \
    --hypothesis_path_file ${OUTPUT_PATH}/hypothesis.csv \
    --output_path ${OUTPUT_PATH}
