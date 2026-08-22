export TEST_PATH=/workspace/SANTA_v2/data/ESCI

# TEST_PATH is the path where the raw test data is located.

export OUTPUT_PATH=/workspace/SANTA_v2/data/processed/product/test
export output_query=product_test_query.jsonl
export output_doc=product_test_doc.jsonl

# OUTPUT_PATH is the path where the processed test data is saved.
# data_type chooses whether to process the query or code.

mkdir -p ${OUTPUT_PATH}

cd ../evaluate_product
python build_test_product.py \
     --input ${TEST_PATH} \
     --output ${OUTPUT_PATH}/${output_query}  \
     --data_type query  \

python build_test_product.py \
     --input ${TEST_PATH} \
     --output ${OUTPUT_PATH}/${output_doc}  \
     --data_type doc  \

