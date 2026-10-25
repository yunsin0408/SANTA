#!/bin/bash
set -e
# Usage: MODEL_NAME=<name from model_configs.json> bash run-xml.sh
# Runs one full model end-to-end: prepare_data -> pretrain -> best-dev-pretrain
# -> finetune -> best-dev-finetune -> index -> evaluate. Assumes Stages 1-3
# (preprocess.py / build_xml_entity.py / build_retrieval_dataset.py) have
# already been run for both the model's pretrain_source and finetune_source
# folders under data/pretrain/<slug>/ and data/retrieval/<slug>/.
#
# To run all models in model_configs.json:
#   for m in $(python3 -c "import json; print('\n'.join(x['name'] for x in json.load(open('processing/XML/model_configs.json'))['models']))"); do
#       MODEL_NAME=$m bash shell/run-xml.sh
#   done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

: "${MODEL_NAME:?set MODEL_NAME to a name in processing/XML/model_configs.json}"
export MODEL_NAME

"${PYTHON}" "${SANTA_DIR}/processing/XML/prepare_data.py" --models "${MODEL_NAME}"
bash "${SCRIPT_DIR}/pretrain-xml.sh"
bash "${SCRIPT_DIR}/dev_xml_pretrain.sh"
bash "${SCRIPT_DIR}/finetune-xml.sh"
bash "${SCRIPT_DIR}/dev_xml_finetune.sh"
bash "${SCRIPT_DIR}/index-xml.sh"
bash "${SCRIPT_DIR}/evaluate_xml.sh"
