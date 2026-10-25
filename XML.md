# SANTA on S1000D XML (t2xml track)

Adapted from [OpenMatch/SANTA](https://arxiv.org/abs/2305.19912) for t2xml's S1000D data, migrated from the
`t2xml` repo. It shares `model.py` / `train_santa.py` / `trainer.py` with the code and product tracks; the XML runs
use extra flags (`--pooling mean`, `--similarity poincare`, `--poincare_scale`, `--temperature`,
`--use_contrastive`) whose defaults keep the paper's behaviour.

## Layout

| Path | What |
| --- | --- |
| `processing/XML/datasets/` | Raw XML folders (`BIKE_7`, `FOSSIG`, `S1000D spec sample`, `s1kd-tools-doc`, `xsd`) and the Stage 1-3 dataset builders -- see its [README](processing/XML/datasets/README.md) |
| `processing/XML/.preprocess_cache/` | LLM response caches the builders replay |
| `processing/XML/prepare_{data,pooled,samedomain}.py`, `model_configs.json` | Tokenize `data/xml/` into `data/xml/tokenized/data*/` (gitignored) |
| `data/xml/` | Built pretrain / retrieval datasets |
| `shell/{pretrain,finetune,index}-xml.sh`, `shell/dev_xml_{pretrain,finetune}.sh`, `shell/evaluate_xml.sh`, `shell/run-xml.sh` | One stage each; `run-xml.sh` runs one model end to end |
| `xml_best_dev/`, `evaluate_xml/` | Best-checkpoint selection and retrieval eval |
| `experiments/common/` | Launchers shared by several experiments (`run_models`, `run_pairs`, `run_nopretrain`, `run_longpretrain`, `run_samedomain`) |
| `experiments/demo/`, `demo/` | Rebuild the demo model; serve the search demo (`GPU=<n> bash demo/start.sh`) |
| `viz/` | Plotting scripts; output in `results/figures/` |
| `runs/xml/` | Checkpoints, TREC files, logs (gitignored) |

Each experiment (launchers, `results/NN_*/`, write-up) lives on its own `exp/NN-*` branch; see
[`results/README.md`](results/README.md).

## Setup

```bash
.venv/bin/pip install -r requirements-xml.txt
# tokenized inputs, e.g. the 12 cross-folder pairs with a 1024-token document cap (data_cnt):
.venv/bin/python processing/XML/prepare_data.py --output-dir data/xml/tokenized/data_cnt --safety-max-len 1024
```

LLM-backed builders (`preprocess.py`, `generate_*_queries.py`, `check_task_queries.py`) read `ANTHROPIC_API_KEY`
from `.env` at the repo root (gitignored).

Open experiment variables for pretrain/finetune hyperparameters are tracked in
[`processing/XML/datasets/README.md`](processing/XML/datasets/README.md#stage-3--training-hyperparameters-pretrainfinetune-santa).
