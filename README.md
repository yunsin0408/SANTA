# Structure Aware Dense Retrieval (SANTA)

Source code and dataset for ACL 2023 Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data.

This fork has two tracks that share the same model code (`model.py`, `train_santa.py`, `trainer.py`):

- **Code and product retrieval** -- SANTA's setup, reproducing CodeSearchNet / AdvTest, ESCI. Instructions
  start at [Environment](#environment) below; [`PIPELINE.md`](PIPELINE.md) is the step-by-step run order and
  [`MODIFICATIONS.md`](MODIFICATIONS.md) lists the changes made to the upstream code.
- **S1000D XML retrieval** -- SANTA applied to S1000D technical documentation. Shared code and data are on
  `master`; each experiment is on its own `exp/NN-*` branch. Setup and layout: [`XML.md`](XML.md).

## XML experiments

Each branch adds `experiments/NN_*/` (a `run.sh` that rebuilds the experiment's results, its launchers and a `README.md` write-up) and
`results/NN_*/` (result JSONs) on top of `master`. Test MRR throughout; GPU training is not bit-exact, so reruns can move single cells by
a few hundredths.

| Branch | Write-up | Experiment | Main finding |
| --- | --- | --- | --- |
| `exp/01-baselines` | [README](https://github.com/yunsin0408/SANTA/blob/exp/01-baselines/experiments/01_baselines/README.md) | T5 vs. CodeT5, `p_max_len` 256 vs. 1024, cross-folder pairs | Everything at chance: first-token pooling gives every document nearly the same vector |
| `exp/02-pooling-fix` | [README](https://github.com/yunsin0408/SANTA/blob/exp/02-pooling-fix/experiments/02_pooling_fix/README.md) | Mean pooling + cosine ("config A") | Lifts S1000D test MRR from chance to ~0.92, but no pretraining (0.94) is just as good |
| `exp/03-query-leakage` | [README](https://github.com/yunsin0408/SANTA/blob/exp/03-query-leakage/experiments/03_query_leakage/README.md) | Queries rewritten without copied identifiers | BM25 still scores 0.98; dense drops to ~0.86, still no pretraining benefit |
| `exp/04-pool-size` | [README](https://github.com/yunsin0408/SANTA/blob/exp/04-pool-size/experiments/04_pool_size/README.md) | Search all 332 S1000D documents instead of 21 | BM25 collapses to 0.13; dense 0.66 / 0.50 (original / anonymized), still no pretraining benefit |
| `exp/05-ablation-study` | [README](https://github.com/yunsin0408/SANTA/blob/exp/05-ablation-study/experiments/05_ablation_study/README.md) | SDA vs. MEP pretraining loss; pretrain epoch sweep 5-100 | Neither loss alone explains it; longer pretraining overfits (best at 10-30 epochs) |
| `exp/06-same-domain` | [README](https://github.com/yunsin0408/SANTA/blob/exp/06-same-domain/experiments/06_same_domain/README.md) | Pretrain and finetune on two halves of S1000D_spec_sample | Pretraining helps the harder half (+0.06 to +0.13) |
| `exp/07-anonymized` | [README](https://github.com/yunsin0408/SANTA/blob/exp/07-anonymized/experiments/07_anonymized/README.md) | Exps 01-06 rerun on the anonymized queries | 01-05 replicate; exp 06's same-domain gain does not |
| `exp/08-pooled` | [README](https://github.com/yunsin0408/SANTA/blob/exp/08-pooled/experiments/08_pooled/README.md) | All four folders pooled into one split (388 / 20 / 100 documents); Poincare similarity | Same-domain pretraining helps a lot (+0.24 to +0.38); Poincare similarity is the best setting |
| `exp/09-advtest` | [README](https://github.com/yunsin0408/SANTA/blob/exp/09-advtest/experiments/09_advtest/README.md) | The exp 08 Poincare recipe on CodeSearchNet -> AdvTest | Doesn't transfer to code: 0.34 vs. 0.43 for the paper's recipe |

To run: `git switch exp/NN-*`, then `bash experiments/NN_*/run.sh [GPU]` 

### XML data

How the
datasets are built: [`processing/XML/datasets/README.md`](processing/XML/datasets/README.md).

| Folder | Contents | Used by |
| --- | --- | --- |
| `pretrain/` | SDA + MEP pretraining pairs per folder (BIKE_7, FOSSIG, S1000D_spec_sample, s1kd-tools-doc), original LLM descriptions | 01-07 (pretraining; S1000D pretrain-dev) |
| `retrieval/` | Retrieval benchmark per folder (corpus / queries / qrels), original queries; S1000D_spec_sample also has the 332-document pool (`*.all332.*`) | 01, 02, 04, 05, 06 (finetuning and test); 07 (untrained zero-shot baseline) |
| `pretrain_anonq/` | Anonymized SDA descriptions for S1000D_spec_sample, the source of `retrieval_anonq/` | 03, 07 |
| `retrieval_anonq/` | S1000D_spec_sample benchmark with anonymized queries, incl. the 332-document pool | 03, 04, 07 |
| `pretrain_samedomain{,_alt}/`, `retrieval_samedomain{,_alt}/` | S1000D_spec_sample's train documents split into halves A / B (original and alternate split) | 06 |
| `pretrain_samedomain{,_alt}_anonq/`, `retrieval_samedomain{,_alt}_anonq/`, `pretrain_samedomain_anonq_src/` | The same halves with anonymized queries (`_src`: intermediate input of `build_samedomain_split_anonq.py`) | 07 |
| `pretrain_pooled{,_anonq}/`, `retrieval_pooled{,_anonq}/` | All four folders pooled, 388 / 20 / 100 documents, halves A / B, original and anonymized queries | 08; the demo (`pretrain_pooled_anonq`, `retrieval_pooled_anonq`) |
| `queries_pooled_taskq/` | Task-style queries for the pooled split (`generate_task_queries.py`) | none yet |
| `retrieval_advtest/` (gitignored) | AdvTest valid / test in the XML retrieval format (`build_advtest_retrieval.py`) | 09 |

Tokenized inputs (`data/xml/tokenized/`, gitignored, built by the `processing/XML/prepare_*.py` scripts):

| Folder | Used by |
| --- | --- |
| `data_t5`, `data_codet5`, `data_cnt` | 01 (`data_cnt` also 02, 04, 05) |
| `data_long` | 02 |
| `data_anonq` | 03, 04 |
| `data_samedomain{,_alt}` | 06 |
| `data_anonq1024`, `data_anonq_t5`, `data_samedomain{,_alt}_anonq` | 07 |
| `data_pooled{,_anonq}` | 08; the demo |

## Environment
(1) Install the following packages using Pip or Conda under this environment:
```
transformers==4.22.2
nltk==3.7
numpy==1.23.2
datasets>=2.4.0
tree-sitter==0.0.5
faiss==1.7.4
scikit-learn>=1.1.2
pandas==1.5.0
pytrec-eval==0.5
tensorboard
```
(2) install openmatch.

If you only need `import openmatch` from another project, install it straight
from GitHub as a library:
```
pip install git+https://github.com/OpenMatch/OpenMatch.git
```

For running the scripts in this repo, use a local development install
instead — the evaluation scripts below `cd` into the OpenMatch checkout
itself, so it needs to exist on disk, not just be importable:
```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```

## Pretrained Checkpoint
#### HuggingFace Link
(1) The checkpoint of the pretrained SANTA model on `Python` data is [here](https://huggingface.co/OpenMatch/santa-code-python-adv).

(2) The checkpoint of the pretrained SANTA model on `ESCI (large)` data is [here](https://huggingface.co/OpenMatch/santa-product-esci).

#### Pretraining Paraments
```
learning_rate=5e-5
num_train_epochs=6
train_n_passages=1 
per_device_train_batch_size=16
```

## Data Download
#### Code
(1) [CodeSearchNet](https://github.com/github/CodeSearchNet)
```
git clone https://github.com/github/CodeSearchNet.git
cd CodeSearchNet/
script/setup
```

(2) [Adv](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv)
```
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
```

(3) [CodeSearch](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)
```
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
```
#### Product
you can download ESCI data from here: [ESCI](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset)

## Process Data
#### Process Code 

(1) Collect pretraining code data.

The pretraining data for the code is sourced from the downloaded CodeSearchNet, which consists of six programming languages. The collection of pretraining data is based on the data statistics provided by [CodeT5](https://arxiv.org/abs/2109.00859) and [CodeRetriever](https://arxiv.org/abs/2201.10866). For the five programming languages other than Python, the `train` from CodeSearchNet will be merged to create the pretraining data `${PRETRAIN_RAW_PATH}/${pretrain_raw_data}`. However, for Python, both the `train` and `test` from CodeSearchNet will be merged as the pretraining data. When selecting a checkpoint for pretraining, the `valid` from CodeSearchNet will be used as the `dev` for all programming languages. 

(2) Process pretraining code data.

process the raw code pretraining data and make it suitable for pretraining inputs `<query, positive, label>`.
Enter the folder `shell` and run the shell script:
```
bash process-pretrain-code.sh
```

(3) Process finetuning code data.

For the `Adv` and `CodeSearch` two code retriever tasks, you can process the raw training file `train ` into input path `${FINETUNE_RAW_PATH}/${finetune_raw_data}`.
```
bash process-finetune-code.sh
```

#### Process Product

(1) Collect pretraining product data.

To use the ESCI (large) data for pretraining, please ensure that the following two files, `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet`, are downloaded and available in the pretraining path `${PRETRAIN_RAW_PATH}`.

(2) Process pretraining produc data.

processing the raw product pretraining data makes it suitable for pretraining inputs `<query, positive, label>` and save dev set into `${PRETRAIN_PATH}/${pretrain_eval_data}` for selecting pretraining checkpoint.
```
bash process-pretrain-product.sh
```

(3) Process finetuning produc data.

For the product search task, we use ESCI (small) data for finetuning. you can process the raw training file which includes  `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet` into input path `${FINETUNE_PATH}/${finetune_data}` for fintuning and eval path `${FINETUNE_PATH}/${finetune_eval_data}` for selecting finetuning checkpoint.

```
bash process-finetune-product.sh
```
> P.S. Due to the absence of product descriptions for some items in the ESCI dataset, when processing the product data for fine-tuning, we concatenate the product title and product description using the following format:  `title: + product title + text: + product description.` If a product doesn't have a description, we concatenate it as follows:  `title: + product title + text:`.

## Pretraining
#### Pretraing for code search

To continue pretraining CodeT5 for different programming languages, utilize the corresponding processed code pretraining data. For instance, if you want to train a Python code retrieval model, only use Python pretraining data for training.
```
bash pretrain-code.sh
```

#### Pretraing for product search

Continuing pretraining T5 using the processed product pretraining data to get a product retrieval model.
```
bash pretrain-product.sh
```

## Finetuning
#### Finetuning for code search

For the pretrained checkpoint, finetuning is performed on downstream tasks related to code retrieval, such as  `Adv ` and `CodeSearch`, useing processed finetuned data. For example, If you have pretrained on Python code, you should also fine-tune on Python code.
```
bash finetune-code.sh
```

#### Finetuning for product search

For the product retrieval task, the pretrained checkpoint is fine-tuned on `ESCI (small)`, useing processed finetuned data. You can find more details about this task [here](https://github.com/amazon-science/esci-data).
```
bash finetune-product.sh
```
> P.S. If you want to use hard negatives, you need to set the parameter `train_n_passages` to n+1, where n is the number of hard negatives.


## Select Checkpoint
The previous pretraining and finetuning tasks saved checkpoints in their respective directories, and this section is about selecting the best checkpoint on their respective dev sets.

#### Best dev checkpoint for code retrieval pretraining
Evaluating all the checkpoints saved after pretraining on the previously processed dev data, and save the best checkpoint.
```
bash dev_code_pretrain.sh
```

#### Best dev checkpoint for code retrieval finetuning
Evaluating all the checkpoints saved after finetuning on the previously processed dev data, and save the best checkpoint.
```
bash dev_code_finetune.sh
```

#### Best dev checkpoint for product retrieval pretraining
Evaluating all the checkpoints saved after pretraining on the previously processed dev data located at `${PRETRAIN_PATH}/${pretrain_eval_data}`, and save the best checkpoint.
```
bash dev_product_pretrain.sh
```

#### Best dev checkpoint for product retrieval finetuning

Evaluating all the checkpoints saved after finetuning on the previously processed dev data located at `${FINETUNE_PATH}/${finetune_eval_data}`, and save the best checkpoint.
```
bash dev_product_finetune.sh
```

## Evaluating

Before evaluating the code and product retrieval tasks, it is necessary to download `OpenMatch` (see the local development install under [Environment](#environment) — a plain `pip install` from GitHub is not enough here, since these scripts `cd` into the checkout).
```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```

#### Evaluating Code Retrieval

For code retrieval tasks, you need to generate test data to conform to OpenMatch's input.
```
bash build-code-test.sh
```

Then, you need to build the Faiss index and obtain the necessary files for inference.

```
bash index-code.sh
```

Evaluate using the obtained inference files.
```
bash evaluate_code.sh
```

#### Evaluating Product Retrieval

For product retrieval task, you need to generate test data to conform to OpenMatch's input.
```
bash build-product-test.sh
```

Encode the query and description of the product as embeddings and save them.
```
bash index-product.sh
```
Calculate scores for the encoded embeddings and sort them to obtain two files `hypothesis.results` and `test.qrels`, which will be used to calculate the NDCG score.
```
bash evaluate_product.sh
```