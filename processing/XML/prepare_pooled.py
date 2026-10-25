#!/usr/bin/env python3
"""Tokenize the pooled split (datasets/build_pooled_split.py output) for both query versions into
data_pooled/ and data_pooled_anonq/, one subfolder per arm:
  full                 finetune on the whole 388-doc train pool (no-pretrain headline number)
  pretrainA_finetuneB  pretrain on halfA, finetune on halfB  (+ finetuneB_only control)
  pretrainB_finetuneA  pretrain on halfB, finetune on halfA  (+ finetuneA_only control)
  pretrain_full        pretrain on the whole 388-doc train pool, no finetuning (pretrain-only arms)
Document cap 1024 tokens, like data_cnt/ and data_anonq1024/. Reuses prepare_data.py's build_* functions."""
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data
from prepare_data import build_pretrain_split, build_finetune_split

prepare_data.SAFETY_MAX_LEN = 1024
REPO_ROOT = Path(__file__).resolve().parents[2]
SLUG = "pooled"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=False)

for suffix in ("", "_anonq"):
    P, R = REPO_ROOT / "data" / "xml" / f"pretrain_pooled{suffix}", REPO_ROOT / "data" / "xml" / f"retrieval_pooled{suffix}"
    out = REPO_ROOT / "data" / "xml" / "tokenized" / f"data_pooled{suffix}"
    arms = {  # name: (pretrain dir or None, retrieval dir)
        "full": (None, R / SLUG),
        "pretrainA_finetuneB": (P / "halfA" / SLUG, R / "halfB" / SLUG),
        "pretrainB_finetuneA": (P / "halfB" / SLUG, R / "halfA" / SLUG),
        "finetuneB_only": (None, R / "halfB" / SLUG),
        "finetuneA_only": (None, R / "halfA" / SLUG),
        "pretrain_full": (P / SLUG, R / SLUG),  # pretrain-only arms (experiments/09_pooled/run_pooled_pretrainonly.sh); finetune files unused
    }
    for name, (pretrain_dir, retrieval_dir) in arms.items():
        out_dir = out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            if pretrain_dir is not None and split != "test":
                build_pretrain_split(tokenizer, split, pretrain_dir, out_dir)
            build_finetune_split(tokenizer, split, retrieval_dir, out_dir)
