#!/usr/bin/env python3
"""Pooled split: all four source folders merged into one corpus, re-split into train/dev/test.

Replaces the per-folder 90/3/7 split (test = 21/6/3/2 docs) with one pooled split whose test set has 100
documents. Test (100) and dev (20) are stratified by source folder (largest-remainder allocation of each
folder's share of the 508 documents); within a folder, documents are ranked by md5("pooled-" + source_file)
and the first ones go to test, the next to dev, the rest to train. The original per-folder split assignment
is ignored.

The train pool is then halved with build_samedomain_split.py's unsalted hash (halfA/halfB) for the
same-domain pretrain -> finetune comparison; dev/test are shared by every arm.

Two query versions, same documents and same split:
  original    data/pretrain/<folder>/          -> data/pretrain_pooled/, data/retrieval_pooled/
  anonymized  data/pretrain_anonq/<folder>/    -> data/pretrain_pooled_anonq/, data/retrieval_pooled_anonq/
              (SDA from data/pretrain_anonq/, MEP from data/pretrain/ unchanged -- MEP never uses the query)

Layout under each output root: pooled/ (full split) and halfA/pooled/, halfB/pooled/ (train halves).

Usage: python datasets/build_pooled_split.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import build_samedomain_split as samedomain
from build_retrieval_dataset import build_split

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLDERS = ["S1000D_spec_sample", "BIKE_7", "s1kd-tools-doc", "FOSSIG"]
SLUG = "pooled"
SPLITS = ("train", "dev", "test")


def load_all(root: Path, folder: str, kind: str):
    rows = []
    for split in SPLITS:
        rows += [json.loads(l) for l in open(root / folder / f"{kind}_pairs.{split}.jsonl")]
    return rows


def allocate(sizes: dict, total: int) -> dict:
    """Largest-remainder allocation of `total` slots proportional to sizes."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def assign_splits(n_test: int, n_dev: int) -> dict:
    """source_file -> split. source_file is unique across all four folders."""
    files = {f: sorted({r["source_file"] for r in load_all(REPO_ROOT / "data" / "xml" / "pretrain", f, "sda")},
                       key=lambda s: hashlib.md5(("pooled-" + s).encode()).hexdigest())
             for f in FOLDERS}
    sizes = {f: len(v) for f, v in files.items()}
    test_n, dev_n = allocate(sizes, n_test), allocate(sizes, n_dev)
    split_of = {}
    for f, ranked in files.items():
        for i, s in enumerate(ranked):
            split_of[s] = "test" if i < test_n[f] else "dev" if i < test_n[f] + dev_n[f] else "train"
    print("folder              total  train  dev  test")
    for f in FOLDERS:
        print(f"{f:20s}{sizes[f]:5d}  {sizes[f] - test_n[f] - dev_n[f]:5d}  {dev_n[f]:3d}  {test_n[f]:4d}")
    return split_of


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(sda_root: Path, split_of: dict, pretrain_out: Path, retrieval_out: Path):
    sda = [r for f in FOLDERS for r in load_all(sda_root, f, "sda")]
    mep = [r for f in FOLDERS for r in load_all(REPO_ROOT / "data" / "xml" / "pretrain", f, "mep")]
    full = pretrain_out / SLUG
    for split in SPLITS:
        write_jsonl(full / f"sda_pairs.{split}.jsonl", [r for r in sda if split_of[r["source_file"]] == split])
        write_jsonl(full / f"mep_pairs.{split}.jsonl", [r for r in mep if split_of[r["source_file"]] == split])
        stats = {}
        build_split(full / f"sda_pairs.{split}.jsonl", retrieval_out / SLUG / f"corpus.{split}.jsonl",
                    retrieval_out / SLUG / f"queries.{split}.jsonl", retrieval_out / SLUG / f"qrels.{split}.tsv", stats)
        print(f"  {split}: {stats['corpus_size']} docs, {stats['queries_out']} queries "
              f"({stats['ambiguous_texts_dropped']} ambiguous dropped)")
    samedomain.FOLDER = SLUG
    samedomain.split_pretrain(full, pretrain_out)
    samedomain.split_retrieval(retrieval_out / SLUG, retrieval_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-dev", type=int, default=20)
    args = ap.parse_args()
    split_of = assign_splits(args.n_test, args.n_dev)
    print("== original queries")
    build(REPO_ROOT / "data" / "xml" / "pretrain", split_of, REPO_ROOT / "data" / "xml" / "pretrain_pooled", REPO_ROOT / "data" / "xml" / "retrieval_pooled")
    print("== anonymized queries")
    build(REPO_ROOT / "data" / "xml" / "pretrain_anonq", split_of, REPO_ROOT / "data" / "xml" / "pretrain_pooled_anonq", REPO_ROOT / "data" / "xml" / "retrieval_pooled_anonq")


if __name__ == "__main__":
    main()
