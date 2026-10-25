"""Convert CodeXGLUE AdvTest (python) valid/test into t2xml's retrieval format, so the t2xml SANTA tooling
(evaluate_xml/index_xml.py, best_dev/evaluate_xml_finetune.py, evaluate_xml/evaluate_xml.py) can score
CodeSearchNet/AdvTest with any t2xml checkpoint, including Poincare ones.

Text fields follow evaluate_code/build_test_code.py: query = ' '.join(docstring_tokens),
doc = ' '.join(function_tokens). Each row is both a query (q_<i>) and a codebase entry (doc_<i>); a query's
relevant docs are all rows with its url, the same ground truth as SANTA's evaluate_code.py (url equality).

Usage: python processing/XML/datasets/build_advtest_retrieval.py [--adv_dir data/AdvTest] [--out data/xml/retrieval_advtest/advtest]
Writes {corpus,queries}.{dev,test}.jsonl and qrels.{dev,test}.tsv (dev = AdvTest valid.jsonl).
"""
import argparse
import json
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def convert(src, out_dir, split):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    by_url = defaultdict(list)
    for i, r in enumerate(rows):
        by_url[r["url"]].append(f"doc_{i}")
    with open(os.path.join(out_dir, f"corpus.{split}.jsonl"), "w") as fc, \
            open(os.path.join(out_dir, f"queries.{split}.jsonl"), "w") as fq, \
            open(os.path.join(out_dir, f"qrels.{split}.tsv"), "w") as fr:
        fr.write("query-id\tcorpus-id\tscore\n")
        for i, r in enumerate(rows):
            code = " ".join(r["code_tokens"] if "code_tokens" in r else r["function_tokens"])
            fc.write(json.dumps({"docid": f"doc_{i}", "structured": code, "url": r["url"]}) + "\n")
            fq.write(json.dumps({"qid": f"q_{i}", "text": " ".join(r["docstring_tokens"]), "url": r["url"]}) + "\n")
            for d in by_url[r["url"]]:
                fr.write(f"q_{i}\t{d}\t1\n")
    dup = sum(len(v) - 1 for v in by_url.values())
    print(f"{split}: {len(rows)} rows, {len(by_url)} unique urls ({dup} duplicate rows) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adv_dir", default=os.path.join(REPO, "data", "AdvTest"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "xml", "retrieval_advtest", "advtest"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert(os.path.join(args.adv_dir, "valid.jsonl"), args.out, "dev")
    convert(os.path.join(args.adv_dir, "test.jsonl"), args.out, "test")


if __name__ == "__main__":
    main()
