"""
Computes NDCG@100 from evaluate_product.py's output (hypothesis.results / test.qrels),
matching the metric SANTA's Table 2 reports (this repo's own pipeline stops short of
this step -- it only produces the two trec-format files).

Reports both settings used in the paper:
  Two-C  : binary relevance, only the Exact label counts as relevant.
  Four-C : the graded relevance already baked into test.qrels by
           evaluate_product.py's esci_label2relevance_pos mapping (E=4, S=2, C=3, I=1).
"""
import argparse

import pytrec_eval


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--measure", default="ndcg_cut_100")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    with open(args.results) as f:
        run = pytrec_eval.parse_run(f)

    # Four-C: use the graded relevance as-is.
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut"})
    results = evaluator.evaluate(run)
    scores = [r[args.measure] for r in results.values()]
    four_c = sum(scores) / len(scores)

    # Two-C: only the Exact label (relevance 4, per evaluate_product.py's mapping) is relevant.
    qrels_two_c = {
        qid: {docid: (1 if rel == 4 else 0) for docid, rel in docs.items()}
        for qid, docs in qrels.items()
    }
    evaluator2 = pytrec_eval.RelevanceEvaluator(qrels_two_c, {"ndcg_cut"})
    results2 = evaluator2.evaluate(run)
    scores2 = [r[args.measure] for r in results2.values()]
    two_c = sum(scores2) / len(scores2)

    print(f"queries evaluated: {len(scores)}")
    print(f"Four-C {args.measure}: {four_c:.4f}  ({four_c*100:.2f})")
    print(f"Two-C  {args.measure}: {two_c:.4f}  ({two_c*100:.2f})")


if __name__ == "__main__":
    main()
