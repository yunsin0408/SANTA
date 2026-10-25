#!/usr/bin/env python3
"""Ambiguity check for the task-style queries (datasets/generate_task_queries.py) on the dev and test splits.

A query is ambiguous when another document in the same candidate pool (the split's own documents: 20 for dev, 100
for test) answers it as well as its source document; scoring it with a single relevant document would then mark
correct retrievals wrong. Hard-but-unique queries are kept. No retriever under evaluation (BM25 over the XML, SANTA)
is used to select or drop queries.

Per source document:
  1. Neighbours: up to --n-neighbours other documents of the same split, ranked by TF-IDF cosine between summaries
     (the SDA descriptions) and between each query and the summaries.
  2. Claude judges each query against the source and its neighbours (whole XML; a neighbour over --max-doc-chars is
     shown by its summary, and the prompt says so) and, for an ambiguous query, proposes a rewrite that only the
     source answers.
  3. Rewrites are judged again with the same neighbours. Final status per query:
       kept            unique as generated
       rewritten       ambiguous, rewrite judged unique (original kept in original_query)
       multi_relevant  ambiguous, no unique rewrite, but realistic with several correct documents: relevant_docs
                       lists them all (qrels then need several positives)
       dropped         none of the above
Writes data/queries_pooled_taskq/pooled/task_queries_checked.<split>.jsonl and prints counts per publication set.

Usage:
  python3 datasets/check_task_queries.py --pilot 5          # print judgements for 5 dev docs, write nothing
  python3 datasets/check_task_queries.py                    # dev and test
"""
import argparse
import hashlib
import json
import math
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CACHE_PATH = SCRIPT_DIR.parent / ".preprocess_cache" / "llm_task_query_check_cache.json"
PROMPT_VERSION = "taskq-check-v1"
LLM_MODEL = "claude-opus-5-5"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

SYSTEM = (
    "You audit evaluation queries for a search engine over S1000D technical publications (XML data modules). Each "
    "query was written for one SOURCE module and will be scored as if the source were its only correct result among "
    "a fixed set of candidate modules. You see the source and the candidates most likely to compete with it.\n\n"
    "For each query decide:\n"
    "- verdict: \"unique\" if the source answers the query clearly better than every candidate, \"ambiguous\" if any "
    "candidate answers it as well as or better than the source. A query that is hard, vague in wording, or phrased "
    "differently from the module is still unique if the source is its best answer; judge only whether another "
    "candidate competes.\n"
    "- also_relevant: ids of the candidates that answer it as well as or better than the source (empty if unique).\n"
    "- rewrite: for an ambiguous query, a new query that only the source answers, keeping the user's intent and "
    "style; otherwise an empty string. Same rules as the original queries: 3 to 15 words, phrased as a user would "
    "type it, no identifiers copied from the module (data module codes, chapter or section numbers, issue numbers, "
    "dates, part numbers, the title verbatim). Leave it empty if the source can only be singled out by such an "
    "identifier.\n"
    "- multi_relevant_ok: true if the original query is realistic and the source together with also_relevant are "
    "genuinely all correct answers to it; false if the query is simply too vague to be useful.\n"
    "- reason: one short sentence."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["unique", "ambiguous"]},
                    "also_relevant": {"type": "array", "items": {"type": "string"}},
                    "rewrite": {"type": "string"},
                    "multi_relevant_ok": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "also_relevant", "rewrite", "multi_relevant_ok", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["judgements"],
    "additionalProperties": False,
}

TOKEN = re.compile(r"[a-z][a-z0-9]+")
STOP = set("the and for with this that are from into its their which module data document documents s1000d "
           "specifies describes documentation technical provides defines".split())


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class TfIdf:
    def __init__(self, texts):
        df = Counter(t for x in texts for t in set(tokens(x)))
        self.idf = {t: math.log(len(texts) / c) + 1 for t, c in df.items()}
        self.vecs = [self.vec(x) for x in texts]

    def vec(self, text):
        tf = Counter(tokens(text))
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}

    def ranked(self, v, exclude):
        sims = [(sum(w * d.get(t, 0.0) for t, w in v.items()), i) for i, d in enumerate(self.vecs) if i != exclude]
        return [i for s, i in sorted(sims, reverse=True) if s > 0]


def neighbours(recs, i, tfidf, k):
    """Up to k other documents: alternately the closest summaries to the source's summary and to each query."""
    lists = [tfidf.ranked(tfidf.vecs[i], i)] + [tfidf.ranked(tfidf.vec(q["query"]), i) for q in recs[i]["queries"]]
    out = []
    for rank in range(len(recs)):
        for lst in lists:
            if rank < len(lst) and lst[rank] not in out:
                out.append(lst[rank])
            if len(out) == k:
                return out
    return out


class Judge:
    def __init__(self, model, cache_path, max_doc_chars):
        self.model, self.cache_path, self.max_doc_chars = model, cache_path, max_doc_chars
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._lock = threading.Lock()
        self._client = anthropic.Anthropic(max_retries=5)

    def _doc(self, tag, r):
        x = r["structured"]
        if len(x) <= self.max_doc_chars:
            return f"<{tag} publication=\"{r['dataset']}\">\n{x}\n</{tag}>"
        return (f"<{tag} publication=\"{r['dataset']}\" shown=\"summary only: the full module is {len(x)} characters\">"
                f"\n{r['text']}\n</{tag}>")

    def judge(self, src, cands, queries):
        """cands: list of (id, record). Returns one judgement per query, in order."""
        listed = "\n".join(f"{i}. [{q['style']}] {q['query']}" for i, q in enumerate(queries))
        content = (self._doc("source", src) + "\n\n"
                   + "\n\n".join(self._doc(f"candidate id=\"{cid}\"", r) for cid, r in cands)
                   + f"\n\nQueries written for the source:\n{listed}")
        key = hashlib.sha256(f"{PROMPT_VERSION}|{self.model}|{self.max_doc_chars}|{content}".encode()).hexdigest()
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused ({resp.stop_details.category if resp.stop_details else 'no category'}), request {resp._request_id}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"hit max_tokens, request {resp._request_id}")
        by_index = {j["index"]: j for j in json.loads(next(b.text for b in resp.content if b.type == "text"))["judgements"]}
        if set(by_index) != set(range(len(queries))):
            raise RuntimeError(f"judgements for {sorted(by_index)}, wanted 0..{len(queries) - 1}, request {resp._request_id}")
        out = [by_index[i] for i in range(len(queries))]
        with self._lock:
            self.cache[key] = out
        return out

    def save_cache(self):
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=0))


def check_doc(judge, recs, i, nbr_idx):
    src = recs[i]
    cands = [(f"C{n + 1}", recs[j]) for n, j in enumerate(nbr_idx)]
    cid_to_file = {cid: r["source_file"] for cid, r in cands}
    first = judge.judge(src, cands, src["queries"])
    to_recheck = [n for n, j in enumerate(first) if j["verdict"] == "ambiguous" and j["rewrite"].strip()]
    second = {}
    if to_recheck:
        rq = [{"query": first[n]["rewrite"].strip(), "style": src["queries"][n]["style"]} for n in to_recheck]
        second = dict(zip(to_recheck, judge.judge(src, cands, rq)))
    out = []
    for n, (q, j) in enumerate(zip(src["queries"], first)):
        also = [cid_to_file[c] for c in j["also_relevant"] if c in cid_to_file]
        row = {"query": q["query"], "style": q["style"], "original_query": q["query"], "relevant_docs": [src["source_file"]],
               "judge": {"verdict": j["verdict"], "also_relevant": also, "reason": j["reason"]}}
        if j["verdict"] == "unique":
            row["status"] = "kept"
        elif n in second and second[n]["verdict"] == "unique":
            row.update(status="rewritten", query=j["rewrite"].strip())
            row["judge"]["rewrite_reason"] = second[n]["reason"]
        elif also and j["multi_relevant_ok"]:
            row.update(status="multi_relevant", relevant_docs=[src["source_file"]] + also)
        else:
            row["status"] = "dropped"
        out.append(row)
    return out, [r["source_file"] for _, r in cands]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "xml" / "queries_pooled_taskq" / "pooled"))
    ap.add_argument("--input-root", default=str(REPO_ROOT / "data" / "xml" / "pretrain_pooled" / "pooled"),
                    help="the documents generate_task_queries.py read, to check its output is complete")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--n-neighbours", type=int, default=5)
    ap.add_argument("--max-doc-chars", type=int, default=60000, help="longer neighbours are shown by their summary")
    ap.add_argument("--pilot", type=int, default=0, help="only the first N docs of the first split; print, write nothing")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    judge = Judge(LLM_MODEL, CACHE_PATH, args.max_doc_chars)
    root = Path(args.root)
    for split in args.splits:
        recs = [json.loads(l) for l in open(root / f"task_queries.{split}.jsonl")]
        n_docs = sum(1 for _ in open(Path(args.input_root) / f"sda_pairs.{split}.jsonl"))
        if len(recs) != n_docs:  # a missing document would also be missing from every neighbour list
            raise SystemExit(f"{split}: queries for {len(recs)}/{n_docs} docs; re-run generate_task_queries.py first")
        tfidf = TfIdf([r["text"] for r in recs])
        idx = list(range(len(recs)))[: args.pilot or None]

        def do_one(i):
            try:
                return i, *check_doc(judge, recs, i, neighbours(recs, i, tfidf, args.n_neighbours)), None
            except (anthropic.APIError, RuntimeError) as e:
                return i, None, None, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(do_one, idx))
        judge.save_cache()

        if args.pilot:
            for i, rows, nbrs, err in results:
                print(f"\n[{recs[i]['dataset']}] {recs[i]['source_file']}")
                if err:
                    print(f"  ERROR {err}")
                    continue
                print(f"  neighbours: {', '.join(nbrs)}")
                for row in rows:
                    print(f"  - {row['status']:<14} {row['original_query']}")
                    if row["status"] == "rewritten":
                        print(f"    -> {row['query']}")
                    if row["judge"]["also_relevant"]:
                        print(f"    competes with: {', '.join(row['judge']['also_relevant'])}")
                    print(f"    ({row['judge']['reason']})")
            return

        counts, n_fail = {}, 0
        with open(root / f"task_queries_checked.{split}.jsonl", "w") as f:
            for i, rows, nbrs, err in results:
                if err:
                    n_fail += 1
                    print(f"  FAILED {recs[i]['source_file']}: {err}")
                    continue
                for row in rows:
                    counts.setdefault(recs[i]["dataset"], Counter())[row["status"]] += 1
                f.write(json.dumps({**recs[i], "queries": rows, "neighbours": nbrs}, ensure_ascii=False) + "\n")
        print(f"\n{split}: {len(recs) - n_fail}/{len(recs)} docs -> {root / f'task_queries_checked.{split}.jsonl'}")
        statuses = ["kept", "rewritten", "multi_relevant", "dropped"]
        print("| publication set | " + " | ".join(statuses) + " |")
        print("| --- " * (len(statuses) + 1) + "|")
        for ds, c in sorted(counts.items()):
            print(f"| {ds} | " + " | ".join(str(c[s]) for s in statuses) + " |")
        if n_fail:
            print(f"{n_fail} failed (re-run to retry; judged docs are cached)")


if __name__ == "__main__":
    main()
