#!/usr/bin/env python3
"""Task-style search queries for the pooled split (datasets/build_pooled_split.py): for every document, ask Claude
for a few short queries a real user would type to find it ("how do I remove the rear wheel", "torque for crank
bolts"), instead of the document-summary descriptions used so far. The summaries stay as SDA pretraining text; these
queries become the finetuning / dev / test queries, so pretraining and finetuning see different text again (as in
SANTA: docstrings for pretraining, search queries downstream).

Reads data/pretrain_pooled/pooled/sda_pairs.{train,dev,test}.jsonl (same documents and split assignment as every pooled
run) and writes data/queries_pooled_taskq/pooled/task_queries.<split>.jsonl: each input record plus
  queries        [{"query": str, "style": "how_to" | "keyword" | "problem" | "lookup"}, ...]
  doc_kind       procedure | description | specification | catalog | front_matter | other
  generated_by   "llm_taskq"
  _leaked_tokens identifier-like tokens copied verbatim from the document (generate_anon_queries.leak_check)
Whole documents are sent, untruncated (the largest is ~1.4M characters).

Responses are cached in .preprocess_cache/llm_task_query_cache.json (key: prompt version + model + document), so
re-runs only pay for documents not seen yet.

Usage:
  python3 datasets/generate_task_queries.py --pilot 5     # print queries for 5 docs, write nothing but the cache
  python3 datasets/generate_task_queries.py               # full run over all splits (--splits dev test to limit)
"""
import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
from generate_anon_queries import leak_check  # noqa: E402

CACHE_PATH = SCRIPT_DIR.parent / ".preprocess_cache" / "llm_task_query_cache.json"
PROMPT_VERSION = "taskq-v2"  # v2: no queries about the module's place in the publication
LLM_MODEL = "claude-opus-5-5"
SPLITS = ("train", "dev", "test")
STYLES = ["how_to", "keyword", "problem", "lookup"]
DOC_KINDS = ["procedure", "description", "specification", "catalog", "front_matter", "other"]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

SYSTEM = (
    "You write evaluation queries for a search engine over S1000D technical publications (XML data modules). "
    "Given one data module, write the short queries that real users would type when the information they need is "
    "in that module. Users are maintenance technicians, operators, and technical writers who know what they want to "
    "do or find out, but not the document's title, code, or structure.\n\n"
    "Rules for each query:\n"
    "- Short and natural: 3 to 15 words, phrased as a user would type it, not as a description of the document. "
    "Never start with phrases like \"This document\" or \"A data module that\".\n"
    "- Specific to this module: name the particular component, task, or topic so the query points here rather than "
    "to a sibling module (\"remove the rear wheel\", not \"remove a wheel\"; \"how to mark applicability on a "
    "procedural step\", not \"applicability\"). If the module covers a narrow aspect of a broad topic, ask about that "
    "aspect.\n"
    "- Answerable from this module's content, not only from its title.\n"
    "- Ask about the content, not about the module itself or where it sits in the publication: no \"which chapter "
    "covers ...\", \"section overview\", or \"where is ... in the spec\". For an overview or introductory module, ask "
    "about what it explains (\"what does information management cover in a CSDB project\").\n"
    "- No identifiers copied from the document: no data module codes, chapter or section numbers, issue numbers, "
    "dates, part numbers, or the module title verbatim. Generic technical terms and product names are fine.\n"
    "- Vary the style across the queries: how_to (\"how do I ...\"), keyword (a few search terms), problem "
    "(a symptom or need: \"brake lever feels spongy\"), lookup (a specific fact: \"tyre pressure for ...\"). "
    "Use only the styles that fit the module; a front-matter or catalog module may only suit lookup and keyword.\n\n"
    "Also classify the module's doc_kind."
)


def schema(n_queries):
    return {
        "type": "object",
        "properties": {
            "doc_kind": {"type": "string", "enum": DOC_KINDS},
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "style": {"type": "string", "enum": STYLES},
                    },
                    "required": ["query", "style"],
                    "additionalProperties": False,
                },
            },  # structured outputs reject minItems > 1, so the count is checked in QueryWriter.write
        },
        "required": ["doc_kind", "queries"],
        "additionalProperties": False,
    }


class QueryWriter:
    def __init__(self, model, cache_path, n_queries):
        self.model, self.cache_path, self.n_queries = model, cache_path, n_queries
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._lock = threading.Lock()
        self._client = anthropic.Anthropic(max_retries=5)

    def _key(self, xml_text):
        return hashlib.sha256(f"{PROMPT_VERSION}|{self.model}|{self.n_queries}|{xml_text}".encode()).hexdigest()

    def write(self, xml_text, dataset):
        key = self._key(xml_text)
        with self._lock:
            if key in self.cache:
                return self.cache[key], True
        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema(self.n_queries)}},
            system=SYSTEM,
            messages=[{"role": "user", "content": (
                f"Write {self.n_queries} queries for this data module (publication set: {dataset}).\n\n"
                f"<document>\n{xml_text}\n</document>")}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused ({resp.stop_details.category if resp.stop_details else 'no category'}), request {resp._request_id}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"hit max_tokens, request {resp._request_id}")
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
        if len(out["queries"]) < self.n_queries:
            raise RuntimeError(f"got {len(out['queries'])} queries, wanted {self.n_queries}, request {resp._request_id}")
        out["queries"] = out["queries"][: self.n_queries]
        with self._lock:
            self.cache[key] = out
        return out, False

    def save_cache(self):
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", default=str(REPO_ROOT / "data" / "xml" / "pretrain_pooled" / "pooled"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "data" / "xml" / "queries_pooled_taskq" / "pooled"))
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--pilot", type=int, default=0, help="only N docs, round-robin across publication sets; print, write no output")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    writer = QueryWriter(LLM_MODEL, CACHE_PATH, args.n_queries)
    src = Path(args.input_root)
    recs = {s: [json.loads(l) for l in open(src / f"sda_pairs.{s}.jsonl")] for s in args.splits}

    def do_one(r):
        try:
            out, cached = writer.write(r["structured"], r["dataset"])
        except (anthropic.APIError, RuntimeError) as e:
            return r, None, [], False, f"{type(e).__name__}: {e}"
        leaks = sorted({t for q in out["queries"] for t in leak_check(q["query"], r["structured"], corpus_dir=src)})
        return r, out, leaks, cached, None

    if args.pilot:
        by_ds = {}
        for r in (r for s in args.splits for r in recs[s]):
            by_ds.setdefault(r["dataset"], []).append(r)
        rng = random.Random(0)
        for rs in by_ds.values():
            rng.shuffle(rs)
        pool = [rs[i] for i in range(args.pilot) for rs in by_ds.values() if i < len(rs)][: args.pilot]  # round-robin
        print(f"pilot: {len(pool)} docs")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for r, out, leaks, cached, err in ex.map(do_one, pool):
                print(f"\n[{r['dataset']}] {r['source_file']}{' (cached)' if cached else ''}")
                if err:
                    print(f"  ERROR {err}")
                    continue
                print(f"  kind: {out['doc_kind']}   summary: {r['text'][:120]}...")
                for q in out["queries"]:
                    print(f"  - [{q['style']}] {q['query']}")
                print(f"  leaked tokens: {leaks or '-'}")
        writer.save_cache()
        return

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_leak = n_fail = n_total = 0
    for s in args.splits:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(do_one, recs[s]))
        writer.save_cache()
        with open(out_dir / f"task_queries.{s}.jsonl", "w") as f:
            for r, out, leaks, _, err in results:
                n_total += 1
                if err:
                    n_fail += 1
                    print(f"  FAILED {r['source_file']}: {err}")
                    continue
                n_leak += bool(leaks)
                f.write(json.dumps({**r, "queries": out["queries"], "doc_kind": out["doc_kind"],
                                    "generated_by": "llm_taskq", "_leaked_tokens": leaks}, ensure_ascii=False) + "\n")
        print(f"{s}: {len(recs[s])} docs -> {out_dir / f'task_queries.{s}.jsonl'}")
    print(f"\ndone. {n_fail} failed (re-run to retry; cached docs are free), "
          f"{n_leak}/{n_total - n_fail} docs with >=1 verbatim identifier in a query")
    if n_fail:
        sys.exit(1)  # output files are incomplete; check_task_queries.py refuses them too


if __name__ == "__main__":
    main()
