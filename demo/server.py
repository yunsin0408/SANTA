"""Search demo for outside testers: type a description, get the S1000D documents ranked by the model.

Serves one checkpoint over the pooled data's documents. Each result says which split it is from, since the model
was pretrained on the train documents and their descriptions. Example queries come from the test split only. Documents are encoded
once at startup; each search encodes the query and ranks every document with model.score.

Endpoints (searches are rate-limited per client IP, see --rate):
  GET  /                  the search page (index.html)
  GET  /api/examples      example queries: selected test-split cases, mostly rank 1, see EXAMPLE_PLAN
  POST /api/search        {"query": str, "k": int} -> ranked documents
  GET  /api/doc/{docid}   one document's XML (docid = "<split>-<docid in that split>", e.g. test-doc_0)
  GET  /api/space         3-D Poincare-ball layout of all documents (Poincare models only), for the interactive view,
                          plus the example queries placed in it (fixed example view)
  GET  /static/...        plotly.js, served locally (the page must not depend on an outside CDN)

3-D view: the documents' true Poincare distances (768-d) are approximated in a 3-D ball by hyperbolic MDS. Each search also places the query in
that ball (documents held fixed, query position fitted to its true distances, nearest documents weighted most).

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python demo/server.py [--port 8600]   (or demo/start.sh)
Queries are appended to demo/logs/queries.jsonl (time, query, top-5 docids) unless --no_log is given.
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

SANTA = Path(__file__).resolve().parent.parent
REPO = SANTA
sys.path.insert(0, str(SANTA))
sys.path.insert(0, str(SANTA / "evaluate_xml"))
from model import SModel, poincare_scores
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments
from index_xml import encode_all

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(SANTA / "runs/xml/pretrain/demo/anonq_poincare_sdamep_s42/checkpoints"))
ap.add_argument("--retrieval_dir", default=str(REPO / "data" / "xml" / "retrieval_pooled_anonq/pooled"))
ap.add_argument("--splits", default="train,dev,test", help="which corpus.<split>.jsonl files to search")
ap.add_argument("--q_max_len", type=int, default=50, help="query token budget; 50 = what the model was trained with")
ap.add_argument("--p_max_len", type=int, default=1024)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8600)
ap.add_argument("--rate", type=int, default=30, help="max searches per client IP per minute")
ap.add_argument("--no_log", action="store_true")
args = ap.parse_args()

# example queries: showcase cases from the test split, chosen by the rank of their own
# document among all searched documents
EXAMPLE_PLAN = {"S1000D_spec_sample": [(1, 1)] * 4 + [(2, 5), (6, 10)],
                "BIKE_7": [(1, 1)] * 2 + [(2, 5)],
                "s1kd-tools-doc": [(1, 1), (2, 5)],
                "FOSSIG": [(6, 10)]}
SOURCE_NAMES = {"S1000D_spec_sample": "S1000D specification sample", "BIKE_7": "Bike sample (S1000D Issue 7)",
                "s1kd-tools-doc": "s1kd-tools documentation", "FOSSIG": "FOSSIG"}


def title_of(xml, fallback):
    def tag(name):
        m = re.search(rf"<{name}\b[^>]*>(.*?)</{name}>", xml, re.S)
        return " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split()) if m else ""
    parts = [p for p in (tag("techName"), tag("infoName")) if p] or [p for p in (tag("pmTitle"), tag("icnTitle")) if p]
    return " — ".join(parts) if parts else fallback


# ---- data -------------------------------------------------------------------------------------------------------
R = Path(args.retrieval_dir)
DOCS = []
for split in args.splits.split(","):  # docids restart at doc_0 in every split file, so prefix them with the split
    for l in open(R / f"corpus.{split}.jsonl"):
        d = json.loads(l)
        d.update(docid=f"{split}-{d['docid']}", split=split, title=title_of(d["structured"], d["source_file"]))
        DOCS.append(d)
DOC_INDEX = {d["docid"]: i for i, d in enumerate(DOCS)}
QRELS = {l.split()[0]: "test-" + l.split()[1] for l in list(open(R / "qrels.test.tsv"))[1:]}
QUERIES = [json.loads(l) for l in open(R / "queries.test.jsonl")]

# ---- model ------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
model = SModel.build(model_args=ModelArguments(model_name_or_path=args.model), santa_args=SantaArguments(use_generate=False))
model.to(device).eval()
print(f"similarity={model.similarity} pooling={model.pooling}; encoding {len(DOCS)} documents ...", flush=True)
DOC_EMBS = encode_all(model, tokenizer, [d["structured"] for d in DOCS], args.p_max_len, 8, device, is_query=False).float()
GPU_LOCK = threading.Lock()


def select_examples():
    with torch.no_grad():
        qe = encode_all(model, tokenizer, [q["text"] for q in QUERIES], args.q_max_len, 16, device, is_query=True).float()
        sc = model.score(qe.to(device), DOC_EMBS.to(device))
    cands = defaultdict(list)
    for q, row in zip(QUERIES, sc):
        gi = DOC_INDEX[QRELS[q["qid"]]]
        rank = int((row > row[gi]).sum()) + 1
        others = row.clone(); others[gi] = -float("inf")
        cands[q["dataset"]].append((rank, -float(row[gi] - others.max()), q, gi))  # margin over the best other document
    picked = []
    for src, plan in EXAMPLE_PLAN.items():
        left = sorted(cands[src], key=lambda c: (c[0], c[1]))  # best rank first, then widest margin
        for lo, hi in plan:
            fit = [c for c in left if lo <= c[0] <= hi] or left[:1]  # no case in that band -> next best of this source
            if fit:
                picked.append((src, fit[0]))
                left.remove(fit[0])
    return [{"query": q["text"], "source": SOURCE_NAMES[src], "expected_docid": DOCS[gi]["docid"],
             "expected_title": DOCS[gi]["title"], "rank": rank} for src, (rank, _, q, gi) in picked]


EXAMPLES = select_examples()
print("examples:", [x["rank"] for x in EXAMPLES], flush=True)


# ---- 3-D layout (Poincare models) ---------------------------------------------------------------------------------
def to_ball3(v):
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    return torch.tanh(n) * v / n


def hyperbolic_mds(D, init, steps=3000):
    """3-D points in the Poincare ball whose pairwise distances match D (n, n). Returns (points (n, 3), stress-1)."""
    iu = torch.triu_indices(len(D), len(D), 1, device=D.device)
    target = D[iu[0], iu[1]]
    v = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.02)
    for _ in range(steps):
        y = to_ball3(v)
        d = -poincare_scores(y, y)[iu[0], iu[1]]
        loss = ((d - target) ** 2).sum() / (target ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return to_ball3(v).detach(), float(loss.detach().sqrt())


SPACE = None
if model.similarity == "poincare":
    with torch.no_grad():
        BALL = model.to_ball(DOC_EMBS.to(device))
        DD = -poincare_scores(BALL, BALL)
    key = hashlib.sha1(DD.cpu().numpy().round(4).tobytes()).hexdigest()[:16]
    cache = SANTA / "demo" / "cache" / f"layout3d_{key}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        LAYOUT, STRESS = torch.tensor(c["points"], device=device), c["stress"]
    else:
        print("fitting 3-D layout (first start only) ...", flush=True)
        X = BALL.double()
        nrm = X.norm(dim=1, keepdim=True)
        V = torch.atanh(nrm) * X / nrm                   # tangent vectors at the origin
        Vc = V - V.mean(0)
        U = Vc @ torch.linalg.svd(Vc, full_matrices=False)[2][:3].T
        U = U / U.norm(dim=1, keepdim=True) * V.norm(dim=1, keepdim=True)  # keep each document's true depth
        with torch.enable_grad():
            LAYOUT, STRESS = hyperbolic_mds(DD.double(), U.float().double())
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({"points": LAYOUT.cpu().tolist(), "stress": STRESS}))
    LAYOUT = LAYOUT.float()
    SPACE = {"stress": round(STRESS, 3), "docs": [
        {"docid": d["docid"], "p": [round(x, 4) for x in LAYOUT[i].tolist()], "title": d["title"],
         "source": SOURCE_NAMES.get(d["dataset"], d["dataset"]), "seen_in_training": d["split"] == "train"}
        for i, d in enumerate(DOCS)]}
    print(f"3-D layout ready (stress {STRESS:.3f})", flush=True)


def place_query(dq, steps=300):
    """Position of a new point in the 3-D layout, given its true distances dq (n,) to every document."""
    w = 1 / dq.clamp(min=0.05) ** 2                       # the nearest documents count most
    w = w / w.sum()
    v0 = LAYOUT[dq.argmin()]
    n = v0.norm().clamp(min=1e-5)
    v = (torch.atanh(n.clamp(max=0.999)) * v0 / n).clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.03)
    for _ in range(steps):
        d = -poincare_scores(to_ball3(v)[None], LAYOUT)[0]
        loss = (w * (d - dq) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [round(x, 4) for x in to_ball3(v).detach().tolist()]


if SPACE is not None:  # fixed example view
    with torch.no_grad():
        ex_embs = encode_all(model, tokenizer, [x["query"] for x in EXAMPLES], args.q_max_len, 16, device, is_query=True).float()
        ex_scores = model.score(ex_embs.to(device), DOC_EMBS.to(device))
    SPACE["examples"] = []
    for x, sc in zip(EXAMPLES, ex_scores):
        gi = DOC_INDEX[x["expected_docid"]]
        with torch.enable_grad():
            pt = place_query(-sc.float())
        SPACE["examples"].append({"p": pt, "source": x["source"], "query": x["query"], "expected_docid": x["expected_docid"],
                                  "expected_title": x["expected_title"], "rank": int((sc > sc[gi]).sum()) + 1})

# ---- app --------------------------------------------------------------------------------------------------------
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")
HITS = defaultdict(deque)
LOG = SANTA / "demo" / "logs" / "queries.jsonl"


def client_ip(request):
    return request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
        or request.client.host


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)


@app.get("/")
def page():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/examples")
def examples():
    sources = [{"name": SOURCE_NAMES.get(k, k), "train": sum(d["dataset"] == k and d["split"] == "train" for d in DOCS),
                "unseen": sum(d["dataset"] == k and d["split"] != "train" for d in DOCS)} for k in SOURCE_NAMES]
    return {"examples": EXAMPLES, "sources": sources, "n_docs": len(DOCS),
            "n_unseen": sum(d["split"] != "train" for d in DOCS),
            "q_max_len": args.q_max_len}


@app.post("/api/search")
def search(body: SearchIn, request: Request):
    ip, now = client_ip(request), time.time()
    hits = HITS[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= args.rate:
        raise HTTPException(429, f"搜尋次數過多：每分鐘上限 {args.rate} 次，請稍候再試。")
    hits.append(now)
    q = " ".join(body.query.split())
    n_tok = len(tokenizer(q)["input_ids"])
    with GPU_LOCK:
        q_emb = encode_all(model, tokenizer, [q], args.q_max_len, 1, device, is_query=True).float()
        with torch.no_grad():
            scores = model.score(q_emb, DOC_EMBS)[0]
        query_point = None
        if SPACE is not None:
            with torch.enable_grad():
                query_point = place_query(-scores.to(device).float())
    order = scores.argsort(descending=True)[:body.k].tolist()
    results = [{"rank": r + 1, "docid": DOCS[i]["docid"], "title": DOCS[i]["title"],
                "source": SOURCE_NAMES.get(DOCS[i]["dataset"], DOCS[i]["dataset"]),
                "file": DOCS[i]["source_file"], "type": DOCS[i]["element"], "seen_in_training": DOCS[i]["split"] == "train",
                "distance": round(-float(scores[i]), 4) if model.similarity == "poincare" else None,
                "score": round(float(scores[i]), 4)} for r, i in enumerate(order)]
    if not args.no_log:
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "query": q,
                                "top5": [x["docid"] for x in results[:5]]}) + "\n")
    return {"results": results, "query_tokens": n_tok, "truncated": n_tok > args.q_max_len,
            "q_max_len": args.q_max_len, "similarity": model.similarity, "query_point": query_point}


@app.get("/api/space")
def space():
    if SPACE is None:
        raise HTTPException(404, "此模型不提供 3D 視覺化。")
    return SPACE


@app.get("/api/doc/{docid}")
def doc(docid: str):
    if docid not in DOC_INDEX:
        raise HTTPException(404, "找不到這份文件。")
    d = DOCS[DOC_INDEX[docid]]
    return {"docid": docid, "title": d["title"], "file": d["source_file"],
            "source": SOURCE_NAMES.get(d["dataset"], d["dataset"]), "xml": d["structured"]}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
