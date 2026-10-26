"""Write <results_dir>/<tag>/<pair>/pretrain_losses.jsonl from runs/xml/<tag>_logs/<tag>_<pair>.pretrain.log, for the 3
S1000D_spec_sample pairs.

Rows: {"kind": "train", "step", "epoch", "loss", "learning_rate"} per step (logging_steps 1),
      {"kind": "eval", "step", "epoch", "eval_loss"} every eval_steps, and one {"kind": "summary", "train_loss"} (mean over the run).
Usage: .venv/bin/python experiments/common/extract_losses.py <results_dir> <tag> [<tag> ...]
  e.g. results/05_pretrain_diagnosis longpretrain10_s42
"""
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAIRS = [f"{s}_to_S1000D_spec_sample" for s in ("BIKE_7", "FOSSIG", "s1kd-tools-doc")]


def extract(log):
    rows, step = [], 0
    for d in re.findall(r"\{'(?:loss|eval_loss|train_runtime)'[^}]*\}", log.read_text()):
        d = ast.literal_eval(d)
        if "loss" in d:
            step += 1
            rows.append({"kind": "train", "step": step, "epoch": d["epoch"], "loss": d["loss"], "learning_rate": d["learning_rate"]})
        elif "eval_loss" in d:
            rows.append({"kind": "eval", "step": step, "epoch": d["epoch"], "eval_loss": d["eval_loss"]})
        else:
            rows.append({"kind": "summary", "step": step, "epoch": d["epoch"], "train_loss": d["train_loss"]})
    return rows


results_dir = REPO / sys.argv[1]
for tag in sys.argv[2:]:
    for pair in PAIRS:
        rows = extract(REPO / "runs" / "xml" / f"{tag}_logs" / f"{tag}_{pair}.pretrain.log")
        out = results_dir / tag / pair / "pretrain_losses.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{out}: {len(rows)} rows")
