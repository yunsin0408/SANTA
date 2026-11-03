"""Collect exp 10's results into results/10_lorentz/lorentz_results.xlsx (gitignored, not committed).

Sheets:
  Summary      test MRR per arm and query version, mean / sd over seeds: Lorentz (results/10_lorentz/) next to
               Poincare (results/08_pooled_poincare/, read from the exp/08-pooled branch with git show)
  Per run      one row per run: test MRR and best dev MRR (finetune and, for half-split arms, pretrain)
  Scale check  results/10_lorentz_scale/ (full arm, anonymized queries, seed 42)
Runs that haven't finished (no eval_test.json) are left out. Re-running overwrites the file.

Run with .venv/bin/python experiments/10_lorentz/make_xlsx.py
"""
import json
import statistics
import subprocess
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "10_lorentz" / "lorentz_results.xlsx"
ARMS = ["full", "pretrainA_finetuneB", "finetuneB_only", "pretrainB_finetuneA", "finetuneA_only", "pretrainonly_sdamep"]
QUERIES = {"": "original", "_anonq": "anonymized"}
SEEDS = [42, 1, 2]


def lorentz(path):
    f = REPO / "results" / "10_lorentz" / path
    return json.loads(f.read_text()) if f.exists() else None


def poincare(path):  # exp 08's results live on its own branch
    r = subprocess.run(["git", "show", f"exp/08-pooled:results/08_pooled_poincare/{path}"], cwd=REPO,
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def runs():
    rows = []
    for sim, load in (("Lorentz", lorentz), ("Poincare (exp 08)", poincare)):
        tag = "lorentz" if sim == "Lorentz" else "poincare"
        for q, qname in QUERIES.items():
            for seed in SEEDS:
                for arm in ARMS:
                    d = f"pooled{q}_{tag}_s{seed}/{arm}"
                    test = load(f"{d}/eval_test.json")
                    if test is None:
                        continue
                    dev_ft, dev_pt = load(f"{d}/best_dev_finetune.json"), load(f"{d}/best_dev_pretrain.json")
                    rows.append([sim, qname, seed, arm, test["eval_mrr"], dev_ft and dev_ft["best_dev_mrr"],
                                 dev_pt and dev_pt["best_dev_mrr"]])
    return rows


def mean_sd(xs):
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else None, len(xs)) if xs else (None, None, 0)


def write(ws, header, rows):
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value, float):
                c.number_format = "0.000"
    for i, h in enumerate(header, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(12, len(h) + 2)
    ws.freeze_panes = "A2"


def main():
    per_run = runs()
    summary = []
    for qname in QUERIES.values():
        for arm in ARMS:
            pick = lambda sim: [r[4] for r in per_run if r[0] == sim and r[1] == qname and r[3] == arm]
            lm, lsd, ln = mean_sd(pick("Lorentz"))
            pm, psd, pn = mean_sd(pick("Poincare (exp 08)"))
            summary.append([qname, arm, lm, lsd, ln, pm, psd, pn, lm - pm if lm is not None and pm is not None else None])

    scale = []
    for d in sorted((REPO / "results" / "10_lorentz_scale").glob("*/")):
        test, dev = d / "eval_test.json", d / "best_dev_finetune.json"
        if test.exists():
            sim, s = d.name.split("_scale")
            scale.append([sim.capitalize(), float(s), json.loads(dev.read_text())["best_dev_mrr"] if dev.exists() else None,
                          json.loads(test.read_text())["eval_mrr"]])

    wb = openpyxl.Workbook()
    write(wb.active, ["Queries", "Arm", "Lorentz mean", "Lorentz sd", "Lorentz seeds", "Poincare mean", "Poincare sd",
                      "Poincare seeds", "Lorentz - Poincare"], summary)
    wb.active.title = "Summary"
    write(wb.create_sheet("Per run"), ["Similarity", "Queries", "Seed", "Arm", "Test MRR", "Best dev MRR (finetune)",
                                        "Best dev MRR (pretrain)"], per_run)
    write(wb.create_sheet("Scale check"), ["Similarity", "Scale", "Best dev MRR", "Test MRR"], scale)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    n = sum(r[0] == "Lorentz" for r in per_run)
    print(f"wrote {OUT}: {n} Lorentz runs, {len(per_run) - n} Poincare runs, {len(scale)} scale-check runs")


if __name__ == "__main__":
    main()
