"""Collect the AdvTest (CodeSearchNet python -> AdvTest) results into results/12_advtest_ablation/advtest_results.xlsx
(gitignored, not committed).

One row per run with its settings, AdvTest test MRR (pretrain only / + finetune) and best AdvTest valid MRR:
exp 09 (read from the exp/09-advtest branch with git show), exp 11 (results/11_advtest_firsttoken/, local), exp 12
(results/12_advtest_ablation/), plus the Table 6 dot-product baselines and the paper's number as reference rows.
Runs without results yet are listed with their status. Re-running overwrites the file.

Run with .venv/bin/python experiments/12_advtest_ablation/make_xlsx.py
"""
import json
import subprocess
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "12_advtest_ablation" / "advtest_results.xlsx"
FILES = ["eval_test_pretrain", "eval_test", "best_dev_pretrain", "best_dev_finetune"]

# (exp, run, source, similarity, pooling, scale, temp, batch, p_max_len, data trunc, compared against, status note)
RUNS = [
    ("09", "advtest_poincare_s42", "branch:exp/09-advtest:results/09_advtest", "poincare", "mean", 0.25, 0.05,
     "16x8", 240, "240/256", "", ""),
    ("11", "advtest_firsttoken_poincare_s42", "local:results/11_advtest_firsttoken", "poincare", "first", 0.006, 1.0,
     "128", 256, "240/256", "", ""),
    ("11", "advtest_firsttoken_poincare_s42 (hard neg)", None, "poincare", "first", 0.006, 1.0, "128", 256, "240/256",
     "exp 11 in-batch", "OOM at 77% (GPU shared with other jobs), not rerun"),
    ("12", "ft_temp005", "local:results/12_advtest_ablation", "poincare", "first", 0.006, 0.05, "128", 256, "240/256",
     "exp 11: temperature", ""),
    ("12", "mean_p256_b128", "local:results/12_advtest_ablation", "poincare", "mean", 0.25, 0.05, "128", 256,
     "240/256", "ft_temp005: pooling; exp 09: batch + p_max_len", ""),
    ("12", "ft_dot", "local:results/12_advtest_ablation", "dot", "first", None, 0.05, "128", 256, "240/256",
     "ft_temp005: similarity", ""),
    ("12", "ft_temp005_hardneg", "local:results/12_advtest_ablation", "poincare", "first", 0.006, 0.05, "128", 256,
     "240/256", "ft_temp005: hard negatives (pretrain reused)", ""),
    ("12", "ft_dot_hardneg", "local:results/12_advtest_ablation", "dot", "first", None, 0.05, "128", 256, "240/256",
     "ft_dot: hard negatives (pretrain reused)", ""),
    ("12", "ft_b16x8", None, "poincare", "first", 0.006, 0.05, "16x8", 256, "240/256", "ft_temp005: batch",
     "stopped (user decision)"),
    ("12", "mean_p1024_b128", None, "poincare", "mean", 0.25, 0.05, "128", 1024, "1024", "mean_p256_b128: length",
     "OOM even with grad ckpt (one layer's attention > 48 GB)"),
]
# (name, similarity, pooling, batch, p_max_len, pretrain-only test MRR, finetune test MRR, note)
REFERENCE = [
    ("Table 6 in-batch (runs/table6)", "dot", "first", "16x8", 240, 0.420, 0.428, "SANTA recipe, temperature 1"),
    ("Table 6 hard neg (runs/table6)", "dot", "first", "16x8", 240, 0.420, 0.419, "SANTA recipe, temperature 1"),
    ("hparamfix (/workspace/SANTA)", "dot", "first", "16x8", 240, None, 0.431, "lr 1e-5, 12 epochs"),
    ("og Poincare (/workspace/SANTA)", "poincare", "first", "16x8", 240, None, 0.0, "no scale: tanh saturated"),
    ("Paper: SANTA (Hard Negative)", "dot", "first", None, None, None, 0.486, "Li et al. 2023, AdvTest"),
]


def load(source, run, name):
    if source is None:
        return None
    kind, *rest = source.split(":")
    if kind == "branch":
        branch, path = rest
        r = subprocess.run(["git", "show", f"{branch}:{path}/{run}/{name}.json"], cwd=REPO, capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else None
    f = REPO / rest[0] / run / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def metric(d):
    if d is None:
        return None
    return d.get("eval_mrr", d.get("best_dev_mrr"))


def autosize(ws):
    for col in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), 60)


def main():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "AdvTest runs"
    head = ["Exp", "Run", "Similarity", "Pooling", "poincare_scale", "Temperature", "Batch", "p_max_len",
            "Data trunc", "Test MRR (pretrain only)", "Test MRR (+ finetune)", "Valid MRR (pretrain)",
            "Valid MRR (finetune)", "Compared against: differs in", "Status"]
    ws.append(head)
    hl = PatternFill("solid", fgColor="FFF2CC")
    best = max((metric(load(s, r, "eval_test")) or 0) for _, r, s, *_ in RUNS)
    for exp, run, src, sim, pool, scale, temp, batch, pmax, trunc, cmp, note in RUNS:
        vals = [metric(load(src, run, n)) for n in FILES]
        status = note or ("done" if vals[1] is not None else "pending")
        ws.append([exp, run, sim, pool, scale, temp, batch, pmax, trunc, *vals, cmp, status])
        if vals[1] is not None and vals[1] == best:
            for c in ws[ws.max_row]:
                c.fill = hl
    ws.append([])
    ws.append(["Reference (not run in this experiment)"])
    ws.cell(ws.max_row, 1).font = Font(bold=True)
    ws.append(["", "Name", "Similarity", "Pooling", "", "", "Batch", "p_max_len", "", "Test MRR (pretrain only)",
               "Test MRR (+ finetune)", "", "", "Note"])
    ws.cell(ws.max_row, 2).font = Font(bold=True)
    for name, sim, pool, batch, pmax, pre, ft, note in REFERENCE:
        ws.append(["", name, sim, pool, None, None, batch, pmax, None, pre, ft, None, None, note])
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row[9:13]:
            if isinstance(c.value, float):
                c.number_format = "0.0000"
    ws.freeze_panes = "C2"
    autosize(ws)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
