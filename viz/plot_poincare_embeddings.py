#!/usr/bin/env python3
"""Visualize the Poincare-ball embeddings of the pooled Poincare runs.

The cached embeddings.npz hold raw encoder reps; they are mapped into the ball exactly as SModel.to_ball does
(x * poincare_scale, then tanh(|x|) x/|x|). Distances are the model's own score (poincare_scores, curvature -1).

Figures (results/figures/):
  fig_poincare_disk_{original,anonymized}.png
      one Poincare disk per arm. 2-D layout by hyperbolic MDS: points are optimized inside the 2-D disk so their
      pairwise Poincare distances match the true 768-d ones (stress in each title; lower = more faithful).
      Colour = source corpus, circle = test document, triangle = query, grey segment = query -> gold document.
      Faint rings mark hyperbolic distance 1, 2, 3 from the origin.
  fig_poincare_ball_3d.png
      plain-language figure for non-technical readers: finetuneB_only vs pretrainA_finetuneB (same finetuning
      data, with / without SDA+MEP pretraining), original queries. Hyperbolic MDS into a 3-D ball, drawn as a
      shaded sphere with depth cues; a line joins each question to its document only when that document is the
      closest one in the true 768-d space.

Usage: .venv/bin/python viz/plot_poincare_embeddings.py [--seed 42]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "09_pooled_poincare"
OUT = REPO / "results" / "figures"
SCALE = 0.25  # --poincare_scale used by run_pooled.sh
EPS = 1e-5    # model.py POINCARE_EPS

ARMS = [("finetuneA_only", "Finetune A only"), ("finetuneB_only", "Finetune B only"),
        ("pretrainonly_mep", "Pretrain only (MEP)"), ("pretrainonly_sdamep", "Pretrain only (SDA+MEP)"),
        ("pretrainA_finetuneB", "Pretrain A → finetune B"), ("pretrainB_finetuneA", "Pretrain B → finetune A"),
        ("full", "Full pool, finetune only")]
SOURCES = ["S1000D_spec_sample", "BIKE_7", "s1kd-tools-doc", "FOSSIG"]
# reference categorical palette, fixed order (same slots as plot_s1000d_figures.py)
SRC_COL = dict(zip(SOURCES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})


def to_ball(x):
    x = x.astype(np.float64) * SCALE
    n = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), EPS, None)
    return np.tanh(n) * x / n


def pdist(a, b):
    """Poincare distance matrix, same formula as model.py poincare_scores (without the sign)."""
    lib = torch if isinstance(a, torch.Tensor) else np
    sa, sb = (a * a).sum(-1), (b * b).sum(-1)
    sq = lib.clip(sa[:, None] + sb[None, :] - 2 * a @ b.T, 0, None)
    den = (1 - lib.clip(sa, None, 1 - EPS))[:, None] * (1 - lib.clip(sb, None, 1 - EPS))[None, :]
    return lib.arccosh(lib.clip(1 + 2 * sq / den, 1 + 1e-7, None))


def hyperbolic_mds(X, steps=3000, seed=0, dim=2):
    """Place points in the dim-D Poincare ball so their pairwise distances match those of X (n, d) in the ball.
    Returns (Y (n, dim), Kruskal stress-1)."""
    D = torch.tensor(pdist(X, X))
    iu = torch.triu_indices(len(X), len(X), 1)
    target = D[iu[0], iu[1]]
    # init: top principal directions of the tangent vectors at the origin, keeping each point's true depth
    V = np.arctanh(np.linalg.norm(X, axis=1, keepdims=True)) * X / np.linalg.norm(X, axis=1, keepdims=True)
    Vc = V - V.mean(0)
    U = Vc @ np.linalg.svd(Vc, full_matrices=False)[2][:dim].T
    U = U / np.linalg.norm(U, axis=1, keepdims=True) * np.linalg.norm(V, axis=1, keepdims=True)
    torch.manual_seed(seed)
    v = torch.tensor(U, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.02)
    for _ in range(steps):
        n = v.norm(dim=1, keepdim=True).clamp(min=EPS)
        y = torch.tanh(n) * v / n
        d = pdist(y, y)[iu[0], iu[1]]
        loss = ((d - target) ** 2).sum() / (target ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        n = v.norm(dim=1, keepdim=True).clamp(min=EPS)
        y = torch.tanh(n) * v / n
    return y.numpy(), float(loss.detach().sqrt())


def load(version, arm, seed):
    suf = "_anonq" if version == "anonymized" else ""
    d = np.load(RES / f"pooled{suf}_poincare_s{seed}" / arm / "embeddings.npz")
    R = REPO / "data" / "xml" / f"retrieval_pooled{suf}" / "pooled"
    src = {json.loads(l)["qid"]: json.loads(l)["dataset"] for l in open(R / "queries.test.jsonl")}
    qrels = {l.split()[0]: l.split()[1] for l in list(open(R / "qrels.test.tsv"))[1:]}
    cids, qids = d["corpus_ids"].tolist(), d["query_ids"].tolist()
    doc_src = {qrels[q]: s for q, s in src.items()}
    gold = np.array([cids.index(qrels[q]) for q in qids])
    return (to_ball(d["corpus_embs"]), to_ball(d["query_embs"]),
            [doc_src[c] for c in cids], [src[q] for q in qids], gold)


def disk_axes(ax):
    for h in (1, 2, 3):
        r = np.tanh(h / 2)  # Euclidean radius at hyperbolic distance h from the origin
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID, lw=0.6, zorder=0))
        ax.text(0, r + 0.012, str(h), color=NEUTRAL, fontsize=6, ha="center", va="bottom")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec=INK2, lw=0.9, zorder=0))
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_disks(version, seed):
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.2))
    for ax, (arm, name) in zip(axes.flat, ARMS):
        P, Q, psrc, qsrc, gold = load(version, arm, seed)
        Y, stress = hyperbolic_mds(np.vstack([P, Q]))
        YP, YQ = Y[:len(P)], Y[len(P):]
        disk_axes(ax)
        for i, g in enumerate(gold):
            ax.plot([YQ[i, 0], YP[g, 0]], [YQ[i, 1], YP[g, 1]], color=NEUTRAL, lw=0.4, alpha=0.6, zorder=1)
        for s in SOURCES[::-1]:  # largest source drawn first, small ones stay visible on top
            m = np.array(psrc) == s
            ax.scatter(YP[m, 0], YP[m, 1], s=16, marker="o", c=SRC_COL[s], ec="white", lw=0.5, zorder=2)
            m = np.array(qsrc) == s
            ax.scatter(YQ[m, 0], YQ[m, 1], s=20, marker="^", c=SRC_COL[s], ec="white", lw=0.5, zorder=3)
        ax.set_title(f"{name}\nstress {stress:.2f}", fontsize=8.5)
        print(f"{version:10s} {arm:22s} stress {stress:.3f}")
    ax = axes.flat[-1]
    ax.axis("off")
    handles = [Line2D([], [], ls="", marker="s", ms=7, mfc=SRC_COL[s], mec="none", label=s) for s in SOURCES]
    handles += [Line2D([], [], ls="", marker="o", ms=6, mfc=INK2, mec="white", label="test document"),
                Line2D([], [], ls="", marker="^", ms=6.5, mfc=INK2, mec="white", label="query"),
                Line2D([], [], color=NEUTRAL, lw=0.8, label="query → gold document")]
    ax.legend(handles=handles, loc="center left", fontsize=8, handletextpad=0.5, labelspacing=0.8)
    fig.suptitle(f"Poincaré-disk view of test embeddings, {version} queries (seed {seed})", x=0.01, ha="left",
                 fontsize=10, weight="bold")
    fig.text(0.01, -0.01, "2-D hyperbolic MDS of the 768-d Poincaré distances; rings = hyperbolic distance 1, 2, 3 "
             "from the origin; stress = Kruskal stress-1 of the 2-D layout.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    path = OUT / f"fig_poincare_disk_{version}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def sphere_backdrop(ax):
    """Shaded ball: radial gradient lit from the upper left, plus an equator and a meridian (back half dashed)."""
    g = np.linspace(-1, 1, 400)
    xx, yy = np.meshgrid(g, g)
    shade = np.clip(1 - 0.55 * np.hypot(xx + 0.35, yy - 0.4), 0, 1)
    rgba = np.zeros((400, 400, 4))
    rgba[..., :3] = (0.86 + 0.13 * shade)[..., None] * np.array([0.93, 0.95, 1.0])
    rgba[..., 3] = 1
    im = ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", zorder=0)
    im.set_clip_path(plt.Circle((0, 0), 1, transform=ax.transData))
    t = np.linspace(0, 2 * np.pi, 400)
    for x, y, z in ((np.cos(t), 0.28 * np.sin(t), np.sin(t)), (0.28 * np.cos(t), np.sin(t), np.cos(t))):
        front = z >= 0
        ax.plot(np.where(front, x, np.nan), np.where(front, y, np.nan), color="#9db0c8", lw=0.7, zorder=1)
        ax.plot(np.where(~front, x, np.nan), np.where(~front, y, np.nan), color="#9db0c8", lw=0.6, ls=(0, (2, 3)),
                zorder=1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#6b7f99", lw=1.2, zorder=1))
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_ball_3d(seed, version="original"):
    panels = [("finetuneB_only", "Trained on questions only"),
              ("pretrainA_finetuneB", "Pre-trained on document structure first")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, (arm, name) in zip(axes, panels):
        P, Q, psrc, qsrc, gold = load(version, arm, seed)
        found = pdist(Q, P).argmin(1) == gold  # true 768-d nearest document is the right one
        Y, stress = hyperbolic_mds(np.vstack([P, Q]), dim=3)
        Yc = Y - Y.mean(0)
        R = np.linalg.svd(Yc, full_matrices=False)[2]  # view: widest spread across the screen, least in depth
        R[2] *= np.sign(np.linalg.det(R))
        tilt = np.array([[1, 0, 0], [0, np.cos(0.35), -np.sin(0.35)], [0, np.sin(0.35), np.cos(0.35)]])
        V = Y @ R.T @ tilt.T
        n = len(P)
        sphere_backdrop(ax)
        depth = (V[:, 2] + 1) / 2  # 0 = back, 1 = front
        items = [(V[i, 2], "pt", i) for i in range(len(V))]
        items += [((V[g, 2] + V[n + q, 2]) / 2 - 1e-3, "ln", q) for q, g in enumerate(gold) if found[q]]
        for z, kind, i in sorted(items):
            d = (z + 1) / 2
            if kind == "ln":
                g = gold[i]
                ax.plot([V[n + i, 0], V[g, 0]], [V[n + i, 1], V[g, 1]], color=INK, lw=0.5 + 0.7 * d,
                        alpha=0.35 + 0.5 * d, zorder=2 + z, solid_capstyle="round")
                continue
            is_q = i >= n
            col = SRC_COL[(qsrc[i - n] if is_q else psrc[i])]
            ax.scatter(V[i, 0], V[i, 1], s=(14 if is_q else 22) * (0.55 + 1.1 * depth[i]),
                       marker="^" if is_q else "o", c=col, alpha=0.35 + 0.65 * depth[i],
                       ec="white", lw=0.4 + 0.5 * depth[i], zorder=2 + z)
        ax.set_title(name, fontsize=10.5, loc="center", pad=22)
        ax.text(0.5, 1.02, f"{found.sum()} of {len(Q)} questions find their own document",
                transform=ax.transAxes, ha="center", fontsize=9, color=INK2)
        print(f"3d {arm:22s} stress {stress:.3f} found {found.sum()}")
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=SRC_COL[s], mec="none", label=s) for s in SOURCES]
    handles += [Line2D([], [], ls="", marker="o", ms=6, mfc="#b9b8b2", mec="white", label="document"),
                Line2D([], [], ls="", marker="^", ms=6.5, mfc="#b9b8b2", mec="white", label="question"),
                Line2D([], [], color=INK, lw=1, label="question matched to its document")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               columnspacing=1.4)
    fig.suptitle("Where the model places questions and documents", x=0.5, y=1.05, fontsize=12.5, weight="bold")
    fig.text(0.5, -0.1, "Each dot is one test question or document; colour = the manual it comes from. Things the "
             "model thinks are related sit close together.\nIllustration: the model's 768-dimensional space is "
             "approximated inside a 3-D ball, so distances are approximate; which pairs are matched uses the "
             "exact distances.", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    path = OUT / "fig_poincare_ball_3d.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for v in ("original", "anonymized"):
        fig_disks(v, a.seed)
    fig_ball_3d(a.seed)
