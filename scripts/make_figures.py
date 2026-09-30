"""Regenerate the README figures.

  python scripts/make_figures.py

Inputs:
  results/holdout_v2_seed42.txt   output of `python sim/validate.py --leagues 300 --seed 42`
  numbers transcribed from sim/README.md (v1 genome table) and
  docs/RECALIBRATION_STUDY.md (backtest table) - cited inline below.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def parse_holdout(path):
    """{block name: {year: title%}} from validate.py output."""
    out, cur = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"=== (.+?)\s+\(\d+ holdout", line)
        if m:
            cur = m.group(1).strip()
            out[cur] = {}
            continue
        m = re.match(r"\s+(\d{4}|ALL)\s+([\d.]+)%", line)
        if m and cur:
            out[cur][m.group(1)] = float(m.group(2))
    return out


def fig_holdout():
    d = parse_holdout(os.path.join(ROOT, "results", "holdout_v2_seed42.txt"))
    champ, dflt = d["evolved champion"], d["hand-tuned default (draftmodel.py weights)"]
    cats = ["2021", "2022", "2023", "2024", "2025", "ALL"]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar([i - w / 2 - 0.01 for i in x], [champ[c] for c in cats], w, color=BLUE,
           label="evolved champion", zorder=3)
    ax.bar([i + w / 2 + 0.01 for i in x], [dflt[c] for c in cats], w, color=ORANGE,
           label="hand-tuned lookahead default", zorder=3)
    ax.axhline(10, color=GRAY, lw=1.5, ls="--", zorder=4, label="luck baseline (10%)")
    for i, c in enumerate(cats):
        ax.text(i - w / 2, champ[c] + 0.6, f"{champ[c]:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x), cats)
    ax.set_ylabel("championship rate (%)")
    ax.set_title("Held-out seasons 2021-25: my drafter vs 9 noisy ESPN-autopick bots\n"
                 "(300 leagues per year; bots jitter ADP by its historical SD and manage rosters)",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "holdout_title_rate.png"), dpi=150)


def fig_lottery():
    # sim/README.md, "Results (2026-08-06 -> 2026-08-10 ...)" table: holdout title%
    # vs 9 deterministic autopick bots, v1 (draft-only) environment.
    rows = [
        ("fixed-run champion, seed-1 rerun", 21.4),
        ("gen-325 loop champion (trained 2010-18)", 14.0),
        ("hand-tuned default (no evolution)", 12.9),
        ("fixed-run champion, different RNG stream", 10.2),
        ("loop champion, guard-selected", 10.1),
        ("gen-91,988 marathon loop champion", 7.7),
        ("loop champion, all-years-selected", 7.5),
    ][::-1]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    colors = [BLUE if "hand-tuned" not in n else ORANGE for n, _ in rows]
    ax.barh([n for n, _ in rows], [v for _, v in rows], color=colors, height=0.6, zorder=3)
    ax.axvline(10, color=GRAY, lw=1.5, ls="--", zorder=4)
    ax.text(10.2, -0.9, "luck = 10%", color=INK2, fontsize=9)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("held-out championship rate (%); SE about 0.8 pts at 1,500 leagues")
    fig.suptitle("Evolution is a lottery: same code, different seeds and run lengths",
                 fontsize=10, x=0.01, ha="left", color=INK)
    ax.set_ylim(-1.2, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "evolution_lottery.png"), dpi=150)


def fig_recal():
    # docs/RECALIBRATION_STUDY.md backtest table (out-of-sample RMSE, points per game).
    rmse = {  # fold -> horizon -> (M0 bench, M1 calib, M2 volume, M2s snap+resid)
        "train 2023 -> test 2024": {"next week": (6.478, 6.443, 6.475, 6.433),
                                    "next 4 weeks": (4.268, 4.226, 4.241, 4.208),
                                    "rest of season": (3.663, 3.655, 3.659, 3.655)},
        "train 2023-24 -> test 2025": {"next week": (6.479, 6.441, 6.487, 6.442),
                                       "next 4 weeks": (4.224, 4.143, 4.175, 4.127),
                                       "rest of season": (3.665, 3.551, 3.590, 3.534)},
    }
    labels = [("calibration only (M1)", BLUE), ("full volume ridge (M2)", ORANGE),
              ("snap share + residual ridge (M2s)", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (fold, hz) in zip(axes, rmse.items()):
        hs = list(hz)
        w = 0.26
        for k, (lab, col) in enumerate(labels):
            vals = [(hz[h][k + 1] / hz[h][0] - 1) * 100 for h in hs]
            ax.bar([i + (k - 1) * (w + 0.02) for i in range(len(hs))], vals, w, color=col,
                   label=lab, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(hs)), hs, fontsize=9)
        ax.set_title(fold, fontsize=9.5, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE change vs consensus (%)\n(below 0 = better)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Can usage data beat the consensus projection? Mostly no - calibration is the edge",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "recal_backtest.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_holdout()
    fig_lottery()
    fig_recal()
    print("wrote figures to", FIG)
