"""Phase 3 report: model comparison, calibration, and example projections."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from draft_dna.config import Settings, get_settings
from draft_dna.eval import backtest as bt
from draft_dna.eval import phase3 as P
from draft_dna.ingest.storage import read_table
from draft_dna.logging_utils import get_logger
from draft_dna.outcomes.tiers import TIERS

log = get_logger(__name__)

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#a3a29d"
RECORD = "Pick only + conformal"
SANITY_IDS = [
    "duranke01",
    "davisan02",
    "gilgesh01",
    "youngtr01",
    "moranja01",
    "halibty01",
    "brunsja01",
    "greendr01",
    "butleji01",
    "lillada01",
    "curryst01",
    "leonaka01",
    "willizi01",
    "bennean01",
    "doncilu01",
    "antetgi01",
    "wembavi01",
    "flaggco01",
    "dybanaj01",
    "jamesle01",
]


def _style(ax: Axes, grid_axis: Literal["both", "x", "y"] = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(tuning: pd.DataFrame, holdout: pd.DataFrame, path: Path) -> None:
    order = list(holdout.sort_values("crps_vs_pick")["model"])
    order = [m for m in order if m != "Pick only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, facecolor=SURFACE)
    for ax, res, title in (
        (axes[0], tuning, "Tuning years 2006-2012 (model selection)"),
        (axes[1], holdout, "Holdout years 2013-2020 (untouched)"),
    ):
        r = res.set_index("model").loc[order]
        y = np.arange(len(order))[::-1]
        for yi, (name, row) in zip(y, r.iterrows(), strict=True):
            color = BLUE if name == RECORD else (ORANGE if name == "Final blend" else MUTED)
            ax.plot(
                [row.crps_vs_pick_lo, row.crps_vs_pick_hi],
                [yi, yi],
                color=color,
                linewidth=2,
                solid_capstyle="round",
            )
            ax.scatter(
                [row.crps_vs_pick],
                [yi],
                s=44,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
            )
        ax.axvline(0, color=TEXT_2, linewidth=1)
        ax.set_yticks(y, order, fontsize=9.5, color=TEXT)
        ax.set_title(title, loc="left", fontsize=10.5, color=TEXT)
        ax.set_xlabel("CRPS difference vs pick only  (left = better)", fontsize=9, color=TEXT_2)
        _style(ax)
    fig.suptitle(
        "Do pre-draft stats beat draft position alone?", x=0.01, ha="left", fontsize=13, color=TEXT
    )
    fig.text(
        0.01,
        -0.02,
        "Rolling-origin backtest by draft year; target = best 3-season value "
        "through year 6. Lines: 95% paired bootstrap intervals. Blue = model of record.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(preds: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), facecolor=SURFACE)
    series = [(RECORD, BLUE), ("Final blend", ORANGE), ("Stats kNN", AQUA)]
    hold = {n: p[p["draft_year"].between(*P.HOLDOUT)] for n, p in preds.items()}
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
    for name, color in series:
        c = P.coverage_curve(hold[name])
        ax.plot(
            c["level"],
            c["observed"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
    ax.set_title("Quantile calibration", loc="left", fontsize=10.5, color=TEXT)
    ax.set_xlabel("Predicted quantile level", fontsize=9, color=TEXT_2)
    ax.set_ylabel("Share of players below it", fontsize=9, color=TEXT_2)
    _style(ax, "both")
    for ax, which, title in (
        (axes[1], "bust", "P(bust or worse)"),
        (axes[2], "allstar", "P(All-Star or better)"),
    ):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
        top = 0.0
        for name, color in series[:2]:
            r = P.reliability(hold[name], which)
            ax.plot(
                r["predicted"], r["observed"], color=color, linewidth=2, marker="o", markersize=4
            )
            top = max(top, r["predicted"].max(), r["observed"].max())
        lim = min(1.0, top * 1.1 + 0.02)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"Reliability: {title}", loc="left", fontsize=10.5, color=TEXT)
        ax.set_xlabel("Predicted probability", fontsize=9, color=TEXT_2)
        ax.set_ylabel("Observed frequency", fontsize=9, color=TEXT_2)
        _style(ax, "both")
    handles = [Line2D([], [], color=c, marker="o", linewidth=2, label=n) for n, c in series]
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.01, 1.04),
        labelcolor=TEXT,
    )
    fig.text(
        0.01,
        -0.03,
        "Holdout draft classes 2013-2020 (480 players). On the diagonal = "
        "calibrated. Probability bins have equal counts.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_class_bands(proj: pd.DataFrame, cuts: list[float], year: int, path: Path) -> None:
    d = proj[proj["draft_year"] == year].sort_values("pick").head(14)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    y = np.arange(len(d))[::-1]
    for c, label in zip(cuts[1:], TIERS[2:], strict=True):
        ax.axvline(c, color=GRID, linewidth=1, zorder=0)
        ax.text(c, len(d) - 0.3, f" {label}", fontsize=8, color=TEXT_2, va="bottom")
    ax.hlines(y, d["floor"], d["ceiling"], color=BLUE, linewidth=4, alpha=0.35)
    ax.scatter(d["median"], y, s=40, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    labels = [f"#{int(p)}  {n}" for p, n in zip(d["pick"], d["player_name"], strict=True)]
    ax.set_yticks(y, labels, fontsize=9.5, color=TEXT)
    for yi, (_, r) in zip(y, d.iterrows(), strict=True):
        ax.text(
            r["ceiling"] + 0.05,
            yi,
            f"{r['p_allstar_plus']:.0%} All-Star+",
            fontsize=8,
            color=TEXT_2,
            va="center",
        )
    ax.set_xlim(0, max(d["ceiling"].max() + 1.2, cuts[-1] + 0.6))
    ax.set_xlabel(
        "Projected peak value through year 6 (floor = 25th pct, dot = median, ceiling = 90th pct)",
        fontsize=9,
        color=TEXT_2,
    )
    _style(ax)
    ax.set_title(
        f"{year} draft: projected outcome ranges (top 14 picks)",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=22,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def comps_table(comps: pd.DataFrame, players: pd.DataFrame) -> str:
    names = players.set_index("bbref_id")
    rows = []
    for pid in SANITY_IDS:
        c = comps[comps["bbref_id"] == pid].sort_values("rank").head(5)
        if c.empty or pid not in names.index:
            continue
        p = names.loc[pid]
        rows.append(
            {
                "prospect": f"{p['player_name']} "
                f"({int(p['draft_year'])}, #{int(p['pick_overall'])})",
                "top 5 comps (peak value thru yr 6)": "; ".join(
                    f"{n} ({v:.1f})" for n, v in zip(c["comp_name"], c["comp_peak6"], strict=True)
                ),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def run(s: Settings | None = None) -> None:
    s = s or get_settings()
    out = s.paths.reports / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    df = bt.modeling_frame(s)
    preds = P.predictions(s, df)
    tuning = P.summarize(preds, P.TUNING)
    holdout = P.summarize(preds, P.HOLDOUT)
    plot_model_comparison(tuning, holdout, out / "model_comparison.png")
    plot_calibration(preds, out / "calibration.png")
    proj = read_table("modeled", "projections", "projections", s)
    comps = read_table("modeled", "projections", "comps", s)
    cuts = P.tier_cuts(s)
    plot_class_bands(proj, cuts, 2026, out / "draft_2026_bands.png")
    players = P.players(s)

    cols = [
        "model",
        "crps",
        "crps_vs_pick",
        "crps_vs_pick_lo",
        "crps_vs_pick_hi",
        "crps_vs_knn",
        "pinball_floor",
        "pinball_median",
        "pinball_ceiling",
        "brier_bust",
        "brier_allstar",
        "below_floor",
        "above_ceiling",
    ]
    top26 = proj[proj["draft_year"] == 2026].sort_values("pick").head(14)
    md = [
        "# Phase 3 results\n",
        "Target: best 3-season value through season 6. Rolling-origin backtest by draft year.\n",
        "## Holdout 2013-2020 (480 players; never used for model selection)\n",
        holdout[cols].round(4).to_markdown(index=False),
        "\n## Tuning 2006-2012 (420 players; used for model selection)\n",
        tuning[cols].round(4).to_markdown(index=False),
        "\n## Comp sanity check (stats kNN, earlier classes only)\n",
        comps_table(comps, players),
        "\n## 2026 draft projections (model of record: pick history + conformal)\n",
        top26[
            [
                "pick",
                "player_name",
                "position",
                "floor",
                "median",
                "ceiling",
                "p_allstar_plus",
                "p_bust_or_worse",
            ]
        ]
        .round(2)
        .to_markdown(index=False),
    ]
    (out / "phase3_results.md").write_text("\n".join(md) + "\n")
    holdout.to_csv(out / "holdout_summary.csv", index=False)
    tuning.to_csv(out / "tuning_summary.csv", index=False)
    log.info("phase 3 report written to %s", out)
