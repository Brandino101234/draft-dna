"""Phase 5 report: pre-registered comparisons, model table, subgroups, exploratory signal."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import spearmanr

from draft_dna.config import Settings, get_settings
from draft_dna.eval import phase5 as P5
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#a3a29d"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_hypotheses(rows: pd.DataFrame, path: Path) -> None:
    """Forest plot: each pre-registered comparison, design B (primary) and A (check)."""
    labels = list(dict.fromkeys(rows["comparison"]))
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(labels) + 1.6), facecolor=SURFACE)
    y0 = np.arange(len(labels))[::-1]
    for design, color, off in (("B", BLUE, 0.16), ("A", ORANGE, -0.16)):
        r = rows[rows["design"] == design].set_index("comparison").reindex(labels)
        ax.hlines(y0 + off, r["ci_lo"], r["ci_hi"], color=color, linewidth=2)
        ax.scatter(
            r["crps_diff"],
            y0 + off,
            s=40,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label="Design B: strict, year 4 (primary)"
            if design == "B"
            else "Design A: year 6 (sensitivity check)",
        )
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y0, labels, fontsize=9.5, color=TEXT)
    ax.set_xlabel(
        "CRPS difference (left = first model better; right = worse)", fontsize=9, color=TEXT_2
    )
    _style(ax)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.14),
        ncol=2,
        labelcolor=TEXT,
    )
    ax.set_title(
        "Does shot data help? Pre-registered comparisons with 95% CIs",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def residual_signal(s: Settings) -> pd.DataFrame:
    """Exploratory (not pre-registered): rank correlation of each shot feature with the
    outcome, and with what is left after out-of-sample pick and pick+stats predictions."""
    preds = P5.predictions(s, "B")
    df = P5.frame(s).set_index("bbref_id")
    y = preds["Pick only + conformal"].set_index("bbref_id")["y"]
    rp = y - preds["Pick only + conformal"].set_index("bbref_id")["q0.50"]
    rs = y - preds["LightGBM pick+stats"].set_index("bbref_id")["q0.50"]
    rows = []
    for f in P5.SHOT:
        x = df.loc[y.index, f]
        ok = x.notna()
        res = spearmanr(x[ok], rs[ok])
        rows.append(
            {
                "feature": f,
                "n": int(ok.sum()),
                "rho_outcome": spearmanr(x[ok], y[ok]).statistic,
                "rho_beyond_pick": spearmanr(x[ok], rp[ok]).statistic,
                "rho_beyond_pick_stats": res.statistic,
                "p_beyond_pick_stats": res.pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values("rho_beyond_pick_stats", key=abs, ascending=False)


LABELS = {
    "rim_fg_eb": "Rim FG% (shrunk)",
    "rim_fg_eb_rel": "Rim FG% vs D-I avg",
    "rim_rate": "Rim attempt rate",
    "dunk_share": "Dunk share of rim attempts",
    "three_rate": "3PA rate",
    "three_rate_rel": "3PA rate vs D-I avg",
    "mid_rate": "Midrange rate",
    "three_fg_eb": "3P% (shrunk)",
    "three_fg_eb_rel": "3P% vs D-I avg",
    "mid_fg_eb": "Midrange FG% (shrunk)",
    "mid_fg_eb_rel": "Midrange FG% vs D-I avg",
    "assisted_rim_share": "Assisted rim makes",
    "assisted_mid_share": "Assisted midrange makes",
    "assisted_three_share": "Assisted threes",
    "unassisted_share": "Self-created makes",
}


def plot_signal(t: pd.DataFrame, path: Path) -> None:
    t = t.sort_values("rho_outcome", key=abs).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
    y = np.arange(len(t))
    ax.barh(
        y + 0.2, t["rho_outcome"], height=0.36, color=MUTED, label="Raw correlation with outcome"
    )
    ax.barh(
        y - 0.2,
        t["rho_beyond_pick_stats"],
        height=0.36,
        color=BLUE,
        label="After removing what pick + stats predict",
    )
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, [LABELS.get(f, f) for f in t["feature"]], fontsize=9, color=TEXT)
    ax.set_xlabel("Spearman correlation with year-4 peak value", fontsize=9, color=TEXT_2)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
    ax.set_title(
        "Exploratory: which shot traits carry signal beyond pick and stats?",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.text(
        0.01,
        -0.02,
        "Design B cohort (414 players). Not pre-registered: 15 features tested, "
        "so treat single correlations as hypotheses (multiple-comparison threshold p < 0.003).",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(s: Settings | None = None) -> dict[str, pd.DataFrame]:
    s = s or get_settings()
    out = s.paths.reports / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    hyp, summ, subs, spat = [], [], [], []
    for key in ("B", "A"):
        d = P5.DESIGNS[key]
        preds = P5.predictions(s, key)
        years = (d.first, d.last)
        h = P5.hypotheses(preds, years).assign(design=key)
        sp, _ = P5.spatial(s, key)
        h = pd.concat(
            [
                h,
                pd.DataFrame(
                    [
                        {
                            "comparison": "Tier B: pick+stats+location vs pick+stats (subset)",
                            "crps_diff": sp["diff"].iloc[0],
                            "ci_lo": sp["ci_lo"].iloc[0],
                            "ci_hi": sp["ci_hi"].iloc[0],
                            "verdict": "no detectable difference"
                            if sp["ci_lo"].iloc[0] < 0 < sp["ci_hi"].iloc[0]
                            else ("better" if sp["ci_hi"].iloc[0] < 0 else "worse"),
                            "design": key,
                        }
                    ]
                ),
            ]
        )
        hyp.append(h)
        summ.append(P5.summary(preds, years).assign(design=key))
        subs.append(
            P5.subgroups(preds, s, years, "LightGBM pick+stats+shot", "LightGBM pick+stats").assign(
                design=key
            )
        )
        spat.append(sp)
    hyp_df = pd.concat(hyp, ignore_index=True)
    plot_hypotheses(hyp_df, out / "preregistered_comparisons.png")
    sig = residual_signal(s)
    plot_signal(sig, out / "exploratory_shot_signal.png")
    summ_df = pd.concat(summ, ignore_index=True)
    w_b, tune_b = P5.tune_comp_weight(P5.predictions(s, "B"), P5.DESIGNS["B"])
    cols = [
        "model",
        "n",
        "crps",
        "crps_vs_pick",
        "ci_lo",
        "ci_hi",
        "pinball_floor",
        "pinball_median",
        "pinball_ceiling",
        "brier_bust",
        "brier_allstar",
        "below_floor",
        "above_ceiling",
    ]
    md = [
        "# Phase 5 results (pre-registered in DECISIONS D027)\n",
        "## Pre-registered comparisons\n",
        hyp_df.round(4).to_markdown(index=False),
        "\n## All models, design B (strict, year-4 outcome)\n",
        summ_df[summ_df["design"] == "B"][cols].round(4).to_markdown(index=False),
        "\n## All models, design A (sensitivity check)\n",
        summ_df[summ_df["design"] == "A"][cols].round(4).to_markdown(index=False),
        f"\n## Comp blend weight (chosen on tuning years): stats weight = {w_b}\n",
        tune_b.round(4).to_markdown(index=False),
        "\n## Subgroups: pick+stats+shot vs pick+stats\n",
        pd.concat(subs).round(4).to_markdown(index=False),
        "\n## Exploratory (not pre-registered): shot-feature signal beyond pick + stats\n",
        sig.round(4).to_markdown(index=False),
    ]
    (out / "phase5_results.md").write_text("\n".join(md) + "\n")
    log.info("phase 5 report written to %s", out)
    return {"hypotheses": hyp_df, "signal": sig}
