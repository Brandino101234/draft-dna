"""Phase 2 report: validation of value definitions, tiers, survival curves.

Writes PNGs and a markdown summary to reports/phase2/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from draft_dna.config import Settings, get_settings
from draft_dna.ingest.storage import read_table, table_path
from draft_dna.logging_utils import get_logger
from draft_dna.outcomes import run as outcomes_run
from draft_dna.outcomes import survival
from draft_dna.outcomes.tiers import TIERS

log = get_logger(__name__)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
CATEGORICAL = {"vorp": "#2a78d6", "blend": "#eb6834", "factor": "#1baf7a"}
CAND_LABEL = {"vorp": "A: VORP", "blend": "B: VORP + WS blend", "factor": "C: Factor score"}
# Ordinal blue ramp (light -> dark), lightest step still >= 2:1 on the surface.
ORDINAL_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ORDINAL_6 = ["#d9d8d3", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_validation(res: pd.DataFrame, path: Path) -> None:
    checks = list(dict.fromkeys(res["check"]))
    fig, axes = plt.subplots(
        len(checks), 1, figsize=(9, 1.25 * len(checks) + 1.2), facecolor=SURFACE
    )
    for ax, check in zip(axes, checks, strict=True):
        sub = res[res["check"] == check].set_index("candidate")
        for i, cand in enumerate(CAND_LABEL):
            r = sub.loc[cand]
            y = len(CAND_LABEL) - 1 - i
            ax.plot(
                [r.ci_low, r.ci_high],
                [y, y],
                color=CATEGORICAL[cand],
                linewidth=2,
                solid_capstyle="round",
            )
            ax.scatter(
                [r.estimate],
                [y],
                s=46,
                color=CATEGORICAL[cand],
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
            )
            ax.text(r.ci_high, y, f"  {r.estimate:.3f}", va="center", fontsize=8.5, color=TEXT_2)
        ax.set_yticks([])
        lo, hi = sub["ci_low"].min(), sub["ci_high"].max()
        pad = (hi - lo) * 0.35 + 0.002
        ax.set_xlim(lo - pad * 0.3, hi + pad)
        ax.set_title(check, loc="left", fontsize=10, color=TEXT, pad=4)
        _style(ax)
    handles = [
        Line2D([], [], marker="o", color=CATEGORICAL[c], linewidth=2, label=CAND_LABEL[c])
        for c in CAND_LABEL
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.01, 0.985),
        labelcolor=TEXT,
    )
    fig.suptitle(
        "Which definition of season value tracks awards and contracts?",
        x=0.01,
        y=1.03,
        ha="left",
        fontsize=13,
        color=TEXT,
    )
    fig.text(
        0.01,
        -0.01,
        "Draft classes 1996-2016, measured 10 seasons after the draft. Lines are 95% "
        "bootstrap intervals (1,000 resamples of players).",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_survival(km: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for color, (_, _, band) in zip(ORDINAL_4, survival.PICK_BANDS, strict=True):
        g = km[km["band"] == band]
        med = g["median"].iloc[0]
        ax.fill_between(
            g["seasons"], g["ci_low"], g["ci_high"], color=color, alpha=0.18, linewidth=0
        )
        ax.step(
            g["seasons"],
            g["survival"],
            where="post",
            color=color,
            linewidth=2,
            label=f"{band}: median {med:.0f} seasons",
        )
    ax.axhline(0.5, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(20.2, 0.515, "half have left", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Seasons after the draft", color=TEXT_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=TEXT)
    ax.set_title(
        "Share of picks whose NBA career lasts beyond k seasons",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.text(
        0.01,
        -0.02,
        "Kaplan-Meier estimates, drafts 1996-2025. Active careers are censored "
        "(counted while observed). Shaded: 95% CI.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_tiers(co: pd.DataFrame, path: Path) -> pd.DataFrame:
    d = co[co["drafted"] & co["draft_year"].between(1996, 2016)].copy()
    d["band"] = survival.pick_band(d["pick_overall"])
    share = pd.crosstab(d["band"], d["tier"], normalize="index").reindex(
        index=[b for _, _, b in survival.PICK_BANDS], columns=TIERS, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    left = np.zeros(len(share))
    y = np.arange(len(share))[::-1]
    for color, tier in zip(ORDINAL_6, TIERS, strict=True):
        w = share[tier].to_numpy()
        ax.barh(y, w, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62)
        for yi, li, wi in zip(y, left, w, strict=True):
            if wi >= 0.06:
                dark = color in ORDINAL_6[3:]
                ax.text(
                    li + wi / 2,
                    yi,
                    f"{wi:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if dark else TEXT,
                )
        left += w
    ax.set_yticks(y, share.index, fontsize=9.5, color=TEXT)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ORDINAL_6]
    ax.legend(
        handles,
        TIERS,
        ncol=6,
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(0, 1.16),
        handlelength=1,
        labelcolor=TEXT,
    )
    ax.set_title("Career tier by draft position", loc="left", fontsize=13, color=TEXT, pad=42)
    fig.text(
        0.01,
        -0.03,
        "Drafts 1996-2016 (10+ seasons observed). Tier = best 3-season value; "
        "cutoffs calibrated to All-NBA, All-Star, starter and rotation roles.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return share


def run(s: Settings | None = None) -> None:
    s = s or get_settings()
    out = s.paths.reports / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    res = outcomes_run.validation_results(s)
    plot_validation(res, out / "value_definition_validation.png")
    careers = read_table("modeled", "outcomes", "careers", s)
    km = survival.kaplan_meier(careers, horizon=25)
    plot_survival(km, out / "career_survival_by_pick.png")
    co = read_table("modeled", "outcomes", "career_outcomes", s)
    share = plot_tiers(co, out / "tiers_by_pick.png")
    cox, conc = survival.cox(careers)
    params = json.loads(
        table_path("modeled", "outcomes", "params", s).with_suffix(".json").read_text()
    )

    md = [
        "# Phase 2 results\n",
        "## Validation of season-value definitions\n",
        "Draft classes 1996-2016, 10 seasons after the draft; 95% bootstrap CIs.\n",
        res.round(3).to_markdown(index=False),
        "\n## Tier cutoffs (best 3-season blend value)\n",
        pd.DataFrame(
            {"tier": params["tier_names"][1:], "min_peak3": params["tier_cuts_peak3"]}
        ).to_markdown(index=False),
        f"\nAgreement with award/role anchors: {params['tier_anchor_agreement']:.1%} exact.\n",
        "\n## Tier mix by pick band (drafts 1996-2016)\n",
        (share * 100).round(0).to_markdown(),
        "\n## Career length (Kaplan-Meier medians, seasons)\n",
        km.groupby("band")[["n", "median"]]
        .first()
        .reindex([b for _, _, b in survival.PICK_BANDS])
        .to_markdown(),
        f"\n## Cox model (concordance {conc:.3f})\n",
        cox.round(3).to_markdown(index=False),
    ]
    (out / "phase2_results.md").write_text("\n".join(md) + "\n")
    res.to_csv(out / "value_definition_validation.csv", index=False)
    log.info("phase 2 report written to %s", out)
