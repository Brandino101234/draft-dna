"""Phase 6 report: who beats their projection, situation effects, SHAP, verdict reversals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from draft_dna.config import Settings, get_settings
from draft_dna.eval import phase6
from draft_dna.eval import phase6_analysis as A
from draft_dna.ingest.storage import read_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#a3a29d"


def _style(ax: Axes, axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)  # type: ignore[arg-type]
    ax.set_axisbelow(True)


def add_segments(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = out["bbref_id"].map(feats["age_at_draft"])
    out["Pick"] = pd.cut(out["pick"], [0, 5, 14, 30, 60], labels=["1-5", "6-14", "15-30", "31-60"])
    out["Age at draft"] = pd.cut(
        out["age"],
        [0, 19.5, 20.5, 21.5, 22.5, 30],
        labels=["<19.5", "19.5-20.5", "20.5-21.5", "21.5-22.5", "22.5+"],
    )
    out["Era"] = pd.cut(
        out["draft_year"], [2000, 2007, 2014, 2021], labels=["2002-07", "2008-14", "2015-20"]
    )
    out["Background"] = out["prospect_source"].map(
        {"college": "College", "high_school": "High school", "other_team": "Intl / pro team"}
    )
    out["Position"] = out["position"].str.capitalize().replace({"Unknown": np.nan})
    return out


def plot_segments(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    groups = ["Pick", "Position", "Age at draft", "Background", "Era"]
    tables = [A.segment_rates(df, g).rename(columns={g: "group"}).assign(split=g) for g in groups]
    t = pd.concat(tables, ignore_index=True).dropna(subset=["group"])
    fig, ax = plt.subplots(figsize=(9, 0.33 * len(t) + 1.8), facecolor=SURFACE)
    y = np.arange(len(t))[::-1]
    ax.axvline(0.5, color=TEXT_2, linewidth=1)
    ax.hlines(y, t["pit_lo"], t["pit_hi"], color=BLUE, linewidth=2)
    ax.scatter(t["mean_pit"], y, s=36, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(
        y,
        [f"{s}: {g}  (n={n})" for s, g, n in zip(t["split"], t["group"], t["n"], strict=True)],
        fontsize=9,
        color=TEXT,
    )
    ax.set_xlabel(
        "Mean PIT: where actual outcomes landed in the draft-night range (0.5 = as projected)",
        fontsize=9,
        color=TEXT_2,
    )
    _style(ax)
    ax.set_title(
        "Who beats their draft-slot projection? (peak through year 6)",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return t


def plot_teams(g: pd.DataFrame, stats: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=SURFACE)
    y = np.arange(len(g))
    ax.axvline(stats["league_mean"], color=TEXT_2, linewidth=1)
    ax.hlines(y, g["raw_lo"], g["raw_hi"], color=MUTED, linewidth=2)
    ax.scatter(g["raw_mean"], y, s=30, color=MUTED, zorder=3, label="Raw team mean (95% CI)")
    ax.scatter(
        g["pooled"],
        y,
        s=34,
        color=BLUE,
        zorder=4,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="After partial pooling",
    )
    ax.set_yticks(
        y, [f"{t} (n={n})" for t, n in zip(g.index, g["n"], strict=True)], fontsize=8.5, color=TEXT
    )
    ax.set_xlabel(
        "Mean PIT of the franchise's picks (0.5 = as projected)", fontsize=9, color=TEXT_2
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
    ax.set_title(
        f"Do some teams develop picks better? Heterogeneity p = {stats['p_value']:.2f}",
        loc="left",
        fontsize=12.5,
        color=TEXT,
        pad=10,
    )
    fig.text(
        0.01,
        -0.02,
        "Drafts 2003-2020, peak through year 6. Team spread is no larger than "
        "luck produces, so partial pooling pulls every team to the league mean.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_situations(r: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(r) + 1.6), facecolor=SURFACE)
    y = np.arange(len(r))[::-1]
    colors = [BLUE if "pre-draft" in t else ORANGE for t in r["timing"]]
    ax.axvline(0, color=TEXT_2, linewidth=1)
    for yi, (_, row), c in zip(y, r.iterrows(), colors, strict=True):
        ax.hlines(yi, row["ipw_lo"], row["ipw_hi"], color=c, linewidth=2)
        ax.scatter(row["ipw_pit_diff"], yi, s=40, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(
            max(row["ipw_hi"], 0) + 0.008,
            yi,
            f"E-value {row['e_value']:.2f}",
            fontsize=8,
            color=TEXT_2,
            va="center",
        )
    ax.set_yticks(y, r["situation"], fontsize=9, color=TEXT)
    ax.set_xlabel(
        "Propensity-weighted difference in mean PIT (treated - untreated)", fontsize=9, color=TEXT_2
    )
    _style(ax)
    handles = [
        Line2D([], [], color=BLUE, marker="o", label="Known before the draft"),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="o",
            label="Happens after the draft (reverse causality likely)",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.16),
        ncol=2,
        labelcolor=TEXT,
    )
    ax.set_title(
        "Situation effects on beating the projection (associations, 95% CIs)",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


NAMES = {
    "weight_lb": "Weight (combine)",
    "height_in": "Height",
    "usg_pct": "Usage rate",
    "ast_per40": "Assists per 40",
    "career_ft_pct": "Career FT%",
    "trb_per40": "Rebounds per 40",
    "log_pick": "Draft pick (log)",
    "ts_rel": "Efficiency vs era",
    "team_sos": "Schedule strength",
    "tov_per40": "Turnovers per 40",
    "blk_per40": "Blocks per 40",
    "age_at_draft": "Age at draft",
    "stl_per40": "Steals per 40",
    "ast_pct": "Assist rate",
    "recruit_rank_top100": "Recruiting rank",
}


def plot_shap(r: dict[str, Any], path: Path, title: str) -> None:
    imp = r["importance"].head(12)[::-1]
    direction = r["direction"]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    colors = [BLUE if direction[f] >= 0 else ORANGE for f in imp.index]
    ax.barh(range(len(imp)), imp.to_numpy(), color=colors, height=0.6)
    ax.set_yticks(range(len(imp)), [NAMES.get(f, f) for f in imp.index], fontsize=9.5, color=TEXT)
    ax.set_xlabel("Mean |SHAP| (average push on predicted PIT)", fontsize=9, color=TEXT_2)
    _style(ax)
    handles = [
        Line2D([], [], color=BLUE, lw=6, label="Higher value -> beats projection more"),
        Line2D([], [], color=ORANGE, lw=6, label="Higher value -> falls short more"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
    ax.set_title(title, loc="left", fontsize=12.5, color=TEXT, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_reversals(table: pd.DataFrame, path: Path) -> None:
    pct = table / table.to_numpy().sum()
    fig, ax = plt.subplots(figsize=(6.6, 5), facecolor=SURFACE)
    ax.imshow(pct.to_numpy(), cmap="Blues", vmin=0, vmax=pct.to_numpy().max())
    for i in range(3):
        for j in range(3):
            v = int(table.iat[i, j])
            ax.text(
                j,
                i,
                f"{v}\n({pct.iat[i, j]:.0%})",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if pct.iat[i, j] > 0.3 else TEXT,
            )
    ax.set_xticks(range(3), table.columns, fontsize=9.5, color=TEXT)
    ax.set_yticks(range(3), table.index, fontsize=9.5, color=TEXT)
    ax.set_xlabel("Career Grade (year 8)", fontsize=10, color=TEXT_2)
    ax.set_ylabel("Year-4 Verdict", fontsize=10, color=TEXT_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "How often is the Year-4 Verdict overturned?", loc="left", fontsize=12.5, color=TEXT, pad=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(s: Settings | None = None) -> dict[str, Any]:
    s = s or get_settings()
    out_dir = s.paths.reports / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = phase6.run(s)
    feats = read_table("modeled", "features", "predraft", s).set_index("bbref_id")
    seg = plot_segments(add_segments(df, feats), out_dir / "who_beats_projection.png")
    team_tables = {n: A.team_effects(df, n) for n in (4, 6, 8)}
    plot_teams(*team_tables[6], out_dir / "team_effects.png")
    sit = A.situation_effects(df, feats)
    plot_situations(sit, out_dir / "situation_effects.png")
    shot = read_table("modeled", "features", "shot_dna", s).set_index("bbref_id")
    feats_all = feats.drop(columns=[c for c in shot.columns if c in feats]).join(shot)
    shap4 = A.shap_overperformance(df, feats_all, 4)
    plot_shap(
        shap4,
        out_dir / "shap_overperformance.png",
        "What predicts beating the draft-slot projection? (year 4, SHAP)",
    )
    rolling = {
        (n, pop): A.rolling_overperformance(
            df if pop == "all" else df[df["prospect_source"] == "college"], feats, n
        )
        for n in (4, 6)
        for pop in ("all", "college")
    }
    rev = A.verdict_reversals(df)
    plot_reversals(rev["table"], out_dir / "verdict_reversals.png")

    rows = [
        {
            "horizon": n,
            "players": pop,
            "pooled_spearman": r.attrs["pooled_spearman"],
            "perm_p": r.attrs["perm_p"],
            "n": r.attrs["n"],
            "positive_years": f"{int((r['spearman'] > 0).sum())}/{len(r)}",
        }
        for (n, pop), r in rolling.items()
    ]
    players = rev["players"]
    md = [
        "# Phase 6 results\n",
        "## Mean PIT by segment (year 6)\n",
        seg.round(3).to_markdown(index=False),
        "\n## Team heterogeneity\n",
        pd.DataFrame([{"horizon": n, **st} for n, (_, st) in team_tables.items()])
        .round(3)
        .to_markdown(index=False),
        "\n## Situation effects (propensity-weighted, year 6)\n",
        sit.round(3).to_markdown(index=False),
        "\n## Overperformance predictability\n",
        f"Leave-classes-out CV (year 4): Spearman {shap4['oof_spearman']:.3f}, "
        f"R2 {shap4['oof_r2']:.3f}.\n",
        "Strict rolling-origin check (each class predicted only from classes with known "
        "outcomes):\n",
        pd.DataFrame(rows).round(3).to_markdown(index=False),
        "\n## Year-4 Verdict vs Career Grade\n",
        rev["table"].to_markdown(),
        f"\nOverturned: {rev['overturned']:.1%} (95% CI {rev['overturned_ci'][0]:.1%}-"
        f"{rev['overturned_ci'][1]:.1%}) of {rev['n']} players.\n",
        "\n### Late bloomers (within band at year 4 -> beat ceiling at year 8)\n",
        players[players["change"] == "late bloomer"]
        .sort_values("actual8", ascending=False)
        .head(15)[["player_name", "draft_year", "pick", "actual4", "actual8"]]
        .round(2)
        .to_markdown(index=False),
        "\n### True late busts (within band at year 4 -> below floor at year 8)\n",
        players[(players["verdict4"] == "within band") & (players["verdict8"] == "below floor")][
            ["player_name", "draft_year", "pick", "actual4", "actual8"]
        ]
        .round(2)
        .to_markdown(index=False),
    ]
    (out_dir / "phase6_results.md").write_text("\n".join(md) + "\n")
    log.info("phase 6 report written to %s", out_dir)
    return {
        "segments": seg,
        "teams": team_tables,
        "situations": sit,
        "rolling": rows,
        "reversals": rev,
    }
