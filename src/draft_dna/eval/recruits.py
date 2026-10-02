"""Were high-school recruiting rankings over- or underrated on draft night? (D035)

For drafted college players with an as-of draft-night projection, PIT says where the
career (graded peak, D033/D034) landed within the range expected for his draft slot:
0.5 = as expected, above = beat the slot. If top recruits average above 0.5, teams
underrated them relative to slot; below 0.5, overrated. Pre-registered in DECISIONS D035.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from draft_dna.config import Settings
from draft_dna.eval.phase6 import pit
from draft_dna.grading.run import QCOLS, asof_prior_grids, peaks_wide
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

GROUPS = ["RSCI 1-10", "RSCI 11-25", "RSCI 26-50", "RSCI 51-100", "Unranked"]
HORIZONS = {4: (2005, 2021), 8: (2005, 2017)}  # primary, secondary (D035)
N_BOOT = 2000
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def recruit_group(rank: pd.Series) -> pd.Series:
    bins = pd.cut(rank, [0, 10, 25, 50, 100], labels=GROUPS[:4])
    return bins.astype("string").fillna("Unranked")


def frame(s: Settings, horizon: int) -> pd.DataFrame:
    """One row per drafted college player in the horizon's classes: recruit group, pick,
    and PIT of his graded peak at `horizon` within the as-of draft-night range."""
    lo, hi = HORIZONS[horizon]
    grids = asof_prior_grids(s, horizon=horizon)  # as-of only: no retrospective classes
    wide = peaks_wide(s)
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    feats = read_table("modeled", "features", "predraft", s).set_index("bbref_id")
    d = pd.DataFrame(index=grids.index)
    d["player_name"] = d.index.map(players["player_name"])
    d["draft_year"] = d.index.map(players["draft_year"]).astype(int)
    d["pick"] = d.index.map(players["pick_overall"]).astype(int)
    d["source"] = d.index.map(feats["prospect_source"])
    d["recruit_rank"] = d.index.map(feats["recruit_rank_top100"])
    d["y"] = wide[horizon].reindex(d.index) if horizon in wide else np.nan
    d = d[(d["source"] == "college") & d["draft_year"].between(lo, hi) & d["y"].notna()]
    d["pit"] = pit(grids.loc[d.index, QCOLS].to_numpy(), d["y"].to_numpy())
    d["group"] = recruit_group(d["recruit_rank"])
    d["horizon"] = horizon
    return d.reset_index(names="bbref_id")


def _boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(d: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for g in GROUPS:
        x = d.loc[d["group"] == g, "pit"].to_numpy()
        if len(x) == 0:
            continue
        lo, hi = _boot_ci(x, rng)
        rows.append(
            {
                "group": g,
                "n": len(x),
                "mean_pit": float(x.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "beat_median": float((x > 0.5).mean()),
                "avg_pick": float(d.loc[d["group"] == g, "pick"].mean()),
                "differs_from_slot": bool(lo > 0.5 or hi < 0.5),
            }
        )
    order = d["group"].map({g: i for i, g in enumerate(GROUPS)})
    rho, p = spearmanr(order, d["pit"])
    return pd.DataFrame(rows), {"spearman_rho": float(rho), "p_value": float(p), "n": len(d)}


def analyze(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players, tables, tests = [], [], []
    for h in HORIZONS:
        d = frame(s, h)
        players.append(d)
        for scope, sub in (("all picks", d), ("first round", d[d["pick"] <= 30])):
            t, test = summarize(sub)
            tables.append(t.assign(horizon=h, scope=scope))
            tests.append({"horizon": h, "scope": scope, **test})
    return (
        pd.concat(players, ignore_index=True),
        pd.concat(tables, ignore_index=True),
        pd.DataFrame(tests),
    )


def plot(table: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    t = table[table["scope"] == "all picks"]
    fig, (ax0, ax) = plt.subplots(
        1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 2.6]}
    )
    y = np.arange(len(GROUPS))[::-1]
    picks = t[t["horizon"] == 4].set_index("group").reindex(GROUPS)
    ax0.barh(y, picks["avg_pick"], color=BLUE, height=0.55)
    for yi, v in zip(y, picks["avg_pick"], strict=True):
        ax0.text(v + 0.8, yi, f"#{v:.0f}", va="center", fontsize=9, color=TEXT)
    ax0.set_yticks(y, [f"{g}  (n={int(n)})" for g, n in zip(GROUPS, _counts(t), strict=True)])
    ax0.set_xlim(0, 45)
    ax0.set_title("Where they were drafted\n(average pick)", fontsize=10, color=TEXT, loc="left")
    ax0.set_xticks([])
    for h, color, off in ((4, BLUE, 0.13), (8, ORANGE, -0.13)):
        r = t[t["horizon"] == h].set_index("group").reindex(GROUPS)
        ax.hlines(y + off, r["ci_lo"], r["ci_hi"], color=color, linewidth=2)
        ax.scatter(
            r["mean_pit"], y + off, s=40, color=color, edgecolor=SURFACE, linewidth=2, zorder=3
        )
        test = tests[(tests["horizon"] == h) & (tests["scope"] == "all picks")].iloc[0]
        ax.scatter(
            [],
            [],
            color=color,
            label=f"Year {h} (n={int(test['n'])}; rank trend rho={test['spearman_rho']:+.2f}, "
            f"p={test['p_value']:.2f})",
        )
    ax.axvline(0.5, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, [])
    ax.set_xlim(0.3, 0.7)
    ax.set_xlabel("Average PIT within draft-slot range (0.5 = career matched his slot)")
    ax.text(0.305, -0.75, "← fell short of slot", color=TEXT_2, fontsize=8.5)
    ax.text(0.695, -0.75, "beat slot →", color=TEXT_2, fontsize=8.5, ha="right")
    ax.set_ylim(-0.9, len(GROUPS) - 0.5)
    ax.set_title("Career vs draft slot, with 95% intervals", fontsize=10, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax0.set_ylim(ax.get_ylim())
    for a in (ax0, ax):
        a.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Teams already price in high-school recruiting rank: top recruits go ~20 picks "
        "earlier, then match their slot",
        fontsize=11.5,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _counts(t: pd.DataFrame) -> list[int]:
    r = t[t["horizon"] == 4].set_index("group").reindex(GROUPS)
    return r["n"].fillna(0).astype(int).tolist()


def _markdown(table: pd.DataFrame, tests: pd.DataFrame) -> str:
    lines = ["# Recruiting rank vs draft slot (D035)", ""]
    for _, test in tests.iterrows():
        h, scope = int(test["horizon"]), test["scope"]
        t = table[(table["horizon"] == h) & (table["scope"] == scope)]
        lines += [
            f"## Year {h}, {scope} (n={int(test['n'])}): Spearman rho = "
            f"{test['spearman_rho']:+.3f}, p = {test['p_value']:.3f}",
            "",
            "| Group | n | Avg pick | Mean PIT | 95% CI | Beat slot median |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in t.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n'])} | {r['avg_pick']:.1f} | {r['mean_pit']:.3f} | "
                f"{r['ci_lo']:.3f}-{r['ci_hi']:.3f} | {r['beat_median']:.0%} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(s: Settings) -> None:
    players, table, tests = analyze(s)
    write_table(players, "modeled", "recruits", "players", s)
    write_table(table, "modeled", "recruits", "summary", s)
    write_table(tests, "modeled", "recruits", "tests", s)
    out = s.paths.reports / "recruits"
    plot(table, tests, out / "recruit_rank_vs_slot.png")
    (out / "results.md").write_text(_markdown(table, tests))
    log.info("recruits: %s", tests.round(3).to_dict("records"))
