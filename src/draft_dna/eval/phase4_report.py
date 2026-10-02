"""Phase 4 report: Shot DNA coverage, styles, and example shot charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from draft_dna.config import Settings, get_settings
from draft_dna.features import shot_styles, shot_xy
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger
from draft_dna.viz.court import draw_folded_half_court

log = get_logger(__name__)

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
EXAMPLES = ["youngtr01", "willizi01", "moranja01", "halibty01", "bookede01", "townska01",
            "flaggco01", "dybanaj01"]  # fmt: skip


def coverage_by_class(s: Settings) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s)
    tier_a = read_table("modeled", "features", "shot_dna", s).set_index("bbref_id")
    cov = read_table("modeled", "features", "shot_coverage", s).set_index("bbref_id")
    d = players[players["drafted"] & (players["prospect_source"] == "college")].set_index(
        "bbref_id"
    )
    d = d[d["draft_year"] >= 2008]
    d["has_shot_type"] = d.index.isin(tier_a.index[tier_a["shot_fga"] >= 100])
    d["has_pbp"] = d.index.isin(cov.index[cov["pbp_fga"] >= 100])
    d["xy_eligible"] = d.index.isin(cov.index[cov["xy_eligible"]])
    return (
        d.groupby("draft_year")
        .agg(
            college_picks=("player_name", "size"),
            shot_type=("has_shot_type", "mean"),
            espn_pbp=("has_pbp", "mean"),
            xy_eligible=("xy_eligible", "mean"),
        )
        .reset_index()
    )


def plot_coverage(cov: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    series = [
        ("shot_type", "Shot-type mix (>=100 FGA)", "#2a78d6"),
        ("espn_pbp", "ESPN play-by-play (>=100 FGA)", "#eb6834"),
        ("xy_eligible", "x/y eligible (>=100 FGA, >=40% covered)", "#1baf7a"),
    ]
    for col, label, color in series:
        ax.plot(
            cov["draft_year"],
            cov[col],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xticks(range(int(cov["draft_year"].min()), int(cov["draft_year"].max()) + 1, 2))
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.set_xlabel("Draft class", fontsize=9, color=TEXT_2)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=3,
        labelcolor=TEXT,
    )
    ax.set_title(
        "Share of drafted college players with each tier of shot data",
        loc="left",
        fontsize=13,
        color=TEXT,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    maps: pd.DataFrame, weights: pd.DataFrame, labels: pd.DataFrame, names: pd.Series, path: Path
) -> None:
    ids = [i for i in EXAMPLES if i in maps.index]
    cols = 4
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 4.1 * rows), facecolor=SURFACE)
    for ax, pid in zip(
        np.atleast_1d(axes).ravel(), ids + [None] * (rows * cols - len(ids)), strict=True
    ):
        if pid is None:
            ax.axis("off")
            continue
        grid = maps.loc[pid].filter(regex=r"^c\d+$").to_numpy(dtype=float)
        grid = grid.reshape(len(shot_xy.GRID_X), len(shot_xy.GRID_Y))
        ax.imshow(
            grid.T,
            origin="lower",
            cmap="Blues",
            interpolation="bilinear",
            extent=(-0.5, 25.5, shot_xy.GRID_Y[0] - 0.5, shot_xy.GRID_Y[-1] + 0.5),
        )
        draw_folded_half_court(ax)
        w = weights.loc[pid].sort_values(ascending=False).head(2)
        mix = "\n".join(
            f"{v:.0%} {labels.set_index('style').loc[k, 'label']}" for k, v in w.items()
        )
        ax.set_title(f"{names.get(pid, pid)}\n{mix}", fontsize=8.5, color=TEXT, loc="left")
    fig.suptitle(
        "College shot maps and style mix (final pre-draft data)",
        x=0.01,
        ha="left",
        fontsize=13,
        color=TEXT,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(s: Settings | None = None) -> dict[str, object]:
    s = s or get_settings()
    out = s.paths.reports / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    cov = coverage_by_class(s)
    plot_coverage(cov, out / "shot_data_by_draft_class.png")

    maps = read_table("modeled", "features", "shot_maps", s).set_index("bbref_id")
    k, curve = shot_styles.choose_k(maps)
    model = shot_xy.fit_styles(maps, k=k)
    labels = shot_styles.describe_styles(model)
    weights = shot_xy.style_weights(model, maps)
    write_table(weights.reset_index(names="bbref_id"), "modeled", "features", "shot_styles", s)
    shot_styles.plot_styles(
        model,
        labels,
        out / "shot_styles.png",
        f"{k} college shot styles (NMF on {len(maps)} players' shot maps)",
    )
    names = read_table("modeled", "core", "players", s).set_index("bbref_id")["player_name"]
    plot_examples(maps, weights, labels, names, out / "example_shot_maps.png")
    (s.paths.modeled / "features" / "shot_styles_meta.json").write_text(
        json.dumps({"k": k, "labels": labels["label"].tolist()}, indent=2)
    )
    md = [
        "# Phase 4 results\n",
        "## Shot data by draft class (drafted college players)\n",
        cov.round(3).to_markdown(index=False),
        "\n## Choosing the number of styles\n",
        curve.round(4).to_markdown(index=False),
        f"\nChosen k = {k}.\n",
        "\n## Styles\n",
        labels.round(3).to_markdown(index=False),
    ]
    (out / "phase4_results.md").write_text("\n".join(md) + "\n")
    log.info("phase 4 report: k=%d, %d players with style maps", k, len(maps))
    return {"coverage": cov, "k": k, "labels": labels, "curve": curve}
