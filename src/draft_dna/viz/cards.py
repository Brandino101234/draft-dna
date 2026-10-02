"""Prospect cards: one shareable PNG per player.

Layout (1600 x 1000 px):
- header: name, draft slot, school, age, size, grade status
- left: the player's college shot map next to his top-3 comps' college shot maps
- right: outcome range (floor / median / ceiling) with tier lines and tier probabilities,
  grade and confidence, then either the trajectory vs projected band (2022-2025 players)
  or "plays like" NBA style comps (2026 rookies)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from draft_dna.config import Settings
from draft_dna.eval import metrics as M
from draft_dna.eval.phase3 import tier_cuts
from draft_dna.features import shot_xy
from draft_dna.grading.extras import nba_maps
from draft_dna.grading.run import QCOLS
from draft_dna.ingest.storage import read_table
from draft_dna.models.projections import tier_probabilities
from draft_dna.outcomes.tiers import TIERS
from draft_dna.viz.court import draw_folded_half_court

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRADE_COLOR = {"A": "#1baf7a", "B": "#2a78d6", "C": "#eda100", "D": "#e34948", "-": "#a3a29d"}


@dataclass
class CardData:
    grades: pd.DataFrame
    players: pd.DataFrame
    feats: pd.DataFrame
    maps: pd.DataFrame
    comps: pd.DataFrame
    plays_like: pd.DataFrame
    bands: pd.DataFrame
    peaks: pd.DataFrame
    cuts: list[float]
    nba_maps: pd.DataFrame

    @classmethod
    def load(cls, s: Settings) -> CardData:
        otn = read_table("modeled", "outcomes", "outcomes_through_n", s)
        return cls(
            grades=read_table("modeled", "grading", "grades", s).set_index("bbref_id"),
            players=read_table("modeled", "core", "players", s).set_index("bbref_id"),
            feats=read_table("modeled", "features", "predraft", s).set_index("bbref_id"),
            maps=read_table("modeled", "features", "shot_maps", s).set_index("bbref_id"),
            comps=read_table("modeled", "projections", "comps", s),
            plays_like=read_table("modeled", "grading", "plays_like", s),
            bands=read_table("modeled", "grading", "trajectory_bands", s),
            peaks=otn.pivot(index="bbref_id", columns="n", values="peak3_blend"),
            cuts=tier_cuts(s),
            nba_maps=nba_maps(s),
        )


def _height(inches: object) -> str:
    if inches is None or pd.isna(inches):
        return "height ?"
    v = float(inches)  # type: ignore[arg-type]
    return f"{int(v // 12)}'{v % 12:.1f}\"".replace('.0"', '"')


def _fmt(value: object, fmt: str, missing: str = "?") -> str:
    return missing if value is None or pd.isna(value) else format(float(value), fmt)  # type: ignore[arg-type]


def _shot_map(
    ax: Axes, d: CardData, pid: str, title: str, big: bool = False, nba: bool = False
) -> None:
    maps = d.nba_maps if nba else d.maps
    if pid in maps.index:
        grid = maps.loc[pid].filter(regex=r"^c\d+$").to_numpy(dtype=float)
        grid = grid.reshape(len(shot_xy.GRID_X), len(shot_xy.GRID_Y))
        ax.imshow(
            grid.T,
            origin="lower",
            cmap="Blues",
            interpolation="bilinear",
            extent=(-0.5, 25.5, shot_xy.GRID_Y[0] - 0.5, shot_xy.GRID_Y[-1] + 0.5),
        )
        draw_folded_half_court(ax, lw=0.8 if big else 0.6)
    else:
        draw_folded_half_court(ax, lw=0.6)
        ax.text(
            12.5,
            12,
            "no college\nshot locations",
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )
    ax.set_title(title, fontsize=11 if big else 8.5, color=TEXT, loc="left")


def _card_comps(d: CardData, pid: str, k: int = 3) -> pd.DataFrame:
    c = d.comps[d.comps["bbref_id"] == pid].sort_values("rank")
    with_map = c[c["comp_id"].isin(d.maps.index)]
    return (with_map if len(with_map) >= k else c).head(k)


def _range(ax: Axes, d: CardData, pid: str) -> None:
    g = d.grades.loc[pid]
    q = g[[f"post_{c}" for c in QCOLS]].to_numpy(dtype=float)[None, :]
    floor, med, ceil = q[0, M.qidx(M.FLOOR)], q[0, M.qidx(M.MEDIAN)], q[0, M.qidx(M.CEILING)]
    top = max(ceil * 1.15, d.cuts[-1] + 0.8)
    for c, label in zip(d.cuts[1:], TIERS[2:], strict=True):
        ax.axvline(c, color=GRID, linewidth=1, zorder=0)
        ax.text(c + 0.03, 1.32, label, fontsize=7.5, color=TEXT_2)
    ax.hlines(0.6, floor, ceil, color=BLUE, linewidth=9, alpha=0.35)
    ax.scatter([med], [0.6], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(floor, 0.15, f"floor {floor:.1f}", fontsize=8, color=TEXT_2, ha="center")
    ax.text(med, 0.95, f"median {med:.1f}", fontsize=9, color=TEXT, ha="center")
    ax.text(ceil, 0.15, f"ceiling {ceil:.1f}", fontsize=8, color=TEXT_2, ha="center")
    ax.set_xlim(0, top)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.set_title(
        "Career peak outcome (best 3-season value through year 8)",
        fontsize=10,
        color=TEXT,
        loc="left",
    )


def _tiers(ax: Axes, d: CardData, pid: str) -> None:
    g = d.grades.loc[pid]
    q = g[[f"post_{c}" for c in QCOLS]].to_numpy(dtype=float)[None, :]
    p = tier_probabilities(q, d.cuts)[0]
    y = np.arange(len(TIERS))[::-1]
    ax.barh(y, p, color=BLUE, height=0.6)
    for yi, v in zip(y, p, strict=True):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, TIERS, fontsize=8.5, color=TEXT)
    ax.set_xlim(0, max(p.max() * 1.3, 0.3))
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Tier probabilities", fontsize=10, color=TEXT, loc="left")


def _grade(ax: Axes, d: CardData, pid: str) -> None:
    g = d.grades.loc[pid]
    ax.axis("off")
    letter = str(g["grade"])
    ax.text(
        0.0,
        0.62,
        letter,
        fontsize=40,
        fontweight="bold",
        color=GRADE_COLOR.get(letter, TEXT),
        va="center",
    )
    ax.text(0.22, 0.78, str(g["status"]), fontsize=10.5, color=TEXT, va="center")
    explain = (
        "Projection only: no NBA games yet"
        if letter == "-"
        else "vs draft-night range: A > ceiling · B > median\nC > floor · D < floor"
    )
    ax.text(0.22, 0.56, explain, fontsize=7.5, color=TEXT_2, va="center", wrap=True)
    conf = float(g["confidence"])
    ax.add_patch(Rectangle((0.22, 0.18), 0.76, 0.12, color=GRID, transform=ax.transAxes))
    ax.add_patch(Rectangle((0.22, 0.18), 0.76 * conf, 0.12, color=BLUE, transform=ax.transAxes))
    ax.text(
        0.22,
        0.02,
        f"Confidence {conf:.0%}  ·  weight on NBA data {g['data_weight']:.0%}",
        fontsize=8,
        color=TEXT_2,
    )


def _trajectory(ax: Axes, d: CardData, pid: str) -> None:
    b = d.bands[d.bands["bbref_id"] == pid].sort_values("n")
    seasons = int(d.grades.loc[pid, "seasons"])
    ax.fill_between(
        b["n"],
        b["floor"],
        b["ceiling"],
        color=BLUE,
        alpha=0.15,
        linewidth=0,
        label="Projected range (draft night)",
    )
    ax.plot(
        b["n"], b["median"], color=BLUE, linewidth=1.5, linestyle="--", label="Projected median"
    )
    if seasons > 0 and pid in d.peaks.index:
        n = np.arange(1, seasons + 1)
        ax.plot(
            n,
            d.peaks.loc[pid, n].to_numpy(dtype=float),
            color=ORANGE,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Actual",
        )
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Seasons after the draft", fontsize=8.5, color=TEXT_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=TEXT)
    ax.set_title("Peak value so far vs projected band", fontsize=10, color=TEXT, loc="left")


def _plays_like(ax: Axes, d: CardData, pid: str) -> None:
    ax.axis("off")
    pl = d.plays_like[d.plays_like["bbref_id"] == pid].sort_values("rank")
    ax.text(0, 0.95, "Plays like (NBA early-career shot style)", fontsize=10, color=TEXT, va="top")
    ax.text(
        0,
        0.80,
        "Stylistic only: where he shoots from, not how good he will be.",
        fontsize=7.5,
        color=TEXT_2,
        va="top",
    )
    if pl.empty:
        ax.text(0, 0.6, "No shot-location data", fontsize=9, color=TEXT_2)
    for i, r in enumerate(pl.itertuples()):
        ax.text(0, 0.62 - 0.13 * i, f"{r.rank}. {r.plays_like_name}", fontsize=10, color=TEXT)
        ax.text(0.75, 0.62 - 0.13 * i, f"{r.style_similarity:.2f}", fontsize=9, color=TEXT_2)


def render(d: CardData, pid: str, path: Path) -> Path:
    p, f, g = d.players.loc[pid], d.feats.loc[pid], d.grades.loc[pid]
    fig = plt.figure(figsize=(16, 10), facecolor=SURFACE)
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[0.5, 2.1, 1.25, 1.25],
        hspace=0.6,
        wspace=0.45,
        width_ratios=[1, 1, 1.15, 1.15, 1, 1],
    )
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.85, str(p["player_name"]), fontsize=26, fontweight="bold", color=TEXT, va="top")
    school = p.get("college_name") or p.get("pre_draft_org") or ""
    head.text(
        0,
        0.2,
        f"{int(p['draft_year'])} draft · pick #{int(p['pick_overall'])} · "
        f"{p['team_id']} · {school} · age {_fmt(f['age_at_draft'], '.1f')} · "
        f"{_height(f['height_in'])} · {str(f['position']).capitalize()}",
        fontsize=12,
        color=TEXT_2,
        va="top",
    )
    head.text(
        1, 0.85, "DRAFT DNA", fontsize=13, color=BLUE, fontweight="bold", ha="right", va="top"
    )

    main = fig.add_subplot(gs[1, 0:2])
    if pid not in d.maps.index and pid in d.nba_maps.index:
        _shot_map(main, d, pid, "NBA shot map (no college data)", big=True, nba=True)
    else:
        _shot_map(main, d, pid, "College shot map", big=True)
    comps = _card_comps(d, pid)
    for i, c in enumerate(comps.itertuples()):
        ax = fig.add_subplot(gs[2 + i // 2, i % 2]) if i < 2 else fig.add_subplot(gs[3, 0])
        use_nba = c.comp_id not in d.maps.index and c.comp_id in d.nba_maps.index
        _shot_map(
            ax,
            d,
            c.comp_id,
            nba=use_nba,
            title=f"{c.comp_name} ('{c.comp_draft_year % 100:02d})"
            f"{' · NBA map' if use_nba else ''}\n"
            f"sim {c.similarity:.0f} · yr-6 peak {c.comp_peak6:.1f}",
        )
    _range(fig.add_subplot(gs[1, 2:6]), d, pid)
    _tiers(fig.add_subplot(gs[2:4, 2:4]), d, pid)
    _grade(fig.add_subplot(gs[2, 4:6]), d, pid)
    if int(g["seasons"]) > 0:
        _trajectory(fig.add_subplot(gs[3, 4:6]), d, pid)
    else:
        _plays_like(fig.add_subplot(gs[3, 4:6]), d, pid)
    fig.text(
        0.01,
        0.005,
        "Projection: draft-slot history + conformal calibration, updated with NBA "
        "play by Bayesian updating. Comps: pre-draft stats, age and size.",
        fontsize=8,
        color=TEXT_2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def render_many(s: Settings, ids: list[str], out_dir: Path) -> list[Path]:
    d = CardData.load(s)
    return [render(d, pid, out_dir / f"{pid}.png") for pid in ids if pid in d.grades.index]
