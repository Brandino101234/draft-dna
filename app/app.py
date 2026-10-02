"""Draft DNA: Streamlit app.

Run with `make app` (or `uv run streamlit run app/app.py`). Reads the built DuckDB tables
and rendered cards; `make refresh` updates them during the season.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # deployed app: package not installed
# With no local build (e.g. Streamlit Community Cloud), read the committed app bundle.
if not (ROOT / "data" / "modeled" / "grading" / "grades.parquet").exists():
    os.environ.setdefault("DRAFT_DNA_MODELED", str(ROOT / "app" / "bundle"))

from draft_dna.config import get_settings  # noqa: E402
from draft_dna.ingest.storage import read_table  # noqa: E402
from draft_dna.viz import cards as card_viz  # noqa: E402

st.set_page_config(page_title="Draft DNA", page_icon="🏀", layout="wide")
S = get_settings()
LOCAL_CARDS = S.paths.modeled / "cards"
CARD_DIR = LOCAL_CARDS if LOCAL_CARDS.exists() else Path(tempfile.gettempdir()) / "draft_dna_cards"
GRADE_COLORS = {"A": "#1baf7a", "B": "#2a78d6", "C": "#eda100", "D": "#e34948", "-": "#a3a29d"}


@st.cache_data(show_spinner=False)
def load() -> dict[str, pd.DataFrame]:
    return {
        "grades": read_table("modeled", "grading", "grades", S),
        "players": read_table("modeled", "core", "players", S),
        "comps": read_table("modeled", "projections", "comps", S),
        "plays_like": read_table("modeled", "grading", "plays_like", S),
        "style_map": read_table("modeled", "grading", "style_map", S),
        "tracker": read_table("modeled", "grading", "rookie_tracker", S),
    }


@st.cache_resource(show_spinner=False)
def card_data() -> card_viz.CardData:
    return card_viz.CardData.load(S)


@st.cache_data(show_spinner="Drawing card...")
def card_path(pid: str) -> str:
    path = CARD_DIR / f"{pid}.png"
    if not path.exists():  # render on demand (always, on the deployed app)
        path.parent.mkdir(parents=True, exist_ok=True)
        card_viz.render(card_data(), pid, path)
    return str(path)


def label(row: pd.Series) -> str:
    return f"{row['player_name']} ({int(row['draft_year'])}, #{int(row['pick'])})"


data = load()
grades = data["grades"].sort_values(["draft_year", "pick"], ascending=[False, True])
options = {label(r): r["bbref_id"] for _, r in grades.iterrows()}

st.sidebar.title("Draft DNA")
st.sidebar.caption("NBA draft comps, outcome ranges and grades from pre-draft data.")
page = st.sidebar.radio("View", ["Player card", "Compare", "Style map", "2026 tracker", "About"])


def summary(pid: str) -> None:
    g = grades.set_index("bbref_id").loc[pid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Grade",
        g["grade"],
        help="vs draft-night range: A > ceiling, B > median, C > floor, D < floor",
    )
    c2.metric("Status", g["status"])
    c3.metric("Confidence", f"{g['confidence']:.0%}")
    c4.metric(
        "Projected peak (median)",
        f"{g['current_median']:.2f}",
        delta=f"{g['current_median'] - g['projected_median']:+.2f} vs draft night",
    )


if page == "Player card":
    choice = st.selectbox("Search a player", list(options), index=0)
    pid = options[choice]
    summary(pid)
    st.image(card_path(pid), use_container_width=True)
    left, right = st.columns(2)
    comps = data["comps"][data["comps"]["bbref_id"] == pid].sort_values("rank")
    left.subheader("Top 15 comps (pre-draft stats, age, size)")
    left.dataframe(
        comps[
            [
                "rank",
                "comp_name",
                "comp_draft_year",
                "comp_pick",
                "similarity",
                "comp_peak6",
                "driving_features",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    pl = data["plays_like"][data["plays_like"]["bbref_id"] == pid].sort_values("rank")
    right.subheader("Plays like (stylistic only)")
    right.caption(
        "Nearest NBA early-career shot styles. Says where he shoots from, not how good he will be."
    )
    right.dataframe(
        pl[["rank", "plays_like_name", "style_similarity", "basis"]],
        hide_index=True,
        use_container_width=True,
    )

elif page == "Compare":
    names = list(options)
    a, b = st.columns(2)
    pa = options[a.selectbox("Player A", names, index=0)]
    pb = options[b.selectbox("Player B", names, index=1)]
    g = grades.set_index("bbref_id")
    cols = [
        "status",
        "grade",
        "confidence",
        "projected_floor",
        "projected_median",
        "projected_ceiling",
        "current_floor",
        "current_median",
        "current_ceiling",
    ]
    table = g.loc[[pa, pb], cols].T.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    table.columns = [g.loc[pa, "player_name"], g.loc[pb, "player_name"]]
    st.dataframe(table, use_container_width=True)
    a.image(card_path(pa), use_container_width=True)
    b.image(card_path(pb), use_container_width=True)

elif page == "Style map":
    st.subheader("Every player's shot style, in 2D")
    st.caption(
        "t-SNE layout of each player's mix of the six NMF shot styles (college maps and NBA "
        "first-four-season maps). Nearby points shoot from similar places."
    )
    sm = data["style_map"].dropna(subset=["player_name"])
    source = st.multiselect(
        "Show", sorted(sm["source"].unique()), default=sorted(sm["source"].unique())
    )
    sm = sm[sm["source"].isin(source)]
    highlight = st.multiselect(
        "Highlight players",
        sorted(sm["player_name"].unique()),
        default=[
            n
            for n in ["AJ Dybantsa", "Cooper Flagg", "Victor Wembanyama"]
            if n in set(sm["player_name"])
        ],
    )
    fig = px.scatter(
        sm,
        x="x",
        y="y",
        color="dominant_style",
        symbol="source",
        opacity=0.55,
        hover_name="player_name",
        hover_data={"x": False, "y": False},
        height=650,
    )
    fig.update_traces(marker={"size": 6})
    hi = sm[sm["player_name"].isin(highlight)]
    fig.add_scatter(
        x=hi["x"],
        y=hi["y"],
        mode="markers+text",
        text=hi["player_name"],
        textposition="top center",
        name="highlighted",
        marker={"size": 13, "color": "white", "line": {"width": 2, "color": "black"}},
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="Dominant style")
    st.plotly_chart(fig, use_container_width=True)

elif page == "2026 tracker":
    st.subheader("2026 class: projection vs reality")
    t = data["tracker"]
    played = int((t["games"] > 0).sum())
    if played == 0:
        st.info(
            "The 2026-27 season hasn't started yet. Projected rookie-season ranges are below; "
            "games are added as the season goes on."
        )
    st.caption(
        "Rookie-season value (VORP + Win Shares blend). Bar = draft-night range (25th-90th "
        "percentile); dot = full-season pace from games played so far."
    )
    top = t.head(30).iloc[::-1]
    fig = px.scatter(
        top,
        x="value_pace",
        y="player_name",
        color="status",
        height=750,
        labels={"value_pace": "Season value pace", "player_name": ""},
    )
    for _, r in top.iterrows():
        fig.add_shape(
            type="line",
            x0=r["projected_floor"],
            x1=r["projected_ceiling"],
            y0=r["player_name"],
            y1=r["player_name"],
            line={"width": 6, "color": "rgba(42,120,214,0.3)"},
        )
    fig.add_scatter(
        x=top["projected_median"],
        y=top["player_name"],
        mode="markers",
        marker={"symbol": "line-ns", "size": 14, "color": "#2a78d6", "line": {"width": 2}},
        name="projected median",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        t[
            [
                "pick",
                "player_name",
                "team",
                "games",
                "minutes",
                "projected_floor",
                "projected_median",
                "projected_ceiling",
                "value_to_date",
                "value_pace",
                "status",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

else:
    # Render README images from the repo (relative links don't resolve inside Streamlit).
    chunk: list[str] = []
    for line in (ROOT / "README.md").read_text().splitlines():
        m = re.fullmatch(r"!\[(.*)\]\((.+)\)", line.strip())
        if m and (ROOT / m.group(2)).exists():
            st.markdown("\n".join(chunk))
            chunk = []
            st.image(str(ROOT / m.group(2)), caption=m.group(1))
        else:
            chunk.append(line)
    st.markdown("\n".join(chunk))
