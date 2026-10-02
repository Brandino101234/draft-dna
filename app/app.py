"""Draft DNA: Streamlit app.

Run with `make app` (or `uv run streamlit run app/app.py`). Reads the built tables (or the
committed app bundle when there is no local build) and draws cards on demand;
`make refresh` updates them during the season.

Links: `?player=<bbref_id>` opens a player's card; `?page=redraft&year=2011` etc.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
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
PAGES = {
    "player": "Player card",
    "redraft": "Redraft",
    "steals": "Steals & busts",
    "teams": "Teams",
    "compare": "Compare",
    "styles": "Style map",
    "tracker": "2026 tracker",
    "about": "About",
}
GRADE_HELP = (
    "Career peak (best 3 seasons, incl. playoffs; All-Star/All-NBA/All-Defense/DPOY set a "
    "minimum) vs the draft-night range for his slot: A beat the ceiling (top 10%), B beat "
    "the median, C beat the floor (25th pct), D below the floor."
)
LIMITED_SOURCES = ("high_school", "other_team")


@st.cache_data(show_spinner=False)
def load() -> dict[str, pd.DataFrame]:
    return {
        "grades": read_table("modeled", "grading", "grades", S),
        "comps": read_table("modeled", "projections", "comps", S),
        "plays_like": read_table("modeled", "grading", "plays_like", S),
        "style_map": read_table("modeled", "grading", "style_map", S),
        "tracker": read_table("modeled", "grading", "rookie_tracker", S),
    }


@st.cache_resource(show_spinner=False)
def card_data() -> card_viz.CardData:
    return card_viz.CardData.load(S)


def card_path(pid: str) -> str:
    # Not cached: the file can disappear (temp-dir cleanup, rebuilds), so check each time.
    path = CARD_DIR / f"{pid}.png"
    if not path.exists():  # render on demand (always, on the deployed app)
        path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Drawing card..."):
            card_viz.render(card_data(), pid, path)
    return str(path)


def label(row: pd.Series) -> str:
    return f"{row['player_name']} ({int(row['draft_year'])}, #{int(row['pick'])})"


data = load()
grades = data["grades"].sort_values(["draft_year", "pick"], ascending=[False, True])
grades["value_vs_slot"] = grades["current_median"] - grades["projected_median"]
grades["finished"] = grades["status"].str.startswith("Career")
grades["round"] = np.where(grades["pick"] <= 30, "1st", "2nd")
# Redraft: each class re-ordered by (projected) career peak; moved = spots gained.
grades["redraft"] = (
    grades.sort_values(["current_median", "pick"], ascending=[False, True])
    .groupby("draft_year")
    .cumcount()
    .add(1)
)
grades["moved"] = (grades["pick"] - grades["redraft"]).astype(int)
G = grades.set_index("bbref_id")
options = {label(r): r["bbref_id"] for _, r in grades.iterrows()}
label_of = {v: k for k, v in options.items()}


def go_to_player(pid: str) -> None:
    # Widgets keep their own state, so navigation is applied at the top of the next run,
    # before any widget is drawn.
    st.session_state["goto"] = pid
    st.rerun()


HEADERS = {
    "player_name": "Player",
    "comp_draft_year": st.column_config.NumberColumn("Year", format="%d"),
    "comp_pick": "Pick",
    "pick": "Pick",
    "grade": "Grade",
    "status": "Status",
    "rank": "#",
}


def clickable(df: pd.DataFrame, key: str, **kwargs: object) -> None:
    """A table whose rows open that player's card (expects a bbref_id column)."""
    kwargs["column_config"] = {**HEADERS, **kwargs.get("column_config", {})}  # type: ignore[dict-item]
    event = st.dataframe(
        df.drop(columns="bbref_id"),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        **kwargs,  # type: ignore[arg-type]
    )
    rows = event.selection.rows  # type: ignore[attr-defined]
    if rows:
        go_to_player(df.iloc[rows[0]]["bbref_id"])


def year_range(key: str, default: tuple[int, int] = (1996, 2026)) -> tuple[int, int]:
    lo, hi = int(grades["draft_year"].min()), int(grades["draft_year"].max())
    picked = st.slider("Draft years", lo, hi, default, key=key)
    return int(picked[0]), int(picked[1])


# ----------------------------------------------------------------------- navigation
qp = st.query_params
slugs = list(PAGES)
target = st.session_state.pop("goto", None)
if target is None and "nav" not in st.session_state and qp.get("player") in G.index:
    target = qp.get("player")  # first load from a shared ?player= link
if target is not None:
    qp.clear()
    qp["player"] = target
    st.session_state["nav"] = PAGES["player"]
    st.session_state["pc_year"] = int(G.loc[target, "draft_year"])
    st.session_state["pc_player"] = label_of[target]
elif "nav" not in st.session_state:
    start = qp.get("page", "player")
    st.session_state["nav"] = PAGES.get(start, PAGES["player"])
st.sidebar.title("Draft DNA")
st.sidebar.caption("NBA draft comps, outcome ranges and grades from pre-draft data.")
page_name = st.sidebar.radio("View", list(PAGES.values()), key="nav")
page = slugs[list(PAGES.values()).index(page_name)]
if page != "player" and qp.get("page") != page:  # keep the URL in step with the sidebar
    qp.clear()
    qp["page"] = page


def summary(pid: str) -> None:
    g = G.loc[pid]
    c1, c2, c3, c4, c5 = st.columns([0.8, 2.2, 1.5, 1.5, 1])
    c1.metric("Grade", g["grade"], help=GRADE_HELP)
    c2.metric("Status", g["status"])
    tier_label = "Outcome tier" if g["finished"] else "Projected tier"
    c3.metric(
        tier_label,
        g["current_tier"],
        delta=f"draft night: {g['projected_tier']}",
        delta_color="off",
        help="Tier of the median projected peak (or the actual peak once his career is done).",
    )
    c4.metric(
        "All-Star or better",
        f"{g['current_p_all_star']:.0%}",
        delta=f"{(g['current_p_all_star'] - g['projected_p_all_star']) * 100:+.0f} pts vs "
        "draft night",
        help="Chance his best 3-season stretch reaches All-Star level or higher.",
    )
    c5.metric(
        "Confidence",
        f"{g['confidence']:.0%}",
        help="How much the range has narrowed since draft night (100% = career done).",
    )


# ----------------------------------------------------------------------- pages
if page == "player":
    years = sorted(grades["draft_year"].unique(), reverse=True)
    c_year, c_player = st.columns([1, 3])
    year = c_year.selectbox("Draft class", ["All years", *years], key="pc_year")
    pool = grades if year == "All years" else grades[grades["draft_year"] == year]
    names = [label(r) for _, r in pool.iterrows()]
    if st.session_state.get("pc_player") not in names:
        st.session_state["pc_player"] = names[0]
    choice = c_player.selectbox("Player (type to search)", names, key="pc_player")
    pid = options[choice]
    if qp.get("player") != pid:
        qp.clear()
        qp["player"] = pid
    summary(pid)
    if G.loc[pid, "projection_type"] == "retrospective":
        st.caption(
            "Retrospective projection: our data starts in 1996, so this class had too few "
            "earlier drafts to project from. Its range comes from how players at the same "
            "draft slot did in other drafts, so it uses hindsight."
        )
    try:
        share = f"{st.context.url.split('?')[0].rstrip('/')}/?player={pid}"
    except Exception:
        share = f"?player={pid}"
    st.caption(f"Link to this card: {share}")
    st.image(card_path(pid), use_container_width=True)

    left, right = st.columns(2)
    comps = data["comps"][data["comps"]["bbref_id"] == pid].sort_values("rank")
    limited = G.loc[pid, "prospect_source"] in LIMITED_SOURCES
    left.subheader("Top 15 comps (pre-draft stats, age, size)")
    if comps.empty:
        left.caption("No comps: there are no earlier draft classes in the data.")
    elif limited:
        left.warning(
            "Limited pre-draft data (high school or overseas/pro team): these comps rest "
            "mostly on age, size and position, so treat them loosely.",
            icon="⚠️",
        )
    comp_cols = ["rank", "comp_name", "comp_draft_year", "comp_pick", "comp_peak6"]
    if not limited:
        comp_cols += ["similarity", "driving_features"]
    with left:
        clickable(
            comps[[*comp_cols, "comp_id"]].rename(columns={"comp_id": "bbref_id"}),
            key=f"comps_{pid}",
            column_config={
                "comp_peak6": st.column_config.NumberColumn("Comp peak (yr 6)", format="%.2f"),
                "comp_name": "Comp",
                "similarity": st.column_config.NumberColumn("Similarity", format="%.0f"),
            },
        )
        st.caption("Click a row to open that player's card.")
    pl = data["plays_like"][data["plays_like"]["bbref_id"] == pid].sort_values("rank")
    right.subheader("Plays like (stylistic only)")
    right.caption(
        "Nearest NBA early-career shot styles. Says where he shoots from, not how good he will be."
    )
    if pl.empty:
        right.caption(
            "Not available: style comps are built for the 2022+ classes with enough "
            "college or NBA shot-location data."
        )
    else:
        right.dataframe(
            pl[["rank", "plays_like_name", "style_similarity", "basis"]],
            hide_index=True,
            use_container_width=True,
        )
    if year != "All years":
        st.subheader(f"The {year} draft class")
        cls = pool[
            [
                "bbref_id",
                "pick",
                "player_name",
                "grade",
                "current_tier",
                "current_p_all_star",
                "status",
            ]
        ]
        clickable(
            cls,
            key=f"class_{year}",
            column_config={
                "current_tier": "Tier",
                "current_p_all_star": st.column_config.ProgressColumn(
                    "All-Star or better", format="percent", min_value=0, max_value=1
                ),
            },
        )

elif page == "redraft":
    st.subheader("Redraft: every class re-ordered by how careers turned out")
    years = sorted(grades["draft_year"].unique(), reverse=True)
    q_year = qp.get("year")
    default_year = int(q_year) if q_year and q_year.isdigit() and int(q_year) in years else 2011
    year = st.selectbox("Draft class", years, index=years.index(default_year))
    qp["year"] = str(year)
    cls = grades[grades["draft_year"] == year].copy()
    if not cls["finished"].all():
        st.info(
            "Careers in this class are still in progress, so the order uses each player's "
            "current projected peak and will change."
        )
    cls = cls.sort_values("redraft")
    fig = px.scatter(
        cls,
        x="pick",
        y="redraft",
        color="grade",
        hover_name="player_name",
        color_discrete_map={"A": "#1baf7a", "B": "#2a78d6", "C": "#eda100", "D": "#e34948"},
        category_orders={"grade": ["A", "B", "C", "D", "-"]},
        labels={"pick": "Actual pick", "redraft": "Redraft position"},
        height=420,
    )
    top = len(cls)
    fig.add_shape(type="line", x0=1, y0=1, x1=top, y1=top, line={"dash": "dot", "color": "#999"})
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above the dotted line = drafted too late; below = drafted too early.")
    show = cls if st.toggle("Show both rounds", value=False) else cls.head(30)
    clickable(
        show[
            [
                "bbref_id",
                "redraft",
                "player_name",
                "pick",
                "moved",
                "grade",
                "current_tier",
                "current_median",
            ]
        ],
        key=f"redraft_{year}",
        column_config={
            "redraft": "Redraft #",
            "pick": "Drafted #",
            "moved": st.column_config.NumberColumn("Moved", format="%+d"),
            "current_tier": "Tier",
            "current_median": st.column_config.NumberColumn("Peak value", format="%.2f"),
        },
    )
    st.caption("Click a row to open that player's card.")

elif page == "steals":
    st.subheader("Steals and busts: who should have gone much higher, or much lower")
    st.caption(
        "Ranked by redraft spots moved: drafted pick minus where he'd go if each class were "
        "redrafted by career peak. Value vs slot = career peak minus the draft-night median "
        "for his slot."
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        lo, hi = year_range("steals_years", (1996, 2021))
    rnd = c2.selectbox("Round", ["Both", "1st", "2nd"])
    finished_only = c3.toggle("Finished careers only", value=True)
    d = grades[grades["draft_year"].between(lo, hi)]
    if rnd != "Both":
        d = d[d["round"] == rnd]
    d = d[d["finished"]] if finished_only else d[d["grade"] != "-"]
    cols = [
        "bbref_id",
        "player_name",
        "draft_year",
        "pick",
        "redraft",
        "moved",
        "grade",
        "value_vs_slot",
    ]
    cfg = {
        "draft_year": st.column_config.NumberColumn("Year", format="%d"),
        "pick": "Drafted #",
        "redraft": "Redraft #",
        "moved": st.column_config.NumberColumn("Moved", format="%+d"),
        "projected_median": st.column_config.NumberColumn("Expected", format="%.2f"),
        "current_median": st.column_config.NumberColumn("Actual", format="%.2f"),
        "value_vs_slot": st.column_config.NumberColumn("Value vs slot", format="%+.2f"),
    }
    left, right = st.columns(2)
    with left:
        st.markdown("**Biggest steals**")
        steals = d.sort_values(["moved", "value_vs_slot"], ascending=False).head(25)
        clickable(steals[cols], key="steals", column_config=cfg)
    with right:
        st.markdown("**Biggest busts**")
        busts = d.sort_values(["moved", "value_vs_slot"]).head(25)
        clickable(busts[cols], key="busts", column_config=cfg)
    st.caption("Click a row to open that player's card.")

elif page == "teams":
    st.subheader("Team draft report cards")
    st.caption(
        "How each franchise's picks did against their draft slot. Phase 6 found these "
        "differences are no bigger than luck would produce (p = 0.92), so read this as a "
        "history of outcomes, not of drafting skill. Team = the team that made the pick."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        lo, hi = year_range("team_years", (1996, 2021))
    finished_only = c2.toggle("Finished careers only", value=True, key="team_fin")
    d = grades[grades["draft_year"].between(lo, hi) & (grades["grade"] != "-")]
    if finished_only:
        d = d[d["finished"]]
    agg = (
        d.groupby("franchise")
        .agg(
            picks=("bbref_id", "size"),
            value_vs_slot=("value_vs_slot", "mean"),
            sd=("value_vs_slot", "std"),
            beat_median=("grade", lambda x: x.isin(["A", "B"]).mean()),
            a_grades=("grade", lambda x: int((x == "A").sum())),
        )
        .reset_index()
    )
    # Relative to the league: careers are skewed (a few stars), so raw averages all sit
    # above the median-based expectation.
    agg["value_vs_slot"] -= d["value_vs_slot"].mean()
    agg["se"] = agg["sd"] / np.sqrt(agg["picks"])
    best = d.loc[d.groupby("franchise")["value_vs_slot"].idxmax(), ["franchise", "player_name"]]
    worst = d.loc[d.groupby("franchise")["value_vs_slot"].idxmin(), ["franchise", "player_name"]]
    agg = agg.merge(best.rename(columns={"player_name": "best_pick"}), on="franchise").merge(
        worst.rename(columns={"player_name": "worst_pick"}), on="franchise"
    )
    agg = agg.sort_values("value_vs_slot", ascending=False)
    fig = px.bar(
        agg,
        x="value_vs_slot",
        y="franchise",
        orientation="h",
        error_x=1.645 * agg["se"],
        height=750,
        labels={
            "value_vs_slot": "Average value vs slot, relative to league average (90% interval)",
            "franchise": "",
        },
    )
    fig.update_traces(marker_color="#2a78d6")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        agg.drop(columns=["sd", "se"]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "value_vs_slot": st.column_config.NumberColumn(
                "Avg value vs slot (vs league)", format="%+.2f"
            ),
            "beat_median": st.column_config.ProgressColumn(
                "Beat slot median (A/B)", format="percent", min_value=0, max_value=1
            ),
            "a_grades": "A grades",
            "best_pick": "Best pick vs slot",
            "worst_pick": "Worst pick vs slot",
        },
    )

elif page == "compare":
    names = list(options)
    a, b = st.columns(2)
    pa = options[a.selectbox("Player A", names, index=0)]
    pb = options[b.selectbox("Player B", names, index=1)]
    rows = {
        "Status": "status",
        "Grade": "grade",
        "Tier (now)": "current_tier",
        "Tier (draft night)": "projected_tier",
        "All-Star or better (now)": "current_p_all_star",
        "All-Star or better (draft night)": "projected_p_all_star",
        "Confidence": "confidence",
        "Peak value: floor / median / ceiling (now)": None,
        "Peak value: floor / median / ceiling (draft night)": None,
    }

    def cell(pid: str, name: str, col: str | None) -> str:
        g = G.loc[pid]
        if col is None:
            k = "current" if "now" in name else "projected"
            return f"{g[f'{k}_floor']:.2f} / {g[f'{k}_median']:.2f} / {g[f'{k}_ceiling']:.2f}"
        v = g[col]
        if col in ("current_p_all_star", "projected_p_all_star", "confidence"):
            return f"{v:.0%}"
        return str(v)

    table = pd.DataFrame(
        {
            G.loc[p, "player_name"]: [cell(p, n, c) for n, c in rows.items()]
            for p in dict.fromkeys([pa, pb])
        },
        index=list(rows),
    )
    st.dataframe(table, use_container_width=True)
    a.image(card_path(pa), use_container_width=True)
    b.image(card_path(pb), use_container_width=True)

elif page == "styles":
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
    hi_pts = sm[sm["player_name"].isin(highlight)]
    fig.add_scatter(
        x=hi_pts["x"],
        y=hi_pts["y"],
        mode="markers+text",
        text=hi_pts["player_name"],
        textposition="top center",
        name="highlighted",
        marker={"size": 13, "color": "white", "line": {"width": 2, "color": "black"}},
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="Dominant style")
    st.plotly_chart(fig, use_container_width=True)

elif page == "tracker":
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
