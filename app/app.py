"""Draft DNA: Streamlit app.

Run with `make app` (or `uv run streamlit run app/app.py`). Reads the built tables (or the
committed app bundle when there is no local build) and draws cards on demand;
`make refresh` updates them during the season.

Links: `?player=<bbref_id>` opens a player's card; `?page=redraft&year=2011` etc.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
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


def _code_and_data_changed() -> bool:
    """Streamlit Cloud reruns app.py after a git push but keeps already-imported modules
    and cached data in memory, so new code could call old modules (AttributeError) and a
    refreshed bundle could keep showing old numbers. Fingerprint the project code and the
    data bundle; when they change, drop the stale modules (the caches are cleared below)."""
    watched = [
        *(ROOT / "src" / "draft_dna").rglob("*.py"),
        ROOT / "app" / "theme.py",
        *(ROOT / "app" / "bundle").rglob("*.parquet"),
    ]
    sig = hashlib.sha1(
        "|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in sorted(watched)).encode()
    ).hexdigest()
    previous = os.environ.get("DRAFT_DNA_CODE_DATA_SIG")
    os.environ["DRAFT_DNA_CODE_DATA_SIG"] = sig
    if previous is None or previous == sig:
        return False
    for name in [m for m in sys.modules if m == "theme" or m.startswith("draft_dna")]:
        del sys.modules[name]
    return True


STALE = _code_and_data_changed()

from draft_dna.config import get_settings  # noqa: E402
from draft_dna.grading import classes as C  # noqa: E402
from draft_dna.ingest.storage import read_table  # noqa: E402
from draft_dna.viz import cards as card_viz  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
import theme  # noqa: E402

st.set_page_config(page_title="Draft DNA", page_icon="🏀", layout="wide")
theme.apply()
S = get_settings()
LOCAL_CARDS = S.paths.modeled / "cards"
CARD_DIR = LOCAL_CARDS if LOCAL_CARDS.exists() else Path(tempfile.gettempdir()) / "draft_dna_cards"
if STALE:  # new code or data deployed: forget cached tables and cards drawn by old code
    st.cache_data.clear()
    st.cache_resource.clear()
    if CARD_DIR != LOCAL_CARDS:
        shutil.rmtree(CARD_DIR, ignore_errors=True)
PAGES = {
    "home": "Home",
    "player": "Player card",
    "redraft": "Redraft",
    "classes": "Draft classes",
    "steals": "Steals & busts",
    "teams": "Teams",
    "leaders": "Leaderboards",
    "pickvalue": "Pick value",
    "recruits": "Recruits",
    "compare": "Compare",
    "styles": "Style map",
    "tracker": "2026 tracker",
    "accuracy": "How accurate is it?",
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
        "recruits": read_table("modeled", "recruits", "players", S),
        "recruit_summary": read_table("modeled", "recruits", "summary", S),
        "recruit_tests": read_table("modeled", "recruits", "tests", S),
        "metrics": read_table("modeled", "grading", "player_metrics", S),
        "acc_models": read_table("modeled", "accuracy", "models", S),
        "acc_calibration": read_table("modeled", "accuracy", "calibration", S),
        "acc_allstar": read_table("modeled", "accuracy", "allstar_reliability", S),
        "acc_sharpening": read_table("modeled", "accuracy", "sharpening", S),
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
# Cross-class scale (D037): which pick's typical career he matched, and all-time rank.
CURVE = C.typical_curve(grades)
grades["equiv_pick"] = C.equivalent_pick(grades["current_median"], CURVE)
grades["played_like"] = grades["equiv_pick"].map(C.equivalent_pick_label)
grades["all_time_rank"] = C.all_time_rank(grades)
CLASSES = C.class_strength(grades, CURVE)
PICK_VALUE = C.pick_value(CURVE)
grades = grades.merge(data["metrics"], on="bbref_id", how="left")
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
        width="stretch",
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
to_page = st.session_state.pop("goto_page", None)
if to_page is not None:  # buttons on the home page
    qp.clear()
    qp["page"] = to_page
    st.session_state["nav"] = PAGES[to_page]
to_class = st.session_state.pop("goto_class", None)
if to_class is not None:  # clicked a class on the Draft classes page
    qp.clear()
    qp["page"], qp["year"] = "redraft", str(to_class)
    st.session_state["nav"] = PAGES["redraft"]
    st.session_state["rd_year"] = int(to_class)
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
    start = qp.get("page", "home")
    st.session_state["nav"] = PAGES.get(start, PAGES["home"])
theme.brand()
page_name = st.sidebar.radio("View", list(PAGES.values()), key="nav")
page = slugs[list(PAGES.values()).index(page_name)]
if page != "player" and qp.get("page") != page:  # keep the URL in step with the sidebar
    qp.clear()
    qp["page"] = page


def summary(pid: str) -> None:
    g = G.loc[pid]
    c1, c2, c3, c4, c5 = st.columns([1.0, 1.7, 1.6, 1.6, 1.25])
    c1.metric("Grade", g["grade"], help=GRADE_HELP)
    c2.metric("Status", g["status"])
    tier_label = "Outcome tier" if g["finished"] else "Projected tier"
    c3.metric(
        tier_label,
        g["current_tier"],
        delta=f"draft night: {g['projected_tier']}",
        delta_color="off",
        delta_arrow="off",
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
    so_far = "" if g["finished"] else " so far (projected)"
    st.caption(
        f"Played like a typical **{g['played_like']}** pick{so_far} · all-time rank "
        f"**{int(g['all_time_rank'])}** of {len(G):,} picks since 1996 (by career peak). "
        "Both compare across draft classes, unlike the redraft."
    )
    more_metrics(g)


def _fmt_or(v: object, fmt: str, missing: str = "—") -> str:
    return missing if v is None or pd.isna(v) else format(v, fmt)  # type: ignore[arg-type]


def surplus_label(v: object) -> str:
    """'+$168M'. The backslash keeps Streamlit from reading '$' as the start of math."""
    if v is None or pd.isna(v):  # type: ignore[arg-type]
        return "—"
    x = float(v)  # type: ignore[arg-type]
    return f"{'+' if x >= 0 else '-'}\\${abs(x):,.0f}M"


def more_metrics(g: pd.Series) -> None:
    with st.expander("More metrics: contract value, development, playoffs, durability"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            "Bust risk (draft night)",
            _fmt_or(g["projected_p_bust"], ".0%"),
            help="Draft-night chance his career never reaches rotation level.",
        )
        c2.metric(
            "Rookie-deal surplus",
            surplus_label(g["surplus_m"]),
            help="Value produced in his first 4 seasons, priced at what veterans with the same "
            "production earn, minus what he was paid. In today's dollars ($M).",
        )
        c3.metric(
            "Late bloomer index",
            _fmt_or(g["late_bloomer"], "+.1f"),
            help="How much his peak grew from year 3 to year 8 beyond players with the same "
            "year-3 level (standard units; + = late bloomer, - = early peaker).",
        )
        c4.metric(
            "Playoff riser",
            _fmt_or(g["playoff_riser"], "+.1f"),
            help="Playoff minus regular-season box plus-minus in the same seasons, per 100 "
            "possessions, shrunk toward 0 for small samples (needs 300+ playoff minutes).",
        )
        c5.metric(
            "Second contract",
            _fmt_or(g["second_pct"], ".0%"),
            delta=None
            if pd.isna(g["second_vs_slot"])
            else f"{g['second_vs_slot'] * 100:+.0f} pts vs slot",
            help="Best salary as a share of the cap in years 5-7 (the second contract), vs the "
            "average for his draft slot.",
        )
        c6.metric(
            "Rotation seasons",
            _fmt_or(g["rotation_seasons"], ".0f"),
            delta=None if pd.isna(g["availability"]) else f"{g['availability']:.0%} of games",
            delta_color="off",
            delta_arrow="off",
            help="Seasons with 1,000+ minutes; delta = share of team games played when in "
            "the league.",
        )


# ----------------------------------------------------------------------- pages
def go_page(slug: str) -> None:
    st.session_state["goto_page"] = slug
    st.rerun()


def page_button(label: str, slug: str, key: str) -> None:
    if st.button(label, key=key, type="secondary"):
        go_page(slug)


if page == "home":
    n_picks = len(G)
    acc = data["acc_models"].set_index("model").loc["Pick only + conformal"]
    inside = 1 - acc["below_floor"] - acc["above_ceiling"]
    best = CLASSES.sort_values("rank").iloc[0]
    sharp = data["acc_sharpening"].set_index("seasons")["crps"]
    rec = data["recruit_tests"].set_index(["horizon", "scope"]).loc[(4, "all picks")]
    theme.hero(
        "NBA draft analytics · 1996-2026",
        "What was expected. What actually happened.",
        f"Every one of the {n_picks:,} NBA draft picks since 1996, projected from draft "
        "night only, then graded season by season as careers unfold. Honest backtests, "
        "pre-registered tests, and no hindsight in the projections.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picks graded", f"{n_picks:,}", help="Both rounds, 1996-2026 drafts.")
    c2.metric(
        "Careers inside the range",
        f"{inside:.0%}",
        delta="target 65%",
        delta_color="off",
        delta_arrow="off",
        help="Held-out 2013-2020 picks finishing between the projected floor (25th "
        "percentile) and ceiling (90th).",
    )
    c3.metric(
        "Strongest draft",
        str(int(best["draft_year"])),
        delta=f"{best['strength']:+.1f} vs an average class",
        delta_color="off",
        delta_arrow="off",
    )
    c4.metric(
        "Forecast error by year 7",
        f"-{1 - sharp.loc[7] / sharp.loc[0]:.0%}",
        delta="vs draft night",
        delta_color="off",
        delta_arrow="off",
        help="How much the range tightens as real seasons replace the projection (CRPS).",
    )
    st.markdown("### What the data says")
    findings = [
        (
            "Draft position is the forecast to beat",
            "College box scores, comps, gradient boosting and a Bayesian model all tied or "
            "lost to how the same draft slots worked out historically.",
            "How accurate is it?",
            "accuracy",
        ),
        (
            "How a prospect scores adds nothing",
            "Shot location and style (where, how, assisted or not) had no detectable value "
            "once draft slot and stats were known.",
            "Style map",
            "styles",
        ),
        (
            "Teams already price in recruiting rank",
            f"Top-10 high-school recruits go about 20 picks earlier, then match their slot "
            f"(rho = {rec['spearman_rho']:+.2f}, p = {rec['p_value']:.2f}).",
            "Recruits",
            "recruits",
        ),
        (
            f"{int(best['draft_year'])} was the best draft",
            f"{best['best_player']} led a class worth {best['strength']:+.1f} over an average "
            "draft. 2000 and 2016 sit at the bottom.",
            "Draft classes",
            "classes",
        ),
        (
            "No team drafts better than luck",
            "Franchise differences against draft slot are no bigger than chance produces, "
            "crediting draft-night trades to the team that got the player.",
            "Teams",
            "teams",
        ),
        (
            "Steals hide everywhere",
            "Isaiah Thomas (#60), Manu Ginobili (#57) and Marc Gasol (#48) played like top-2 "
            "picks; Thabeet and Bennett went the other way.",
            "Steals & busts",
            "steals",
        ),
    ]
    for row in range(0, len(findings), 3):
        cols = st.columns(3)
        for col, (title, body, button, slug) in zip(cols, findings[row : row + 3], strict=False):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
                page_button(f"{button} →", slug, key=f"home_{slug}")
    st.markdown("### Featured prospects")
    featured = [p for p in ("dybanaj01", "flaggco01", "wembavi01") if p in G.index]
    cols = st.columns(len(featured))
    for col, pid in zip(cols, featured, strict=True):
        with col:
            sample = ROOT / "reports" / "cards" / f"{pid}.png"
            st.image(str(sample if sample.exists() else card_path(pid)), width="stretch")
            g = G.loc[pid]
            st.caption(
                f"**{g['player_name']}** · {int(g['draft_year'])} #{int(g['pick'])} · "
                f"{g['current_tier']} projected"
            )
            if st.button("Open card →", key=f"feat_{pid}"):
                go_to_player(pid)
    st.caption(
        "Method: draft-slot history with conformal calibration on draft night, then Bayesian "
        "updating as seasons are played. Full write-up on the About page."
    )

elif page == "accuracy":
    models = data["acc_models"]
    cal = data["acc_calibration"]
    allstar = data["acc_allstar"]
    sharp = data["acc_sharpening"]
    rec_row = models.set_index("model").loc["Pick only + conformal"]
    theme.hero(
        "How accurate is it?",
        "Judged only on players it never saw",
        "Every number on this page comes from the 2013-2020 draft classes, predicted by models "
        "trained only on earlier drafts and never used to choose a model. Outcome: best "
        "3-season stretch through year 6.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Finished below the floor",
        f"{rec_row['below_floor']:.1%}",
        delta="target 25%",
        delta_color="off",
        delta_arrow="off",
    )
    c2.metric(
        "Finished above the ceiling",
        f"{rec_row['above_ceiling']:.1%}",
        delta="target 10%",
        delta_color="off",
        delta_arrow="off",
    )
    c3.metric(
        "Inside the range",
        f"{1 - rec_row['below_floor'] - rec_row['above_ceiling']:.1%}",
        delta="target 65%",
        delta_color="off",
        delta_arrow="off",
    )
    c4.metric("Held-out players", f"{int(cal['n'].iloc[0])}")

    st.markdown("### Are the ranges honest?")
    st.caption(
        "For every predicted percentile, the share of careers that actually finished below "
        "it. On the dotted line = perfectly calibrated. The floor (25th), median and ceiling "
        "(90th) land almost exactly on target. The lowest percentiles sit above the line "
        "because many careers end at exactly zero value (out of the league), which a smooth "
        "range can't split finely."
    )
    fig = px.line(
        cal,
        x="level",
        y="observed",
        markers=True,
        labels={"level": "Predicted percentile", "observed": "Share of careers below it"},
        height=420,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": theme.MUTED})
    for lvl, name in ((0.25, "floor"), (0.5, "median"), (0.9, "ceiling")):
        obs = float(cal.loc[(cal["level"] - lvl).abs().idxmin(), "observed"])
        fig.add_annotation(
            x=lvl,
            y=obs,
            text=f"{name}: {obs:.0%}",
            showarrow=True,
            arrowhead=0,
            ax=40,
            ay=-30,
            font={"color": theme.TEXT},
        )
    fig.update_traces(line={"width": 3, "color": theme.BLUE}, marker={"size": 8})
    fig.update_layout(
        xaxis={"tickformat": ".0%", "range": [0, 1]}, yaxis={"tickformat": ".0%", "range": [0, 1]}
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Does anything beat draft position?")
    st.caption(
        "Forecast error (CRPS, lower is better) for every model, held out. Bars show the "
        "difference from plain draft-slot history with a 95% interval: nothing is reliably "
        "better, and stats-only models are reliably worse. So the simplest model ships."
    )
    m = models.copy()
    m["diff"] = m["crps_vs_pick"]
    m["err_hi"] = m["crps_vs_pick_hi"] - m["diff"]
    m["err_lo"] = m["diff"] - m["crps_vs_pick_lo"]
    m = m.sort_values("diff", ascending=False)
    fig = px.bar(
        m,
        x="diff",
        y="label",
        orientation="h",
        error_x="err_hi",
        error_x_minus="err_lo",
        labels={"diff": "Extra forecast error vs draft slot (lower is better)", "label": ""},
        height=380,
    )
    fig.update_traces(
        marker_color=[
            theme.BLUE if "model of record" in lbl else "rgba(10,132,255,0.35)"
            for lbl in m["label"]
        ]
    )
    fig.add_vline(x=0, line_color=theme.MUTED, line_width=1)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### The forecast sharpens as seasons are played")
    st.caption(
        "Forecast error of the year-8 career peak after N seasons, on held-out 2011-2018 "
        "picks. Each season of real play replaces more of the draft-night guess."
    )
    s2 = sharp.assign(weight=sharp["data_weight"].map(lambda w: f"{w:.0%} real data"))
    fig = px.line(
        s2,
        x="seasons",
        y="crps",
        markers=True,
        text="weight",
        height=380,
        labels={"seasons": "Seasons played", "crps": "Forecast error (CRPS)"},
    )
    fig.update_traces(
        line={"width": 3, "color": theme.BLUE},
        marker={"size": 9},
        textposition="top center",
        textfont={"color": theme.TEXT_2, "size": 11},
    )
    fig.update_layout(xaxis={"range": [-0.4, 7.6], "dtick": 1})
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Where it misses: All-Star odds for top picks run high")
    st.caption(
        "Predicted chance of an All-Star-or-better career vs how often it happened. Low "
        "odds are accurate; when the model gave a 35-50% chance (top picks), only about 1 "
        "in 6 got there in the 2013-2020 drafts. Treat high All-Star odds as optimistic."
    )
    nice = {
        "(-0.001, 0.05]": "0-5%",
        "(0.05, 0.1]": "5-10%",
        "(0.1, 0.2]": "10-20%",
        "(0.2, 0.35]": "20-35%",
        "(0.35, 0.5]": "35-50%",
        "(0.5, 1.0]": "50%+",
    }
    a = allstar.assign(bucket=allstar["bin"].map(nice).fillna(allstar["bin"]))
    a = a.melt(
        id_vars=["bucket", "n"],
        value_vars=["predicted", "observed"],
        var_name="series",
        value_name="share",
    )
    a["series"] = a["series"].map({"predicted": "Predicted", "observed": "Actually happened"})
    fig = px.bar(
        a,
        x="bucket",
        y="share",
        color="series",
        barmode="group",
        height=380,
        text=a["share"].map(lambda v: f"{v:.0%}"),
        color_discrete_map={"Predicted": theme.BLUE, "Actually happened": theme.HERO_GOLD},
        labels={"bucket": "Predicted chance of All-Star or better", "share": "", "series": ""},
        hover_data={"n": True},
    )
    fig.update_traces(textposition="outside", textfont={"color": theme.TEXT_2})
    fig.update_layout(
        yaxis={"tickformat": ".0%", "range": [0, 0.65]},
        legend={"orientation": "h", "y": 1.12, "x": 0, "title": ""},
        bargap=0.25,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Full method and every model's numbers: DECISIONS D020-D024 and D031-D034 in the "
        "repository, and the About page."
    )

elif page == "player":
    years = sorted(grades["draft_year"].unique(), reverse=True)
    c_year, c_player = st.columns([1, 3])
    year = c_year.selectbox("Draft class", ["All years", *years], key="pc_year")
    pool = grades if year == "All years" else grades[grades["draft_year"] == year]
    names = [label(r) for _, r in pool.iterrows()]
    if st.session_state.get("pc_player") not in names:
        st.session_state["pc_player"] = names[0]
    choice = c_player.selectbox("Player (type to search)", names, key="pc_player")
    pid = options[choice]
    gp = G.loc[pid]
    theme.hero(
        f"{int(gp['draft_year'])} draft · pick #{int(gp['pick'])} · {gp['team']}",
        str(gp["player_name"]),
    )
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
    st.image(card_path(pid), width="stretch")

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
            pl[["rank", "plays_like_name", "style_similarity", "style_percentile", "basis"]],
            column_config={
                "rank": "#",
                "plays_like_name": "Plays like",
                "style_similarity": st.column_config.NumberColumn(
                    "Style match",
                    format="percent",
                    help="How alike the two shot-style mixes are (cosine similarity). Runs "
                    "high: two random NBA players are about 82% alike.",
                ),
                "style_percentile": st.column_config.NumberColumn(
                    "Closer than",
                    format="percent",
                    help="Share of all NBA player pairs that are less alike than this match.",
                ),
                "basis": "Based on",
            },
            hide_index=True,
            width="stretch",
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
    theme.hero(
        "Redraft",
        "If teams could do it again",
        "Every class re-ordered by how careers actually turned out.",
    )
    years = sorted(grades["draft_year"].unique(), reverse=True)
    q_year = qp.get("year")
    default_year = int(q_year) if q_year and q_year.isdigit() and int(q_year) in years else 2011
    if "rd_year" not in st.session_state:
        st.session_state["rd_year"] = default_year
    year = st.selectbox("Draft class", years, key="rd_year")
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
        color_discrete_map=theme.GRADE_COLORS,
        category_orders={"grade": ["A", "B", "C", "D", "-"]},
        labels={"pick": "Actual pick", "redraft": "Redraft position"},
        height=420,
    )
    top = len(cls)
    fig.add_shape(
        type="line", x0=1, y0=1, x1=top, y1=top, line={"dash": "dot", "color": theme.MUTED}
    )
    fig.update_traces(marker={"size": 9, "line": {"width": 1, "color": theme.BG}})
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
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
                "played_like",
                "all_time_rank",
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
            "played_like": "Played like",
            "all_time_rank": st.column_config.NumberColumn("All-time rank", format="%d"),
        },
    )
    strength = CLASSES.set_index("draft_year").loc[year]
    st.caption(
        f"Click a row to open that player's card. Redraft order is within this class; "
        f"'Played like' and 'All-time rank' compare across classes. The {year} class ranks "
        f"#{int(strength['rank'])} of {len(CLASSES)} by strength (Draft classes page)."
    )

elif page == "classes":
    theme.hero(
        "Draft classes",
        "Which drafts were strongest?",
        "Every class against an average draft from the same 60 picks.",
    )
    st.caption(
        "Strength = total career peak a class produced minus what an average class produces "
        "from the same 60 picks (0 = an average draft; positive = stronger). Unfinished "
        "careers count at their expected peak, so recent classes (lighter bars) are "
        "provisional and move toward reality each season. Stars = All-Star tier or better; "
        "starters+ = Starter tier or better."
    )
    cl = CLASSES.sort_values("draft_year")
    cl["status"] = np.where(cl["provisional"], "provisional", "complete")
    fig = px.bar(
        cl,
        x="draft_year",
        y="strength",
        color="status",
        color_discrete_map={"complete": theme.BLUE, "provisional": "rgba(10,132,255,0.35)"},
        hover_data={"best_player": True, "stars": True, "starters_plus": True},
        labels={"draft_year": "Draft class", "strength": "Value vs an average class"},
        height=420,
    )
    fig.add_hline(y=0, line_color=theme.MUTED, line_width=1)
    fig.update_layout(legend={"orientation": "h", "y": 1.08, "x": 0, "title": ""})
    st.plotly_chart(fig, width="stretch")
    table = CLASSES.sort_values("rank")[
        [
            "rank",
            "draft_year",
            "strength",
            "lottery",
            "later_picks",
            "stars",
            "starters_plus",
            "best_player",
            "provisional",
        ]
    ]
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="classes_table",
        column_config={
            "rank": "#",
            "draft_year": st.column_config.NumberColumn("Class", format="%d"),
            "strength": st.column_config.NumberColumn("Strength", format="%+.1f"),
            "lottery": st.column_config.NumberColumn("From picks 1-14", format="%+.1f"),
            "later_picks": st.column_config.NumberColumn("From picks 15-60", format="%+.1f"),
            "stars": "Stars",
            "starters_plus": "Starters+",
            "best_player": "Best player",
            "provisional": "Provisional",
        },
    )
    if event.selection.rows:  # type: ignore[attr-defined]
        st.session_state["goto_class"] = int(table.iloc[event.selection.rows[0]]["draft_year"])  # type: ignore[attr-defined]
        st.rerun()
    st.caption("Click a class to open its redraft.")

elif page == "leaders":
    theme.hero(
        "Leaderboards",
        "Bargains, bloomers and big-game players",
        "Six ways to rank every pick since 1996 beyond the grade.",
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        lo, hi = year_range("leader_years", (1996, 2026))
    rnd = c2.selectbox("Round", ["Both", "1st", "2nd"], key="leader_round")
    d = grades[grades["draft_year"].between(lo, hi)]
    if rnd != "Both":
        d = d[d["round"] == rnd]
    base = ["bbref_id", "player_name", "draft_year", "pick"]
    cfg = {
        "draft_year": st.column_config.NumberColumn("Year", format="%d"),
        "surplus_m": st.column_config.NumberColumn("Surplus ($M, today)", format="%+.0f"),
        "late_bloomer": st.column_config.NumberColumn("Late bloomer", format="%+.1f"),
        "playoff_riser": st.column_config.NumberColumn("Playoff riser", format="%+.1f"),
        "second_pct": st.column_config.NumberColumn("2nd contract (% cap)", format="percent"),
        "second_vs_slot": st.column_config.NumberColumn("vs slot", format="percent"),
        "rotation_seasons": "Rotation seasons",
        "availability": st.column_config.NumberColumn("Games played", format="percent"),
        "projected_p_bust": st.column_config.NumberColumn("Bust risk", format="percent"),
        "projected_p_all_star": st.column_config.NumberColumn("All-Star odds", format="percent"),
    }
    boards = {
        "Bargains": (
            "surplus_m",
            "Rookie-contract surplus: production in years 1-4 priced at veteran market "
            "rates, minus actual pay, in today's dollars. Top picks earn more, but stars on "
            "rookie deals are the best value in basketball.",
        ),
        "Late bloomers": (
            "late_bloomer",
            "Peak growth from year 3 to year 8 beyond players with the same year-3 level "
            "(players in the NBA for 2+ of their first 3 seasons). Sort the other way for "
            "early peakers.",
        ),
        "Playoff risers": (
            "playoff_riser",
            "Playoff minus regular-season box plus-minus in the same seasons (per 100 "
            "possessions), 300+ playoff minutes, shrunk toward 0. Sort the other way for "
            "players who fade in the playoffs.",
        ),
        "Second contracts": (
            "second_vs_slot",
            "Best salary in years 5-7 as a share of the cap, minus the average for his slot: "
            "how teams valued him when the rookie deal ended.",
        ),
        "Durability": (
            "rotation_seasons",
            "Seasons with 1,000+ minutes, with the share of team games played.",
        ),
        "Bust risk": (
            "projected_p_bust",
            "Draft-night chance a career never reaches rotation level, from the draft-slot "
            "model. Highest risk first; pick a single class with the year slider.",
        ),
    }
    tabs = st.tabs(list(boards))
    for tab, (col, note) in zip(tabs, boards.values(), strict=True):
        with tab:
            st.caption(note)
            flip = st.toggle("Reverse order", key=f"flip_{col}")
            extra = {
                "second_vs_slot": ["second_pct", "second_vs_slot"],
                "rotation_seasons": ["rotation_seasons", "availability"],
                "projected_p_bust": ["projected_p_bust", "projected_p_all_star"],
            }.get(col, [col])
            rows = d.dropna(subset=[col]).sort_values(col, ascending=flip).head(30)
            clickable(rows[[*base, *extra, "grade"]], key=f"lb_{col}", column_config=cfg)
    st.caption("Click a row to open that player's card.")

elif page == "pickvalue":
    theme.hero(
        "Pick value",
        "What is each pick worth?",
        "Every slot's average career value since 1996, with the #1 pick = 100.",
    )
    pv = PICK_VALUE.copy()
    st.caption(
        "Average best-3-season career value by pick (classes 1996-2017, smoothed so it never "
        "rises with pick number), scaled so the #1 pick = 100. The drop is steep: a #10 pick "
        "is worth well under half of a #1, and late second-rounders a small fraction."
    )
    fig = px.area(
        pv,
        x="pick",
        y="value",
        labels={"pick": "Pick", "value": "Value (#1 = 100)"},
        height=380,
    )
    fig.update_traces(line={"color": theme.BLUE, "width": 3}, fillcolor="rgba(10,132,255,0.18)")
    for p in (1, 5, 10, 20, 30, 45):
        v = float(pv.loc[pv["pick"] == p, "value"].iloc[0])
        fig.add_annotation(
            x=p,
            y=v,
            text=f"#{p}: {v:.0f}",
            showarrow=True,
            arrowhead=0,
            ay=-28,
            font={"color": theme.TEXT},
        )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Trade calculator")
    st.caption("Which side of a pick swap is worth more, on average?")
    picks = list(range(1, 61))
    a, b = st.columns(2)
    side_a = a.multiselect("Side A gets", picks, default=[5], key="trade_a")
    side_b = b.multiselect("Side B gets", picks, default=[12, 20], key="trade_b")
    val = pv.set_index("pick")["value"]
    va, vb = float(val.reindex(side_a).sum()), float(val.reindex(side_b).sum())
    a.metric("Side A value", f"{va:.0f}")
    b.metric("Side B value", f"{vb:.0f}")
    if va or vb:
        lead, diff = ("A", va - vb) if va >= vb else ("B", vb - va)
        st.markdown(
            f"**Side {lead} wins by {diff:.0f}** "
            f"(about the value of pick #{int(val[val <= max(diff, 0.01)].index.min())})."
            if diff > 0
            else "**Even trade.**"
        )
    st.dataframe(
        pv[["pick", "value"]].T.round(0).astype(int),
        width="stretch",
        hide_index=False,
    )

elif page == "steals":
    theme.hero(
        "Steals & busts",
        "Who should have gone higher, or lower",
        "The biggest gaps between draft slot and redraft position.",
    )
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
        "played_like",
        "grade",
        "value_vs_slot",
    ]
    cfg = {
        "played_like": "Played like",
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
    theme.hero(
        "Teams",
        "Team draft report cards",
        "How each franchise's picks did against their draft slots.",
    )
    st.caption(
        "How each franchise's picks did against their draft slot. Phase 6 found these "
        "differences are no bigger than luck would produce (p = 0.85), so read this as a "
        "history of outcomes, not of drafting skill. Team = the team that got the player: "
        "picks traded on draft night count for the acquiring team (Shai: Clippers, not "
        "Charlotte)."
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
    fig.update_traces(marker_color=theme.BLUE)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        agg.drop(columns=["sd", "se"]),
        hide_index=True,
        width="stretch",
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

elif page == "recruits":
    theme.hero(
        "Recruits",
        "Did teams misjudge high-school rankings?",
        "Top recruits vs their draft slot, 2005-2021.",
    )
    tests = data["recruit_tests"].set_index(["horizon", "scope"])
    t4 = tests.loc[(4, "all picks")]
    st.markdown(
        f"**No.** Top-10 high-school recruits (RSCI) were drafted about 20 picks earlier than "
        f"unranked players, and after that their careers matched their draft slot like "
        f"everyone else's. There is no trend from recruiting rank to beating the slot "
        f"(year 4: rho = {t4['spearman_rho']:+.2f}, p = {t4['p_value']:.2f}; "
        f"n = {int(t4['n'])} college picks, 2005-2021). Teams already price recruiting "
        "pedigree into the pick. The analysis plan was written down before it ran "
        "(DECISIONS D035)."
    )
    chart = ROOT / "reports" / "recruits" / "recruit_rank_vs_slot.png"
    if chart.exists():
        st.image(str(chart), width="stretch")
    summ = data["recruit_summary"]
    show = summ[(summ["horizon"] == 4) & (summ["scope"] == "all picks")]
    st.dataframe(
        show[["group", "n", "avg_pick", "mean_pit", "ci_lo", "ci_hi", "beat_median"]],
        hide_index=True,
        width="stretch",
        column_config={
            "group": "Recruit rank",
            "avg_pick": st.column_config.NumberColumn("Avg pick", format="%.0f"),
            "mean_pit": st.column_config.NumberColumn(
                "Avg PIT (0.5 = matched slot)", format="%.3f"
            ),
            "ci_lo": st.column_config.NumberColumn("95% CI low", format="%.3f"),
            "ci_hi": st.column_config.NumberColumn("95% CI high", format="%.3f"),
            "beat_median": st.column_config.ProgressColumn(
                "Beat slot median", format="percent", min_value=0, max_value=1
            ),
        },
    )
    rec = data["recruits"]
    rec4 = rec[rec["horizon"] == 4].set_index("bbref_id")
    joined = grades.set_index("bbref_id").join(rec4[["recruit_rank", "group", "pit"]], how="inner")
    joined = joined.reset_index()
    cols = ["bbref_id", "player_name", "draft_year", "pick", "grade", "current_tier"]
    cfg = {
        "draft_year": st.column_config.NumberColumn("Year", format="%d"),
        "current_tier": "Tier",
        "recruit_rank": st.column_config.NumberColumn("RSCI rank", format="%d"),
    }
    left, right = st.columns(2)
    with left:
        st.markdown("**Unranked in high school, became stars**")
        stars = joined[joined["group"] == "Unranked"].nlargest(15, "current_median")
        clickable(stars[cols], key="unranked_stars", column_config=cfg)
    with right:
        st.markdown("**Top-10 recruits who fell furthest short of their slot**")
        short = joined[joined["group"] == "RSCI 1-10"].nsmallest(15, "pit")
        clickable(short[[*cols, "recruit_rank"]], key="top_recruit_misses", column_config=cfg)
    st.caption("Click a row to open that player's card. Lists cover 2005-2021 college picks.")

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
    st.dataframe(table, width="stretch")
    a.image(card_path(pa), width="stretch")
    b.image(card_path(pb), width="stretch")

elif page == "styles":
    theme.hero(
        "Style map",
        "Where every player shoots from",
        "About 1,900 college and NBA shot diets on one map.",
    )
    sm = data["style_map"].dropna(subset=["player_name"])
    view = st.radio(
        "Layout",
        ["Rim vs three (readable axes)", "Style similarity map (t-SNE)"],
        horizontal=True,
    )
    c1, c2 = st.columns([1, 2])
    source = c1.multiselect(
        "Show", sorted(sm["source"].unique()), default=sorted(sm["source"].unique())
    )
    sm = sm[sm["source"].isin(source)]
    highlight = c2.multiselect(
        "Highlight players",
        sorted(sm["player_name"].unique()),
        default=[
            n
            for n in ["AJ Dybantsa", "Cooper Flagg", "Victor Wembanyama", "Stephen Curry"]
            if n in set(sm["player_name"])
        ],
    )
    readable = view.startswith("Rim")
    xcol, ycol = ("rim_share", "three_share") if readable else ("x", "y")
    if readable:
        sm = sm.dropna(subset=["rim_share", "three_share"])
        st.caption(
            "Each dot is one player's shot diet: **across** = share of shots at the rim "
            "(within 4 ft), **up** = share of shots from three. College dots use pre-draft "
            "shots; NBA dots use his first four seasons. Top-left = shooters, bottom-right = "
            "rim attackers, bottom-left = midrange-heavy."
        )
    else:
        st.caption(
            "**The axes have no units.** t-SNE places players with a similar mix of the six "
            "shot styles near each other, so only closeness matters: neighbors shoot from "
            "similar places, while left/right, up/down and long distances mean nothing (a "
            "rerun could rotate or flip the picture). Use the readable layout for "
            "interpretable axes."
        )
    hover = {"x": False, "y": False, "rim_share": ":.0%", "three_share": ":.0%"}
    fig = px.scatter(
        sm,
        x=xcol,
        y=ycol,
        color="dominant_style",
        symbol="source",
        opacity=0.55,
        hover_name="player_name",
        hover_data=hover,
        labels={
            "rim_share": "Share of shots at the rim",
            "three_share": "Share of shots from three",
            "dominant_style": "Dominant style",
            "source": "Shots from",
        },
        height=650,
    )
    fig.update_traces(marker={"size": 8})
    hi_pts = sm[sm["player_name"].isin(highlight)]
    fig.add_scatter(
        x=hi_pts[xcol],
        y=hi_pts[ycol],
        mode="markers+text",
        text=hi_pts["player_name"]
        + np.where(hi_pts["source"] == "college", " (college)", " (NBA)"),
        textposition="top center",
        name="highlighted",
        marker={"size": 13, "color": theme.TEXT, "line": {"width": 3, "color": theme.HERO_GOLD}},
    )
    if readable:
        fig.update_layout(
            xaxis={"tickformat": ".0%", "range": [0, 1]},
            yaxis={"tickformat": ".0%", "range": [0, 0.9]},
        )
    else:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    fig.update_layout(legend_title_text="Dominant style")
    st.plotly_chart(fig, width="stretch")
    if not readable:
        st.caption(
            "A player can appear twice: once for his college shots and once for his NBA "
            "shots (e.g. Cooper Flagg)."
        )

elif page == "tracker":
    theme.hero(
        "2026 tracker",
        "The rookies, live",
        "Draft-night projection vs what they're actually doing this season.",
    )
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
            line={"width": 6, "color": "rgba(10,132,255,0.35)"},
        )
    fig.add_scatter(
        x=top["projected_median"],
        y=top["player_name"],
        mode="markers",
        marker={"symbol": "line-ns", "size": 14, "color": theme.BLUE, "line": {"width": 2}},
        name="projected median",
    )
    st.plotly_chart(fig, width="stretch")
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
        width="stretch",
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
