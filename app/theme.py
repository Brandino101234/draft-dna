"""Draft DNA look: "Apple meets superhero".

Apple: system-style type (Inter), generous spacing, quiet frosted-glass panels, restraint.
Superhero: near-black stage with a spotlight glow, one bold hero gradient (red to gold) on
the wordmark and headlines, electric-blue accents.

Base colors come from .streamlit/config.toml; this adds the CSS and a matching Plotly
template. Chart colors were checked with the dataviz palette validator on the dark surface.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, RAISED = "#07070b", "#14141c", "#1c1c26"
TEXT, TEXT_2, MUTED, LINE = "#f5f5f7", "#a1a1aa", "#6b6b76", "rgba(255,255,255,0.08)"
BLUE, HERO_RED, HERO_GOLD = "#0a84ff", "#ff375f", "#ff9f0a"
# Categorical chart colors (validated: lightness band, CVD separation, 3:1 on SURFACE).
CHART_COLORS = ["#0a84ff", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
GRADE_COLORS = {"A": "#30d158", "B": "#0a84ff", "C": "#ffd60a", "D": "#ff453a", "-": MUTED}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Helvetica Neue",
    Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}}

/* Stage: near-black with a soft spotlight from the top and a faint hero glow */
.stApp {{
  background:
    radial-gradient(1200px 520px at 50% -180px, rgba(10,132,255,0.16), transparent 70%),
    radial-gradient(700px 380px at 92% 0%, rgba(255,55,95,0.10), transparent 70%),
    {BG};
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}

/* Type: big, tight, confident */
h1, h2, h3 {{ letter-spacing: -0.025em; font-weight: 800 !important; color: {TEXT}; }}
h3 {{ font-size: 1.35rem !important; }}
p, li {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {TEXT_2} !important; }}

/* Hero header */
.dd-hero {{ margin: 0.2rem 0 1.4rem; }}
.dd-eyebrow {{
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
  color: {BLUE}; margin-bottom: 0.35rem;
}}
.dd-title {{
  font-size: clamp(2rem, 4.2vw, 3.1rem); font-weight: 900; line-height: 1.04;
  letter-spacing: -0.035em; margin: 0; display: inline-block; padding-bottom: 0.08em;
  background: linear-gradient(92deg, {TEXT} 0%, {TEXT} 45%, {HERO_GOLD} 78%, {HERO_RED} 100%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent !important; -webkit-text-fill-color: transparent !important;
}}
.dd-sub {{ font-size: 1.08rem; color: {TEXT_2}; margin-top: 0.6rem; max-width: 760px;
  line-height: 1.5; }}

/* Sidebar: frosted glass, gradient wordmark, pill navigation */
[data-testid="stSidebar"] {{
  background: rgba(12,12,18,0.72);
  backdrop-filter: saturate(160%) blur(20px); -webkit-backdrop-filter: saturate(160%) blur(20px);
  border-right: 1px solid {LINE};
}}
.dd-brand {{
  font-size: 1.7rem; font-weight: 900; letter-spacing: -0.04em; margin: 0.2rem 0 0.1rem;
  background: linear-gradient(90deg, {HERO_RED}, {HERO_GOLD}); display: inline-block;
  -webkit-background-clip: text; background-clip: text;
  color: transparent !important; -webkit-text-fill-color: transparent !important;
}}
.dd-tagline {{ color: {TEXT_2}; font-size: 0.86rem; margin-bottom: 1.2rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 0.42rem 0.7rem; border-radius: 10px; margin: 1px 0; width: 100%;
  transition: background 0.15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.05); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: rgba(10,132,255,0.16);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
  color: {TEXT}; font-weight: 700;
}}

/* Metric tiles: glass cards with a hero edge on hover */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  border: 1px solid {LINE}; border-radius: 18px; padding: 1rem 1.1rem;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
  border-color: rgba(255,159,10,0.45); transform: translateY(-2px);
  box-shadow: 0 10px 30px -12px rgba(255,55,95,0.35);
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.7rem !important; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: {TEXT_2} !important;
}}
[data-testid="stMetric"] label, [data-testid="stMetric"] label *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"] *,
[data-testid="stMetricValue"] * {{
  max-width: none !important;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
[data-testid="stMetricValue"] {{
  font-size: clamp(1.4rem, 2.2vw, 2rem) !important; line-height: 1.15;
  font-weight: 800; letter-spacing: -0.03em;
}}

/* Tables, charts and images sit on rounded glass panels */
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stImage"] img {{
  border-radius: 16px; overflow: hidden;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; }}
[data-testid="stImage"] img {{
  border: 1px solid {LINE}; box-shadow: 0 30px 80px -40px rgba(10,132,255,0.45);
}}

/* Bordered containers (home findings): glass panels */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
  border-radius: 18px !important; border-color: {LINE} !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.012));
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
  border-color: rgba(10,132,255,0.45) !important;
  box-shadow: 0 16px 40px -24px rgba(10,132,255,0.6);
}}

/* Buttons: pill, quiet until hovered, then the hero gradient */
.stButton > button {{
  border-radius: 999px; border: 1px solid {LINE}; background: rgba(255,255,255,0.04);
  color: {TEXT}; font-weight: 600; padding: 0.35rem 1rem; transition: all 0.18s ease;
}}
.stButton > button:hover {{
  border-color: transparent; color: #fff;
  background: linear-gradient(90deg, {HERO_RED}, {HERO_GOLD});
  box-shadow: 0 8px 24px -10px rgba(255,55,95,0.6);
}}

/* Inputs: soft, rounded */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  border-radius: 12px !important; background: {SURFACE} !important; border-color: {LINE} !important;
}}
[data-baseweb="tag"] {{
  border-radius: 999px !important; background: rgba(10,132,255,0.22) !important;
}}
[data-testid="stAlert"] {{ border-radius: 14px; border: 1px solid {LINE}; }}
hr {{ border-color: {LINE}; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  [data-testid="stMetric"] {{ padding: 0.8rem; }}
}}
</style>
"""


def apply() -> None:
    """Inject the CSS and make the hero Plotly template the default."""
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["draft_dna"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, -apple-system, Helvetica Neue, Arial", "color": TEXT_2},
            colorway=CHART_COLORS,
            xaxis={"gridcolor": "rgba(255,255,255,0.06)", "zerolinecolor": LINE, "linecolor": LINE},
            yaxis={"gridcolor": "rgba(255,255,255,0.06)", "zerolinecolor": LINE, "linecolor": LINE},
            legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT_2}},
            hoverlabel={"bgcolor": RAISED, "bordercolor": LINE, "font": {"color": TEXT}},
            margin={"t": 30, "r": 10, "b": 40, "l": 10},
        )
    )
    pio.templates.default = "plotly_dark+draft_dna"


def hero(eyebrow: str, title: str, sub: str = "") -> None:
    """Page header: small blue eyebrow, big gradient headline, quiet subtitle."""
    st.markdown(
        f'<div class="dd-hero"><div class="dd-eyebrow">{eyebrow}</div>'
        f'<div class="dd-title" role="heading" aria-level="1">{title}</div>'
        + (f'<div class="dd-sub">{sub}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.sidebar.markdown(
        '<div class="dd-brand">DRAFT DNA</div>'
        '<div class="dd-tagline">Every NBA pick since 1996: what was expected, '
        "what happened.</div>",
        unsafe_allow_html=True,
    )
