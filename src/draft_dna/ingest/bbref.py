"""Basketball-Reference ingestion: drafts, season stats, awards, coaches, player bios.

All pages go through the shared Fetcher (cached, <=~17 req/min). Parsers return
string-valued DataFrames; typing and cleaning happen in staging.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag

from draft_dna.ingest.fetcher import Fetcher, NotFoundError
from draft_dna.ingest.html_tables import parse_table, table_ids
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

BASE = "https://www.basketball-reference.com"
SOURCE = "sports_reference"
# Pages for the season in progress change daily; everything older is final.
IN_SEASON_MAX_AGE = timedelta(hours=20)


def _max_age(season: int, current_season: int) -> timedelta | None:
    return IN_SEASON_MAX_AGE if season >= current_season else None


# --------------------------------------------------------------------------- drafts
def parse_draft(html: bytes, year: int) -> pd.DataFrame:
    """One row per pick, with round taken from the page's 'Round N' header rows."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="stats")
    if not isinstance(table, Tag):
        raise ValueError(f"draft {year}: no #stats table")
    body = table.find("tbody")
    assert isinstance(body, Tag)
    rows: list[dict[str, Any]] = []
    rnd = 1
    for tr in body.find_all("tr"):
        classes = tr.get("class") or []
        if "over_header" in classes:
            m = re.search(r"Round (\d+)", tr.get_text())
            if m:
                rnd = int(m.group(1))
            continue
        if "thead" in classes:
            continue
        cells = {
            str(c.get("data-stat")): c for c in tr.find_all(["th", "td"]) if c.get("data-stat")
        }
        if "pick_overall" not in cells or not cells["pick_overall"].get_text(strip=True):
            continue
        player = cells.get("player")
        link = player.find("a") if player else None
        college = cells.get("college_name")
        team = cells.get("team_id")
        rows.append(
            {
                "draft_year": year,
                "round": rnd,
                "pick_overall": int(cells["pick_overall"].get_text(strip=True)),
                "team_id": team.get_text(strip=True) if team else None,
                "player_name": player.get_text(strip=True) if player else None,
                "bbref_id": (
                    re.search(r"/players/\w/([\w.]+)\.html", str(link["href"])).group(1)  # type: ignore[union-attr]
                    if isinstance(link, Tag)
                    else None
                ),
                "college_name": college.get_text(strip=True) or None if college else None,
            }
        )
    return pd.DataFrame(rows)


def ingest_drafts(fetcher: Fetcher, years: Iterable[int], current_season: int) -> pd.DataFrame:
    frames = []
    for y in years:
        html = fetcher.get(f"{BASE}/draft/NBA_{y}.html", max_age=_max_age(y, current_season - 1))
        frames.append(parse_draft(html, y))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- seasons
SEASON_PAGES: dict[str, tuple[str, dict[str, str]]] = {
    # name: (url template, {table_id: phase})
    "totals": (
        "/leagues/NBA_{s}_totals.html",
        {"totals_stats": "regular", "totals_stats_post": "playoffs"},
    ),
    "advanced": (
        "/leagues/NBA_{s}_advanced.html",
        {"advanced": "regular", "advanced_post": "playoffs"},
    ),
    "per_poss": (
        "/leagues/NBA_{s}_per_poss.html",
        {"per_poss": "regular", "per_poss_post": "playoffs"},
    ),
}


def ingest_season_player_stats(
    fetcher: Fetcher, seasons: Iterable[int], current_season: int
) -> dict[str, pd.DataFrame]:
    """Player-season tables (regular season + playoffs). `season` = ending year."""
    out: dict[str, list[pd.DataFrame]] = {name: [] for name in SEASON_PAGES}
    for s in seasons:
        for name, (tmpl, tables) in SEASON_PAGES.items():
            try:
                html = fetcher.get(BASE + tmpl.format(s=s), max_age=_max_age(s, current_season))
            except NotFoundError:
                log.warning("no %s page for %d", name, s)
                continue
            for tid, phase in tables.items():
                df = parse_table(html, tid)
                if df.empty:
                    continue
                df.insert(0, "phase", phase)
                df.insert(0, "season", s)
                out[name].append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def ingest_team_seasons(
    fetcher: Fetcher, seasons: Iterable[int], current_season: int
) -> pd.DataFrame:
    """Team ratings (wins, SRS, pace, ORtg/DRtg) from the league season page."""
    frames = []
    for s in seasons:
        html = fetcher.get(f"{BASE}/leagues/NBA_{s}.html", max_age=_max_age(s, current_season))
        df = parse_table(html, "advanced-team")
        if df.empty:
            continue
        df = df[df["team"].str.len() > 0]
        df = df[df["team"] != "League Average"]
        df.insert(0, "season", s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ingest_coaches(fetcher: Fetcher, seasons: Iterable[int], current_season: int) -> pd.DataFrame:
    frames = []
    for s in seasons:
        html = fetcher.get(
            f"{BASE}/leagues/NBA_{s}_coaches.html", max_age=_max_age(s, current_season)
        )
        df = parse_table(html, "NBA_coaches")
        df.insert(0, "season", s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


AWARD_TABLES = {
    "mvp": "mvp",
    "roy": "roy",
    "dpoy": "dpoy",
    "smoy": "smoy",
    "mip": "mip",
    "leading_all_nba": "all_nba",
    "leading_all_defense": "all_defense",
    "leading_all_rookie": "all_rookie",
}


def ingest_awards(fetcher: Fetcher, seasons: Iterable[int]) -> pd.DataFrame:
    """Award voting (MVP, ROY, ...) and All-NBA/All-Defense/All-Rookie teams.

    Only completed seasons have award pages.
    """
    frames = []
    for s in seasons:
        try:
            html = fetcher.get(f"{BASE}/awards/awards_{s}.html")
        except NotFoundError:
            log.warning("no awards page for %d", s)
            continue
        for tid, award in AWARD_TABLES.items():
            df = parse_table(html, tid)
            if df.empty:
                continue
            df.insert(0, "award", award)
            df.insert(0, "season", s)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ingest_all_stars(fetcher: Fetcher, seasons: Iterable[int]) -> pd.DataFrame:
    """All-Star rosters (one table per All-Star team; 1999 had no game)."""
    frames = []
    for s in seasons:
        try:
            html = fetcher.get(f"{BASE}/allstar/NBA_{s}.html")
        except NotFoundError:
            log.info("no All-Star game in %d", s)
            continue
        for tid in table_ids(html):
            if tid == "line_score":
                continue
            df = parse_table(html, tid)
            if df.empty or "player__id" not in df:
                continue
            df = df[df["player__id"].notna()]
            df.insert(0, "all_star_team", tid)
            df.insert(0, "season", s)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ingest_league_averages(fetcher: Fetcher, current_season: int) -> pd.DataFrame:
    html = fetcher.get(f"{BASE}/leagues/NBA_stats_per_game.html", max_age=IN_SEASON_MAX_AGE)
    df = parse_table(html, "stats-Regular-Season")
    return df[df["season"].str.len() > 0]


def ingest_salary_cap(fetcher: Fetcher) -> pd.DataFrame:
    """League salary cap by season (used to express salaries as % of cap)."""
    html = fetcher.get(f"{BASE}/contracts/salary-cap-history.html", max_age=IN_SEASON_MAX_AGE)
    return parse_table(html, "salary_cap_history")


# --------------------------------------------------------------------------- players
def ingest_player_index(fetcher: Fetcher) -> pd.DataFrame:
    """Every NBA/ABA player: years active, position, size, birth date, colleges."""
    frames = []
    for letter in string.ascii_lowercase:
        try:
            html = fetcher.get(f"{BASE}/players/{letter}/")
        except NotFoundError:
            continue  # there is no 'x' index on some versions of the site
        frames.append(parse_table(html, "players"))
    return pd.concat(frames, ignore_index=True)


def player_url(bbref_id: str) -> str:
    return f"{BASE}/players/{bbref_id[0]}/{bbref_id}.html"


_CBB_LINK = re.compile(r"https?://www\.sports-reference\.com/cbb/players/([\w-]+)\.html")


def parse_transactions(html: bytes, bbref_id: str) -> pd.DataFrame:
    """Transactions block: date, kind (drafted / traded / other) and the team codes from
    the links' data-attr-from / data-attr-to (first pair = this player's own move)."""
    # The block often sits inside an HTML comment (rendered by JavaScript), so parse
    # just that fragment from the raw text.
    raw = html.decode("utf-8", "ignore")
    start = raw.find('id="div_transactions"')
    rows: list[dict[str, Any]] = []
    if start < 0:
        return pd.DataFrame(rows)
    end = raw.find("</div>", start)
    soup = BeautifulSoup("<div " + raw[start : end + 6], "lxml")
    block = soup.find(id="div_transactions")
    if not isinstance(block, Tag):
        return pd.DataFrame(rows)
    for p in block.find_all("p", class_="transaction"):
        date = p.find("strong")
        text = p.get_text(" ", strip=True)
        frm = p.find(attrs={"data-attr-from": True})
        to = p.find(attrs={"data-attr-to": True})
        kind = (
            "drafted"
            if re.search(r"\bDrafted by\b", text)
            else "traded"
            if re.search(r"\btraded by\b", text, re.IGNORECASE)
            else "other"
        )
        rows.append(
            {
                "bbref_id": bbref_id,
                "date": pd.to_datetime(date.get_text(strip=True), errors="coerce")
                if isinstance(date, Tag)
                else pd.NaT,
                "kind": kind,
                "team_from": frm["data-attr-from"] if isinstance(frm, Tag) else None,
                "team_to": to["data-attr-to"] if isinstance(to, Tag) else None,
                "text": text[:300],
            }
        )
    return pd.DataFrame(rows)


def parse_player_page(html: bytes, bbref_id: str) -> dict[str, Any]:
    """Bio fields from the #meta block plus the college-stats link."""
    soup = BeautifulSoup(html, "lxml")
    meta = soup.find(id="meta")
    text = meta.get_text(" ", strip=True) if isinstance(meta, Tag) else ""
    bio: dict[str, Any] = {"bbref_id": bbref_id}

    birth = soup.find(id="necro-birth")
    bio["birth_date"] = birth.get("data-birth") if isinstance(birth, Tag) else None
    m = re.search(r"Position:\s*(.+?)\s*(?:▪|Shoots:)", text)
    bio["position"] = m.group(1).strip() if m else None
    m = re.search(r"Shoots:\s*(Left|Right)", text)
    bio["shoots"] = m.group(1) if m else None
    m = re.search(r"\((\d+)cm,\s*(\d+)kg\)", text)
    bio["height_cm"], bio["weight_kg"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"(\d)-(\d{1,2})\s*,\s*(\d+)lb", text)
    bio["height_in"] = int(m.group(1)) * 12 + int(m.group(2)) if m else None
    bio["weight_lb"] = int(m.group(3)) if m else None
    m = re.search(r"College:\s*(.+?)\s*(?:High School:|Recruiting Rank:|Draft:|NBA Debut:|$)", text)
    bio["colleges"] = m.group(1).strip() if m else None
    m = re.search(r"High School:\s*(.+?)\s*(?:Recruiting Rank:|Draft:|NBA Debut:|$)", text)
    bio["high_school"] = m.group(1).strip() if m else None
    m = re.search(r"Recruiting Rank:\s*(\d{4})\s*\((\d+)\)", text)
    bio["recruit_year"], bio["recruit_rank"] = (
        (int(m.group(1)), int(m.group(2))) if m else (None, None)
    )
    m = re.search(r"Draft:\s*(.+?)\s*(?:NBA Debut:|Experience:|Career Length:|$)", text)
    bio["draft_text"] = m.group(1).strip() if m else None
    m = re.search(r"NBA Debut:\s*(\w+ \d{1,2}, \d{4})", text)
    bio["nba_debut"] = m.group(1) if m else None
    m = re.search(
        r"Born:.*?\bin\s+(.+?)\s+(?:College:|High School:|Draft:|NBA Debut:|Died:|$)", text
    )
    # The country flag code ("us") trails the place name; it is stored separately.
    bio["birthplace"] = re.sub(r"\s+[a-z]{2}$", "", m.group(1).strip()) if m else None
    flag = meta.find("span", class_=re.compile(r"f-i f-")) if isinstance(meta, Tag) else None
    bio["birth_country"] = flag.get_text(strip=True) if isinstance(flag, Tag) else None
    m = _CBB_LINK.search(html.decode("utf-8", errors="replace"))
    bio["cbb_id"] = m.group(1) if m else None
    return bio


def ingest_player_pages(fetcher: Fetcher, bbref_ids: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Bio, college box stats (as listed on BBRef), salary history and transactions."""
    bios: list[dict[str, Any]] = []
    transactions: list[pd.DataFrame] = []
    colleges: list[pd.DataFrame] = []
    salaries: list[pd.DataFrame] = []
    ids = sorted(set(bbref_ids))
    for i, pid in enumerate(ids, 1):
        if i % 100 == 0 or not fetcher.is_cached(player_url(pid)):
            log.info("player pages: %d/%d (%s)", i, len(ids), pid)
        try:
            html = fetcher.get(player_url(pid))
        except NotFoundError:
            log.warning("no player page for %s", pid)
            continue
        bios.append(parse_player_page(html, pid))
        tx = parse_transactions(html, pid)
        if not tx.empty:
            transactions.append(tx)
        for tid, sink in (("all_college_stats", colleges), ("all_salaries", salaries)):
            df = parse_table(html, tid)
            if not df.empty:
                df.insert(0, "bbref_id", pid)
                sink.append(df)
    return {
        "player_bios": pd.DataFrame(bios),
        "player_college_box": pd.concat(colleges, ignore_index=True)
        if colleges
        else pd.DataFrame(),
        "player_salaries": pd.concat(salaries, ignore_index=True) if salaries else pd.DataFrame(),
        "player_transactions": pd.concat(transactions, ignore_index=True)
        if transactions
        else pd.DataFrame(),
    }
