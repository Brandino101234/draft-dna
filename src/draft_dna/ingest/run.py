"""Orchestrate ingestion per source. Every step is resumable: cached pages are free."""

from __future__ import annotations

import pandas as pd

from draft_dna.config import Settings, get_settings
from draft_dna.ingest import barttorvik, bbref, cbb, espn, nba_stats
from draft_dna.ingest.fetcher import Fetcher, FetchError
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)


def nba_seasons(s: Settings) -> list[int]:
    """NBA seasons (ending year) from the first training class's rookie year to now."""
    return list(range(s.draft_classes.training[0] + 1, s.current_nba_season + 1))


def completed_seasons(s: Settings) -> list[int]:
    return [y for y in nba_seasons(s) if y < s.current_nba_season]


def run_bbref_league(settings: Settings | None = None) -> None:
    """Drafts, season stats, team ratings, coaches, awards, All-Stars, player index."""
    s = settings or get_settings()
    f = Fetcher(bbref.SOURCE, settings=s)
    cur = s.current_nba_season

    write_table(
        bbref.ingest_drafts(f, s.draft_classes.all_years(), cur), "raw", "bbref", "draft", s
    )
    write_table(bbref.ingest_player_index(f), "raw", "bbref", "player_index", s)
    write_table(bbref.ingest_league_averages(f, cur), "raw", "bbref", "league_averages", s)
    write_table(bbref.ingest_salary_cap(f), "raw", "bbref", "salary_cap", s)
    for name, df in bbref.ingest_season_player_stats(f, nba_seasons(s), cur).items():
        write_table(df, "raw", "bbref", f"season_{name}", s)
    write_table(
        bbref.ingest_team_seasons(f, nba_seasons(s), cur), "raw", "bbref", "team_seasons", s
    )
    write_table(bbref.ingest_coaches(f, nba_seasons(s), cur), "raw", "bbref", "coaches", s)
    write_table(bbref.ingest_awards(f, completed_seasons(s)), "raw", "bbref", "awards", s)
    write_table(bbref.ingest_all_stars(f, completed_seasons(s)), "raw", "bbref", "all_stars", s)
    log.info("bbref league pages done (%d network requests)", f.network_requests)


def player_page_ids(s: Settings) -> list[str]:
    """Everyone drafted in our classes plus everyone who debuted in our NBA seasons.

    The second group contains the undrafted players; bios tell us which is which.
    """
    draft = read_table("raw", "bbref", "draft", s)
    index = read_table("raw", "bbref", "player_index", s)
    debut_ok = pd.to_numeric(index["year_min"], errors="coerce") >= nba_seasons(s)[0]
    ids = set(draft["bbref_id"].dropna()) | set(index.loc[debut_ok, "player__id"].dropna())
    return sorted(ids)


def run_bbref_players(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if not table_path("raw", "bbref", "draft", s).exists():
        raise RuntimeError("run the bbref league step first")
    f = Fetcher(bbref.SOURCE, settings=s)
    ids = player_page_ids(s)
    log.info("player pages: %d players (cached ones are free)", len(ids))
    for name, df in bbref.ingest_player_pages(f, ids).items():
        write_table(df, "raw", "bbref", name, s)
    log.info("bbref player pages done (%d network requests)", f.network_requests)


def run_barttorvik(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    f = Fetcher(barttorvik.SOURCE, settings=s)
    last_college = s.draft_classes.all_years()[-1]  # college season ending in the draft year
    seasons = range(barttorvik.FIRST_SEASON, last_college + 1)
    write_table(barttorvik.ingest_players(f, seasons), "raw", "barttorvik", "players", s)
    write_table(barttorvik.ingest_teams(f, seasons), "raw", "barttorvik", "teams", s)


def run_nba_api(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    write_table(nba_stats.draft_history(s), "raw", "nba_api", "draft_history", s)
    write_table(nba_stats.combine(s), "raw", "nba_api", "combine", s)
    write_table(nba_stats.all_players(s), "raw", "nba_api", "all_players", s)


def college_seasons(s: Settings) -> list[int]:
    """College seasons (ending year) that can precede our draft classes: up to 4 years."""
    years = s.draft_classes.all_years()
    return list(range(years[0] - 3, years[-1] + 1))


def run_cbb_seasons(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    f = Fetcher(cbb.SOURCE, settings=s)
    for name, df in cbb.ingest_team_seasons(f, college_seasons(s)).items():
        write_table(df, "raw", "cbb", name, s)


def run_cbb_players(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if not table_path("raw", "bbref", "player_bios", s).exists():
        raise RuntimeError("run the bbref-players step first (it provides college links)")
    bios = read_table("raw", "bbref", "player_bios", s)
    f = Fetcher(cbb.SOURCE, settings=s)
    for name, df in cbb.ingest_players(f, bios["cbb_id"].dropna()).items():
        write_table(df, "raw", "cbb", name, s)


def run_espn_shots(settings: Settings | None = None) -> None:
    """Every game of every drafted player's pre-draft team-season, 2008-2026 (ESPN PBP).

    Writes one parquet per season so a long pull keeps its progress; reruns are cheap
    because every response is cached.
    """
    from draft_dna.eval.shot_audit import SEASONS, team_seasons

    s = settings or get_settings()
    f = Fetcher(espn.SOURCE, settings=s)
    ts, _ = team_seasons(s)
    ts = ts.dropna(subset=["espn_team_id"])
    write_table(ts, "raw", "espn", "team_seasons", s)
    now = pd.Timestamp.now(tz="UTC")
    for season in SEASONS:
        shots, rosters, games = [], [], []
        for r in ts[ts["season"] == season].itertuples():
            sched = espn.schedule(f, int(r.espn_team_id), season)
            if sched.empty:
                continue
            sched = sched[pd.to_datetime(sched["date"], utc=True) < now]
            games.append(sched.assign(school_slug=r.school_slug))
            for gid in sched["game_id"]:
                sh, ro, _ = espn.game_shots(f, int(gid))
                shots.append(sh)
                rosters.append(ro)
        if not games:
            continue
        g = pd.concat(games, ignore_index=True)
        sh = pd.concat(shots, ignore_index=True).drop_duplicates()
        ro = pd.concat(rosters, ignore_index=True).drop_duplicates()
        write_table(g.assign(season=season), "raw", "espn", f"games_{season}", s)
        write_table(sh.assign(season=season), "raw", "espn", f"shots_{season}", s)
        write_table(ro.assign(season=season), "raw", "espn", f"rosters_{season}", s)
        log.info(
            "espn %d: %d games, %d shots (%d network requests so far)",
            season,
            len(g),
            len(sh),
            f.network_requests,
        )


NBA_SHOT_SEASONS = 4  # early-career seasons used for "plays like" style comps


def run_nba_shots(settings: Settings | None = None) -> None:
    """stats.nba.com shot charts for each player's first NBA seasons (1996-97 onward)."""
    s = settings or get_settings()
    xw = read_table("modeled", "xwalk", "players", s).dropna(subset=["nba_person_id"])
    seasons = read_table("modeled", "core", "nba_player_seasons", s)
    seasons = seasons[seasons["bbref_id"].isin(xw["bbref_id"]) & (seasons["mp"] > 0)]
    seasons = seasons[seasons["season"] < s.current_nba_season]
    early = seasons.sort_values("season").groupby("bbref_id").head(NBA_SHOT_SEASONS)
    pid = xw.set_index("bbref_id")["nba_person_id"].astype(int)
    frames = []
    skipped = 0
    for i, r in enumerate(early.itertuples(), 1):
        try:
            df = nba_stats.shot_chart(int(pid[r.bbref_id]), int(r.season), s)
        except FetchError as exc:  # persistent throttling: skip; a rerun retries it
            log.warning("skipping %s %s: %s", r.bbref_id, r.season, exc)
            skipped += 1
            continue
        if not df.empty:
            frames.append(df.assign(bbref_id=r.bbref_id, season=int(r.season)))
        if i % 500 == 0:
            log.info("nba shots: %d/%d player-seasons", i, len(early))
    write_table(pd.concat(frames, ignore_index=True), "raw", "nba_api", "shots_early_career", s)
    log.info("nba shots done; %d player-seasons skipped (rerun to retry)", skipped)


STEPS = {
    "bbref-league": run_bbref_league,
    "bbref-players": run_bbref_players,
    "cbb-seasons": run_cbb_seasons,
    "cbb-players": run_cbb_players,
    "barttorvik": run_barttorvik,
    "nba-api": run_nba_api,
    "espn-shots": run_espn_shots,
    "nba-shots": run_nba_shots,
}
