"""Player ID crosswalk anchored on the Basketball-Reference ID.

Universe: every draft pick in our classes plus every undrafted player who debuted
in our NBA seasons. For each player we resolve:

- nba_person_id  (stats.nba.com): draft slot for picks; name + debut season otherwise
- cbb_id         (Sports-Reference CBB): the link on the BBRef bio page
- bart_pid       (Barttorvik, 2008+): birthdate, NBA pick, name, school and season
- combine        via nba_person_id, else name + combine year

Every link records a method and score. Manual overrides (committed CSV) win.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from draft_dna.config import Settings
from draft_dna.crosswalk.names import last_name, normalize_name
from draft_dna.ingest import nba_stats
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

OVERRIDES = Path(__file__).parent / "overrides.csv"
NAME_MIN = 80  # minimum name similarity to accept a candidate with other evidence
FUZZY_MIN = 92  # minimum similarity to accept on name alone


def name_sim(a: str, b: str) -> float:
    return float(fuzz.token_sort_ratio(normalize_name(a), normalize_name(b)))


_SCHOOL_SUBS = [
    (r"\(.*?\)", ""),  # "(CA)", "(FL)"
    (r"^st\b", "saint"),  # leading "St." means Saint (St. John's)
    (r"\bst\b", "state"),  # elsewhere it means State (Kansas St.)
    (r"\buniv(ersity)?\b", ""),
    (r"\bof\b", ""),
    (r"\bthe\b", ""),
]
SCHOOL_ALIASES = {
    "uconn": "connecticut",
    "unc": "north carolina",
    "lsu": "louisiana state",
    "usc": "southern california",
    "ucla": "ucla",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "unlv": "nevada las vegas",
    "vcu": "virginia commonwealth",
    "utep": "texas el paso",
    "uab": "alabama birmingham",
    "ucf": "central florida",
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "umass": "massachusetts",
    "miami fl": "miami",
    "miami (fl)": "miami",
}


def normalize_school(name: str | None) -> str:
    if not isinstance(name, str) or not name:
        return ""
    s = re.sub(r"\(.*?\)", "", name)
    s = normalize_name(s, drop_suffix=False)
    s = SCHOOL_ALIASES.get(s, s)
    for pat, rep in _SCHOOL_SUBS:
        s = re.sub(pat, rep, s)
    return " ".join(s.split())


def school_sim(a: str | None, b: str | None) -> float:
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return 0.0
    # token_sort, not token_set: "kansas" must not fully match "kansas state".
    return float(fuzz.token_sort_ratio(normalize_school(a), normalize_school(b)))


# ------------------------------------------------------------------------- universe
def player_universe(s: Settings) -> pd.DataFrame:
    """One row per player: drafted (our classes) + undrafted debuting in our seasons."""
    draft = read_table("staging", "bbref", "draft_picks", s)
    index = read_table("staging", "bbref", "player_index", s)
    first_season = s.draft_classes.training[0] + 1
    drafted = draft[["bbref_id", "player_name", "draft_year", "round", "pick_overall",
                     "team_id", "college_name"]].copy()  # fmt: skip
    drafted["drafted"] = True

    late = index[
        (index["first_season"] >= first_season) & ~index["bbref_id"].isin(drafted["bbref_id"])
    ]
    undrafted = late[["bbref_id", "player_name"]].copy()
    undrafted["drafted"] = False
    if table_path("staging", "bbref", "player_bios", s).exists():
        bios = read_table("staging", "bbref", "player_bios", s).set_index("bbref_id")
        draft_text = undrafted["bbref_id"].map(bios["draft_text"])
        # Drafted before our first class (e.g. a 1993 pick debuting in 1997): out of scope.
        early = draft_text.fillna("").str.contains(r"\d{4} NBA Draft")
        undrafted = undrafted[~early]
    out = pd.concat([drafted, undrafted], ignore_index=True)
    index_cols = ["bbref_id", "first_season", "last_season", "birth_date", "height_in",
                  "weight_lb", "pos"]  # fmt: skip
    out = out.merge(index[index_cols], on="bbref_id", how="left")
    if table_path("staging", "bbref", "player_bios", s).exists():
        bios = read_table("staging", "bbref", "player_bios", s)
        out = out.merge(
            bios[["bbref_id", "cbb_id", "colleges", "birth_country"]], on="bbref_id", how="left"
        )
        out["birth_date"] = out["birth_date"].fillna(
            out["bbref_id"].map(bios.set_index("bbref_id")["birth_date"])
        )
    else:
        out["cbb_id"] = pd.NA
        out["colleges"] = out.get("college_name", pd.NA)
        out["birth_country"] = pd.NA
    index_colleges = index.set_index("bbref_id")["colleges"].replace("", pd.NA)
    out["colleges"] = (
        out["colleges"].fillna(out["college_name"]).fillna(out["bbref_id"].map(index_colleges))
    )
    return out


# ---------------------------------------------------------------------- nba person id
DRAFT_FUZZY_MIN = 75
ORG_TYPE_TO_SOURCE = {
    "College/University": "college",
    "High School": "high_school",
    "Other Team/Club": "other_team",
}


def link_draft_history(drafted: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Match BBRef picks to stats.nba.com draft rows within each draft year.

    Order: exact normalized name; best fuzzy name (greedy, >= DRAFT_FUZZY_MIN); then
    same pick number for whatever is left on both sides (flagged by a low score).
    """
    out = []
    for year, picks in drafted.groupby("draft_year"):
        hist = history[history["draft_year"] == year].copy()
        hist["norm"] = hist["nba_player_name"].map(normalize_name)
        left = picks.assign(norm=picks["player_name"].map(normalize_name))
        used: set[int] = set()

        def take(
            bbref_id: str, h: pd.Series, method: str, score: float, used: set[int] = used
        ) -> None:
            used.add(int(h.name))
            out.append({"bbref_id": bbref_id, **h.drop("norm").to_dict(),
                        "draft_link_method": method, "draft_link_score": score})  # fmt: skip

        pending = []
        for r in left.itertuples():
            hits = hist[(hist["norm"] == r.norm) & ~hist.index.isin(used)]
            if len(hits) == 1:
                take(r.bbref_id, hits.iloc[0], "name", 100.0)
            else:
                pending.append(r)
        scored = []
        for r in pending:
            for idx, h in hist[~hist.index.isin(used)].iterrows():
                scored.append((name_sim(r.player_name, h["nba_player_name"]), r.bbref_id, idx))
        done: set[str] = set()
        for score, bid, idx in sorted(scored, reverse=True):
            if score < DRAFT_FUZZY_MIN or bid in done or idx in used:
                continue
            take(bid, hist.loc[idx], "fuzzy_name", score)
            done.add(bid)
        for r in pending:
            if r.bbref_id in done:
                continue
            slot = hist[(hist["pick_overall"] == r.pick_overall) & ~hist.index.isin(used)]
            if len(slot) == 1:
                h = slot.iloc[0]
                take(r.bbref_id, h, "pick_slot", name_sim(r.player_name, h["nba_player_name"]))
            else:
                out.append({"bbref_id": r.bbref_id, "draft_link_method": "unresolved",
                            "draft_link_score": 0.0})  # fmt: skip
    res = pd.DataFrame(out).drop(columns=["draft_year", "pick_overall"], errors="ignore")
    res["nba_person_id"] = res["nba_person_id"].astype("Int64")
    return res


def link_undrafted(
    universe: pd.DataFrame,
    nba_players: pd.DataFrame,
    birth_lookup: Callable[[int], str | None] | None = None,
) -> pd.DataFrame:
    """Link undrafted players to stats.nba.com by name and debut season.

    stats.nba.com debut years sometimes differ from BBRef by a season or two (signed
    but inactive), so windows are +/-2. Same-name candidates are split by birthdate.
    """
    nba = nba_players.assign(
        norm=nba_players["player_name"].map(normalize_name),
        last=nba_players["player_name"].map(last_name),
    )
    uni = universe.assign(last=universe["player_name"].map(last_name))
    rows = []
    for r in uni.itertuples():
        if pd.isna(r.first_season):
            rows.append((r.bbref_id, pd.NA, "unresolved", 0.0))
            continue
        norm = normalize_name(r.player_name)
        near = nba[(nba["first_season"] - r.first_season).abs() <= 2]
        # Exact names get a wider window (signed-but-inactive years shift debut dates).
        same = nba[(nba["norm"] == norm) & ((nba["first_season"] - r.first_season).abs() <= 3)]
        exact = same[same["first_season"] == r.first_season]
        if len(exact) == 1:
            rows.append((r.bbref_id, int(exact.iloc[0].nba_person_id), "name_season", 100.0))
            continue
        if len(same) == 1:
            rows.append((r.bbref_id, int(same.iloc[0].nba_person_id), "name_season", 100.0))
            continue
        if len(same) > 1 and birth_lookup is not None and pd.notna(r.birth_date):
            bd = pd.Timestamp(r.birth_date).strftime("%Y-%m-%d")
            hit = [int(c) for c in same["nba_person_id"] if birth_lookup(int(c)) == bd]
            if len(hit) == 1:
                rows.append((r.bbref_id, hit[0], "name_birthdate", 100.0))
                continue
        # Unique last name on both sides within the window (nicknames: Pooh/Eugene).
        # Require the same first initial (Isaac/Ike, Stanislav/Slava) so that
        # "Willie Reed" cannot fall onto "Davon Reed"; true nicknames go in overrides.
        initial = norm[:1]
        nba_last = near[(near["last"] == r.last) & (near["norm"].str[:1] == initial)]
        uni_last = uni[
            (uni["last"] == r.last) & ((uni["first_season"] - r.first_season).abs() <= 2)
        ]
        if len(nba_last) == 1 and len(uni_last) == 1:
            score = float(fuzz.token_set_ratio(norm, nba_last.iloc[0].norm))
            rows.append((r.bbref_id, int(nba_last.iloc[0].nba_person_id), "lastname_season", score))
            continue
        sims = near["player_name"].map(lambda n, name=r.player_name: name_sim(name, n))
        best = float(sims.max()) if len(sims) else 0.0
        if best >= FUZZY_MIN and (sims == best).sum() == 1:
            rows.append(
                (r.bbref_id, int(near.loc[sims.idxmax()].nba_person_id), "fuzzy_season", best)
            )
        else:
            rows.append((r.bbref_id, pd.NA, "unresolved", best))
    return pd.DataFrame(rows, columns=["bbref_id", "nba_person_id", "nba_method", "nba_score"])


def link_nba(
    universe: pd.DataFrame,
    history: pd.DataFrame,
    nba_players: pd.DataFrame,
    birth_lookup: Callable[[int], str | None] | None = None,
) -> pd.DataFrame:
    drafted = universe[universe["drafted"]]
    d = link_draft_history(drafted, history)
    d["nba_method"] = "draft_" + d["draft_link_method"]
    d["nba_score"] = d["draft_link_score"]
    d.loc[d["draft_link_method"] == "unresolved", "nba_method"] = "unresolved"
    u = link_undrafted(universe[~universe["drafted"]], nba_players, birth_lookup)
    out = pd.concat(
        [d.drop(columns=["draft_link_method", "draft_link_score"]), u], ignore_index=True
    )
    out["prospect_source"] = out["pre_draft_org_type"].map(ORG_TYPE_TO_SOURCE).fillna("unknown")
    return out


# ------------------------------------------------------------------------ barttorvik
@dataclass
class BartMatch:
    bart_pid: int | None
    method: str
    score: float


def _college_window(r: pd.Series) -> tuple[int, int]:
    if r["drafted"]:
        return int(r["draft_year"]) - 5, int(r["draft_year"])
    fs = r["first_season"]
    return (int(fs) - 7, int(fs) - 1) if pd.notna(fs) else (0, -1)


def best_school_sim(colleges: str | None, team: str) -> float:
    """Best match of a Barttorvik team against any school in a comma-separated list."""
    if not isinstance(colleges, str):
        return 0.0
    return max(school_sim(c.strip(), team) for c in colleges.split(","))


def _resolve_transfer(hit: pd.DataFrame) -> int | None:
    """Barttorvik issues a new player ID at each school. If several candidates have
    non-overlapping season ranges, they are one transfer: keep the latest school's ID.
    """
    if len(hit) < 2:
        return None
    ordered = hit.sort_values("first_season")
    if (
        ordered["first_season"].iloc[1:].to_numpy() > ordered["last_season"].iloc[:-1].to_numpy()
    ).all():
        return int(ordered.index[-1])
    return None


def match_bart(r: pd.Series, bart: pd.DataFrame) -> BartMatch:
    lo, hi = _college_window(r)
    cands = bart[(bart["season"] >= lo) & (bart["season"] <= hi)]
    if cands.empty:
        return BartMatch(None, "no_candidates", 0.0)
    last = r["player_last"]
    cands = (
        cands[
            cands["last_norm"].str.contains(last, regex=False)
            | (cands["first_norm"] == r["player_first"])
        ]
        if last
        else cands
    )
    if cands.empty:
        return BartMatch(None, "unresolved", 0.0)
    per = cands.groupby("bart_pid").agg(
        player_name=("player_name", "last"),
        birth_date=("birth_date", "max"),
        nba_pick=("nba_pick", "max"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        team=("team", "last"),
        last=("last_norm", "last"),
    )
    per["name_score"] = per["player_name"].map(lambda n: name_sim(r["player_name"], n))
    # Nicknames ("Bam" Adebayo is "Edrice" on Barttorvik) fail name similarity, so a
    # hard identifier (birthdate, NBA pick) plus an exact last name is also accepted.
    same_last = per["last"] == last
    plausible = (per["name_score"] >= NAME_MIN - 10) | same_last

    if pd.notna(r["birth_date"]):
        hit = per[(per["birth_date"] == r["birth_date"]) & plausible]
        if len(hit) == 1:
            return BartMatch(int(hit.index[0]), "birthdate", float(hit.iloc[0].name_score))
        if (pid := _resolve_transfer(hit)) is not None:
            return BartMatch(pid, "birthdate_transfer", float(hit["name_score"].min()))
    if r["drafted"]:
        hit = per[(per["nba_pick"] == r["pick_overall"]) & (per["last_season"] == r["draft_year"])
                  & ((per["name_score"] >= NAME_MIN) | same_last)]  # fmt: skip
        if len(hit) == 1:
            return BartMatch(int(hit.index[0]), "nba_pick", float(hit.iloc[0].name_score))
    per = per[per["name_score"] >= NAME_MIN - 10]
    if per.empty:
        return BartMatch(None, "unresolved", 0.0)
    per["school_score"] = per["team"].map(lambda t: best_school_sim(r["colleges"], t))
    hit = per[(per["name_score"] >= NAME_MIN) & (per["school_score"] >= 85)]
    if len(hit) == 1:
        return BartMatch(int(hit.index[0]), "name_school", float(hit.iloc[0].name_score))
    hit = hit[hit["name_score"] >= FUZZY_MIN]
    if (pid := _resolve_transfer(hit)) is not None:
        return BartMatch(pid, "name_school_transfer", float(hit["name_score"].min()))
    hit = per[per["name_score"] >= FUZZY_MIN]
    if len(hit) == 1 and r["drafted"] and pd.notna(r["colleges"]):
        return BartMatch(int(hit.index[0]), "fuzzy_name", float(hit.iloc[0].name_score))
    return BartMatch(
        None, "unresolved" if len(hit) == 0 else "ambiguous", float(per["name_score"].max())
    )


def link_bart(universe: pd.DataFrame, bart: pd.DataFrame, first_bart_season: int) -> pd.DataFrame:
    bart = bart.assign(
        last_norm=bart["player_name"].map(
            lambda n: normalize_name(n).split()[-1] if normalize_name(n) else ""
        ),
        first_norm=bart["player_name"].map(
            lambda n: normalize_name(n).split()[0] if normalize_name(n) else ""
        ),
    )
    rows = []
    for _, r in universe.iterrows():
        _, hi = _college_window(r)
        eligible = hi >= first_bart_season and r.get("prospect_source") in ("college", "unknown")
        if not eligible:
            rows.append((r["bbref_id"], pd.NA, "not_applicable", 0.0))
            continue
        norm = normalize_name(r["player_name"]).split()
        r = r.copy()
        r["player_last"] = norm[-1] if norm else ""
        r["player_first"] = norm[0] if norm else ""
        m = match_bart(r, bart)
        rows.append(
            (r["bbref_id"], m.bart_pid if m.bart_pid is not None else pd.NA, m.method, m.score)
        )
    out = pd.DataFrame(rows, columns=["bbref_id", "bart_pid", "bart_method", "bart_score"])
    out["bart_pid"] = out["bart_pid"].astype("Int64")
    return out


# --------------------------------------------------------------------------- combine
def link_combine(universe: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Return (bbref_id, combine_year, method) for each combine attendance we can place."""
    by_id = universe.dropna(subset=["nba_person_id"]).merge(
        combine.dropna(subset=["nba_person_id"]), on="nba_person_id", suffixes=("", "_c")
    )
    by_id = by_id.assign(combine_method="nba_person_id")[
        ["bbref_id", "combine_year", "combine_method"]
    ]
    unlinked = combine[combine["nba_person_id"].isna()]
    rows = []
    for c in unlinked.itertuples():
        cands = universe[(universe["draft_year"] == c.combine_year)]
        sims = cands["player_name"].map(lambda n, cn=c.player_name: name_sim(n, cn))
        if len(sims) and sims.max() >= FUZZY_MIN:
            rows.append((cands.loc[sims.idxmax(), "bbref_id"], c.combine_year, "name_year"))
    extra = pd.DataFrame(rows, columns=["bbref_id", "combine_year", "combine_method"])
    return pd.concat([by_id, extra], ignore_index=True).drop_duplicates()


# ---------------------------------------------------------------------------- build
def apply_overrides(xw: pd.DataFrame, path: Path = OVERRIDES) -> pd.DataFrame:
    if not path.exists():
        return xw
    ov = pd.read_csv(path, dtype=str, comment="#").fillna("")
    xw = xw.copy()
    for o in ov.itertuples():
        mask = xw["bbref_id"] == o.bbref_id
        if not mask.any():
            log.warning("override for unknown bbref_id %s", o.bbref_id)
            continue
        value = pd.NA if o.value == "" else o.value
        if o.field in ("nba_person_id", "bart_pid"):
            value = pd.NA if value is pd.NA else int(value)
        xw.loc[mask, o.field] = value
        method_col = {
            "nba_person_id": "nba_method",
            "bart_pid": "bart_method",
            "cbb_id": "cbb_method",
        }.get(o.field)
        if method_col:
            xw.loc[mask, method_col] = "manual"
    return xw


def build_crosswalk(s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = player_universe(s)
    nba = link_nba(
        universe,
        read_table("staging", "nba_api", "draft_history", s),
        read_table("staging", "nba_api", "players", s),
        birth_lookup=lambda pid: nba_stats.player_birth_date(pid, s),
    )
    universe = universe.merge(nba, on="bbref_id", how="left")
    bart = read_table("staging", "barttorvik", "player_seasons", s)
    universe = universe.merge(
        link_bart(universe, bart, int(bart["season"].min())), on="bbref_id", how="left"
    )
    universe["cbb_method"] = universe["cbb_id"].notna().map({True: "bbref_link", False: "none"})
    xw = apply_overrides(universe)
    combine = link_combine(xw, read_table("staging", "nba_api", "combine", s))
    return xw, combine


def run(s: Settings) -> None:
    xw, combine = build_crosswalk(s)
    write_table(xw, "modeled", "xwalk", "players", s)
    write_table(combine, "modeled", "xwalk", "combine_links", s)
    for col in ("nba_method", "bart_method", "cbb_method"):
        log.info("%s: %s", col, xw[col].value_counts(dropna=False).to_dict())
