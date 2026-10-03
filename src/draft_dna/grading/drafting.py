"""Team drafting ability: judge each pick against who was still on the board (D043).

For a pick at slot p, the alternatives are everyone drafted after p in the same draft.
- better_after: how many of them had clearly better careers.
- missed_weight: the same, but each better player counts by how easy he was to find:
  exp(-k / 15) if he went k picks later, so obvious misses (stars taken right after)
  cost far more than hidden gems nobody took until the second round.
- Draft IQ for the pick = the typical missed_weight for that slot (across all drafts)
  minus his, so taking the best player left scores high and a bust with stars taken
  right behind him scores low.
  (A share-of-later-picks score was tried first: it rated Kwame Brown at #1 above most
  picks because most later picks never played, so it was dropped.)
- Close calls: when two careers are within CLOSE of each other (career peak, D033), the
  player with more championships counts as better; equal rings = a tie.
- best_after: the best player still available, and missed_star flags a pick that never
  reached the All-Star tier when an All-NBA-or-better player went within the next 10
  picks (an obvious miss, not a second-round gem nobody saw).

The team credited is the one that actually got the player (draft-night trades, D036).
Team drafting ability = average Draft IQ weighted by the slot's average value (high picks
are bigger decisions), on finished careers. A permutation test (shuffling teams within
each draft) says whether teams differ more than luck would produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_dna.config import Settings
from draft_dna.eval.phase6 import FRANCHISE
from draft_dna.ingest.storage import read_table, table_path, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

CLOSE = 0.25
EASE_SCALE = 15  # a better player taken k picks later counts exp(-k / 15): obvious misses cost more
STAR_WINDOW = 10
STAR_PLUS = {"All-NBA", "Superstar", "MVP", "Legend"}
LOW = {"Out of league", "Bust", "Rotation", "Starter"}


def championships(s: Settings) -> pd.Series:
    """Rings per player: seasons on the champion's playoff roster."""
    if not table_path("raw", "bbref", "finals_mvp", s).exists():
        return pd.Series(dtype=float)
    fm = read_table("raw", "bbref", "finals_mvp", s)
    if "champion" not in fm:
        return pd.Series(dtype=float)
    champ = fm.set_index("season")["champion"]
    st = read_table("staging", "bbref", "nba_team_stints", s)
    po = st[(st["phase"] == "playoffs") & (st["games"].fillna(0) > 0)]
    won = po[po["team"] == po["season"].map(champ)]
    return won.groupby("bbref_id")["season"].nunique()


def beats(v_a: float, r_a: float, v_b: float, r_b: float) -> float:
    """1 if career a beats b, 0 if it loses, 0.5 for a tie (rings break close calls)."""
    if v_a > v_b + CLOSE:
        return 1.0
    if v_b > v_a + CLOSE:
        return 0.0
    if r_a != r_b:
        return 1.0 if r_a > r_b else 0.0
    return 0.5


def build(s: Settings) -> pd.DataFrame:
    g = read_table("modeled", "grading", "grades", s)
    g = g[g["pick"].notna()].copy()
    rings = championships(s)
    g["rings"] = g["bbref_id"].map(rings).fillna(0).astype(int)
    g["franchise"] = g["team"].replace(FRANCHISE)
    rows = []
    for _, cls in g.groupby("draft_year"):
        cls = cls.sort_values("pick")
        v = cls["current_median"].to_numpy(dtype=float)
        r = cls["rings"].to_numpy(dtype=float)
        tiers = cls["current_tier"].to_numpy()
        ids = cls["bbref_id"].to_numpy()
        for i in range(len(cls)):
            later = np.arange(i + 1, len(cls))
            if len(later) == 0:
                rows.append({"bbref_id": ids[i], "pick_score": np.nan})
                continue
            wins = [beats(v[i], r[i], v[j], r[j]) for j in later]
            picks = cls["pick"].to_numpy(dtype=float)
            # How easy each miss was: better players who went right after count fully,
            # ones nobody took until much later (hidden gems) count little.
            missed_weight = sum(
                np.exp(-(picks[j] - picks[i]) / EASE_SCALE)
                for j, w in zip(later, wins, strict=True)
                if w == 0.0
            )
            best = later[np.argmax(v[later] + 1e-3 * r[later])]
            rows.append(
                {
                    "bbref_id": ids[i],
                    "pick_score": float(np.mean(wins)),
                    "better_after": int(sum(w == 0.0 for w in wins)),
                    "missed_weight": float(missed_weight),
                    "best_after_id": ids[best],
                    "best_after_gap": float(v[best] - v[i]),
                    # an obvious miss: an All-NBA-or-better player within the next 10 picks
                    "missed_star": bool(
                        tiers[i] in LOW
                        and any(
                            tiers[j] in STAR_PLUS and picks[j] - picks[i] <= STAR_WINDOW
                            for j in later
                        )
                    ),
                }
            )
    keep = ["bbref_id", "player_name", "draft_year", "pick", "franchise", "team"]
    keep += ["current_median", "current_tier", "rings", "status"]
    out = g[keep].merge(pd.DataFrame(rows), on="bbref_id")
    names = g.set_index("bbref_id")
    out["best_after"] = out["best_after_id"].map(names["player_name"])
    out["best_after_pick"] = out["best_after_id"].map(names["pick"])
    out["best_after_tier"] = out["best_after_id"].map(names["current_tier"])
    log.info(
        "drafting: %d picks scored; %d missed stars",
        out["pick_score"].notna().sum(),
        int(out["missed_star"].astype("boolean").fillna(False).sum()),
    )
    return out


FIRST_FINISHED, LAST_FINISHED = 1996, 2021
N_PERM = 2000


def add_draft_iq(d: pd.DataFrame) -> pd.DataFrame:
    fin = d[d["status"].str.startswith("Career") & d["pick_score"].notna()]
    fin = fin[fin["draft_year"].between(FIRST_FINISHED, LAST_FINISHED)]
    typical = fin.groupby("pick")["missed_weight"].mean()
    typical = typical.rolling(5, center=True, min_periods=1).mean()
    d["typical_missed_weight"] = d["pick"].map(typical)
    d["draft_iq"] = d["typical_missed_weight"] - d["missed_weight"]
    return d


def team_table(d: pd.DataFrame, weights: pd.Series, seed: int = 0) -> tuple[pd.DataFrame, float]:
    fin = d[
        d["status"].str.startswith("Career")
        & d["draft_iq"].notna()
        & d["draft_year"].between(FIRST_FINISHED, LAST_FINISHED)
    ].copy()
    fin["w"] = fin["pick"].map(weights)

    def stats(x: pd.DataFrame) -> pd.Series:
        m = np.average(x["draft_iq"], weights=x["w"])
        n_eff = x["w"].sum() ** 2 / (x["w"] ** 2).sum()
        sd = np.sqrt(np.average((x["draft_iq"] - m) ** 2, weights=x["w"]))
        return pd.Series(
            {
                "draft_iq": m,
                "se": sd / np.sqrt(n_eff),
                "picks": len(x),
                "missed_stars": int(x["missed_star"].sum()),
                "best_available": int((x["better_after"] == 0).sum()),
            }
        )

    teams = fin.groupby("franchise").apply(stats, include_groups=False)
    # Permutation test: spread of team means vs shuffling team labels within each draft.
    rng = np.random.default_rng(seed)

    def spread(labels: np.ndarray) -> float:
        tmp = fin.assign(f=labels)
        means = tmp.groupby("f").apply(
            lambda x: np.average(x["draft_iq"], weights=x["w"]), include_groups=False
        )
        return float(means.var())

    observed = spread(fin["franchise"].to_numpy())
    years = fin["draft_year"].to_numpy()
    labels = fin["franchise"].to_numpy()
    perm = []
    for _ in range(N_PERM):
        shuffled = labels.copy()
        for y in np.unique(years):
            idx = np.nonzero(years == y)[0]
            shuffled[idx] = rng.permutation(shuffled[idx])
        perm.append(spread(shuffled))
    p = float((np.sum(np.array(perm) >= observed) + 1) / (N_PERM + 1))
    return teams.sort_values("draft_iq", ascending=False).reset_index(), p


def run(s: Settings) -> None:
    from draft_dna.grading import classes as C

    d = add_draft_iq(build(s))
    write_table(d, "modeled", "grading", "drafting", s)
    grades = read_table("modeled", "grading", "grades", s)
    weights = C.typical_curve(grades).set_index("pick")["average_peak"]
    teams, p = team_table(d, weights)
    teams["perm_p"] = p
    write_table(teams, "modeled", "grading", "drafting_teams", s)
    log.info("drafting teams: spread vs luck p = %.3f", p)
