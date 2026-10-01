# Decisions

Each methodological or architectural choice, why it was made, and what was rejected. Newest at the bottom.

---

### D001: Scope of college shot data (Phase 0)
**Decision:** Use college shot data wherever it exists, with two layers: shot *type* parsed from play-by-play text (rim/jumper/3, assisted), with broader coverage from about 2009, and shot *x/y location* (ESPN via cbbpy), roughly 2010–12 onward and mostly televised/high-major games. Every player gets `shot_coverage_pct` and a coverage tier. Earlier drafts are explicitly labeled as having inconsistent shot data, and low-coverage players fall back to stats-only.
**Why:** Shot coordinates for 1996–2009 college players don't exist publicly. Pretending otherwise would bias the headline comparison toward eras with data.
**Consequence:** The shot-vs-stats headline test (Phase 5) runs on the coverage-eligible cohort (≈2012–2021 drafts), a few hundred players across about 4–6 backtest folds. Expect wide confidence intervals, and a null result is plausible.

### D002: Year-4 Verdict instead of "Rookie Deal Verdict"
**Why:** Second-round picks aren't on 4-year rookie-scale contracts. The checkpoint is the same, but the name is accurate for both rounds.

### D003: Anchor player ID = Basketball-Reference ID
**Why:** BBRef covers drafted, undrafted, NBA and (via Sports-Reference CBB links) college players with stable IDs. nba_api, ESPN, Barttorvik and combine IDs map onto it through the crosswalk.

### D004: Pre-draft data gaps are flagged, not imputed away
**Decision:** A `prospect_source` flag (college / high school / international / G-League). High school and most international players get age, size, pick and combine features only, with explicit missingness indicators. College BPM (available ~2010+) is never imputed. TS%, usage, AST%, etc. are derived from box score plus team totals where Sports-Reference lacks them.
**Why:** Dropping preps-to-pros players (Kobe, Garnett, LeBron…) would remove many of the best outcomes, and naive imputation would invent signal.

### D005: Injury effects use games missed, not injury type
**Why:** No clean public injury history exists. Games missed is observable. Situation effects in Phase 6 are reported as associations with stated assumptions, not causal claims.

### D006: Contracts are for validation only
**Decision:** Second-contract salary as % of the salary cap validates the outcome composite in Phase 2 but is never a model input or target.
**Why:** Contracts reflect market and team context, not just player value, and raw dollars are distorted by cap growth.

### D007: Tooling
`uv` for environment and lockfile, DuckDB over parquet for storage, pydantic-validated YAML config, `mypy --strict`, ruff. CI runs unit tests only and never scrapes.

### D008: Barttorvik placeholder birthdates are treated as missing
Barttorvik fills unknown birthdates with October 15 of an estimated year (e.g. Michael Beasley shows 1988-10-15). All October 15 dates are set to missing. This loses ~0.3% of real birthdays but avoids false crosswalk matches and wrong ages. Real birthdates remain for 55–86% of player-seasons from 2010 on.

### D009: Combine attendance is kept per year
64 players attended the combine twice (withdrew, returned). Staging keeps one row per player per combine year; pre-draft features use the latest combine on or before the draft year.

### D010: Crosswalk matching order
- **NBA ID:** draft slot (year + pick) for drafted players; normalized name + debut season (±1) for undrafted; fuzzy name (≥92) only when unique within the debut window.
- **College (Sports-Reference):** the link on the Basketball-Reference bio. No name matching needed.
- **Barttorvik:** birthdate + name ≥70 → NBA pick + final season + name ≥80 → name ≥80 + school ≥85 → fuzzy name ≥92 (drafted college players only). Ambiguous cases stay unresolved rather than guessed.
- School names use `token_sort` similarity, not `token_set`, because token_set scores "Kansas" vs "Kansas State" as a perfect match.
- Manual fixes live in `src/draft_dna/crosswalk/overrides.csv` and always win.

### D011: Type checking treats pandas as untyped
`mypy --strict` on our own code, with pandas ignored. pandas-stubs rejects many idiomatic expressions in a scraping/cleaning pipeline and added noise without catching real bugs. Data correctness is enforced by data-quality tests instead.

### D012: Draft picks link to stats.nba.com by name within draft year, not by pick number
Basketball-Reference and stats.nba.com number second-round picks differently in 1997, 2001 and 2002 (e.g. 2001 #30 is Trenton Hassell on BBRef, Gilbert Arenas on stats.nba.com). Joining on (year, pick) silently mislinked about 85 players; the name-agreement data-quality test caught it. Linking order within each draft year: exact normalized name → best fuzzy name (≥75) → same pick number for leftovers. Leftovers are two players whose stats.nba.com names are garbled ("Ha Ha", "Sun Sun"). Undrafted players: exact name within ±3 seasons of debut, birthdate to split same-name pairs, and unique last name + same first initial within ±2 seasons for name variants (Isaac/Ike). Pure nicknames (Pooh/Eugene Jeter, Flip/Ronald Murray) are manual overrides.

### D013: Era adjustment in Phase 1
Counting stats are kept per 100 possessions (pace-neutral). Efficiency (TS%) and shot mix (3PA rate) are stored relative to that season's league average, separately for the NBA (Basketball-Reference league averages) and Division I (computed from all-team totals). The NCAA 3-point line moved twice (19'9" → 20'9" in 2008-09, → 22'1.75" in 2019-20), and the NBA line was shortened through 1996-97; both are recorded per season. Z-scoring and any further normalization are fit inside each backtest fold in Phase 3, because league-wide distributions can leak future information.

### D014: International and G League pre-draft stats deferred
There is no reliable, scrapeable source of international club stats covering 1996–2026; Basketball-Reference's international coverage is sparse. International and pro-team prospects (~18% of picks) get age, size, pick and combine features plus a `prospect_source` flag, and the model handles their missing college stats explicitly. If a reliable source turns up it can be added as a new ingest step without changing the pipeline shape.

### D015: In-season updates
Pages for the current NBA season are re-fetched when their cache is older than 20 hours; all other pages are cached forever. Rerunning `draft-dna ingest bbref-league && draft-dna transform` during 2026-27 updates season totals, advanced stats, team records and coaches. Game-level updates for the live tracker are part of Phase 7's `make refresh`.
