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

### D016: Career value = VORP + Win Shares blend, 75% peak / 25% total
Three per-season value definitions were compared: **A** VORP, **B** the mean of z-scored VORP and Win Shares, **C** a one-factor model of VORP, WS, minutes, shrunken BPM and starts. Each was turned into a career composite and validated for the 1996–2016 classes, 10 seasons after the draft, with 95% bootstrap CIs (1,000 resamples of players):

| check | A: VORP | B: blend | C: factor |
|---|---|---|---|
| AUC, ever All-NBA | 0.990 | 0.991 | 0.991 |
| AUC, ever All-Star | 0.982 | 0.984 | 0.983 |
| Spearman, All-Star selections | 0.486 | 0.487 | 0.485 |
| Spearman, peak pay % of cap (yrs 5–10) | 0.766 | **0.886** | 0.818 |
| Spearman, 2nd-contract pay vs value through yr 4 | 0.633 | **0.828** | 0.723 |

All three tie on awards. On contracts, B beats A by +0.12 to +0.20, with CIs on the difference excluding zero. Why: VORP rates a below-replacement player who keeps getting minutes *below* a player who is out of the league, but teams keep paying the former, and staying in the league is part of NBA success. Win Shares rarely goes far below zero, which fixes that. C does not beat B, so the simpler B wins.

**Peak vs total weighting.** Peak-heavy weightings score higher on every check, but the targets (All-Star selections, peak pay) are themselves peak-oriented, so they cannot judge how much longevity should count. 75% peak / 25% total keeps nearly all of the validated signal (peak pay 0.886 vs 0.907 for peak alone) while still crediting long careers. Peak, total, longevity and playoff value are all stored separately for later analysis.

Seasons outside the NBA count as zero (a bust is an outcome, not missing data). Composites are z-scored at each N against training-class players only, so later classes never shift the scale.

### D017: Tiers are cutoffs on peak value, calibrated to role and award anchors
Tiers are defined on the same quantity the models will predict (best 3-season blend value), so a predicted distribution converts directly into tier probabilities. Cutoffs maximize balanced accuracy between neighboring anchor groups (All-NBA selection; All-Star selection; best 3-year stretch averaging ≥41 starts and ≥1,800 min; ≥1,000 min; ≥1,000 career minutes): 0.01 / 0.32 / 1.01 / 1.76 / 2.37. Agreement with anchors: 67% exact, 97% within one tier. Disagreements are intended: a tier means "played like a typical X", not "was voted X" (e.g. Andrew Wiggins, a one-time All-Star, sits at the Rotation/Starter line). "Out of league" means no positive NBA value (peak ≈ 0), not only zero games.

### D018: Career end and censoring
A career has ended if the player has not appeared in either of the last two completed seasons (2024–25, 2025–26); otherwise it is censored. One season of slack avoids counting a player who missed a year injured as retired. Never-played picks have duration 0 and an ended career.

### D019: Cox model split at the rookie deal
A single Cox model fails the proportional-hazards test for draft pick (p < 0.0001): pick matters far more during the rookie deal than afterwards. Fitting seasons 1–4 and seasons 5+ separately: each doubling of pick number multiplies the yearly hazard of a career ending by 2.07 (1.83–2.34) in seasons 1–4 but only 1.16 (1.09–1.24) afterwards; age at draft matters more later (1.26 per year vs 1.11). The seasons-5+ model passes the assumption test; the seasons 1–4 hazard ratio for pick is an average over those years (second-rounders mostly leave in years 1–2).
