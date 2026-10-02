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

### D020: Phase 3 target and evaluation design
- **Target:** best 3-season value through season 6 after the draft. By year 6 the median player has reached his career peak, and the year-6 peak ranks players almost identically to the career peak (Spearman 0.98 vs. through year 14), while letting classes through 2020 be scored.
- **No leakage:** the model for draft class Y trains only on classes with c + 6 ≤ Y (outcome complete by draft night). `split()` asserts this; a data-quality test checks every projection row. Tier cutoffs and the blend metric's scale are fixed definitions computed in Phase 2 (they use all training classes but no player-level information).
- **Tuning vs holdout:** every modeling choice (hyperparameters, transforms, blend weights) was made on test years 2006–2012 only. 2013–2020 was scored once, at the end. This mattered: the blend's 5% edge on the tuning years disappeared on the holdout.

### D021: Model of record = draft-slot history + conformal calibration
Holdout (2013–2020, 480 players), CRPS difference vs pick only, 95% paired bootstrap CI:

| model | CRPS | vs pick only | below floor (25%) | above ceiling (10%) |
|---|---|---|---|---|
| Pick only | 0.331 | — | 23.5% | 10.2% |
| Pick only + conformal | 0.332 | +0.002 (−0.002, +0.005) | 23.7% | 9.8% |
| Final blend (pick + LightGBM + Bayes) | 0.331 | +0.000 (−0.009, +0.008) | 31.3% | 8.3% |
| LightGBM quantile (pick + stats) | 0.334 | +0.003 (−0.009, +0.014) | 31.0% | 9.2% |
| Bayesian hierarchical Tobit | 0.360 | +0.029 (+0.011, +0.047) | 27.2% | 6.9% |
| Stats kNN (baseline b) | 0.371 | +0.040 (+0.018, +0.064) | 26.5% | 8.4% |
| NGBoost | 0.384 | +0.053 (+0.036, +0.072) | 51.0% | 19.0% |

No model using pre-draft box-score stats beat the draft pick alone on the holdout, so per the baselines-first rule the simpler model is kept. Conformal calibration is added because it keeps the accuracy of pick-only while giving the best calibration in both periods (tuning: 23.7% / 10.5%; holdout: 23.7% / 9.8%). Known weakness: for the top picks, P(All-Star or better) runs high in the holdout (predicted ~43%, observed ~20%), consistent with 2013–2020 top picks reaching All-Star value by year 6 less often than history implied.

### D022: Comps = standardized stats kNN with a feature-overlap guard
The first comps failed the sanity check: players missing most college stats (JUCO, Division II, never played) were "close" to everyone on the one or two features they had ("hubness") and appeared as comps for Curry, Zion, Durant, etc. Fix: a historical player can only be a comp if he shares ≥70% of the prospect's available features (others are pushed behind every eligible comp). Comps became sensible (Durant → Carmelo/Pierce; Anthony Davis → Bosh/Aldridge/Brand; SGA → Westbrook/Wall/Rose). The fix made the comps' outcome spread slightly *less* predictive (holdout CRPS 0.361 → 0.371): the hubs were mostly busts and pulled predictions toward realistic outcomes for the wrong reason. Learned similarity: tree-proximity comps beat stats kNN on the tuning years (−0.024, CI excludes 0) but not on the holdout (−0.013, CI −0.028 to +0.002), so the simpler kNN is kept. Comps are descriptive context; the forecast comes from D021.

### D023: Imputation inside folds neutralizes absent features
Features with fewer than 30 observed values in a fold's training data (e.g. BPM before 2008) are set to the training median for that fold, and standardized values are clipped to ±5. Without this, a test player's raw BPM entered the Bayesian model unscaled and produced ceilings in the thousands.

### D024: Non-college prospects
International and high-school prospects have only age, size and (sometimes) combine data, so their stats comps are physical-profile comps (LeBron → other high-school bigs; Wembanyama → tall international players). Their projections come from draft slot like everyone else's.

### D025: College shot-data audit findings (Phase 4a)
Stratified sample of ESPN play-by-play: 20 drafted-player team-seasons × 8 games for each college season 2008–2026 (3,148 games, 140,000+ field-goal attempts). Full report: `reports/phase4/shot_audit.md`.
- **Shot type** (layup, dunk, tip, jumper, three) is present on 100% of field-goal attempts in every game that has play-by-play, from 2008 onward.
- **Play-by-play itself** exists for 77–100% of games, except a gap in 2012–2013 (37–40%).
- **x/y coordinates** do not exist before 2014. From 2014–2025 only 11–67% of shots have them; coverage *fell* recently (2023: 17%, 2024: 11%), then reached 100% in 2025-26. They come mostly from televised games (48% of televised vs 9% of untelevised games) and high-major conferences (~45% Big Ten / ACC / SEC / Big 12; ≤10% A-10, MWC, CAA, MAC, MVC).
- **Per drafted player** (≥20 sampled shots): median 21% of shots carry coordinates; for college seasons 2014–2020, 69% of players reach ≥40% coverage. That leaves roughly 230 coordinate-eligible drafted players with 6-year outcomes, skewed toward televised high-major programs, which is a selection bias.
- **Shot-type splits** (rim / midrange / three / dunk) already exist from Barttorvik for 521 of 535 drafted college players in the 2010–2020 classes.
- **NBA shot charts** (stats.nba.com) are complete from 1996-97 (e.g. Kobe Bryant's 422 rookie FGA all present with zones and x/y).
- A full ESPN pull of every drafted-player team-season 2008–2026 is ~41,600 games, about 13 hours at 1 request/second. ESPN's API is unofficial (no robots.txt; not a licensed feed): personal research only, cached, polite.
Decision (user, after review): two-tier design (D026).


### D026: Shot DNA design (two tiers)
**Tier A, shot type (headline test in Phase 5):**
- **Sources:** Barttorvik rim / midrange / three / dunk splits (2010+) plus ESPN play-by-play assisted vs. unassisted makes (2008+).
- **Shot mix** comes from the final pre-draft season.
- **Zone efficiency** uses all pre-draft seasons, shrunk with empirical Bayes toward that season's Division I average. The beta prior is fit by method of moments on every D-I player that season, so it uses pre-draft information only. Prior strength is about 50 shots at the rim, about 160 in the midrange, and 300–600 from three. Three-point percentage is the noisiest, which matches the research consensus.

**Tier B, coordinates (cards and a secondary test):**
- **Court frame:** ESPN x/y is in feet; the basket sits at y = 1.0, fit from the data. That placement classifies twos vs. threes 98–100% of the time in every line era.
- **Consistency filter:** coordinates that contradict the shot's value (e.g. a three logged at the rim) are dropped. That's about 1% of shots, mostly 2020–2025.
- **Zones:** rim < 4 ft, short mid 4–14 ft, long mid 14 ft+, corner three (|dx| ≥ 20 and dy ≤ 8), above-the-break three. On NBA data, these agree with stats.nba.com's own zones 98–100%.
- **Era normalization:** style maps rescale distance so the 3-point line sits at 22.15 ft in every era. Without it, pooled seasons produced a spurious "threes from the old 20.75 ft line" style.
- **Shot maps:** each player's map is a Gaussian KDE (bandwidth 1.5 ft) on a 1-ft grid, folded left/right.
- **Eligibility:** ≥100 shots with valid coordinates and ≥40% coordinate coverage. That's 395 drafted players, including 53–92% of the 2014–2020 classes and 92% of the 2026 class.

**Shot styles:** NMF with k = 6, chosen where the gain per added style drops from 9–12% to 5–7%. A k = 10 fit split one style into near-duplicate rim blobs and three-point angles. Style weights are mixture weights, not shot shares; use tier-A or zone shares for "% of shots from X." For the Phase 5 backtest, NMF is refit inside each fold.

**Comps guardrail:** shot-based comps must be within 3 inches of height and share the position bucket (`GuardedKnn`).

**NBA shot charts:** stats.nba.com shot charts for each player's first 4 NBA seasons (2.24M shots, 2,424 players) are standardized into the same frame for Phase 7's "plays like" comps. stats.nba.com throttles sustained pulls, so it is accessed at 3.5 s/request with retries, and any player-season that keeps failing is skipped.

### D027: Phase 5 pre-registration (written and committed before any Phase 5 model was run)
**Question.** Does *how* a prospect scores (Shot DNA) predict NBA success better than box-score stats, and does it add anything beyond draft position?

**Cohort.** Drafted college players with ≥100 shot-type field-goal attempts (Barttorvik splits, available from the 2010 class). Every model is trained and tested on this cohort only, so the comparison is identical across models.

**Design B (primary, user choice).** Target = best 3-season value through year 4 (`y_peak4`). Strict leakage rule: class c trains test year Y only if c + 4 ≤ Y (outcome known by draft night). Test years 2014–2022 (~417 players).

**Design A (sensitivity check).** Target = year-6 peak (`y_peak6`). Train on classes drafted before Y (c < Y), even though their 6-year outcomes complete later. Test years 2012–2020. It uses information that was not available on draft night, so it is a robustness check only.

**Feature sets.**
- `stats`: Phase 3 pre-draft stats, age, size, combine
- `shot`: tier-A Shot DNA (rim/mid/three mix, dunk share, era-relative 3PA rate, EB zone FG% and their era-relative versions, assisted shares by type, unassisted share)
- `stats+shot`

**Models** (hyperparameters frozen from Phase 3; nothing tuned except where stated):
1. Pick only + conformal (Phase 3 model of record; baseline a)
2. kNN comps on `stats` (baseline b), on `shot` (size/position guardrail), and on `stats+shot`, where the combined distance is w·d_stats + (1−w)·d_shot, with w ∈ {0, 0.25, 0.5, 0.75, 1} chosen on test years 2014–2017 and reported on 2018–2022
3. LightGBM quantile (Phase 3 config) on `stats`, `shot`, `stats+shot` without pick
4. LightGBM quantile on pick + `stats`, pick + `shot`, pick + `stats+shot`

**Pre-registered comparisons** (CRPS; 95% paired bootstrap CI over players):
- **H1, beyond draft position:** LightGBM pick + `stats+shot` vs pick + `stats`
- **H2, how vs how much:** LightGBM `shot` vs `stats` vs `stats+shot` (no pick)
- **H3, the bar:** the best shot-informed model vs pick only + conformal
- **Secondary metrics:** pinball loss (floor, median, ceiling), Brier score for P(bust) and P(All-Star level), coverage
- **Subgroups (descriptive, with CIs):** position (guard / forward / big) and pick band (1–14, 15–30, 31–60)
- **Tier-B spatial test (exploratory):** coordinate-eligible subset, adding zone shares and NMF style weights (NMF refit inside each fold) to pick + `stats`

**Decision rule.** Shot data "helps" only if a model's CRPS beats its no-shot counterpart with a 95% CI that excludes zero under design B. The model of record changes only if a shot-informed model also beats pick only + conformal that way. Results are reported whether positive, null or negative.

### D028: Phase 5 results: shot data does not add detectable predictive value
All comparisons as pre-registered in D027 (CRPS difference, 95% paired bootstrap CI; negative = better):

| comparison | Design B (primary, strict, year 4) | Design A (check, year 6) |
|---|---|---|
| H1: pick + stats + shot vs pick + stats | −0.0001 (−0.0016, +0.0013) no difference | +0.0009 (−0.0014, +0.0032) no difference |
| H2a: shot vs stats (no pick) | +0.0154 (+0.0042, +0.0267) **worse** | +0.0261 (+0.0098, +0.0419) **worse** |
| H2b: stats + shot vs stats (no pick) | −0.0006 (−0.0023, +0.0010) no difference | −0.0005 (−0.0029, +0.0018) no difference |
| H2c: kNN shot vs kNN stats | +0.0164 (+0.0083, +0.0246) **worse** | +0.0200 (+0.0100, +0.0301) **worse** |
| H3: pick + stats + shot vs pick only | +0.0013 (−0.0066, +0.0092) tie | +0.0096 (+0.0004, +0.0185) worse |
| Tier B: pick + stats + location vs pick + stats (coordinate subset) | +0.0017 (−0.0005, +0.0038) no difference | +0.0018 (−0.0007, +0.0044) no difference |

- **Comp blend weight:** chosen on tuning years, it is 100% stats / 0% shot in both designs.
- **Subgroups:** none of the 24 subgroup intervals (position × pick band × design) excludes zero.
- **Conclusion:** within this cohort (~415 drafted college players, 2010+), *how* a prospect scores, measured by shot type, assisted rate or shot location, adds no detectable information beyond draft position plus box-score stats. On its own it is a weaker signal than box-score stats. The intervals on H1 are narrow (about ±0.0015 CRPS, ~0.5% of the score), so any real effect is very small. The model of record is unchanged (D021).
- **Why plausible:** shot profile is largely downstream of things box scores and scouts already capture (size, athleticism, role). Scoring *volume* and *efficiency* matter; *where* the shots come from adds little once those are known.

### D029: Exploratory signal (rim finishing) and its pre-registered confirmation test
Not pre-registered, so a hypothesis only: among 15 shot features, shrunken rim FG% (relative to the D-I average) is the one with signal beyond out-of-sample pick + stats predictions (Spearman ρ = 0.14, p = 0.0032, n = 414, vs. a multiple-comparison threshold of 0.0033). The pre-registered model, given all 15 shot features, did not turn it into better forecasts, plausibly because 14 weak features diluted it.
**Confirmation test (fixed now, run when data exists):**
- **Cohort:** drafted college players from the 2023–2026 classes, never used in any Phase 5 analysis.
- **When:** as each class's year-4 outcome completes (2023 class: after the 2026-27 season).
- **Test 1:** Spearman ρ between `rim_fg_eb_rel` and the residual of a LightGBM pick + stats model trained only on classes ≤ 2022.
- **Test 2:** CRPS of LightGBM pick + stats + `rim_fg_eb_rel` vs pick + stats.
- **Confirmation requires:** ρ > 0 with a 95% CI excluding zero *and* a CRPS improvement whose 95% CI excludes zero. Otherwise the signal is treated as noise.

### D030: Phase 6 methods and findings (who beats their projection, and why)
**Measure.** For every drafted player with an as-of-draft projection (model of record, D021), the PIT score is where his actual peak landed inside his projected range at years 4, 6 and 8. Point masses at zero use the midpoint. Overall mean PIT is 0.49–0.51 at every horizon (calibrated), so group differences are meaningful. "Below floor" rates sit below 25% because many floors are exactly 0, which cannot be undershot.

**Segments (year 6, mean PIT, 95% CI):**
- International / pro-team picks **fall short of their slot**: 0.40 (0.37–0.44), n = 218. Partly stash risk: second-round internationals who never come over are zeros.
- Late first-rounders (picks 15–30) slightly beat their slot: 0.54 (0.51–0.58).
- Forwards: 0.54 (0.51–0.56).
- The youngest draftees (< 19.5) sit lower (0.46) than 19.5–20.5-year-olds (0.55).
- No trend by era.
- These are descriptive comparisons across ~17 groups; single intervals near 0.5 should not be over-read.

**Drafting franchise: no detectable effect.** Cochran's Q heterogeneity test: p = 0.82 / 0.92 / 0.96 at years 4 / 6 / 8. The estimated between-team variance is 0, so partial pooling returns every team to the league mean. The raw "best" teams (Houston 0.62, Indiana 0.59, ~30 picks each) are within what luck produces.

**Situation effects** (inverse-propensity weighting on pick, age, size, key stats, background, position and draft year; propensities trimmed to 0.05–0.95; 300 bootstrap resamples; E-values on the risk ratio of beating the projected median):
- **Known before the draft:**
  - bottom-third team: −0.008 (−0.046, +0.028)
  - deep at his position: −0.013 (−0.051, +0.022)
  - No detectable effect from either. Balance after weighting: max |SMD| 0.05 and 0.16; the depth comparison is not fully balanced.
- **After the draft:**
  - head-coach change in years 1–3: +0.012 (−0.031, +0.045), no effect
  - traded in years 1–3: −0.059 (−0.098, −0.022), E-value 1.49
  - missed ≥ 25% of games in years 1–2: −0.254 (−0.287, −0.209), E-value 3.19
  - The last two are **associations, not effects**: underperformance plausibly *causes* trades and lost minutes (reverse causality). The low E-value for trades means even weak confounding could explain it. There is no reliable public injury history, so "missed games" mixes injury, G League assignment and DNPs (D005).

**SHAP / overperformance predictability:**
- **Leave-classes-out CV:** a LightGBM model of PIT from pre-draft traits reaches Spearman 0.26 (year 4).
- **Strict rolling-origin check** (each class predicted only from classes with known outcomes):
  - year 4: 0.19 (p < 0.001; college-only 0.15, p < 0.001)
  - year 6: 0.12 (p = 0.004; college-only 0.04, p = 0.43)
- **Conclusion:** pre-draft traits predict who outperforms their slot *early* (through year 4), but for college players the edge fades by year 6. That is consistent with Phases 3 and 5 (year-6 target).
- **SHAP drivers (year 4):** height, assists, rebounds, career FT% and efficiency push toward beating the slot; high usage and turnovers push toward falling short. One reading: teams overvalue scoring volume relative to passing, size and shooting touch, but only for the first few years.

**Verdict reversals:** of 840 players with both a Year-4 Verdict and a Career Grade (years 4 and 8, each vs. the same-point projected range), 14.2% (12.0–16.7%) changed category:
- **Upgrades: 89 (3× more common than downgrades)**
  - 61 went from below floor to within band (late-developing role players)
  - 28 went from within band to beat ceiling (Curry, Shai Gilgeous-Alexander, Harden, Sabonis, Derrick White, Lowry)
- **Downgrades: 30**
  - 23 are early overachievers who plateaued (beat ceiling at 4, within band at 8: Anthony Davis, Paul George, Deron Williams, Trae Young). They didn't decline; the projected ceiling for top picks rises through year 8.
  - Only 7 (0.8%) are true late busts (within band at 4, below floor at 8).
- **Implication for the grading system (Phase 7):** a year-4 verdict is right ~86% of the time, and when it is wrong it is usually too pessimistic.

### D031: Phase 7 grading model, cards and app

**Grading (Bayesian updating).** On a square-root scale of year-8 peak value:
- **Prior:** the draft-night projection, which is the model of record's quantile grid.
- **Measurement model:** each season's observed peak-so-far is s(peak_N) = a_N + b_N·s(peak_8) + noise, fit on history.
- **Posterior:** normal–normal, so the weight on the data grows each season (0.35 / 0.54 / 0.67 / 0.79 / 0.89 / 0.96 / 0.99 after 1–7 seasons).
- **Calibration:** empirical residual quantiles in 5 bins of the posterior mean for each N, fit on classes ≤ 2010 and validated on the held-out 2011–18 classes (480 players).
- **Rules:**
  - A peak can never fall below the peak already reached.
  - Grade = midpoint PIT of the posterior median vs the draft-night range: A ≥ 0.9, B ≥ 0.5, C ≥ 0.25, else D. On finished careers that gives A 9.6% / B 36.9% / C 28.6% / D 24.9%.
  - Confidence = 1 − posterior sd / prior sd.

**Validation:**
- CRPS falls from 0.387 (draft night) to 0.025 (7 seasons).
- Calibrated ceilings are exceeded 7–10% of the time (target 10%).
- **Known limit:** floors run high in years 1–2 (35–38% finish strictly below, target 25%) and come closer at about 24–30% in years 3–7. Treat early floors as optimistic.

**Plays-like comps:** NBA early-career shot maps (≥ 200 FGA) are projected onto the college NMF styles, with the 3-point line rescaled. Matches use cosine similarity and must be within 3 inches of height. They are **stylistic only**: Phase 5 found style has no predictive value. The "Rim attacker" style dominates the style map, so coarse labels hide a lot of variety.

**Rookie tracker / refresh:** a rookie's projected range is 3 × the year-1 peak band. Pace = value-to-date × 82 / team games played. `draft-dna refresh` re-fetches only current-season pages (older pages are cached), then rebuilds, regrades and redraws cards. Draft-night projections never change.
