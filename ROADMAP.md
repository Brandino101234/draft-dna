# Roadmap

## Phase 0: Repo setup (`phase-0-scaffold`)
- [x] Repo, GitHub remote, branch-per-phase workflow
- [x] Package skeleton, typed YAML config, logging, CLI stub
- [x] `.gitignore` for secrets and data; `.env.example`
- [x] CLAUDE.md, ROADMAP.md, DECISIONS.md
- [x] CI: ruff, mypy, pytest

## Phase 1: Data pipeline (`phase-1-data-pipeline`)
- [x] Shared fetcher: on-disk cache, per-source rate limit, retries, resumable
- [x] Draft picks 1996–2026 (BBRef + nba_api DraftHistory, linked by name within year)
- [x] Undrafted NBA players (debuted 1997+)
- [x] NBA season stats: box, advanced, per-100, playoffs, awards, All-Stars, team, coach, games absent
- [x] College stats (SR CBB) + Barttorvik (SOS, rim/mid splits); usage/AST% derived pre-2010
- [x] Combine measurements and drills
- [x] `prospect_source` flag (college / high school / international or pro team)
- [ ] International / G League stats (deferred: no reliable source; see DECISIONS D014)
- [x] Player ID crosswalk with gold-set match-quality tests
- [x] Era tables (NBA and NCAA pace, 3PA rate, TS%, line distances) and era-relative columns
- [x] Data-quality tests: row counts, nulls, duplicate IDs, join coverage
- [x] `make data` end-to-end; coverage report in `reports/phase1/`

## Phase 2: Define "better" (`phase-2-outcomes`)
- [x] `outcomes_through_n` table (same-point cumulative and peak value, zero-filled seasons)
- [x] Three candidate composites; validated vs awards and 2nd-contract % of cap (bootstrap CIs)
- [x] Early-career outcomes (Y3, Y4)
- [x] Tier mapping: Out of league / Bust / Rotation / Starter / All-Star / All-NBA (calibrated cutoffs)
- [x] Survival analysis of career length with censoring (Kaplan-Meier, Cox split at rookie deal)

## Phase 3: Stats comps and outcome bands, MVP (`phase-3-comps-bands`)
- [x] Rolling-origin backtest harness with leakage tests (tuning 2006-2012, holdout 2013-2020)
- [x] Baselines: pick-only, standardized kNN
- [x] Learned similarity (weighted kNN, tree proximity) + sanity-check comps
- [x] LightGBM quantile, NGBoost, PyMC hierarchical (Tobit), quantile blends
- [x] Conformal calibration of intervals
- [x] Evaluation: pinball, CRPS, Brier, calibration, coverage, all vs baselines with bootstrap CIs
- [x] Per-player output: top-15 comps, floor/median/ceiling, tier probabilities

## Phase 4: Shot audit and Shot DNA (`phase-4-shot-dna`)
- [x] Coverage audit by season, conference, player → stopped and reported (D025)
- [x] Court coordinate standardization; NCAA 3PT line eras (2008-09, 2019-20); era-normalized style maps
- [x] Zone frequencies, FT rate, EB-shrunk zone efficiency, assisted rates
- [x] KDE heatmaps → NMF shot styles (k = 6)
- [x] Size/position guardrail for comps
- [x] NBA early-career shot charts (for Phase 7 "plays like")

## Phase 5: Does shot data help? (`phase-5-shot-vs-stats`)
- [x] Pre-registered comparison in DECISIONS (D027, committed before any model ran)
- [x] Stats-only vs shot-only vs combined backtest (comps with tuned blend weight + LightGBM), with CIs
- [x] Results by position and pick band; tier-B shot-location test
- [x] Result: no detectable added value (D028); rim finishing flagged for confirmation on 2023+ classes (D029)
- [ ] Run the D029 confirmation test as 2023-2026 classes reach year 4

## Phase 6: Who beats their projection (`phase-6-over-under`)
- [x] Beat-ceiling / miss-floor rates and mean PIT by pick, position, age, era, background, team
- [x] Team effects with partial pooling + heterogeneity test (no detectable team effect)
- [x] Situation effects (propensity weighting, balance checks, E-values; stated limits)
- [x] SHAP for overperformance drivers, with leave-classes-out and strict rolling checks
- [x] Year-4 Verdict -> Career Grade overturn rate (14%)

## Phase 7: Cards and app (`phase-7-app`)
- [x] Grading state machine with Bayesian updating
- [x] Prospect cards (PNG export)
- [x] NBA "plays like" style comps (stylistic only)
- [x] Streamlit app: search, cards, compare, style map, 2026 tracker
- [x] `make refresh` for in-season updates
- [x] Final README writeup
