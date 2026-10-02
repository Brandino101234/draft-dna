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
- [ ] Coverage audit by season, conference, player → **stop and report**
- [ ] Court coordinate standardization; NCAA 3PT line eras (2008–09, 2019–20)
- [ ] Zone frequencies, FT rate, EB-shrunk zone efficiency, assisted rates
- [ ] KDE heatmaps → NMF shot styles
- [ ] Size/position guardrail for comps

## Phase 5: Does shot data help? (`phase-5-shot-vs-stats`)
- [ ] Pre-registered comparison in DECISIONS.md
- [ ] Stats-only vs shot-only vs combined backtest, with CIs
- [ ] Results by position and archetype

## Phase 6: Who beats their projection (`phase-6-over-under`)
- [ ] Beat-ceiling / miss-floor rates by segment
- [ ] Situation effects (matching, stated assumptions, sensitivity)
- [ ] SHAP for overperformance drivers
- [ ] Year-4 Verdict → Career Grade overturn rate

## Phase 7: Cards and app (`phase-7-app`)
- [ ] Grading state machine with Bayesian updating
- [ ] Prospect cards (PNG export)
- [ ] NBA "plays like" style comps (stylistic only)
- [ ] Streamlit app: search, cards, compare, style map, 2026 tracker
- [ ] `make refresh` for in-season updates
- [ ] Final README writeup
