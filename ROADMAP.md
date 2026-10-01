# Roadmap

## Phase 0: Repo setup (`phase-0-scaffold`)
- [x] Repo, GitHub remote, branch-per-phase workflow
- [x] Package skeleton, typed YAML config, logging, CLI stub
- [x] `.gitignore` for secrets and data; `.env.example`
- [x] CLAUDE.md, ROADMAP.md, DECISIONS.md
- [x] CI: ruff, mypy, pytest

## Phase 1: Data pipeline (`phase-1-data-pipeline`)
- [ ] Shared fetcher: on-disk cache, per-source rate limit, retries, resumable
- [ ] Draft picks 1996–2026 (BBRef + nba_api DraftHistory)
- [ ] Undrafted NBA players (rosters minus draft list)
- [ ] NBA season stats: box, advanced, playoffs, awards, team, coach, games missed
- [ ] College stats (SR CBB) + Barttorvik (SOS, splits); derived advanced stats pre-2010
- [ ] Combine measurements and drills
- [ ] International / G-League stats where available; `prospect_source` flag
- [ ] Player ID crosswalk with gold-set match-quality tests
- [ ] Era tables (NBA and NCAA pace, 3PA rate, line distances)
- [ ] Data-quality tests: row counts, nulls, duplicate IDs, join coverage
- [ ] `make data` end-to-end; coverage report in `reports/`

## Phase 2: Define "better" (`phase-2-outcomes`)
- [ ] `outcomes_through_n` table (same-point cumulative and peak value)
- [ ] Three candidate composites; validate vs awards and 2nd-contract % of cap
- [ ] Early-career outcomes (Y3, Y4)
- [ ] Tier mapping: Out of league / Bust / Rotation / Starter / All-Star / All-NBA
- [ ] Survival analysis of career length with censoring

## Phase 3: Stats comps and outcome bands, MVP (`phase-3-comps-bands`)
- [ ] Rolling-origin backtest harness with leakage tests
- [ ] Baselines: pick-only, standardized kNN
- [ ] Learned similarity (weighted kNN, tree proximity) + sanity-check comps
- [ ] LightGBM quantile, NGBoost, PyMC hierarchical
- [ ] Conformal calibration of intervals
- [ ] Evaluation: pinball, CRPS, Brier, calibration, coverage, all vs baselines
- [ ] Per-player output: top-15 comps, floor/median/ceiling, tier probabilities

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
