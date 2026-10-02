# CLAUDE.md: Draft DNA project conventions

Read this, ROADMAP.md and DECISIONS.md at the start of every session.

## What this project is
NBA draft analytics. For every prospect (draft classes 1996–2026), find historical comps from **pre-draft data only**, predict floor/median/ceiling outcomes, and grade players against those projections as their careers unfold. Headline question: does *how* a prospect scores (Shot DNA) beat traditional stats?

## Methodological ground rules (non-negotiable)
1. **No leakage.** A model or comp for draft year Y uses only players drafted before Y, and only outcomes observable by the start of season Y. Everything fitted (scalers, PCA/NMF bases, empirical-Bayes priors, conformal calibration) is fit inside the fold. Use the rolling-origin harness in `src/draft_dna/eval/`. Never hand-roll splits.
2. **Baselines first.** Compare every model to (a) draft pick alone and (b) standardized stats-only kNN. If a complex method doesn't win, say so and keep the simpler one.
3. **Uncertainty everywhere.** Output ranges and probabilities, and check calibration.
4. **Same-point comparisons.** A player with N NBA seasons is compared to comps' outcomes through N seasons (`outcomes_through_n`), never full careers.
5. **Small n (~1,500).** Prefer regularized, hierarchical or Bayesian methods. No deep learning without written justification in DECISIONS.md.
6. **Censoring.** 2022–2025 careers are unfinished. Use survival methods for career length and never label them "short careers".
7. Players with zero NBA minutes stay in the data as "Out of league". Never drop them.

## Outcome conventions (Phase 2)
- The canonical outcome is `career_value` / `composite_blend`: 75% z(best 3-season blend value) + 25% z(total), at the same N, scaled on training classes (DECISIONS D016).
- Tiers are cutoffs on `peak3_blend` (D017); read cutoffs from `data/modeled/outcomes/params.json`, never hardcode them.
- Seasons out of the league count as zero. Always use `modeled.outcomes__outcomes_through_n` at matching N.

## Modeling conventions (Phase 3)
- Target `y_peak6` = best 3-season blend value through season 6. Backtests use `eval.backtest` (class c trains year Y only if c + 6 <= Y).
- Choose models on test years 2006-2012 (`phase3.TUNING`); report 2013-2020 (`phase3.HOLDOUT`) only as the final, untouched comparison. Never tune on the holdout.
- Model of record: `phase3.pick_conformal` (D021). A new model replaces it only if it beats it on the holdout with a CI excluding zero.
- Every new comparison reports CRPS, pinball (floor/median/ceiling), Brier (bust, All-Star) and coverage vs both baselines.

## Draft class roles
- 1996–2021: training and backtesting
- 2022–2025: in-progress, provisional grades
- 2026: live projections
Configured in `config/settings.yaml`. Don't hardcode years.

## Grading terms
Projection (0 seasons) → Provisional-low (after Y1) → Provisional-medium (Y2–3) → **Year-4 Verdict** (not "Rookie Deal Verdict": 2nd-rounders aren't on 4-year deals) → Career Grade (Y8+ or retired).

## Data
- Layers: `data/raw` (cached source responses, never re-fetched) → `data/staging` (typed, deduped) → `data/modeled` → DuckDB at `data/draft_dna.duckdb`. All gitignored and rebuilt with `make data`.
- Rate limits in config. Sports-Reference must stay under 20 req/min. Always go through the shared cached fetcher, never call `requests` directly from an ingest module.
- Anchor player ID: Basketball-Reference ID. All cross-source joins go through the crosswalk. Manual overrides live in a committed CSV.
- Shot data coverage varies by era. Every player carries a coverage tier, and low-coverage players fall back to stats-only. Earlier drafts are labeled as having inconsistent shot data.

## Engineering
- Python 3.12, `uv`. Layout: `src/draft_dna/{ingest,crosswalk,features,outcomes,similarity,models,grading,eval,viz}`, plus `sql/`, `tests/`, `notebooks/` (exploration only, no pipeline logic), `app/`, `reports/`.
- Config-driven (YAML via `draft_dna.config.get_settings()`), type hints (`mypy --strict`), `logging` via `draft_dna.logging_utils` (no print in library code), pytest.
- `make check` (ruff + mypy + unit tests) must pass before every commit. Data-quality tests are marked `@pytest.mark.dq` and run with `make dq` against a built DB. CI never scrapes.

## Git workflow
- One branch per phase: `phase-N-<slug>` off up-to-date `main`.
- Commit at each working milestone with a clear message (e.g. "Add player ID crosswalk with match-quality tests"), then push the branch to origin.
- At the end of a phase: open a PR summarizing what was built, results vs baselines, and charts. Then **stop for the user's review**. The user merges, and Claude never merges or enables auto-merge.
- Never commit secrets (`.env` is gitignored; `.env.example` documents variables). Never commit raw or large data or `*.duckdb`.
- Do commit portfolio outputs: charts in `reports/`, README, sample prospect cards.

## Working with the user
- The user knows Python and data well but is new to Bayesian models, conformal prediction, survival analysis, NMF, KDE and empirical Bayes. Explain each simply (what it does, why here, what could go wrong) before implementing it, and record the choice in DECISIONS.md.
- Work one phase at a time. Update ROADMAP.md checkboxes as items land.
