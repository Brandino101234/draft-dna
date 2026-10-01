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
