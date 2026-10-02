# Draft DNA

NBA draft analytics: historical comps from pre-draft data, floor/median/ceiling projections, and season-by-season grading of players against those projections.

**Headline question:** does *how* a prospect scores (shot location and style) predict NBA success better than traditional stats?

Work in progress. See [ROADMAP.md](ROADMAP.md) for status and [DECISIONS.md](DECISIONS.md) for every methodological choice.

## Quickstart

```bash
make setup      # install environment (requires uv)
make check      # lint, type check, unit tests
make data       # download everything (first run: several hours, rate-limited) and build the database
make transform  # rebuild staging -> modeled -> DuckDB from cached downloads (minutes)
make dq         # data-quality tests against the built database
```

All raw data is rebuilt from scripts; nothing under `data/` is committed.

## Phase 1: Data pipeline

### What's in the database
| | |
|---|---|
| Players | 2,715: 1,833 draft picks (1996–2026, both rounds) plus 882 undrafted players who debuted 1997 or later |
| NBA player-seasons | 12,415 (1997–2026), with playoffs, awards, All-Star selections, games absent, team quality and head coach |
| College player-seasons | 6,992 for 2,214 players, with Barttorvik advanced stats from 2008 and strength of schedule |
| Combine | 1,330 players with measurements and/or athletic tests (2000–2026) |
| Salaries | 12,998 player-seasons, used only to validate the outcome definition in Phase 2 |

Draft classes: **1996–2021** for training and backtesting (1,540 picks), **2022–2025** in progress (233), **2026** live (60).

### Sources
Basketball-Reference (drafts, NBA stats, awards, bios), Sports-Reference College Basketball, Barttorvik (college advanced stats and rim/midrange shot splits), and stats.nba.com through `nba_api` (draft history, combine). Every request goes through one cached, rate-limited fetcher. Sports-Reference is held under its ~20 requests/minute limit, and no page is ever downloaded twice.

### What data exists for each draft class

![Coverage by draft class](reports/phase1/coverage_by_draft_year.png)

How to read it:
- **College advanced stats depend on era.** Barttorvik (college BPM, adjusted ratings) starts in 2008, and rim/midrange shot splits start in 2010. Usage and assist rate go back to 1996 because they are derived from box-score totals where Sports-Reference doesn't publish them. Derived values match the published ones to within 0.06 points wherever both exist.
- **About 1 in 5 picks didn't play college basketball.** These are international or pro-team prospects (9–33% per class) and, before the 2006 age rule, high-schoolers (up to 15%). They are kept, with a `prospect_source` flag and explicit missing data. Dropping them would remove some of the best outcomes in the sample (Kobe, Garnett, LeBron, Dirk, Giannis).
- **Combine data starts in 2000, and top prospects often skip athletic testing.** Measurements are much better covered than athletic tests.

### Linking players across sources
Each source has its own player IDs, and names don't line up cleanly. The crosswalk anchors on the Basketball-Reference ID and records how every link was made. Problems it handles:
- **Pick numbering:** stats.nba.com and Basketball-Reference number second-round picks differently in 1997, 2001 and 2002. Joining on pick number silently swapped about 85 players (2001 #30 is Trenton Hassell on one site and Gilbert Arenas on the other), so picks are linked by name within each draft year.
- **Name variants:** legal names and nicknames (Barttorvik's "Edrice" Adebayo, "Nah'Shon" Hyland), suffixes (Jr., III), accents, name changes (Metta World Peace, Enes Freedom) and garbled source names ("Ha Ha", "Cui Cui").
- **Same names:** same-name players who debuted the same season (two Tony Mitchells in 2014) are told apart by birth date.
- **Transfers:** Barttorvik assigns a new player ID at each school, so college stats are linked season by season.
- **Post-draft college seasons:** drafted players who later play college ball (James Nnaji: drafted 2023, Baylor 2025–26) have those seasons flagged so they can never leak into pre-draft features.

Match quality is tested against a hand-verified gold set of 62 hard cases, plus coverage thresholds: 98.5% of drafted college players are linked to their college stats, and 99.3% of drafted college players since 2010 are linked to Barttorvik.

### Era adjustment
Counting stats are kept per 100 possessions. Shooting efficiency (TS%) and 3-point attempt rate are stored relative to that season's league average, for both the NBA and Division I. The NCAA 3-point line moved twice (2008–09 and 2019–20) and the NBA's was shortened through 1996–97; both are recorded per season.

### Tests
- **Unit tests (46):** fetcher caching and rate limiting, HTML/CSV parsers, name and school normalization, crosswalk rules, derived-rate formulas.
- **Data-quality tests (17, `make dq`):** draft class sizes, unique keys, cross-source name agreement, ID coverage thresholds, the gold set, no NBA seasons before the draft, flagged post-draft college seasons, award counts, derived-vs-published rates, and null checks.

### Known limits
- No international or G League stats (see DECISIONS D014). Those prospects are modeled on age, size, pick and combine data.
- Opponent-dependent college rates (rebound, steal and block percentage) are unavailable before about 2010 and are not imputed.
- "Games absent" counts every game a player didn't play (injury, rest, coach's decision, G League), not injuries alone.

## Phase 2: What counts as a "better" career?

### Career value
Each NBA season gets a value score: the average of z-scored **VORP** and **Win Shares**, shifted so a season out of the league scores zero. From those season scores, every player gets these outcomes *through N seasons after the draft*:

- **Peak:** best 3-season stretch
- **Total:** sum of all seasons
- **Longevity:** seasons with 500+ minutes
- **Playoffs:** playoff value
- **Career value:** 75% peak + 25% total, scaled against the 1996–2021 draft classes at the same N

Measuring "through N seasons" is what lets a player with 3 seasons be compared fairly to his comps' first 3 seasons.

![Validation of value definitions](reports/phase2/value_definition_validation.png)

Three definitions of season value were tested against things the metric never sees: All-NBA and All-Star voting, and how much teams paid players on their second contracts. All three predict awards equally well (AUC ≈ 0.98–0.99). The VORP + Win Shares blend tracks contracts much better (Spearman 0.83 vs 0.63 for VORP alone). Plain VORP ranks a below-replacement player who keeps getting minutes *below* someone out of the league, while teams keep paying the former. A data-driven factor score didn't beat the simple blend, so the blend was kept (DECISIONS D016).

### Tiers
| Tier | Best 3-season value | Calibrated to |
|---|---|---|
| Out of league | < 0.01 | no positive NBA value |
| Bust | 0.01–0.32 | in the league, never a rotation player |
| Rotation | 0.32–1.01 | 1,000+ minutes a season |
| Starter | 1.01–1.76 | half the games started, 1,800+ minutes |
| All-Star | 1.76–2.37 | All-Star selection |
| All-NBA | ≥ 2.37 | All-NBA selection |

Tiers describe how a player *played*, not how he was voted (67% exact agreement with the award and role anchors, 97% within one tier).

![Tiers by pick](reports/phase2/tiers_by_pick.png)

For the 1996–2016 drafts, 37% of top-5 picks reached All-NBA level, versus 13% for picks 6–14 and 4% for picks 15–30. Among second-rounders, 56% never established an NBA career, yet 10% became starters or better.

### Career length (survival analysis)
Many careers from recent drafts are still going, so their length is only a lower bound. Kaplan–Meier curves use those *censored* careers correctly instead of treating them as short.

![Survival by pick](reports/phase2/career_survival_by_pick.png)

Median career length: 14 seasons for picks 1–5, 11 for 6–14, 8 for 15–30, and 3 for 31–60. A Cox model shows *when* draft position matters. During the rookie deal (seasons 1–4), each doubling of pick number (e.g. #10 → #20) **doubles** the yearly chance of a career ending (hazard ratio 2.07). After season 4 that drops to +16%, and age at draft matters more (+26% per year older). Teams give high picks extra chances early; once players reach their second contract, performance matters far more than draft slot.

### Limits
- The validation targets (awards, peak pay) favor peak performance, so they can't fully settle how much longevity should count. The 75/25 weighting is a judgment call, recorded in D016.
- A career counts as "ended" when the player has missed two straight seasons, so a player on a long injury absence could be misclassified.

## Phase 3: Comps and outcome ranges from pre-draft stats

**Question:** Using only what's known on draft night, can we predict a range of outcomes for each prospect better than draft position alone?

**Setup:**
- **Target:** best 3-season value through year 6, which ranks players almost identically to their career peak.
- **Backtest:** rolling-origin, one fold per draft year. A model for class Y trains only on classes whose 6-year outcomes were complete by draft night of Y.
- **Selection vs. testing:** every modeling choice was made on 2006–2012. The 2013–2020 holdout was scored once.

**Baselines:**
- **(a)** draft-slot history: the outcomes of the 60 historically nearest picks
- **(b)** k-nearest-neighbor comps on standardized pre-draft stats, age and physical profile

**Models tried:**
- LightGBM quantile regression
- NGBoost
- Bayesian hierarchical censored (Tobit) regression with partial pooling by position, era and pick range
- Learned-similarity comps (ridge-weighted and tree-proximity)
- Quantile blends
- Conformal calibration

![Model comparison](reports/phase3/model_comparison.png)

**Finding: pre-draft box-score stats did not beat draft position.**
- On the holdout, the best stats-based model (a blend of pick history, LightGBM and the Bayesian model) tied draft-slot history (CRPS 0.331 vs 0.331; difference CI −0.009 to +0.008).
- On the tuning years the same blend looked 5% better, an illusion the held-out years exposed.
- Stats-only comps were significantly *worse* than draft slot (+0.040).
- Teams' draft order already absorbs what college box scores say. This sets up the headline question for Phases 4–5: does *how* a prospect scores add information that box scores don't?

The model of record is therefore the simplest one: **draft-slot history with conformal calibration.** It is also the best calibrated: on the holdout, 23.7% of players finished below their floor (target 25%) and 9.8% above their ceiling (target 10%).

![Calibration](reports/phase3/calibration.png)

**Where it misses:** for top picks, P(All-Star or better) ran high in 2013–2020. Predictions near 43% came true about 20% of the time.

### Comps
Each prospect gets 15 comps from earlier draft classes, ranked by similarity in pre-draft stats, age and size. Each comp comes with a similarity score and the features that make the two players alike. A first version failed the sanity check: players missing most stats looked similar to everyone. A feature-overlap rule fixed it:

| Prospect | Top comps |
|---|---|
| Kevin Durant | Carmelo Anthony, Paul Pierce, Tim Thomas |
| Anthony Davis | Chris Bosh, LaMarcus Aldridge, Elton Brand |
| Shai Gilgeous-Alexander | Russell Westbrook, John Wall, Derrick Rose, Kyle Lowry |
| Cooper Flagg | Luol Deng, Carmelo Anthony, Thaddeus Young |
| AJ Dybantsa | RJ Barrett, Jabari Parker, Carmelo Anthony |

Comps are context, not the forecast. Stats-only comps are a poor guide to outcomes: Curry's and Lillard's nearest comps mostly flamed out.

### 2026 class
![2026 bands](reports/phase3/draft_2026_bands.png)

Full tables are in [reports/phase3/phase3_results.md](reports/phase3/phase3_results.md).

## Phase 4: Shot DNA, how each prospect scores

**Audit first.** Before building anything, I sampled 3,148 college games (2008–2026) to see what shot data really exists.

![Shot data coverage](reports/phase4/shot_coverage_by_season.png)

- **Shot type is complete:** every field-goal attempt in ESPN play-by-play is labeled layup, dunk, tip, jumper or three, from 2008 on.
- **Shot location (x/y) is not.** It doesn't exist before 2014. From 2014–2025 only 11–67% of shots have it, mostly from televised high-major games (48% of televised games vs 9% of others). Coverage fell to 11% in 2024, then jumped to 100% in 2025–26.

That shaped a two-tier design:

| Tier | What it measures | Coverage | Used for |
|---|---|---|---|
| A: shot type | rim / midrange / three / dunk mix; shooting by zone (empirical Bayes); assisted vs. self-created makes | 90–100% of drafted college players, 2010+ | Phase 5 headline test |
| B: shot location | 5 court zones; smoothed shot maps; NMF shot styles | 395 players (53–92% of the 2014–2020 classes, 92% of 2026) | player cards, secondary test |

![Shot data by class](reports/phase4/shot_data_by_draft_class.png)

### Six shot styles
Each eligible player's shots become a smoothed heat map. Non-negative matrix factorization then finds six building-block "styles," so every player is a mix of them. Shot distances are rescaled so the 3-point line sits in the same place in every era; otherwise the 2019–20 line change creates a fake "old-line threes" style.

![Shot styles](reports/phase4/shot_styles.png)

![Example shot maps](reports/phase4/example_shot_maps.png)

A style mix is a *shape* description, not a shot count. Trae Young's diffuse deep threes carry less weight than his concentrated rim attempts. For "% of shots from X," use the shot-type numbers: Trae took 53% of his shots from three, and only 11% of his rim makes were assisted (versus 49% for Zion).

The test of whether any of this predicts NBA success beyond draft position is Phase 5.

## Phase 5: Does *how* a prospect scores predict NBA success?

**The headline question.** Phase 3 found that college box-score stats don't beat draft position. Does shot data (where and how a prospect scores) add what box scores miss?

**Method.** The full analysis plan was written and committed before any model ran ([DECISIONS D027](DECISIONS.md)).
- **Cohort:** every model sees the same ~415 drafted college players from 2010 on with shot data.
- **Primary design (B):** fully leak-free. Careers are judged at year 4, and a model trains only on classes whose year-4 outcomes were known by draft night.
- **Sensitivity check (A):** year-6 outcomes and more training data, at the cost of using outcomes not yet known on draft night.
- **Models:** stats-only, shot-only and combined versions of both comps and LightGBM, each with and without the draft pick. The comps' stats/shot blend weight was tuned on early years only.

![Pre-registered comparisons](reports/phase5/preregistered_comparisons.png)

**Finding: no. Shot data added no detectable predictive value, in either design.**
- **Beyond pick + stats:** adding shot type, assisted rate and shot mix changed forecast error by −0.0001 (95% CI −0.0016 to +0.0013). That interval is narrow enough to rule out anything but a tiny effect.
- **On its own:** shot data is a *weaker* signal than box-score stats (+0.015 worse for models, +0.016 for comps). When the comp blend was tuned, it gave shot data zero weight.
- **Shot location:** on the 157–215 players with coordinates, adding zone shares and the six NMF shot styles didn't help either.
- **Subgroups:** none of the 24 position × pick-band subgroups showed a detectable effect.
- **Draft position:** nothing beat it. It remains the model of record.

The most plausible reading: a prospect's shot profile is mostly *downstream* of things box scores and scouts already capture (size, athleticism, role). How much and how efficiently a player scores matters; where the shots come from adds little once those are known.

![Exploratory signal](reports/phase5/exploratory_shot_signal.png)

**One lead worth following (exploratory, not confirmed).** Of 15 shot traits, only **rim finishing** (shrunken rim FG% relative to the Division I average) still correlates with outcomes after removing what pick + stats predict (ρ = 0.14, p = 0.0032). That's right at the multiple-testing threshold, so it's a hypothesis, not a finding. A confirmation test is pre-registered ([D029](DECISIONS.md)) on the 2023–2026 classes, which no Phase 5 analysis has touched, to run as their year-4 outcomes arrive.

**Limits:**
- **Sample size:** shot data starts with the 2010 class, so the cohort is ~415 players. Effects smaller than about 0.5% of forecast error can't be detected.
- **Coordinates:** location data covers the 2014+ classes and leans toward televised games.
- **Outcome:** "success" means peak value by year 4 (primary) or year 6 (check). A trait that only pays off late in a career would be missed.
