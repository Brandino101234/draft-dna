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
