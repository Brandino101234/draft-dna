# Phase 2 results

## Validation of season-value definitions

Draft classes 1996-2016, 10 seasons after the draft; 95% bootstrap CIs.

| check                                         | candidate   |   estimate |   ci_low |   ci_high |   diff_vs_vorp |   diff_ci_low |   diff_ci_high |
|:----------------------------------------------|:------------|-----------:|---------:|----------:|---------------:|--------------:|---------------:|
| AUC: ever All-NBA (10 yrs)                    | vorp        |      0.99  |    0.985 |     0.994 |        nan     |       nan     |        nan     |
| AUC: ever All-NBA (10 yrs)                    | blend       |      0.991 |    0.986 |     0.995 |          0.001 |        -0.001 |          0.003 |
| AUC: ever All-NBA (10 yrs)                    | factor      |      0.991 |    0.986 |     0.995 |          0.001 |        -0     |          0.001 |
| AUC: ever All-Star (10 yrs)                   | vorp        |      0.982 |    0.972 |     0.99  |        nan     |       nan     |        nan     |
| AUC: ever All-Star (10 yrs)                   | blend       |      0.984 |    0.976 |     0.991 |          0.002 |        -0.001 |          0.006 |
| AUC: ever All-Star (10 yrs)                   | factor      |      0.983 |    0.974 |     0.991 |          0.001 |        -0     |          0.002 |
| Spearman: All-Star selections                 | vorp        |      0.486 |    0.445 |     0.524 |        nan     |       nan     |        nan     |
| Spearman: All-Star selections                 | blend       |      0.487 |    0.446 |     0.524 |          0.001 |        -0.002 |          0.005 |
| Spearman: All-Star selections                 | factor      |      0.485 |    0.445 |     0.524 |         -0     |        -0.001 |          0.001 |
| Spearman: peak pay % of cap (yrs 5-10)        | vorp        |      0.766 |    0.729 |     0.799 |        nan     |       nan     |        nan     |
| Spearman: peak pay % of cap (yrs 5-10)        | blend       |      0.886 |    0.869 |     0.9   |          0.119 |         0.093 |          0.15  |
| Spearman: peak pay % of cap (yrs 5-10)        | factor      |      0.818 |    0.79  |     0.843 |          0.051 |         0.036 |          0.069 |
| Spearman: 2nd-contract pay vs value thru yr 4 | vorp        |      0.633 |    0.588 |     0.676 |        nan     |       nan     |        nan     |
| Spearman: 2nd-contract pay vs value thru yr 4 | blend       |      0.828 |    0.805 |     0.847 |          0.194 |         0.16  |          0.231 |
| Spearman: 2nd-contract pay vs value thru yr 4 | factor      |      0.723 |    0.688 |     0.756 |          0.089 |         0.067 |          0.113 |

## Tier cutoffs (best 3-season blend value)

| tier     |   min_peak3 |
|:---------|------------:|
| Bust     |        0.01 |
| Rotation |        0.32 |
| Starter  |        1.01 |
| All-Star |        1.76 |
| All-NBA  |        2.37 |

Agreement with award/role anchors: 66.9% exact.


## Tier mix by pick band (drafts 1996-2016)

| band        |   Out of league |   Bust |   Rotation |   Starter |   All-Star |   All-NBA |
|:------------|----------------:|-------:|-----------:|----------:|-----------:|----------:|
| Picks 1-5   |               4 |     10 |         18 |        19 |         11 |        37 |
| Picks 6-14  |               8 |     18 |         27 |        22 |         12 |        13 |
| Picks 15-30 |              19 |     25 |         26 |        19 |          7 |         4 |
| Picks 31-60 |              56 |     22 |         11 |         6 |          2 |         2 |

## Career length (Kaplan-Meier medians, seasons)

| band        |   n |   median |
|:------------|----:|---------:|
| Picks 1-5   | 150 |       14 |
| Picks 6-14  | 270 |       11 |
| Picks 15-30 | 477 |        8 |
| Picks 31-60 | 876 |        3 |

## Cox model (concordance 0.688)

| period      | covariate     |   hazard_ratio |   ci_low |   ci_high |     p |   n_players |
|:------------|:--------------|---------------:|---------:|----------:|------:|------------:|
| seasons 1-4 | log2_pick     |          2.069 |    1.832 |     2.338 | 0     |        1561 |
| seasons 1-4 | age_at_draft  |          1.106 |    1.026 |     1.192 | 0.009 |        1561 |
| seasons 1-4 | draft_year_10 |          0.866 |    0.773 |     0.97  | 0.013 |        1561 |
| seasons 1-4 | non_college   |          1.065 |    0.824 |     1.376 | 0.63  |        1561 |
| seasons 5+  | log2_pick     |          1.163 |    1.09  |     1.241 | 0     |         893 |
| seasons 5+  | age_at_draft  |          1.259 |    1.178 |     1.346 | 0     |         893 |
| seasons 5+  | draft_year_10 |          0.951 |    0.836 |     1.083 | 0.45  |         893 |
| seasons 5+  | non_college   |          1.097 |    0.884 |     1.36  | 0.402 |         893 |
