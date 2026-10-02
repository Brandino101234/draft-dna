# Phase 6 results

## Mean PIT by segment (year 6)

| group           |   n |   beat_ceiling |   bc_lo |   bc_hi |   below_floor |   bf_lo |   bf_hi |   mean_pit |   pit_lo |   pit_hi | split        |
|:----------------|----:|---------------:|--------:|--------:|--------------:|--------:|--------:|-----------:|---------:|---------:|:-------------|
| 1-5             |  90 |          0.111 |   0.061 |   0.193 |         0.267 |   0.186 |   0.366 |      0.494 |    0.432 |    0.556 | Pick         |
| 6-14            | 162 |          0.056 |   0.03  |   0.102 |         0.216 |   0.16  |   0.286 |      0.492 |    0.452 |    0.533 | Pick         |
| 15-30           | 288 |          0.125 |   0.092 |   0.168 |         0.17  |   0.131 |   0.218 |      0.542 |    0.508 |    0.575 | Pick         |
| 31-60           | 537 |          0.136 |   0.11  |   0.168 |         0.019 |   0.01  |   0.034 |      0.489 |    0.464 |    0.513 | Pick         |
| Big             | 117 |          0.12  |   0.073 |   0.191 |         0.188 |   0.128 |   0.268 |      0.484 |    0.428 |    0.54  | Position     |
| Forward         | 435 |          0.14  |   0.111 |   0.176 |         0.08  |   0.058 |   0.11  |      0.536 |    0.51  |    0.561 | Position     |
| Guard           | 434 |          0.108 |   0.082 |   0.141 |         0.129 |   0.101 |   0.164 |      0.517 |    0.49  |    0.545 | Position     |
| <19.5           | 163 |          0.135 |   0.091 |   0.196 |         0.215 |   0.159 |   0.284 |      0.463 |    0.416 |    0.511 | Age at draft |
| 19.5-20.5       | 201 |          0.139 |   0.098 |   0.194 |         0.119 |   0.082 |   0.172 |      0.547 |    0.507 |    0.586 | Age at draft |
| 20.5-21.5       | 213 |          0.113 |   0.077 |   0.162 |         0.094 |   0.062 |   0.141 |      0.515 |    0.478 |    0.552 | Age at draft |
| 21.5-22.5       | 310 |          0.106 |   0.077 |   0.146 |         0.077 |   0.053 |   0.113 |      0.501 |    0.47  |    0.533 | Age at draft |
| 22.5+           | 190 |          0.111 |   0.073 |   0.163 |         0.079 |   0.048 |   0.126 |      0.486 |    0.445 |    0.526 | Age at draft |
| College         | 830 |          0.119 |   0.099 |   0.143 |         0.106 |   0.087 |   0.129 |      0.526 |    0.507 |    0.545 | Background   |
| High school     |  27 |          0.333 |   0.186 |   0.522 |         0.111 |   0.039 |   0.281 |      0.62  |    0.492 |    0.749 | Background   |
| Intl / pro team | 218 |          0.087 |   0.057 |   0.132 |         0.124 |   0.087 |   0.174 |      0.402 |    0.365 |    0.44  | Background   |
| 2002-07         | 297 |          0.168 |   0.13  |   0.215 |         0.104 |   0.075 |   0.144 |      0.516 |    0.481 |    0.551 | Era          |
| 2008-14         | 420 |          0.095 |   0.071 |   0.127 |         0.093 |   0.069 |   0.124 |      0.512 |    0.486 |    0.539 | Era          |
| 2015-20         | 360 |          0.106 |   0.078 |   0.142 |         0.133 |   0.102 |   0.172 |      0.484 |    0.455 |    0.513 | Era          |

## Team heterogeneity

|   horizon |   teams |      Q |   df |   p_value |   tau |   league_mean |
|----------:|--------:|-------:|-----:|----------:|------:|--------------:|
|         4 |      30 | 22.006 |   29 |     0.82  |     0 |         0.492 |
|         6 |      30 | 19.096 |   29 |     0.919 |     0 |         0.504 |
|         8 |      30 | 17.304 |   29 |     0.957 |     0 |         0.511 |

## Situation effects (propensity-weighted, year 6)

| situation                                                      | timing                   |    n |   treated_share |   raw_pit_diff |   ipw_pit_diff |   ipw_lo |   ipw_hi |   rr_above_median |   rr_lo |   rr_hi |   e_value |   e_value_ci |   max_abs_smd_before |   max_abs_smd_after |
|:---------------------------------------------------------------|:-------------------------|-----:|----------------:|---------------:|---------------:|---------:|---------:|------------------:|--------:|--------:|----------:|-------------:|---------------------:|--------------------:|
| Drafted by a bottom-third team (by SRS)                        | pre-draft                | 1077 |           0.397 |         -0.017 |         -0.008 |   -0.046 |    0.028 |             1.018 |   0.907 |   1.148 |     1.152 |        1     |                0.655 |               0.053 |
| Drafting team deep at his position (top half of minutes share) | pre-draft                | 1077 |           0.458 |          0.019 |         -0.013 |   -0.051 |    0.022 |             0.942 |   0.826 |   1.068 |     1.316 |        1     |                0.424 |               0.158 |
| Drafting team changed head coach in years 1-3                  | post-draft, team-level   | 1077 |           0.666 |          0.022 |          0.012 |   -0.031 |    0.045 |             1.043 |   0.915 |   1.175 |     1.253 |        1     |                0.195 |               0.011 |
| Traded / moved teams in years 1-3                              | post-draft, player-level |  921 |           0.585 |         -0.051 |         -0.059 |   -0.098 |   -0.022 |             0.892 |   0.791 |   0.996 |     1.491 |        1.064 |                0.362 |               0.01  |
| Missed >= 25% of games in years 1-2 (injury, G League or DNP)  | post-draft, player-level |  899 |           0.627 |         -0.182 |         -0.254 |   -0.287 |   -0.209 |             0.529 |   0.463 |   0.611 |     3.186 |        2.66  |                0.918 |               0.27  |

## Overperformance predictability

Leave-classes-out CV (year 4): Spearman 0.261, R2 0.072.

Strict rolling-origin check (each class predicted only from classes with known outcomes):

|   horizon | players   |   pooled_spearman |   perm_p |   n | positive_years   |
|----------:|:----------|------------------:|---------:|----:|:-----------------|
|         4 | all       |             0.187 |    0     | 778 | 12/13            |
|         4 | college   |             0.153 |    0     | 630 | 10/13            |
|         6 | all       |             0.125 |    0.004 | 600 | 8/10             |
|         6 | college   |             0.04  |    0.431 | 435 | 7/9              |

## Year-4 Verdict vs Career Grade

| verdict4     |   below floor |   within band |   beat ceiling |
|:-------------|--------------:|--------------:|---------------:|
| below floor  |            69 |            61 |              0 |
| within band  |             7 |           590 |             28 |
| beat ceiling |             0 |            23 |             62 |

Overturned: 14.2% (95% CI 12.0%-16.7%) of 840 players.


### Late bloomers (within band at year 4 -> beat ceiling at year 8)

| player_name             |   draft_year |   pick |   actual4 |   actual8 |
|:------------------------|-------------:|-------:|----------:|----------:|
| Stephen Curry           |         2009 |      7 |      2.24 |      5.47 |
| Shai Gilgeous-Alexander |         2018 |     11 |      1.58 |      5.41 |
| James Harden            |         2009 |      3 |      2.94 |      5.2  |
| Domantas Sabonis        |         2016 |     11 |      1.78 |      3.55 |
| Derrick White           |         2017 |     29 |      0.97 |      2.51 |
| Kyle Lowry              |         2006 |     24 |      0.95 |      2.37 |
| Hassan Whiteside        |         2010 |     33 |      0.01 |      2.26 |
| Jeff Teague             |         2009 |     19 |      1.23 |      2.1  |
| Tobias Harris           |         2011 |     19 |      0.94 |      2.03 |
| Dejounte Murray         |         2016 |     29 |      0.57 |      2.02 |
| Isaiah Hartenstein      |         2017 |     43 |      0.21 |      1.72 |
| Reggie Jackson          |         2011 |     24 |      1.1  |      1.62 |
| Ersan İlyasova          |         2005 |     36 |      0.03 |      1.43 |
| Tiago Splitter          |         2007 |     28 |      0.15 |      1.42 |
| DeMarre Carroll         |         2009 |     27 |      0.31 |      1.42 |

### True late busts (within band at year 4 -> below floor at year 8)

| player_name        |   draft_year |   pick |   actual4 |   actual8 |
|:-------------------|-------------:|-------:|----------:|----------:|
| Greg Oden          |         2007 |      1 |      0.5  |      0.5  |
| Víctor Claver      |         2009 |     22 |      0    |      0.01 |
| Ekpe Udoh          |         2010 |      6 |      0.28 |      0.31 |
| Jimmer Fredette    |         2011 |     10 |      0.13 |      0.13 |
| Fab Melo           |         2012 |     22 |      0    |      0    |
| Nemanja Nedović    |         2013 |     30 |      0    |      0    |
| Guerschon Yabusele |         2016 |     16 |      0.06 |      0.06 |
