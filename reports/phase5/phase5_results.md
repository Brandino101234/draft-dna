# Phase 5 results (pre-registered in DECISIONS D027)

## Pre-registered comparisons

| comparison                                         |   crps_diff |   ci_lo |   ci_hi | verdict                  | design   |
|:---------------------------------------------------|------------:|--------:|--------:|:-------------------------|:---------|
| H1: pick+stats+shot vs pick+stats                  |     -0.0001 | -0.0016 |  0.0013 | no detectable difference | B        |
| H2a: shot vs stats (no pick)                       |      0.0154 |  0.0042 |  0.0267 | worse                    | B        |
| H2b: stats+shot vs stats (no pick)                 |     -0.0006 | -0.0023 |  0.001  | no detectable difference | B        |
| H2c: kNN shot vs kNN stats                         |      0.0164 |  0.0083 |  0.0246 | worse                    | B        |
| H3: pick+stats+shot vs pick only                   |      0.0013 | -0.0066 |  0.0092 | no detectable difference | B        |
| H3: pick+shot vs pick only                         |      0.0043 | -0.0023 |  0.0103 | no detectable difference | B        |
| Tier B: pick+stats+location vs pick+stats (subset) |      0.0017 | -0.0005 |  0.0038 | no detectable difference | B        |
| H1: pick+stats+shot vs pick+stats                  |      0.0009 | -0.0014 |  0.0032 | no detectable difference | A        |
| H2a: shot vs stats (no pick)                       |      0.0261 |  0.0098 |  0.0419 | worse                    | A        |
| H2b: stats+shot vs stats (no pick)                 |     -0.0005 | -0.0029 |  0.0018 | no detectable difference | A        |
| H2c: kNN shot vs kNN stats                         |      0.02   |  0.01   |  0.0301 | worse                    | A        |
| H3: pick+stats+shot vs pick only                   |      0.0096 |  0.0004 |  0.0185 | worse                    | A        |
| H3: pick+shot vs pick only                         |      0.0105 |  0.0023 |  0.0186 | worse                    | A        |
| Tier B: pick+stats+location vs pick+stats (subset) |      0.0018 | -0.0007 |  0.0044 | no detectable difference | A        |

## All models, design B (strict, year-4 outcome)

| model                    |   n |   crps |   crps_vs_pick |   ci_lo |   ci_hi |   pinball_floor |   pinball_median |   pinball_ceiling |   brier_bust |   brier_allstar |   below_floor |   above_ceiling |
|:-------------------------|----:|-------:|---------------:|--------:|--------:|----------------:|-----------------:|------------------:|-------------:|----------------:|--------------:|----------------:|
| Pick only + conformal    | 414 | 0.2711 |         0      |  0      |  0      |          0.1106 |           0.1851 |            0.1114 |       0.2123 |          0.0383 |        0.2089 |          0.1039 |
| kNN stats                | 414 | 0.2887 |         0.0176 |  0.0054 |  0.0313 |          0.1131 |           0.1919 |            0.1326 |       0.2275 |          0.0423 |        0.221  |          0.1159 |
| kNN shot                 | 414 | 0.3052 |         0.0341 |  0.0205 |  0.0489 |          0.1141 |           0.2038 |            0.1399 |       0.2413 |          0.0435 |        0.2114 |          0.1159 |
| kNN stats+shot w=0.25    | 414 | 0.3014 |         0.0303 |  0.0168 |  0.0452 |          0.1139 |           0.2006 |            0.1407 |       0.2359 |          0.0434 |        0.2029 |          0.1087 |
| kNN stats+shot w=0.5     | 414 | 0.3    |         0.0289 |  0.0155 |  0.0434 |          0.1139 |           0.2    |            0.14   |       0.2353 |          0.0429 |        0.1993 |          0.1184 |
| kNN stats+shot w=0.75    | 414 | 0.297  |         0.0259 |  0.0128 |  0.0402 |          0.1141 |           0.1976 |            0.1384 |       0.2332 |          0.0428 |        0.2089 |          0.1232 |
| LightGBM stats           | 414 | 0.2872 |         0.0161 |  0.0039 |  0.0288 |          0.1134 |           0.1937 |            0.1286 |       0.2268 |          0.0413 |        0.2729 |          0.1063 |
| LightGBM pick+stats      | 414 | 0.2725 |         0.0014 | -0.0065 |  0.0093 |          0.1101 |           0.1816 |            0.123  |       0.2101 |          0.0404 |        0.2343 |          0.1111 |
| LightGBM shot            | 414 | 0.3026 |         0.0316 |  0.0175 |  0.0465 |          0.114  |           0.2007 |            0.1396 |       0.24   |          0.0431 |        0.2343 |          0.1232 |
| LightGBM pick+shot       | 414 | 0.2754 |         0.0043 | -0.0023 |  0.0103 |          0.1112 |           0.1839 |            0.1246 |       0.2128 |          0.0411 |        0.215  |          0.1377 |
| LightGBM stats+shot      | 414 | 0.2866 |         0.0155 |  0.0033 |  0.0281 |          0.1133 |           0.1926 |            0.1277 |       0.2265 |          0.0411 |        0.285  |          0.1135 |
| LightGBM pick+stats+shot | 414 | 0.2723 |         0.0013 | -0.0066 |  0.0092 |          0.11   |           0.1814 |            0.1234 |       0.2107 |          0.0402 |        0.244  |          0.1111 |

## All models, design A (sensitivity check)

| model                    |   n |   crps |   crps_vs_pick |   ci_lo |   ci_hi |   pinball_floor |   pinball_median |   pinball_ceiling |   brier_bust |   brier_allstar |   below_floor |   above_ceiling |
|:-------------------------|----:|-------:|---------------:|--------:|--------:|----------------:|-----------------:|------------------:|-------------:|----------------:|--------------:|----------------:|
| Pick only + conformal    | 417 | 0.3313 |         0      |  0      |  0      |          0.1357 |           0.2207 |            0.1434 |       0.1917 |          0.0646 |        0.2014 |          0.0959 |
| kNN stats                | 417 | 0.3719 |         0.0406 |  0.0245 |  0.0571 |          0.144  |           0.2535 |            0.1634 |       0.2391 |          0.0706 |        0.241  |          0.1079 |
| kNN shot                 | 417 | 0.3919 |         0.0606 |  0.0429 |  0.0788 |          0.1459 |           0.2649 |            0.1812 |       0.2525 |          0.0739 |        0.2278 |          0.1103 |
| kNN stats+shot w=0.25    | 417 | 0.3849 |         0.0536 |  0.0356 |  0.072  |          0.145  |           0.2595 |            0.1755 |       0.2471 |          0.0727 |        0.2074 |          0.1079 |
| kNN stats+shot w=0.5     | 417 | 0.378  |         0.0467 |  0.029  |  0.0642 |          0.1445 |           0.2556 |            0.1702 |       0.2402 |          0.0728 |        0.2086 |          0.1079 |
| kNN stats+shot w=0.75    | 417 | 0.3768 |         0.0455 |  0.0279 |  0.0631 |          0.1442 |           0.2529 |            0.1712 |       0.2378 |          0.0736 |        0.2038 |          0.1127 |
| LightGBM stats           | 417 | 0.364  |         0.0327 |  0.018  |  0.0485 |          0.1456 |           0.2471 |            0.1604 |       0.2234 |          0.0698 |        0.3237 |          0.0983 |
| LightGBM pick+stats      | 417 | 0.34   |         0.0087 | -0.0007 |  0.0181 |          0.1356 |           0.2258 |            0.155  |       0.1986 |          0.0679 |        0.2974 |          0.1223 |
| LightGBM shot            | 417 | 0.3901 |         0.0588 |  0.0386 |  0.0776 |          0.1464 |           0.2658 |            0.1758 |       0.2492 |          0.0719 |        0.2938 |          0.1175 |
| LightGBM pick+shot       | 417 | 0.3418 |         0.0105 |  0.0023 |  0.0186 |          0.1353 |           0.2271 |            0.1561 |       0.2007 |          0.0687 |        0.247  |          0.1367 |
| LightGBM stats+shot      | 417 | 0.3635 |         0.0322 |  0.0174 |  0.0474 |          0.1455 |           0.2461 |            0.1599 |       0.2234 |          0.0699 |        0.3309 |          0.1031 |
| LightGBM pick+stats+shot | 417 | 0.3409 |         0.0096 |  0.0004 |  0.0185 |          0.1352 |           0.226  |            0.1544 |       0.1988 |          0.0676 |        0.2926 |          0.1343 |

## Comp blend weight (chosen on tuning years): stats weight = 1.0

|   w_stats | model                 |   crps_tune |   crps_report |
|----------:|:----------------------|------------:|--------------:|
|      1    | kNN stats             |      0.2881 |        0.2892 |
|      0.75 | kNN stats+shot w=0.75 |      0.2979 |        0.2962 |
|      0.5  | kNN stats+shot w=0.5  |      0.3027 |        0.2979 |
|      0.25 | kNN stats+shot w=0.25 |      0.3027 |        0.3004 |
|      0    | kNN shot              |      0.3057 |        0.3047 |

## Subgroups: pick+stats+shot vs pick+stats

| split     | group       |   n |   crps_diff |   ci_lo |   ci_hi | design   |
|:----------|:------------|----:|------------:|--------:|--------:|:---------|
| position  | big         |  22 |     -0.0011 | -0.0061 |  0.004  | B        |
| position  | forward     | 172 |      0.0012 | -0.0012 |  0.0035 | B        |
| position  | guard       | 220 |     -0.001  | -0.0031 |  0.001  | B        |
| pick band | picks 1-14  | 104 |      0.0003 | -0.0033 |  0.0041 | B        |
| pick band | picks 15-30 | 115 |      0.001  | -0.0023 |  0.0044 | B        |
| pick band | picks 31-60 | 195 |     -0.001  | -0.0025 |  0.0005 | B        |
| position  | big         |  27 |     -0.0029 | -0.0116 |  0.005  | A        |
| position  | forward     | 182 |      0.0026 | -0.0007 |  0.0058 | A        |
| position  | guard       | 208 |     -0      | -0.0035 |  0.003  | A        |
| pick band | picks 1-14  | 110 |      0.0028 | -0.002  |  0.0076 | A        |
| pick band | picks 15-30 | 111 |      0.0007 | -0.0043 |  0.0055 | A        |
| pick band | picks 31-60 | 196 |      0      | -0.0028 |  0.0027 | A        |

## Exploratory (not pre-registered): shot-feature signal beyond pick + stats

| feature              |   n |   rho_outcome |   rho_beyond_pick |   rho_beyond_pick_stats |   p_beyond_pick_stats |
|:---------------------|----:|--------------:|------------------:|------------------------:|----------------------:|
| rim_fg_eb_rel        | 414 |        0.2791 |            0.1866 |                  0.1444 |                0.0032 |
| rim_fg_eb            | 414 |        0.2759 |            0.1827 |                  0.1416 |                0.0039 |
| rim_rate             | 414 |        0.1689 |            0.1192 |                  0.0648 |                0.1879 |
| three_rate           | 414 |       -0.1077 |           -0.1259 |                 -0.0584 |                0.2358 |
| three_rate_rel       | 414 |       -0.1151 |           -0.1246 |                 -0.0504 |                0.3067 |
| three_fg_eb_rel      | 414 |       -0.0356 |           -0.0536 |                 -0.0413 |                0.4023 |
| dunk_share           | 414 |        0.1653 |            0.0828 |                  0.0412 |                0.4028 |
| mid_fg_eb            | 414 |        0.0382 |           -0.0156 |                 -0.0385 |                0.4349 |
| assisted_three_share | 383 |        0.0669 |            0.0826 |                  0.0364 |                0.4778 |
| mid_fg_eb_rel        | 414 |        0.0274 |           -0.0019 |                 -0.0273 |                0.5799 |
| three_fg_eb          | 414 |       -0.042  |           -0.0401 |                 -0.0273 |                0.5802 |
| unassisted_share     | 412 |       -0.0596 |           -0.0373 |                  0.0209 |                0.6724 |
| mid_rate             | 414 |       -0.0633 |            0.0275 |                 -0.0059 |                0.904  |
| assisted_rim_share   | 412 |        0.0992 |            0.0626 |                 -0.0058 |                0.9071 |
| assisted_mid_share   | 412 |        0.0489 |            0.0743 |                 -0.0006 |                0.99   |
