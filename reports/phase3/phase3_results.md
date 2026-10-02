# Phase 3 results

Target: best 3-season value through season 6. Rolling-origin backtest by draft year.

## Holdout 2013-2020 (480 players; never used for model selection)

| model                   |   crps |   crps_vs_pick |   crps_vs_pick_lo |   crps_vs_pick_hi |   crps_vs_knn |   pinball_floor |   pinball_median |   pinball_ceiling |   brier_bust |   brier_allstar |   below_floor |   above_ceiling |
|:------------------------|-------:|---------------:|------------------:|------------------:|--------------:|----------------:|-----------------:|------------------:|-------------:|----------------:|--------------:|----------------:|
| Pick only               | 0.3308 |         0      |            0      |            0      |       -0.0404 |          0.1308 |           0.2167 |            0.1495 |       0.1953 |          0.0645 |        0.2354 |          0.1021 |
| Pick only + conformal   | 0.3323 |         0.0015 |           -0.0023 |            0.0048 |       -0.0389 |          0.1282 |           0.2186 |            0.1558 |       0.1937 |          0.0661 |        0.2365 |          0.0979 |
| Stats kNN               | 0.3712 |         0.0404 |            0.0181 |            0.0638 |        0      |          0.1336 |           0.2465 |            0.1766 |       0.2296 |          0.0694 |        0.2646 |          0.0844 |
| Learned-weight kNN      | 0.3851 |         0.0543 |            0.0302 |            0.0785 |        0.0139 |          0.1424 |           0.2615 |            0.1726 |       0.2375 |          0.0732 |        0.3406 |          0.0719 |
| Tree-proximity kNN      | 0.3587 |         0.0279 |            0.0062 |            0.0508 |       -0.0125 |          0.1345 |           0.2406 |            0.1651 |       0.221  |          0.0648 |        0.2615 |          0.1552 |
| LightGBM quantile       | 0.334  |         0.0031 |           -0.0086 |            0.0141 |       -0.0372 |          0.1282 |           0.2204 |            0.153  |       0.1957 |          0.0626 |        0.3104 |          0.0917 |
| NGBoost                 | 0.3838 |         0.053  |            0.0355 |            0.0717 |        0.0126 |          0.1672 |           0.2392 |            0.184  |       0.2255 |          0.0683 |        0.5104 |          0.1896 |
| Bayesian hierarchical   | 0.3602 |         0.0294 |            0.0106 |            0.0468 |       -0.011  |          0.1334 |           0.2297 |            0.1821 |       0.1908 |          0.074  |        0.2719 |          0.0688 |
| Final blend             | 0.3312 |         0.0004 |           -0.0086 |            0.0084 |       -0.04   |          0.1276 |           0.2157 |            0.1545 |       0.1882 |          0.0652 |        0.3125 |          0.0833 |
| Final blend + conformal | 0.3309 |         0      |           -0.0095 |            0.0084 |       -0.0403 |          0.1255 |           0.2156 |            0.1562 |       0.1861 |          0.0655 |        0.3271 |          0.0854 |

## Tuning 2006-2012 (420 players; used for model selection)

| model                   |   crps |   crps_vs_pick |   crps_vs_pick_lo |   crps_vs_pick_hi |   crps_vs_knn |   pinball_floor |   pinball_median |   pinball_ceiling |   brier_bust |   brier_allstar |   below_floor |   above_ceiling |
|:------------------------|-------:|---------------:|------------------:|------------------:|--------------:|----------------:|-----------------:|------------------:|-------------:|----------------:|--------------:|----------------:|
| Pick only               | 0.362  |         0      |            0      |            0      |       -0.0447 |          0.1433 |           0.2406 |            0.1645 |       0.1869 |          0.0801 |        0.2429 |          0.1286 |
| Pick only + conformal   | 0.3615 |        -0.0005 |           -0.0056 |            0.0039 |       -0.0451 |          0.1428 |           0.2378 |            0.1686 |       0.1869 |          0.0802 |        0.2369 |          0.1048 |
| Stats kNN               | 0.4067 |         0.0447 |            0.0198 |            0.0721 |        0      |          0.1521 |           0.2751 |            0.1862 |       0.2258 |          0.0859 |        0.206  |          0.1167 |
| Learned-weight kNN      | 0.3911 |         0.029  |            0.0052 |            0.054  |       -0.0156 |          0.1501 |           0.2604 |            0.1785 |       0.2142 |          0.0849 |        0.2369 |          0.1024 |
| Tree-proximity kNN      | 0.3825 |         0.0205 |           -0.0026 |            0.0434 |       -0.0242 |          0.1463 |           0.2503 |            0.1835 |       0.2076 |          0.0818 |        0.219  |          0.1357 |
| LightGBM quantile       | 0.3548 |        -0.0072 |           -0.0185 |            0.0055 |       -0.0519 |          0.1433 |           0.2361 |            0.1565 |       0.1828 |          0.0774 |        0.269  |          0.1167 |
| NGBoost                 | 0.4038 |         0.0418 |            0.0201 |            0.0652 |       -0.0028 |          0.1681 |           0.2393 |            0.2172 |       0.2155 |          0.0848 |        0.4714 |          0.2655 |
| Bayesian hierarchical   | 0.3667 |         0.0047 |           -0.0194 |            0.0289 |       -0.04   |          0.1437 |           0.2384 |            0.1789 |       0.1797 |          0.0768 |        0.2405 |          0.0607 |
| Final blend             | 0.343  |        -0.019  |           -0.0308 |           -0.0062 |       -0.0637 |          0.1414 |           0.2282 |            0.1502 |       0.1762 |          0.0745 |        0.269  |          0.069  |
| Final blend + conformal | 0.3434 |        -0.0186 |           -0.032  |           -0.0051 |       -0.0633 |          0.1402 |           0.2269 |            0.1558 |       0.1745 |          0.0759 |        0.3167 |          0.0714 |

## Comp sanity check (stats kNN, earlier classes only)

| prospect                            | top 5 comps (peak value thru yr 6)                                                                                |
|:------------------------------------|:------------------------------------------------------------------------------------------------------------------|
| Kevin Durant (2007, #2)             | Jumaine Jones (0.7); Tim Thomas (1.0); Carmelo Anthony (2.4); Paul Pierce (3.9); Keith Van Horn (1.3)             |
| Anthony Davis (2012, #1)            | Brandan Wright (0.8); Stromile Swift (0.9); Elton Brand (3.1); LaMarcus Aldridge (2.6); Chris Bosh (2.9)          |
| Shai Gilgeous-Alexander (2018, #11) | Wade Baldwin (0.0); Russell Westbrook (3.0); John Wall (2.6); Kyle Lowry (1.6); Derrick Rose (2.9)                |
| Trae Young (2018, #5)               | Pierre Jackson (0.0); Luke Ridnour (1.1); Stephen Curry (4.6); Jawun Evans (0.0); Jordan Farmar (0.6)             |
| Ja Morant (2019, #2)                | Eric Maynor (0.2); Cameron Payne (0.4); Nick Calathes (0.2); Pierre Jackson (0.0); Trae Young (2.7)               |
| Tyrese Haliburton (2020, #12)       | Gabe Pruitt (0.0); Lonzo Ball (0.9); Jarrett Jack (0.9); Shane Larkin (0.2); Patrick McCaw (0.1)                  |
| Jalen Brunson (2018, #33)           | Wayne Ellington (0.4); Luke Kennard (0.9); Luther Head (1.2); Reggie Jackson (1.6); Nolan Smith (0.0)             |
| Draymond Green (2012, #35)          | Eduardo Nájera (0.7); Jeff Green (1.2); Caron Butler (1.9); P.J. Tucker (0.0); Kawhi Leonard (4.1)                |
| Jimmy Butler (2011, #30)            | Andre Emmett (0.0); Brandon Roy (3.3); Quincy Pondexter (0.4); Chris Douglas-Roberts (0.2); Ryan Robertson (0.0)  |
| Damian Lillard (2012, #6)           | Marcus Brown (0.0); James Cotton (0.0); Charles Jenkins (0.0); Bryce Drew (0.1); Quincy Douby (0.0)               |
| Stephen Curry (2009, #7)            | Rodney Stuckey (1.1); Marcus Brown (0.0); Quincy Douby (0.0); Jameer Nelson (1.5); Luke Ridnour (1.1)             |
| Kawhi Leonard (2011, #15)           | Wilson Chandler (0.8); Quentin Richardson (0.9); Marcus Williams (0.0); Carmelo Anthony (2.4); Rick Rickert (0.0) |
| Zion Williamson (2019, #1)          | Stromile Swift (0.9); Brandan Wright (0.8); Andrew Bogut (1.5); Paul Pierce (3.9); Chris Bosh (2.9)               |
| Anthony Bennett (2013, #1)          | Tobias Harris (1.7); Chris Bosh (2.9); Derrick Brown (0.3); Jared Sullinger (1.0); Nikola Vučević (1.6)           |
| Luka Dončić (2018, #3)              | Sasha Pavlović (0.2); Sergey Karasev (0.0); Furkan Korkmaz (0.4); Evan Fournier (0.9); Álex Abrines (0.3)         |
| Giannis Antetokounmpo (2013, #15)   | Serge Ibaka (2.2); Dāvis Bertāns (0.2); Darko Miličić (0.3); Peja Stojaković (2.3); Nikoloz Tskitishvili (-0.0)   |
| Victor Wembanyama (2023, #1)        | Pavel Podkolzin (0.0); Kristaps Porziņģis (1.2); Peter John Ramos (0.0); Slavko Vraneš (0.0); Alexis Ajinça (0.1) |
| Cooper Flagg (2025, #1)             | Luol Deng (2.1); Carmelo Anthony (2.4); Grant Williams (0.8); Mike Miller (1.8); Thaddeus Young (1.8)             |
| AJ Dybantsa (2026, #1)              | RJ Barrett (0.5); Jabari Parker (0.7); Carmelo Anthony (2.4); Marcus Williams (0.0); T.J. Warren (1.0)            |
| LeBron James (2003, #1)             | Kwame Brown (0.5); Tyson Chandler (1.7); Amar'e Stoudemire (2.3); Eddy Curry (0.8); DeSagana Diop (0.6)           |

## 2026 draft projections (model of record: pick history + conformal)

|   pick | player_name       | position   |   floor |   median |   ceiling |   p_allstar_plus |   p_bust_or_worse |
|-------:|:------------------|:-----------|--------:|---------:|----------:|-----------------:|------------------:|
|      1 | AJ Dybantsa       | forward    |    0.72 |     1.78 |      3.78 |             0.51 |              0.09 |
|      2 | Darryn Peterson   | guard      |    0.87 |     1.75 |      3.29 |             0.5  |              0.1  |
|      3 | Cameron Boozer    | forward    |    0.75 |     1.62 |      3.19 |             0.44 |              0.15 |
|      4 | Caleb Wilson      | forward    |    0.69 |     1.57 |      3.2  |             0.43 |              0.2  |
|      5 | Keaton Wagler     | guard      |    0.38 |     1.28 |      3.22 |             0.36 |              0.22 |
|      6 | Mikel Brown Jr.   | guard      |    0.54 |     1.2  |      3.04 |             0.32 |              0.17 |
|      7 | Darius Acuff Jr.  | guard      |    0.38 |     0.94 |      2.28 |             0.21 |              0.21 |
|      8 | Kingston Flemings | guard      |    0.49 |     1    |      2.67 |             0.27 |              0.16 |
|      9 | Morez Johnson Jr. | forward    |    0.39 |     0.94 |      2.73 |             0.31 |              0.22 |
|     10 | Brayden Burries   | guard      |    0.22 |     0.64 |      2.73 |             0.3  |              0.29 |
|     11 | Yaxel Lendeborg   | forward    |    0.14 |     0.59 |      2.11 |             0.17 |              0.35 |
|     12 | Aday Mara         | big        |    0.11 |     0.63 |      2.15 |             0.14 |              0.4  |
|     13 | Nate Ament        | forward    |    0.16 |     0.68 |      2.17 |             0.14 |              0.41 |
|     14 | Hannes Steinbach  | forward    |    0.19 |     0.71 |      2.38 |             0.16 |              0.37 |
