# NNUE model comparison

Baseline: `benchmarks/results/nnue-cp-simple-public-5m-handoff-baseline-large.json`  
Candidate: `benchmarks/results/nnue-cp-simple-public-50m-large.json`

Lower centipawn losses, mistake counts, failures, and runtime are better. Higher agreement and depth are better.

## Move quality by category

| Group | Model | Positions | Valid | Best % | Top-3 % | Mean cp | Median cp | P90 cp | P95 cp | >=100 cp | >=200 cp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | baseline | 400 | 400 | 35.00 | 61.25 | 85.27 | 13 | 181.00 | 356.00 | 70 | 33 |
| overall | candidate | 400 | 400 | 38.00 | 63.50 | 74.60 | 9 | 162.00 | 292.00 | 60 | 31 |
| endgame | baseline | 100 | 100 | 27.00 | 49.00 | 157.58 | 9.50 | 209.50 | 388.50 | 16 | 10 |
| endgame | candidate | 100 | 100 | 30.00 | 51.00 | 148.72 | 8.00 | 205.00 | 375.50 | 17 | 9 |
| middlegame | baseline | 200 | 200 | 34.00 | 58.50 | 76.39 | 23 | 188.20 | 331.50 | 47 | 18 |
| middlegame | candidate | 200 | 200 | 39.50 | 65.50 | 58.17 | 11 | 159.80 | 272.40 | 32 | 15 |
| opening | baseline | 50 | 50 | 50.00 | 82.00 | 13.34 | 0.00 | 58.10 | 62.30 | 0 | 0 |
| opening | candidate | 50 | 50 | 44.00 | 72.00 | 14.28 | 1.00 | 49.10 | 58.55 | 1 | 0 |
| tactical | baseline | 50 | 50 | 40.00 | 76.00 | 67.40 | 14.50 | 179.20 | 372.50 | 7 | 5 |
| tactical | candidate | 50 | 50 | 42.00 | 72.00 | 71.52 | 23.50 | 216.70 | 302.70 | 10 | 7 |

## Mate and draw-changing outcomes

| Group | Model | Missed win mate | Allowed loss mate | Worse mate distance | Est. draw-to-loss | Est. win-to-draw | Immediate draws |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | baseline | 0 | 3 | 3 | 11 | 8 | 0 |
| overall | candidate | 0 | 3 | 0 | 14 | 9 | 0 |

## Search and runtime

| Group | Model | Mean depth | Completed nodes | Total nodes | Runtime seconds | Failures |
|---|---|---:|---:|---:|---:|---:|
| overall | baseline | 4.62 | 10674.85 | 17057.20 | 209.41 | 0 |
| overall | candidate | 4.68 | 11202.40 | 16936.86 | 209.37 | 0 |
| endgame | baseline | 6.18 | 12748.83 | 18102.08 | 49.74 | 0 |
| endgame | candidate | 6.22 | 12802.46 | 17336.19 | 49.00 | 0 |
| middlegame | baseline | 4.08 | 9647.02 | 16593.21 | 105.89 | 0 |
| middlegame | candidate | 4.17 | 10595.27 | 16748.76 | 106.48 | 0 |
| opening | baseline | 4.10 | 10057.28 | 17434.64 | 26.92 | 0 |
| opening | candidate | 4.08 | 10290.64 | 17756.54 | 27.31 | 0 |
| tactical | baseline | 4.20 | 11255.76 | 16445.96 | 26.75 | 0 |
| tactical | candidate | 4.24 | 11342.52 | 16070.88 | 26.48 | 0 |

## Paired uncertainty

```json
{
  "common_cp_positions": 381,
  "baseline_mean_loss_cp": 85.26509186351706,
  "candidate_mean_loss_cp": 74.60367454068242,
  "candidate_lower_cp_loss": 101,
  "baseline_lower_cp_loss": 80,
  "equal_cp_loss": 200,
  "game_clusters": 102,
  "bootstrap_replicates": 2000,
  "candidate_exact_agreement_advantage_ci95": [
    -0.02083820093457944,
    0.08311025722408681
  ],
  "candidate_cp_loss_reduction_ci95": [
    0.28933297204534164,
    22.808938505088534
  ]
}
```

Positive confidence-interval values favor the candidate. Intervals resample source games and measure suite sampling uncertainty, not engine-label or timing uncertainty.
