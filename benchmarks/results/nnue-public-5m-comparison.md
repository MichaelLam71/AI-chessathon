# NNUE model comparison

Baseline: `benchmarks\results\nnue-cp-simple-current-step3-large.json`  
Candidate: `benchmarks\results\nnue-cp-simple-public-5m-large.json`

Lower centipawn losses, mistake counts, failures, and runtime are better. Higher agreement and depth are better.

## Move quality by category

| Group | Model | Positions | Valid | Best % | Top-3 % | Mean cp | Median cp | P90 cp | P95 cp | >=100 cp | >=200 cp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | baseline | 400 | 400 | 33.00 | 56.50 | 120.36 | 17.00 | 421.10 | 597.90 | 90 | 58 |
| overall | candidate | 400 | 400 | 33.75 | 59.75 | 88.85 | 16 | 191.00 | 360.00 | 73 | 37 |
| endgame | baseline | 100 | 100 | 32.00 | 58.00 | 140.09 | 5.00 | 237.00 | 448.50 | 13 | 9 |
| endgame | candidate | 100 | 100 | 25.00 | 47.00 | 159.37 | 10.00 | 209.50 | 388.50 | 17 | 10 |
| middlegame | baseline | 200 | 200 | 34.50 | 55.00 | 114.33 | 28.50 | 523.50 | 636.50 | 50 | 32 |
| middlegame | candidate | 200 | 200 | 33.50 | 58.00 | 76.20 | 23 | 203.00 | 338.50 | 44 | 20 |
| opening | baseline | 50 | 50 | 28.00 | 54.00 | 89.94 | 18.00 | 250.50 | 584.00 | 9 | 6 |
| opening | candidate | 50 | 50 | 48.00 | 80.00 | 17.68 | 0.50 | 58.10 | 62.30 | 1 | 1 |
| tactical | baseline | 50 | 50 | 34.00 | 62.00 | 140.48 | 41.00 | 528.20 | 589.00 | 18 | 11 |
| tactical | candidate | 50 | 50 | 38.00 | 72.00 | 88.02 | 27.00 | 272.50 | 404.50 | 11 | 6 |

## Mate and draw-changing outcomes

| Group | Model | Missed win mate | Allowed loss mate | Worse mate distance | Est. draw-to-loss | Est. win-to-draw | Immediate draws |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | baseline | 1 | 2 | 2 | 22 | 12 | 1 |
| overall | candidate | 0 | 3 | 3 | 11 | 8 | 0 |

## Search and runtime

| Group | Model | Mean depth | Completed nodes | Total nodes | Runtime seconds | Failures |
|---|---|---:|---:|---:|---:|---:|
| overall | baseline | 4.26 | 7219.65 | 11209.93 | 248.06 | 0 |
| overall | candidate | 4.31 | 7305.54 | 10757.08 | 247.45 | 0 |
| endgame | baseline | 5.81 | 8872.65 | 11632.56 | 58.60 | 0 |
| endgame | candidate | 5.72 | 8582.72 | 11705.67 | 59.86 | 0 |
| middlegame | baseline | 3.75 | 6688.74 | 11269.34 | 126.84 | 0 |
| middlegame | candidate | 3.85 | 6918.91 | 10392.67 | 124.67 | 0 |
| opening | baseline | 3.78 | 7597.38 | 10352.34 | 30.60 | 0 |
| opening | candidate | 3.86 | 7147.64 | 10482.44 | 31.10 | 0 |
| tactical | baseline | 3.68 | 5659.58 | 10984.60 | 31.82 | 0 |
| tactical | candidate | 3.78 | 6455.60 | 10592.14 | 31.62 | 0 |

## Paired uncertainty

```json
{
  "common_cp_positions": 381,
  "baseline_mean_loss_cp": 120.67716535433071,
  "candidate_mean_loss_cp": 88.84514435695539,
  "candidate_lower_cp_loss": 111,
  "baseline_lower_cp_loss": 102,
  "equal_cp_loss": 168,
  "game_clusters": 102,
  "bootstrap_replicates": 2000,
  "candidate_exact_agreement_advantage_ci95": [
    -0.04250385071090048,
    0.05405405405405406
  ],
  "candidate_cp_loss_reduction_ci95": [
    14.47756995877038,
    49.548019021739144
  ]
}
```

Positive confidence-interval values favor the candidate. Intervals resample source games and measure suite sampling uncertainty, not engine-label or timing uncertainty.
