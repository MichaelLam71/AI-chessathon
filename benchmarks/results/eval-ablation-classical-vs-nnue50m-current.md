# Large fixed-position comparison

400 matched positions; 1000 ms wall ceiling, 10000 ms remaining-clock input. Fresh processes and existing diagnostics enabled for both evaluators.

Ordinary cp statistics and mistake/blunder counts exclude mate-involving comparisons and failed replies. Agreement includes all positions. Blunders are included in mistakes. Tags overlap the primary categories.

## Move quality

| Group | Evaluator | N | CP N | Best % | Top 3 % | Mean cp | Median cp | P90 cp | P95 cp | >=100 cp | >=200 cp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | classical | 400 | 383 | 34.8 | 58.5 | 83.00 | 15 | 201.00 | 320.50 | 73 | 39 |
| overall | nnue | 400 | 382 | 39.0 | 63.2 | 79.58 | 9.00 | 169.80 | 320.35 | 66 | 35 |
| endgame | classical | 100 | 86 | 27.0 | 53.0 | 154.08 | 8.00 | 267.50 | 506.75 | 15 | 10 |
| endgame | nnue | 100 | 86 | 32.0 | 51.0 | 154.84 | 8.00 | 240.50 | 390.75 | 19 | 10 |
| middlegame | classical | 200 | 197 | 37.5 | 59.0 | 62.80 | 20 | 174.00 | 274.00 | 39 | 19 |
| middlegame | nnue | 200 | 196 | 41.0 | 65.5 | 62.08 | 11.00 | 166.00 | 295.75 | 35 | 18 |
| opening | classical | 50 | 50 | 30.0 | 54.0 | 58 | 18.00 | 121.30 | 185.20 | 8 | 3 |
| opening | nnue | 50 | 50 | 42.0 | 70.0 | 22.64 | 5.50 | 51.70 | 92.00 | 3 | 1 |
| tactical | classical | 50 | 50 | 44.0 | 72.0 | 65.34 | 17.50 | 216.30 | 240.25 | 11 | 7 |
| tactical | nnue | 50 | 50 | 42.0 | 72.0 | 75.66 | 17.00 | 216.70 | 394.30 | 9 | 6 |
| defensive | classical | 100 | 100 | 37.0 | 61.0 | 78.29 | 21.50 | 238.00 | 329.70 | 28 | 13 |
| defensive | nnue | 100 | 100 | 37.0 | 65.0 | 82.40 | 17.00 | 247.60 | 381.65 | 26 | 14 |
| imbalanced | classical | 133 | 117 | 43.6 | 65.4 | 133.30 | 10 | 222.60 | 476.40 | 28 | 14 |
| imbalanced | nnue | 133 | 116 | 44.4 | 63.2 | 136.16 | 12.00 | 197.50 | 350.50 | 26 | 12 |
| in_check | classical | 15 | 15 | 66.7 | 93.3 | 18.87 | 0 | 51.20 | 117.90 | 1 | 0 |
| in_check | nnue | 15 | 15 | 60.0 | 86.7 | 78.93 | 0 | 236.40 | 459.90 | 2 | 2 |
| no_local_training_overlap | classical | 388 | 371 | 34.8 | 59.0 | 71.66 | 15 | 202.00 | 318.50 | 71 | 38 |
| no_local_training_overlap | nnue | 388 | 370 | 39.7 | 63.4 | 66.46 | 9.00 | 168.20 | 315.15 | 64 | 33 |
| training_overlap | classical | 12 | 12 | 33.3 | 41.7 | 433.75 | 16.50 | 153.70 | 2249.45 | 2 | 1 |
| training_overlap | nnue | 12 | 12 | 16.7 | 58.3 | 483.92 | 13.50 | 248.90 | 2592.75 | 2 | 2 |

## Mate and estimated draw transitions

| Group | Evaluator | Mate samples | Missed winning mate | Allowed losing mate | Worse mate distance | Est. draw to loss | Est. win to draw | Immediate draws |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | classical | 17 | 1 | 1 | 1 | 10 | 15 | 0 |
| overall | nnue | 18 | 0 | 2 | 1 | 13 | 11 | 1 |
| endgame | classical | 14 | 1 | 1 | 0 | 1 | 2 | 0 |
| endgame | nnue | 14 | 0 | 1 | 1 | 2 | 2 | 1 |
| middlegame | classical | 3 | 0 | 0 | 1 | 6 | 7 | 0 |
| middlegame | nnue | 4 | 0 | 1 | 0 | 10 | 7 | 0 |
| opening | classical | 0 | 0 | 0 | 0 | 2 | 3 | 0 |
| opening | nnue | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tactical | classical | 0 | 0 | 0 | 0 | 1 | 3 | 0 |
| tactical | nnue | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| defensive | classical | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| defensive | nnue | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| imbalanced | classical | 16 | 1 | 1 | 1 | 3 | 0 | 0 |
| imbalanced | nnue | 17 | 0 | 2 | 1 | 2 | 0 | 0 |
| in_check | classical | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| in_check | nnue | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| no_local_training_overlap | classical | 17 | 1 | 1 | 1 | 10 | 15 | 0 |
| no_local_training_overlap | nnue | 18 | 0 | 2 | 1 | 13 | 11 | 1 |
| training_overlap | classical | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| training_overlap | nnue | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Search and runtime

| Group | Evaluator | Depth | Total nodes | Completed nodes | Seconds | Failures |
|---|---|---:|---:|---:|---:|---:|
| overall | classical | 4.41 | 8668.13 | 6356.88 | 257.67 | 0 |
| overall | nnue | 4.88 | 9559.82 | 6677.44 | 263.78 | 0 |
| endgame | classical | 6.30 | 12167.88 | 9188.32 | 62.84 | 0 |
| endgame | nnue | 6.32 | 9387.24 | 7239.55 | 63.56 | 0 |
| middlegame | classical | 3.82 | 7731.71 | 5507.56 | 130.33 | 0 |
| middlegame | nnue | 4.37 | 9589.82 | 6494.52 | 133.33 | 0 |
| opening | classical | 3.50 | 6585.12 | 4567.90 | 32.50 | 0 |
| opening | nnue | 4.32 | 10222.34 | 6212.20 | 33.90 | 0 |
| tactical | classical | 3.90 | 7497.32 | 5880.30 | 31.79 | 0 |
| tactical | nnue | 4.64 | 9122.48 | 6750.14 | 32.79 | 0 |
| defensive | classical | 4.10 | 8305.23 | 6363.07 | 64.64 | 0 |
| defensive | nnue | 4.72 | 9671.93 | 7040.65 | 66.18 | 0 |
| imbalanced | classical | 5.11 | 9569.15 | 7297.26 | 84.64 | 0 |
| imbalanced | nnue | 5.55 | 9843.32 | 6955.77 | 87.16 | 0 |
| in_check | classical | 5.07 | 8399.73 | 7241 | 9.21 | 0 |
| in_check | nnue | 5.33 | 9497.20 | 6624.87 | 10.26 | 0 |
| no_local_training_overlap | classical | 4.41 | 8662.06 | 6388.70 | 249.38 | 0 |
| no_local_training_overlap | nnue | 4.88 | 9507.97 | 6685.74 | 255.20 | 0 |
| training_overlap | classical | 4.25 | 8864.25 | 5328.08 | 8.09 | 0 |
| training_overlap | nnue | 4.83 | 11236.17 | 6409.25 | 8.37 | 0 |

## Matched ordinary-cp subset and uncertainty

This subset excludes any position involving mate for either evaluator, so both cp means use exactly the same FENs. Confidence intervals resample whole source games (2,000 replicates, seed 19); positive differences favor NNUE. They describe sampling uncertainty within this suite, not label error or wall-clock run-to-run variation.

```json
{
  "common_cp_positions": 382,
  "common_cp_classical_mean": 83.21989528795811,
  "common_cp_nnue_mean": 79.57591623036649,
  "nnue_lower_cp_loss": 125,
  "classical_lower_cp_loss": 113,
  "equal_cp_loss": 144,
  "game_clusters": 102,
  "bootstrap_replicates": 2000,
  "nnue_exact_agreement_advantage_ci95": [
    -0.009805737109658678,
    0.09653465346534654
  ],
  "nnue_cp_loss_reduction_ci95": [
    -10.209552113088698,
    18.544307542867383
  ]
}
```

### Excluding local training overlap

```json
{
  "common_cp_positions": 370,
  "common_cp_classical_mean": 71.85135135135135,
  "common_cp_nnue_mean": 66.46216216216216,
  "nnue_lower_cp_loss": 121,
  "classical_lower_cp_loss": 108,
  "equal_cp_loss": 141,
  "game_clusters": 102,
  "bootstrap_replicates": 2000,
  "nnue_exact_agreement_advantage_ci95": [
    0.0,
    0.10133424657534247
  ],
  "nnue_cp_loss_reduction_ci95": [
    -7.976277994157741,
    19.58477213916045
  ]
}
```

No-local-overlap means no identical piece-placement and side-to-move input in the scanned local training chunks. It does not rule out nearby positions or other training sources.

## Interpretation limits

Draw-to-loss means best-move Stockfish WDL draw probability >=90% and chosen-move loss probability >=50%. Win-to-draw means best-move win probability >=50% and chosen-move draw probability >=90%. These are model estimates, not tablebase or repetition proofs. FENs omit game history.

Mate events use Stockfish's finite-search mate reports. Missing a mate does not necessarily lose the game. Mate-distance changes are counted separately from abandoning or allowing mate. A transition can overlap multiple outcome counters.

Most positions come from newly generated low-node Stockfish self-play with varied openings. Categories use board-based heuristics; tactical does not mean a verified unique solution. This is a broader diagnostic holdout, not an Elo estimate or proof of competition strength.
