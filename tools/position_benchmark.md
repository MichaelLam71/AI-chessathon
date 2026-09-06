# Fixed-position benchmark

## Large suite (400 positions)

The prepared `benchmarks/positions-large-labelled.json` is shared by both evaluators:

```powershell
uv run python tools/position_benchmark.py test benchmarks/positions-large-labelled.json --evaluator classical --output benchmarks/results/classical-large.json
uv run python tools/position_benchmark.py test benchmarks/positions-large-labelled.json --evaluator nnue --output benchmarks/results/nnue-large.json
uv run python tools/position_benchmark.py compare benchmarks/results/classical-large.json benchmarks/results/nnue-large.json
```

Those output names may already contain the completed comparison. Use new output
names to repeat; no command overwrites existing result files.

The large suite has 50 opening/early middlegame, 200 normal middlegame, 50 tactical,
and 100 endgame positions from 102 games (two existing Chessathon games and 100 new
Stockfish 19 self-play games). There are 206 White-to-move and 194 Black-to-move
positions; 133 are tagged as materially imbalanced. `benchmarks/positions-large.json` stores
source game, ply, category, tags, seed and source hashes. `benchmarks/positions-large.txt` is
the equivalent FEN list; label the JSON to preserve its category metadata.

Rebuild with new output names if desired:

```powershell
uv run python tools/position_benchmark.py build --output benchmarks/positions-another.json
uv run python tools/position_benchmark.py label benchmarks/positions-another.json benchmarks/positions-another-labelled.json --nodes 2000000
```

`build` reuses `benchmarks/benchmark-holdout.pgn` when present. Otherwise it
generates 100 games using seed 19, 20 opening lines, 10,000 nodes per move, and
one thread/128 MB hash. Early moves and occasional later moves vary among the
top three within 100 cp of the best low-node score. Games stop at a claimable
outcome or 240 plies. The generation settings are much cheaper than labelling.
The new games were generated independently of the evaluator tests. The training
NPZ files were not used to generate or select positions. A subsequent audit of
all 270 local chunks (13.5 million rows) found 12 matching board-and-side feature
inputs; the other 388 positions are reported separately under
`no_local_training_overlap`. This rules out identical local training inputs, not
nearby positions or other training sources. Fixed-node self-play improves
repeatability, but the saved PGN/JSON files
are the durable reference.

The optional training audit is reproducible without rerunning Stockfish:

```powershell
uv run python tools/audit_position_training.py benchmarks/positions-large.json --output benchmarks/new-training-audit.json
uv run python tools/audit_position_training.py benchmarks/positions-large.json --output benchmarks/new-training-audit.json --annotate-labels benchmarks/positions-large-labelled.json
```

The second command adds audit tags/metadata to an existing label JSON and keeps
all Stockfish scores intact. Annotate before running either evaluator, so their
label hashes match. The original completed audit is saved in
`benchmarks/position-large-training-audit.json` with hashes of all scanned chunks.

Selection uses at least four legal moves, no available mate-in-one, no terminal
or claimable-draw positions, a maximum eight positions per game, and at least ten
plies between samples from one game. Counter-only duplicates are removed.
Near duplicates with the same side to move and at most four changed square/piece
entries are rejected, even across games. All 400 FENs were checked against their
source PGNs. Positions are shuffled deterministically before tests.

Primary categories are heuristic and mutually exclusive: endgames have at most
four non-pawn/non-king pieces or at most twelve total pieces; openings are the
remaining positions through ply 24; tactical positions are in check or have at
least three legal captures and a checking move; the rest are middlegames.
Imbalanced tags mean a material difference >=150 cp. Defensive tags are added
from label best scores between -600 and -50 cp. Tags overlap primary categories.

The large labels use **2 million nodes per position**, shared over all legal root
moves, with Stockfish 19, one thread and 128 MB hash. Raw WDL estimates, depths,
scores and PVs are retained. `*.checkpoint.jsonl` saves completed positions as
they finish: rerun the identical label command to resume if the final JSON does
not exist. Settings, input and binary hashes must match. Keep the checkpoint;
it also provides per-position analysis times. A partially written/corrupt journal
line is rejected rather than silently discarding evidence.

## Small suite and general options

Run from the repository root in PowerShell:

```powershell
uv run python tools/position_benchmark.py label benchmarks/positions.txt benchmarks/positions-labelled.json
uv run python tools/position_benchmark.py test benchmarks/positions-labelled.json
```

Select classical or NNUE explicitly without editing `agent.py`:

```powershell
uv run python tools/position_benchmark.py test benchmarks/positions-labelled.json --evaluator classical --output benchmarks/results/position-classical.json
uv run python tools/position_benchmark.py test benchmarks/positions-labelled.json --evaluator nnue --output benchmarks/results/position-nnue.json
```

The selection changes `USE_LEARNED_EVAL` only in each child process before the
existing harness runner calls the agent. Default `--evaluator current` uses the
source setting. Reports include the evaluator selection and NNUE weights hash.

The label step automatically finds the executable under `stockfish/`. Override it
with `--stockfish "C:\path with spaces\stockfish.exe"`. It uses one thread, 128 MB
hash, a fresh game/hash per FEN, and 10 million nodes per position across MultiPV
for **all legal moves**. Use `--nodes 20000000` or `--depth 22` to change the limit.
Labelling is the expensive one-time step. Fixed nodes, executable, options, and
positions improve repeatability; retain the JSON as the reference across tests.
The node budget is shared across root moves, not granted separately to each move.
MultiPV may finish with differing depths between lines when the node limit hits;
each line's depth is recorded. Labels are finite-search estimates, not ground truth.

JSON contains each FEN, best move, top three moves (fewer if necessary), every
legal move's score, mate distance, PV, depth and nodes, plus engine identity,
binary hash and analysis settings. All scores use the original side-to-move POV.
Labelling every legal move lets tests calculate losses entirely offline without
Stockfish. Equal scores retain Stockfish's ordering; exact agreement is with the
single recorded best move, so an equally scored alternative can have zero loss.

The test step imports the current `agent.py` through the existing harness runner
in a fresh process for every FEN. This keeps TT/history state independent of
position order. It enables existing `CHESS_SEARCH_STATS` diagnostics and fixes
`PYTHONHASHSEED=0` in child processes. Explicit evaluator selection changes only
the existing toggle in each child; engine source and search logic are unchanged.
OMP/OpenBLAS/MKL/NumExpr thread-count environment variables are set to one in
each worker before import and recorded in the report. CPU affinity is not pinned.
Diagnostics add overhead; use identical settings when comparing versions.

Defaults are a **1,000 ms wall-time ceiling per position**, a separate 60-second
import timeout, and **10,000 ms passed to get_move**. The engine's own time manager
still decides how much of that remaining clock to use. The ceiling includes IPC
and response handling after initialization; overruns fail and the worker is killed.
This measures cold-position behavior, without normal between-move cache reuse.

```powershell
uv run python tools/position_benchmark.py test benchmarks/positions-labelled.json --budget-ms 2000 --time-left-ms 40000 --output benchmarks/results/position-results-v2.json
```

Console output and `position-results.json` include exact and top-three agreement
as fractions of **all** positions; failed positions count as disagreement.
Report schema version 2 makes mean/median/P90/P95 loss and mistake counts cover
valid replies where **both best and chosen scores are ordinary cp**, with failures
and mate-involving comparisons reported separately. Loss is
`max(0, best score - chosen score)` in centipawns. Percentiles use linear interpolation.
Large mistakes default to >=100 cp and blunders to >=200 cp; blunders are included
in the large-mistake count. Override with `--mistake-cp` and `--blunder-cp`.
Mate reports separately count missed winning mates, newly allowed losing mates,
and worsened mate distances. `synthetic_mean_loss_cp` and per-position `loss_cp`
retain the old 100,000-cp mate mapping for traceability; use `mean_loss_cp` and
`ordinary_loss_cp` for ordinary loss. Earlier report files retain their original
semantics and are not rewritten.

Estimated draw-to-loss transitions require best-move WDL draw probability >=90%
and chosen-move loss probability >=50%. Estimated win-to-draw transitions require
best-move win probability >=50% and chosen-move draw probability >=90%. These
are Stockfish model estimates, not proven game outcomes. No tablebases are used;
FEN tests cannot reconstruct repetition history. Immediate board draws are also
counted. Outcome counters can overlap. Old labels without WDL cannot identify
these estimated transitions; relabel under a new filename if needed.

Reports include `by_category` and `by_tag` summaries with the same metrics and
per-group runtimes. `compare` writes readable tables and paired statistics to
`benchmarks/results/position-large-results.md` and a JSON companion. It checks matching
engine/weights/label hashes and budgets. It also compares loss on the common
ordinary-cp subset and gives game-cluster bootstrap intervals (2,000 replicates).
Intervals assess suite sampling uncertainty, not Stockfish label uncertainty,
machine load or search timing variance. Single-pass fixed-position quality is
evidence about move selection; it does not directly measure Elo or game strength.

Average completed depth, completed nodes and total nodes use available diagnostics
from valid replies. Total nodes include unfinished search iterations. Total runtime
includes child startup, imports and cleanup. Individual results preserve diagnostics,
failure details and move wall time where available. Reports record agent and label
hashes. Output files are created exclusively: use new filenames for later runs.
Final labels are written after all analyses; completed analyses are checkpointed.

`benchmarks/positions.txt` contains the 12 original diagnostic positions. They
cover openings, tactical middlegames, mixed-piece endings and pawn endings with
both sides to move. This is a starter regression set, not a balanced strength test.
Add one valid nonterminal FEN per line; blank lines and `#` comments are accepted,
and duplicate FEN strings are removed during labelling.

The tool and Stockfish remain outside submission/runtime code. The existing default
packager includes root Python files and `weights/`, not `tools/` or `stockfish/`.
