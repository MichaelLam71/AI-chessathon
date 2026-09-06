# Combined search experiment -- unbenchmarked

The previous complete agent is saved as search-before-combined.txt in this folder.
This is the immediately preceding instrumented agent with its narrow pawn-capture
filter, not the older bare 66% baseline. No imported or ported engine code is used.
The new helpers are original Python implementations of general search techniques.

## Changes

- PVS: first move full window; later moves zero window, with full-window re-search
  for results strictly between alpha and beta.
- LMR: one ply only, non-PV nodes at depth >= 4, fifth move or later. Excludes
  captures, promotions, checks/evasions, root, TT/killer moves, positive-history
  quiets, all pawn moves, and castling. Reduced alpha-raisers are re-searched at
  full depth before any cutoff. No aggressive reduction formula is used.
- Aspiration: depth >= 4 around the last completed score, initially +/-75 cp,
  doubling on failure and falling back to the full window after three failures.
  An interrupted re-search leaves the last completed iteration's move intact.
- Ordering: promotion priority, victim-first capture ordering with en-passant
  value, bounded colour-specific history carried across turns with decay,
  failed-quiet penalties, and a countermove table. Killers remain per search.
- TT: 262144 hashed slots with full-key collision checks, depth/exact/generation
  replacement, mate-score normalization, and halfmove-clock/null-subtree keys.
  Entries never grow beyond the slot limit. Root uses TT for ordering, not return.
- Qsearch: replace the narrow pawn-only filter with a legal least-valuable-attacker
  exchange estimate on unfavorable captures. Skip only estimates below -100 cp.
  Checks, recaptures, promotions, en passant and advanced pawn captures are exempt.
  The estimator returns unknown (keeps the move) for checking exchange sequences,
  ambiguous equal-type recapturers, promotions, or exchanges exceeding 12 replies.
- Qsearch delta pruning: outside check/endgames/mate windows, skip unprotected
  non-checking captures when stand-pat + victim value + 250 cp is below alpha.
  This is a search margin, not a change to evaluation.
- Quiet promotions added to qsearch. Terminal mate/stalemate checks precede
  stand-pat. Draw checks cover qsearch too; mate scores use distance from root.
  Main search uses mate-distance bounds. Recursion is capped at 64 plies globally.
- Null move: only non-PV, non-root nodes with sufficient material, static score
  >= beta, and halfmove clock below 90; disabled in endgames/check/mate windows.
  No further null pruning or draw claims inside the synthetic null subtree;
  its TT entries have separate keys. Null changes no NNUE piece features.
- Skip impossible repetition/fifty-move claim work using the halfmove count.

NNUE weights, evaluate(), handcrafted fallback, and all time allocation formulas
are unchanged. Existing 512-node clock checks remain; exchange estimation and
aspiration retries additionally check the SAME hard deadline. Diagnostics remain
optional with the same fields. No harness files changed.

## User-run verification and benchmark

The assistant performed AST syntax/preservation checks and git diff --check ONLY.
The following regression checks have been written but NOT executed:

```powershell
Remove-Item Env:CHESS_SEARCH_STATS -ErrorAction SilentlyContinue
uv run python -m unittest discover -s tests -p test_search_regressions.py -v
uv run python -m harness.arena --opponent baselines/numba --games 100 2>&1 | Tee-Object combined-arena-100.log
```

Regression checks cover legal/pinned exchanges, en passant, quiet promotion,
terminal qsearch scores, shallow PVS versus a full-width reference, TT replacement,
and board/NNUE restoration after nested timeouts. They do not prove tactical
strength or exhaustively verify LMR/aspiration behavior. Existing repository-wide
lint/type errors were not repaired by changing evaluation or NNUE code.

Compare against the supplied +46 =40 -14, 66.0%, 0 flags. Reject reliability
regressions. A higher score is provisional evidence until repeated; a similar
score needs clear efficiency gains to justify keeping this larger change.
All heuristics interacting at once makes attribution harder.

For diagnostics, use the same two harness.play commands documented in
../SEARCH_STATS.md, with CHESS_SEARCH_STATS=1 and NEW log/PGN filenames prefixed
combined-. The regular arena hides diagnostics. Disable stats for strength tests.
Qsearch counts omit temporary exchange-estimator board moves (they do no NNUE
search); elapsed time includes that work. Consequently fewer nodes alone is not
proof of efficiency. Compare depth and elapsed time for matching FENs/clocks.

## Risks and rollback

This is an experiment, not a proven improvement. PVS/aspiration can spend time
re-searching; LMR/delta/SEE can miss sacrifices; legal SEE and terminal checks have
Python overhead. Broad changes could score below the prior engine. Search memory
is bounded, but wall-clock reliability must still be measured on your machine.

To revert the whole combined experiment while keeping diagnostics and the earlier
pawn-only filter (this overwrites agent.py; preserve any later edits first):

```powershell
Copy-Item experiments/search-before-combined.txt agent.py
```

## Reading used for ideas only

- Stockfish search architecture: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp
- Sunfish Python search organization: https://github.com/thomasahle/sunfish/blob/master/sunfish.py
- Competition documentation: https://aichessathon.com/docs

No engine source, tuned constants, evaluation terms, or model weights were copied
from these projects. The requested markdown rule URLs could not be fetched; the
public documentation page was accessible instead.
