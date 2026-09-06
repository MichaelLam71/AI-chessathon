# Search diagnostics

> The current agent includes the combined search experiment described in
> [experiments/combined-search.md](experiments/combined-search.md). The instrumentation
> itself remains unchanged; the paragraph below describes its original addition.

Instrumentation is disabled by default. Only agent.py contains runtime changes.
No search windows, depths, ordering, evaluation, pruning, TT policy, or time
allocations were changed. Disabled mode adds boolean checks but no diagnostic
counter updates or output. Enabled mode costs time and can affect completed depth
and moves under a wall clock; use it for diagnosis, not strength comparisons.

## Enable and collect

In PowerShell, from the repository root:

```powershell
$env:CHESS_SEARCH_STATS = '1'
uv run python -m harness.play --white . --black baselines/numba --base-ms 10000 --increment-ms 100 --pgn stats-white.pgn 2>&1 | Tee-Object stats-white.log
uv run python -m harness.play --white baselines/numba --black . --base-ms 10000 --increment-ms 100 --pgn stats-black.pgn 2>&1 | Tee-Object stats-black.log
```

These time controls match the existing arena defaults used for the supplied
baseline. Start with these two games, then repeat if the samples disagree.
The harness displays the captured stderr after each game, not live per move.
Each agent move emits one `SEARCH_STATS ` line followed by a JSON object.
The FEN, move, and remaining clock identify the position in each sample.
Normal harness.arena does not display these stderr records.

For more diverse samples, use harness.play's --fen argument with saved middlegame
and endgame positions, playing both colours with identical time settings. Keep the
same positions, clocks, machine load, and instrumentation settings across runs.
Retain the PGNs and logs. Full games preserve the normal between-move TT state.
Avoid drawing conclusions from a starting-position-only sample or one aggregate
that mixes very different completed depths and remaining clock times.

Disable before normal strength benchmarking (new processes read this at import):

```powershell
Remove-Item Env:CHESS_SEARCH_STATS -ErrorAction SilentlyContinue
uv run python -m harness.arena --opponent baselines/numba --games 100
```

Setting the variable to '0' also disables it. No source edit is needed.

## Read the output

Parse the per-move records in PowerShell:

```powershell
$records = @(Get-Content stats-white.log, stats-black.log |
    Where-Object { $_.StartsWith('SEARCH_STATS ') } |
    ForEach-Object { $_.Substring(13) | ConvertFrom-Json })
$records | Select-Object move, completed_depth, total_nodes, q_nodes, tt_probes, tt_hits, tt_cutoffs, beta_cutoffs, beta_move_1, beta_move_2, beta_move_3, beta_move_4plus, null_attempts, null_cutoffs, timed_out | Format-Table
```

Counters reset per get_move and cover ALL iterative-deepening iterations,
including an interrupted last iteration and null-move subtrees. They count visits,
not distinct board positions. A node that triggers the time check is included.

- raw_node_calls: original node_count, unchanged for time-check scheduling.
- q_entries: alpha-beta depth-zero calls that enter quiescence.
- q_nodes: all quiescence calls, including its initial entries.
- total_nodes: raw_node_calls - q_entries, avoiding double-counting the
  alpha-beta-to-quiescence transition.
- main_nodes: total_nodes - q_nodes (includes early draw and TT returns).
- completed_depth: last fully returned iterative-deepening depth; 0 if none.
- completed_nodes: cumulative total_nodes at that depth's completion, including
  all earlier iterations. total_nodes - completed_nodes measures work in the
  unfinished iteration, if any.
- timed_out: the search caught its internal TimeoutError, NOT a game flag.
- elapsed_ms: elapsed since the existing search clock started; excludes board/NNUE
  initialization and JSON formatting/output, so it is not full move wall time.
- tt_probes: actual dictionary lookups; depth-zero and early-draw nodes do not probe.
- tt_hits: any existing entry, including ones too shallow for a return.
- tt_depth_hits: hits with stored depth >= requested depth.
- tt_cutoffs: all immediate TT returns, including EXACT returns.
  Equals tt_exact_returns + tt_lower_cutoffs + tt_upper_cutoffs.
- beta_cutoffs: normal alpha-beta legal-move-loop cutoffs only.
- beta_move_1/2/3/4plus: one-based position in the actual sorted move list for
  each such cutoff. The four counters sum to beta_cutoffs.
- q_beta_cutoffs and q_beta_move_*: separate quiescence move-loop equivalents.
- q_stand_pat_cutoffs: quiescence returns before searching moves because its
  static score already meets beta. No move index applies.
- null_attempts/null_cutoffs: existing null-move searches started / successful
  pruning returns. An interrupted attempt counts only as an attempt.

## Interpret before changing search

Compute ratios from summed counts within comparable groups, rather than averaging
per-move percentages. Treat zero denominators as unavailable, not zero percent.

| Diagnostic | Calculation | What to investigate |
|---|---|---|
| First-move cutoff share | beta_move_1 / beta_cutoffs | Low relative to comparable samples, with many late cutoffs, suggests ordering is leaving refutations late. |
| Late cutoff share | beta_move_4plus / beta_cutoffs | A large share means at least three earlier moves were searched before those cutoffs. Inspect the corresponding positions. |
| TT hit rate | tt_hits / tt_probes | Low reuse can suggest limited transpositions or retention issues; it alone does not diagnose a replacement bug. |
| Depth-usable hit share | tt_depth_hits / tt_hits | Low values mean many hits are too shallow for returns. Such entries can still help ordering during iterative deepening. |
| TT return rate | tt_cutoffs / tt_probes | Low despite many depth-usable hits suggests stored bounds seldom satisfy the current window. |
| Quiescence share | q_nodes / total_nodes | A dominant or growing share alongside stalled depth warrants inspecting qsearch work. High share alone is not proof of waste. |
| Qsearch expansion | q_nodes / q_entries | Average calls per qsearch entry; rising values locate expanding tactical subtrees. |
| Null success rate | null_cutoffs / null_attempts | Low success warrants examining the cost of failed attempts before considering any change. |

Use the separate qsearch cutoff histogram for capture ordering; mixing it with
main-search cutoffs would obscure both. Stand-pat cutoffs are cheap early exits,
so many q_nodes do not imply the same cost as many expanded nodes. These are work
counts, not a time profiler. There are no universal pass/fail percentages.

Current validation: text diff reviewed and git diff --check passed. The engine,
benchmarks, and tests have not been run by the assistant, as requested.
