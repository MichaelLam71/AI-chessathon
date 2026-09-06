"""Compare matched fixed-position reports, retaining categories and outcome errors."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from typing import Any

from position_benchmark import percentile


def paired_statistics(classical: dict[str, Any], nnue: dict[str, Any]) -> dict[str, Any]:
    if any(report.get("report_schema_version") != 2 for report in (classical, nnue)):
        raise ValueError("Comparison requires report schema 2 (separate ordinary cp and mates)")
    for field in ("agent_sha256", "nnue_eval_sha256", "weights_sha256", "labels_sha256",
                  "budget_ms", "time_left_ms", "mistake_cp", "blunder_cp", "mate_cp"):
        if classical[field] != nnue[field]:
            raise ValueError(f"Reports differ in {field}")
    if classical.get("worker_environment") != nnue.get("worker_environment"):
        raise ValueError("Worker environments differ")
    if classical["evaluator"] != "classical" or nnue["evaluator"] != "nnue":
        raise ValueError("Expected classical and nnue reports in that order")
    c_rows = {r["fen"]: r for r in classical["results"]}
    n_rows = {r["fen"]: r for r in nnue["results"]}
    if (c_rows.keys() != n_rows.keys() or len(c_rows) != len(classical["results"])
            or len(n_rows) != len(nnue["results"])):
        raise ValueError("Position sets differ or contain duplicates")
    groups: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    common_cp: list[tuple[float, float]] = []
    for fen, c in c_rows.items():
        n = n_rows[fen]
        exact_delta = float(n.get("best_agreement", False)) - float(c.get("best_agreement", False))
        cp_delta = None
        if c.get("ordinary_loss_cp") is not None and n.get("ordinary_loss_cp") is not None:
            common_cp.append((c["ordinary_loss_cp"], n["ordinary_loss_cp"]))
            cp_delta = c["ordinary_loss_cp"] - n["ordinary_loss_cp"]
        groups[c.get("game_id") or fen].append((exact_delta, cp_delta))
    rng = random.Random(19)
    keys = list(groups)
    exact_bootstrap: list[float] = []
    cp_bootstrap: list[float] = []
    for _ in range(2000):
        resampled = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        exact_bootstrap.append(statistics.mean(row[0] for row in resampled))
        values = [row[1] for row in resampled if row[1] is not None]
        if values:
            cp_bootstrap.append(statistics.mean(values))
    return {"common_cp_positions": len(common_cp),
            "common_cp_classical_mean": (statistics.mean(c for c, _ in common_cp)
                                         if common_cp else None),
            "common_cp_nnue_mean": statistics.mean(n for _, n in common_cp) if common_cp else None,
            "nnue_lower_cp_loss": sum(n < c for c, n in common_cp),
            "classical_lower_cp_loss": sum(c < n for c, n in common_cp),
            "equal_cp_loss": sum(c == n for c, n in common_cp),
            "game_clusters": len(groups), "bootstrap_replicates": 2000,
            "nnue_exact_agreement_advantage_ci95": [percentile(exact_bootstrap, p)
                                                    for p in (2.5, 97.5)],
            "nnue_cp_loss_reduction_ci95": [percentile(cp_bootstrap, p) for p in (2.5, 97.5)]}


def compare(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classical = json.loads(args.classical.read_text(encoding="utf-8"))
    nnue = json.loads(args.nnue.read_text(encoding="utf-8"))
    paired = paired_statistics(classical, nnue)
    unseen_paired = None
    if "no_local_training_overlap" in classical["by_tag"]:
        unseen_paired = paired_statistics(
            dict(classical, results=[r for r in classical["results"]
                                     if "no_local_training_overlap" in r["tags"]]),
            dict(nnue, results=[r for r in nnue["results"]
                                if "no_local_training_overlap" in r["tags"]]))
    lines = ["# Large fixed-position comparison", "",
             f"{classical['summary']['positions']} matched positions; "
             f"{classical['budget_ms']} ms wall ceiling, "
             f"{classical['time_left_ms']} ms remaining-clock input. "
             "Fresh processes and existing diagnostics enabled for both evaluators.", "",
             "Ordinary cp statistics and mistake/blunder counts exclude mate-involving "
             "comparisons and failed replies. Agreement includes all positions. "
             "Blunders are included in mistakes. Tags overlap the primary categories.", ""]
    groups = [("overall", classical["summary"], nnue["summary"])]
    for section in ("by_category", "by_tag"):
        for name in classical[section]:
            groups.append((name, classical[section][name], nnue[section][name]))
    sections = [
        ("Move quality", [("positions", "N"), ("cp_samples", "CP N"),
                          ("best_agreement", "Best %"), ("top3_agreement", "Top 3 %"),
                          ("mean_loss_cp", "Mean cp"), ("median_loss_cp", "Median cp"),
                          ("p90_loss_cp", "P90 cp"), ("p95_loss_cp", "P95 cp"),
                          ("large_mistakes", ">=100 cp"), ("blunders", ">=200 cp")]),
        ("Mate and estimated draw transitions", [("mate_involved", "Mate samples"),
             ("missed_winning_mate", "Missed winning mate"),
             ("allowed_losing_mate", "Allowed losing mate"),
             ("mate_distance_worsened", "Worse mate distance"),
             ("estimated_draw_to_loss", "Est. draw to loss"),
             ("estimated_win_to_draw", "Est. win to draw"), ("immediate_draw", "Immediate draws")]),
        ("Search and runtime", [("mean_completed_depth", "Depth"),
             ("mean_total_nodes", "Total nodes"), ("mean_completed_nodes", "Completed nodes"),
             ("runtime_seconds", "Seconds"), ("failures", "Failures")]),
    ]
    for title, fields in sections:
        lines.extend([f"## {title}", "", "| Group | Evaluator | " + " | ".join(
            name for _, name in fields) + " |", "|---|---|" + "---:|" * len(fields)])
        for name, c, n in groups:
            for evaluator, summary in (("classical", c), ("nnue", n)):
                cells = []
                for key, _ in fields:
                    value = summary[key]
                    if value is None:
                        cells.append("N/A")
                    elif "agreement" in key:
                        cells.append(f"{value * 100:.1f}")
                    elif isinstance(value, int):
                        cells.append(str(value))
                    else:
                        cells.append(f"{value:.2f}")
                lines.append(f"| {name} | {evaluator} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Matched ordinary-cp subset and uncertainty", "",
                  "This subset excludes any position involving mate for either evaluator, "
                  "so both cp means use exactly the same FENs. Confidence intervals resample "
                  "whole source games (2,000 replicates, seed 19); "
                  "positive differences favor NNUE. "
                  "They describe sampling uncertainty within this suite, not label error or "
                  "wall-clock run-to-run variation.", "", "```json", json.dumps(paired, indent=2),
                  "```", "", "### Excluding local training overlap", "", "```json",
                  json.dumps(unseen_paired, indent=2), "```", "",
                  "No-local-overlap means no identical piece-placement and side-to-move "
                  "input in the scanned local training chunks. It does not rule out "
                  "nearby positions or other training sources.", "",
                  "## Interpretation limits", "",
                  "Draw-to-loss means best-move Stockfish WDL draw probability >=90% and "
                  "chosen-move loss probability >=50%. Win-to-draw means best-move win "
                  "probability >=50% and chosen-move draw probability >=90%. These are "
                  "model estimates, not tablebase or repetition proofs. "
                  "FENs omit game history.", "",
                  "Mate events use Stockfish's finite-search mate reports. Missing a mate "
                  "does not necessarily lose the game. Mate-distance changes are counted "
                  "separately from abandoning or allowing mate. A transition can overlap "
                  "multiple outcome counters.", "",
                  "Most positions come from newly generated low-node Stockfish self-play "
                  "with varied openings. Categories use board-based heuristics; tactical "
                  "does not mean a verified unique solution. This is a broader diagnostic "
                  "holdout, not an Elo estimate or proof of competition strength.", ""])
    with args.output.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    with args.output.with_suffix(".json").open("x", encoding="utf-8") as output:
        json.dump({"paired": paired, "paired_no_local_training_overlap": unseen_paired,
                   "classical": classical["summary"],
                   "nnue": nnue["summary"]}, output, indent=2)
    print(json.dumps(paired, indent=2))
    print(f"Saved {args.output}")
