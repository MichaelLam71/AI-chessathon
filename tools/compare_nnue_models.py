"""Compare two matched NNUE fixed-position reports with different weights."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from position_benchmark import percentile

MATCH_FIELDS = (
    "agent_sha256", "nnue_eval_sha256", "labels_sha256", "budget_ms", "time_left_ms",
    "mistake_cp", "blunder_cp", "mate_cp", "worker_environment",
)
QUALITY_FIELDS = (
    ("positions", "Positions"), ("valid", "Valid"),
    ("best_agreement", "Best %"), ("top3_agreement", "Top-3 %"),
    ("mean_loss_cp", "Mean cp"), ("median_loss_cp", "Median cp"),
    ("p90_loss_cp", "P90 cp"), ("p95_loss_cp", "P95 cp"),
    ("large_mistakes", ">=100 cp"), ("blunders", ">=200 cp"),
)
OUTCOME_FIELDS = (
    ("missed_winning_mate", "Missed win mate"),
    ("allowed_losing_mate", "Allowed loss mate"),
    ("mate_distance_worsened", "Worse mate distance"),
    ("estimated_draw_to_loss", "Est. draw-to-loss"),
    ("estimated_win_to_draw", "Est. win-to-draw"),
    ("immediate_draw", "Immediate draws"),
)
SEARCH_FIELDS = (
    ("mean_completed_depth", "Mean depth"),
    ("mean_completed_nodes", "Completed nodes"),
    ("mean_total_nodes", "Total nodes"),
    ("runtime_seconds", "Runtime seconds"), ("failures", "Failures"),
)


def verify(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    if any(report.get("report_schema_version") != 2 for report in (baseline, candidate)):
        raise ValueError("Comparison requires report schema 2")
    if baseline.get("evaluator") != "nnue" or candidate.get("evaluator") != "nnue":
        raise ValueError("Both reports must use --evaluator nnue")
    for field in MATCH_FIELDS:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Reports differ in {field}")
    if baseline.get("weights_sha256") == candidate.get("weights_sha256"):
        raise ValueError("Reports use identical NNUE weights")


def paired_statistics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {row["fen"]: row for row in baseline["results"]}
    candidate_rows = {row["fen"]: row for row in candidate["results"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("Reports contain different position sets")
    groups: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    common: list[tuple[float, float]] = []
    for fen, base in baseline_rows.items():
        other = candidate_rows[fen]
        exact_delta = float(other.get("best_agreement", False)) - float(
            base.get("best_agreement", False)
        )
        loss_delta = None
        if base.get("ordinary_loss_cp") is not None and other.get("ordinary_loss_cp") is not None:
            common.append((base["ordinary_loss_cp"], other["ordinary_loss_cp"]))
            loss_delta = base["ordinary_loss_cp"] - other["ordinary_loss_cp"]
        groups[base.get("game_id") or fen].append((exact_delta, loss_delta))
    rng = random.Random(19)
    exact_bootstrap: list[float] = []
    loss_bootstrap: list[float] = []
    group_names = list(groups)
    for _ in range(2000):
        rows = [
            row for name in rng.choices(group_names, k=len(group_names)) for row in groups[name]
        ]
        exact_bootstrap.append(statistics.mean(row[0] for row in rows))
        losses = [row[1] for row in rows if row[1] is not None]
        if losses:
            loss_bootstrap.append(statistics.mean(losses))
    return {
        "common_cp_positions": len(common),
        "baseline_mean_loss_cp": statistics.mean(row[0] for row in common),
        "candidate_mean_loss_cp": statistics.mean(row[1] for row in common),
        "candidate_lower_cp_loss": sum(other < base for base, other in common),
        "baseline_lower_cp_loss": sum(base < other for base, other in common),
        "equal_cp_loss": sum(base == other for base, other in common),
        "game_clusters": len(groups), "bootstrap_replicates": 2000,
        "candidate_exact_agreement_advantage_ci95": [
            percentile(exact_bootstrap, percent) for percent in (2.5, 97.5)
        ],
        "candidate_cp_loss_reduction_ci95": [
            percentile(loss_bootstrap, percent) for percent in (2.5, 97.5)
        ],
    }


def display(value: Any, key: str) -> str:
    if value is None:
        return "N/A"
    if "agreement" in key:
        return f"{value * 100:.2f}"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def add_table(lines: list[str], title: str,
              groups: list[tuple[str, dict[str, Any], dict[str, Any]]],
              fields: tuple[tuple[str, str], ...]) -> None:
    lines.extend([
        f"## {title}", "",
        "| Group | Model | " + " | ".join(label for _, label in fields) + " |",
        "|---|---|" + "---:|" * len(fields),
    ])
    for group, baseline, candidate in groups:
        for model, summary in (("baseline", baseline), ("candidate", candidate)):
            cells = [display(summary.get(key), key) for key, _ in fields]
            lines.append(f"| {group} | {model} | " + " | ".join(cells) + " |")
    lines.append("")


def compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    verify(baseline, candidate)
    paired = paired_statistics(baseline, candidate)
    groups = [("overall", baseline["summary"], candidate["summary"])]
    for name in sorted(baseline["by_category"]):
        groups.append((name, baseline["by_category"][name], candidate["by_category"][name]))
    lines = [
        "# NNUE model comparison", "", f"Baseline: `{args.baseline}`  ",
        f"Candidate: `{args.candidate}`", "",
        "Lower centipawn losses, mistake counts, failures, and runtime are better. "
        "Higher agreement and depth are better.", "",
    ]
    add_table(lines, "Move quality by category", groups, QUALITY_FIELDS)
    add_table(lines, "Mate and draw-changing outcomes", groups[:1], OUTCOME_FIELDS)
    add_table(lines, "Search and runtime", groups, SEARCH_FIELDS)
    lines.extend([
        "## Paired uncertainty", "", "```json", json.dumps(paired, indent=2), "```", "",
        "Positive confidence-interval values favor the candidate. Intervals resample source "
        "games and measure suite sampling uncertainty, not engine-label or timing uncertainty.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    companion = args.output.with_suffix(".json")
    with companion.open("x", encoding="utf-8") as output:
        json.dump(
            {"paired": paired, "baseline": baseline["summary"],
             "candidate": candidate["summary"]},
            output,
            indent=2,
        )
        output.write("\n")
    print(json.dumps(paired, indent=2))
    print(f"Saved {args.output} and {companion}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/results/nnue-model-comparison.md"))
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
