"""Local fixed-position labels and tests; never imported by the submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
MATE_CP = 100_000
ACTIVE_NNUE_WEIGHTS = "nnue_weights_cp_simple_public.npz"
WORKER_ENV = {"CHESS_SEARCH_STATS": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def board_for(fen: str) -> chess.Board:
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError(f"Expected a valid, nonterminal FEN: {fen}")
    return board


def stockfish_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise ValueError(f"Stockfish executable not found: {path}")
        return path
    candidates = sorted((ROOT / "stockfish").rglob("*.exe"))
    if os.name != "nt":
        candidates = [p for p in (ROOT / "stockfish").rglob("stockfish*")
                      if p.is_file() and os.access(p, os.X_OK) and not p.suffix]
    if len(candidates) != 1:
        raise ValueError("Expected one executable under stockfish/; specify --stockfish PATH")
    return candidates[0].resolve()


def label(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.positions.read_text(encoding="utf-8-sig")
    if args.positions.suffix == ".json":
        entries = json.loads(raw)["positions"]
    else:
        fens = [line.split("#", 1)[0].strip() for line in raw.splitlines()]
        entries = [{"fen": fen} for fen in dict.fromkeys(f for f in fens if f)]
    if not entries:
        raise ValueError("No FENs supplied")
    boards = [board_for(entry["fen"]) for entry in entries]
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    executable = stockfish_path(args.stockfish)
    limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(
        nodes=args.nodes)
    settings = {"positions_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "stockfish_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "limit": {"depth": args.depth} if args.depth else {"nodes": args.nodes},
                "threads": 1, "hash_mb": args.hash_mb, "multipv": "all legal moves"}
    checkpoint = args.output.with_suffix(".checkpoint.jsonl")
    records: list[dict[str, Any]] = []
    if checkpoint.exists():
        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        if json.loads(lines[0]) != settings:
            raise ValueError("Checkpoint settings differ; choose a new output name")
        records = [json.loads(line) for line in lines[1:]]
        if [r["fen"] for r in records] != [e["fen"] for e in entries[:len(records)]]:
            raise ValueError("Checkpoint positions differ")
    else:
        with checkpoint.open("x", encoding="utf-8") as output:
            output.write(json.dumps(settings) + "\n")
    started = time.perf_counter()
    with (checkpoint.open("a", encoding="utf-8") as journal,
          chess.engine.SimpleEngine.popen_uci(str(executable)) as engine):
        engine.configure({"Threads": 1, "Hash": args.hash_mb, "UCI_ShowWDL": True})
        for index in range(len(records), len(entries)):
            board = boards[index]
            position_started = time.perf_counter()
            infos = engine.analyse(board, limit, multipv=board.legal_moves.count(), game=object())
            moves: list[dict[str, Any]] = []
            for info in infos:
                score = info["score"].pov(board.turn)
                wdl = info["wdl"].pov(board.turn) if "wdl" in info else None
                moves.append({"move": info["pv"][0].uci(), "cp": score.score(),
                              "mate": score.mate(), "score_cp": score.score(mate_score=MATE_CP),
                              "wdl": list(wdl) if wdl is not None else None,
                              "depth": info.get("depth"), "nodes": info.get("nodes"),
                              "pv": [move.uci() for move in info["pv"]]})
            if {m["move"] for m in moves} != {m.uci() for m in board.legal_moves}:
                raise ValueError(f"Incomplete Stockfish labels at position {index + 1}")
            moves.sort(key=lambda item: item["score_cp"], reverse=True)
            tags = list(entries[index].get("tags", []))
            best_cp = moves[0]["cp"]
            if best_cp is not None and -600 <= best_cp <= -50:
                tags.append("defensive")
            record = dict(entries[index], best_move=moves[0]["move"],
                          top3=[m["move"] for m in moves[:3]], moves=moves,
                          tags=sorted(set(tags)),
                          label_seconds=time.perf_counter() - position_started)
            records.append(record)
            journal.write(json.dumps(record) + "\n")
            journal.flush()
            print(f"Labelled {index + 1}/{len(boards)}: {moves[0]['move']}", flush=True)
        document = dict(settings, schema_version=1, stockfish=engine.id,
                        score_pov="side to move", mate_cp=MATE_CP,
                        runtime_seconds=sum(r.get("label_seconds", 0) for r in records),
                        current_session_seconds=time.perf_counter() - started, positions=records)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def read_lines(stream: TextIO, messages: queue.Queue[str]) -> None:
    for line in stream:
        messages.put(line)
    messages.put("")


def sample(fen: str, args: argparse.Namespace, agent_root: Path = ROOT) -> dict[str, Any]:
    sample_started = time.perf_counter()
    result: dict[str, Any] = {"fen": fen, "status": "error"}
    env = dict(os.environ, **WORKER_ENV)
    if agent_root != ROOT:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
    command = [sys.executable, "-m", "harness.runner", str(agent_root)]
    if args.evaluator != "current":
        bootstrap = (
            "import contextlib, runpy, sys\n"
            "with contextlib.redirect_stdout(sys.stderr):\n"
            "    import agent\n"
            "    agent.USE_LEARNED_EVAL = sys.argv[1] == 'nnue'\n"
            "sys.argv = ['harness.runner', sys.argv[2]]\n"
            "runpy.run_module('harness.runner', run_name='__main__')\n"
        )
        command = [sys.executable, "-c", bootstrap, args.evaluator, str(agent_root)]
    # Use the existing runner's import and get_move protocol, including fd redirection.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command, cwd=agent_root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True,
        )
        assert process.stdout is not None and process.stdin is not None
        messages: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(target=read_lines, args=(process.stdout, messages), daemon=True)
        reader.start()
        phase = "import"
        try:
            if json.loads(messages.get(timeout=args.import_timeout))["ready"] is not True:
                raise ValueError("Runner did not become ready")
            phase = "move"
            started = time.perf_counter()
            process.stdin.write(json.dumps({"fen": fen, "time_left_ms": args.time_left_ms}) + "\n")
            process.stdin.flush()
            response = messages.get(timeout=args.budget_ms / 1000)
            elapsed = (time.perf_counter() - started) * 1000
            result["elapsed_ms"] = elapsed
            move = json.loads(response)["move"]
            if elapsed > args.budget_ms:
                result["status"] = "timeout"
            elif (not isinstance(move, str)
                  or chess.Move.from_uci(move) not in board_for(fen).legal_moves):
                result.update(status="illegal", move=move)
            else:
                result.update(status="ok", move=move)
        except queue.Empty:
            result.update(status=f"{phase}_timeout")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["error"] = str(exc)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join(timeout=2)
            process.stdin.close()
            process.stdout.close()
        errors.seek(0)
        stderr = errors.read()
        for line in stderr.splitlines():
            if line.startswith("SEARCH_STATS "):
                result["diagnostics"] = json.loads(line.removeprefix("SEARCH_STATS "))
        if result["status"] != "ok":
            result["stderr"] = stderr[-4000:]
    result["runtime_seconds"] = time.perf_counter() - sample_started
    return result


def score_result(position: dict[str, Any], result: dict[str, Any]) -> None:
    """Separate finite cp comparisons from mate outcomes and WDL draw estimates."""
    moves = {m["move"]: m for m in position["moves"]}
    best = moves[position["best_move"]]
    chosen = moves[result["move"]]
    best_mate, chosen_mate = best["mate"], chosen["mate"]
    cp_loss = (max(0, best["cp"] - chosen["cp"])
               if best["cp"] is not None and chosen["cp"] is not None else None)
    best_wdl, chosen_wdl = best.get("wdl"), chosen.get("wdl")
    result.update(
        best_agreement=result["move"] == position["best_move"],
        top3_agreement=result["move"] in position["top3"],
        loss_cp=max(0, best["score_cp"] - chosen["score_cp"]),
        ordinary_loss_cp=cp_loss, best_cp=best["cp"], chosen_cp=chosen["cp"],
        best_mate=best_mate, chosen_mate=chosen_mate,
        mate_involved=best_mate is not None or chosen_mate is not None,
        missed_winning_mate=(best_mate is not None and best_mate > 0
                             and (chosen_mate is None or chosen_mate <= 0)),
        allowed_losing_mate=(chosen_mate is not None and chosen_mate < 0
                             and (best_mate is None or best_mate >= 0)),
        mate_distance_worsened=(best_mate is not None and chosen_mate is not None
                                and best_mate * chosen_mate > 0
                                and chosen["score_cp"] < best["score_cp"]),
        estimated_draw_to_loss=bool(best_wdl and chosen_wdl
                                   and best_wdl[1] >= 900 and chosen_wdl[2] >= 500),
        estimated_win_to_draw=bool(best_wdl and chosen_wdl
                                  and best_wdl[0] >= 500 and chosen_wdl[1] >= 900),
    )
    board = chess.Board(position["fen"])
    board.push_uci(result["move"])
    outcome = board.outcome()
    result["immediate_draw"] = outcome is not None and outcome.winner is None


def percentile(values: list[float], percent: float) -> float | None:
    """Linear interpolation between sorted samples, including endpoints."""
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * percent / 100
    lower = int(offset)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (offset - lower)


def summarize(results: list[dict[str, Any]], mistake_cp: int,
              blunder_cp: int) -> dict[str, Any]:
    valid = [r for r in results if r["status"] == "ok"]
    losses = [r["ordinary_loss_cp"] for r in valid if r["ordinary_loss_cp"] is not None]
    diagnostics = [r["diagnostics"] for r in valid if "diagnostics" in r]
    summary: dict[str, Any] = {
        "positions": len(results), "valid": len(valid), "failures": len(results) - len(valid),
        "best_agreement": sum(r["best_agreement"] for r in valid) / len(results),
        "top3_agreement": sum(r["top3_agreement"] for r in valid) / len(results),
        "cp_samples": len(losses),
        "mean_loss_cp": statistics.mean(losses) if losses else None,
        "median_loss_cp": statistics.median(losses) if losses else None,
        "p90_loss_cp": percentile(losses, 90), "p95_loss_cp": percentile(losses, 95),
        "large_mistakes": sum(loss >= mistake_cp for loss in losses),
        "blunders": sum(loss >= blunder_cp for loss in losses),
        "synthetic_mean_loss_cp": statistics.mean(r["loss_cp"] for r in valid) if valid else None,
        "diagnostic_samples": len(diagnostics),
        "runtime_seconds": sum(r.get("runtime_seconds", 0) for r in results),
    }
    for field in ("mate_involved", "missed_winning_mate", "allowed_losing_mate",
                  "mate_distance_worsened", "estimated_draw_to_loss",
                  "estimated_win_to_draw", "immediate_draw"):
        summary[field] = sum(r[field] for r in valid)
    for field in ("completed_depth", "completed_nodes", "total_nodes"):
        values = [d[field] for d in diagnostics if field in d]
        summary[f"mean_{field}"] = statistics.mean(values) if values else None
    return summary


def test(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    document = json.loads(args.labels.read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1 or document.get("score_pov") != "side to move":
        raise ValueError("Unsupported label schema or score perspective")
    positions = document["positions"]
    if not positions:
        raise ValueError("Empty label set")
    for position in positions:
        board = board_for(position["fen"])
        labels = position["moves"]
        if len(labels) != board.legal_moves.count() or {m["move"] for m in labels} != {
            move.uci() for move in board.legal_moves
        }:
            raise ValueError("Labels must cover every legal move exactly once; rerun label")
        if any(not isinstance(m["score_cp"], (int, float)) for m in labels):
            raise ValueError("Missing numerical score")
    weights_path = ((args.nnue_weights or ROOT / "weights" / ACTIVE_NNUE_WEIGHTS)
                    .resolve())
    if not weights_path.is_file():
        raise ValueError(f"NNUE weights not found: {weights_path}")
    if args.nnue_weights is not None and args.evaluator != "nnue":
        raise ValueError("--nnue-weights requires --evaluator nnue")
    results: list[dict[str, Any]] = []
    # An alternate model is staged under the filename expected by nnue_eval.py. Source and
    # incumbent weights remain untouched, and every child imports this isolated copy.
    staging = tempfile.TemporaryDirectory() if args.nnue_weights is not None else None
    agent_root = ROOT
    if staging is not None:
        agent_root = Path(staging.name)
        (agent_root / "weights").mkdir()
        shutil.copy2(ROOT / "agent.py", agent_root / "agent.py")
        shutil.copy2(ROOT / "nnue_eval.py", agent_root / "nnue_eval.py")
        shutil.copy2(weights_path, agent_root / "weights" / ACTIVE_NNUE_WEIGHTS)
    # Reserve output before doing any searches.
    with args.output.open("x", encoding="utf-8") as output:
        for index, position in enumerate(positions, 1):
            result = sample(position["fen"], args, agent_root)
            result.update(category=position.get("category", "uncategorized"),
                          tags=position.get("tags", []), id=position.get("id", str(index)),
                          game_id=position.get("game_id"))
            if result["status"] == "ok":
                score_result(position, result)
            results.append(result)
            print(f"Tested {index}/{len(positions)}: {result.get('move', '-')} "
                  f"{result['status']}, loss {result.get('loss_cp', 'N/A')} cp", flush=True)
        summary = summarize(results, args.mistake_cp, args.blunder_cp)
        summary["runtime_seconds"] = time.perf_counter() - started
        categories = {name: summarize([r for r in results if r["category"] == name],
                                      args.mistake_cp, args.blunder_cp)
                      for name in sorted({r["category"] for r in results})}
        tags = {name: summarize([r for r in results if name in r["tags"]],
                                args.mistake_cp, args.blunder_cp)
                for name in sorted({tag for r in results for tag in r["tags"]})}
        report = {"report_schema_version": 2, "summary": summary,
                  "worker_environment": WORKER_ENV,
                  "by_category": categories, "by_tag": tags,
                  "evaluator": args.evaluator, "budget_ms": args.budget_ms,
                  "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                  "time_left_ms": args.time_left_ms, "mistake_cp": args.mistake_cp,
                  "blunder_cp": args.blunder_cp, "mate_cp": document["mate_cp"],
                  "agent_sha256": hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest(),
                  "nnue_eval_sha256": hashlib.sha256(
                      (ROOT / "nnue_eval.py").read_bytes()).hexdigest(),
                  "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                  "results": results}
        json.dump(report, output, indent=2)
        output.write("\n")
    if staging is not None:
        staging.cleanup()
    print(json.dumps(summary, indent=2))
    print("Ordinary cp losses exclude mate scores and failures; mate events are separate.")
    print("Agreement includes failures. Draw transitions are WDL estimates, not proven outcomes.")
    print(f"Saved {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Build a diverse suite from saved and new games")
    builder.add_argument("--generated-pgn", type=Path,
                         default=ROOT / "benchmarks" / "benchmark-holdout.pgn")
    builder.add_argument("--games", type=positive, default=100)
    builder.add_argument("--seed", type=int, default=19)
    builder.add_argument("--stockfish", type=Path)
    builder.add_argument("--output", type=Path, default=Path("benchmarks/positions-large.json"))
    comparison = commands.add_parser("compare", help="Compare evaluator reports by category")
    comparison.add_argument("classical", type=Path)
    comparison.add_argument("nnue", type=Path)
    comparison.add_argument("--output", type=Path,
                            default=Path("benchmarks/results/position-large-results.md"))
    label_parser = commands.add_parser("label", help="Create reusable Stockfish labels")
    label_parser.add_argument("positions", type=Path)
    label_parser.add_argument("output", type=Path)
    label_parser.add_argument("--stockfish", type=Path)
    limits = label_parser.add_mutually_exclusive_group()
    limits.add_argument("--nodes", type=positive, default=10_000_000)
    limits.add_argument("--depth", type=positive)
    label_parser.add_argument("--hash-mb", type=positive, default=128)
    test_parser = commands.add_parser("test", help="Test agent.py without Stockfish")
    test_parser.add_argument("labels", type=Path)
    test_parser.add_argument("--evaluator", choices=("current", "classical", "nnue"),
                             default="current")
    test_parser.add_argument("--nnue-weights", type=Path)
    test_parser.add_argument("--budget-ms", type=positive, default=1000)
    test_parser.add_argument("--time-left-ms", type=positive, default=10000)
    test_parser.add_argument("--import-timeout", type=positive, default=60)
    test_parser.add_argument("--mistake-cp", type=positive, default=100)
    test_parser.add_argument("--blunder-cp", type=positive, default=200)
    test_parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/position-results.json")
    )
    args = parser.parse_args()
    if args.command == "compare":
        from compare_position_results import compare

        compare(args)
        return
    if args.command == "build":
        from build_position_suite import build

        build(args)
        return
    if args.command == "test" and args.blunder_cp < args.mistake_cp:
        parser.error("--blunder-cp must be >= --mistake-cp")
    try:
        (label if args.command == "label" else test)(args)
    except (ValueError, OSError, chess.engine.EngineError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
