"""User-run, fresh-process evaluator comparison. Never imported by the agent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FENS = ROOT / "benchmarks" / "positions.txt"


def worker(evaluator: str, fen: str, clock: int) -> None:
    # Set before import: stats-only imports are conditional in agent.py.
    os.environ["CHESS_SEARCH_STATS"] = "1"
    sys.path.insert(0, str(ROOT))
    import chess
    import agent

    board = chess.Board(fen)
    if not board.is_valid() or not any(board.legal_moves):
        raise ValueError("Expected a valid, nonterminal FEN")
    agent.USE_LEARNED_EVAL = evaluator == "nnue"
    move = agent.get_move(fen, clock)
    if chess.Move.from_uci(move) not in board.legal_moves:
        raise ValueError(f"Illegal result: {move}")


def run(arguments: argparse.Namespace) -> None:
    fens = [line.strip() for line in arguments.fens.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not fens:
        raise ValueError("No FENs supplied")
    if arguments.time_left_ms <= 0 or arguments.repeat <= 0:
        raise ValueError("Clock and repeat count must be positive")
    digest = hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest()
    env = dict(os.environ, CHESS_SEARCH_STATS="1", PYTHONHASHSEED="0")
    # Refuse to overwrite evidence from a previous run.
    with arguments.out.open("x", encoding="utf-8") as output:
        for repeat in range(1, arguments.repeat + 1):
            for position, fen in enumerate(fens, 1):
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "_worker",
                     arguments.evaluator, fen, str(arguments.time_left_ms)],
                    cwd=ROOT, env=env, capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(f"Position {position} failed:\n{result.stderr}")
                lines = [line[len("SEARCH_STATS "):] for line in result.stderr.splitlines()
                         if line.startswith("SEARCH_STATS ")]
                if len(lines) != 1:
                    raise RuntimeError(f"Expected one stats record, received {len(lines)}")
                record = json.loads(lines[0])
                record.update(position=position, repeat=repeat, agent_sha256=digest)
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(f"{arguments.evaluator} {position}/{len(fens)} repeat {repeat}: "
                      f"depth {record['completed_depth']}, "
                      f"{record['nodes_per_second']:.0f} nodes/s", flush=True)


def read_log(path: Path, evaluator: str) -> dict:
    records = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["evaluation"] != evaluator:
            raise ValueError(f"{path} contains a different evaluator")
        key = (record["position"], record["repeat"])
        if key in records:
            raise ValueError(f"Duplicate sample in {path}: {key}")
        records[key] = record
    if not records:
        raise ValueError(f"Empty log: {path}")
    return records


def compare(arguments: argparse.Namespace) -> None:
    classical = read_log(arguments.classical, "classical")
    nnue = read_log(arguments.nnue, "nnue")
    if classical.keys() != nnue.keys():
        raise ValueError("Logs do not contain the same samples (possibly an incomplete run)")
    print("pos rep | depth C/N | nodes C/N | main C/N | q C/N | ms C/N | NPS C/N | timeout C/N")
    for key in sorted(classical):
        c, n = classical[key], nnue[key]
        for field in ("fen", "time_left_ms", "agent_sha256"):
            if c[field] != n[field]:
                raise ValueError(f"Sample {key} differs in {field}")
        pairs = []
        for field in ("completed_depth", "total_nodes", "main_nodes", "q_nodes",
                      "elapsed_ms", "nodes_per_second", "timed_out"):
            pairs.append(f"{c[field]:.0f}/{n[field]:.0f}")
        print(f"{key[0]:3} {key[1]:3} | " + " | ".join(pairs))
    print("\nAggregate throughput = sum(nodes) / sum(search time); C=classical, N=NNUE.")
    for label, records in (("classical", classical), ("nnue", nnue)):
        values = list(records.values())
        nodes = sum(r["total_nodes"] for r in values)
        elapsed = sum(r["elapsed_ms"] for r in values)
        qnodes = sum(r["q_nodes"] for r in values)
        nps = 1000 * nodes / elapsed if elapsed else 0
        share = qnodes / nodes if nodes else 0
        depths = sum(r["completed_depth"] for r in values) / len(values)
        timeouts = sum(r["timed_out"] for r in values)
        print(f"{label}: {nps:.0f} nodes/s, mean depth {depths:.2f}, "
              f"q share {share:.1%}, internal timeouts {timeouts}/{len(values)}")
    c_deeper = sum(classical[k]["completed_depth"] > nnue[k]["completed_depth"] for k in classical)
    n_deeper = sum(classical[k]["completed_depth"] < nnue[k]["completed_depth"] for k in classical)
    print(f"Deeper: classical {c_deeper}, NNUE {n_deeper}, equal {len(classical)-c_deeper-n_deeper}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--evaluator", choices=("classical", "nnue"), required=True)
    runner.add_argument("--fens", type=Path, default=DEFAULT_FENS)
    runner.add_argument("--time-left-ms", type=int, default=10000)
    runner.add_argument("--repeat", type=int, default=1)
    runner.add_argument("--out", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--classical", type=Path, required=True)
    comparison.add_argument("--nnue", type=Path, required=True)
    child = commands.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("evaluator", choices=("classical", "nnue"))
    child.add_argument("fen")
    child.add_argument("clock", type=int)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments)
    elif arguments.command == "compare":
        compare(arguments)
    else:
        worker(arguments.evaluator, arguments.fen, arguments.clock)


if __name__ == "__main__":
    main()
