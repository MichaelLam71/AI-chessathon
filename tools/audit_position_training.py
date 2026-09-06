"""Check board-and-side feature overlap with local NNUE training chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def annotate(args: argparse.Namespace) -> None:
    report = json.loads(args.output.read_text(encoding="utf-8"))
    labels = json.loads(args.annotate_labels.read_text(encoding="utf-8"))
    raw_digest = hashlib.sha256(args.positions.read_bytes()).hexdigest()
    text_digest = hashlib.sha256(
        args.positions.read_text(encoding="utf-8-sig").encode()).hexdigest()
    if (labels["positions_sha256"] != text_digest
            or report["positions_sha256"] != raw_digest):
        raise ValueError("Audit and labels have different position sources")
    audited = {r["fen"]: r for r in report["positions"]}
    if set(audited) != {r["fen"] for r in labels["positions"]}:
        raise ValueError("Audit and label position sets differ")
    for row in labels["positions"]:
        tag = ("training_overlap" if audited[row["fen"]]["training_overlap"]
               else "no_local_training_overlap")
        row["tags"] = sorted(set(row.get("tags", [])) | {tag})
    labels["training_audit"] = {"path": str(args.output), "sha256": hashlib.sha256(
        args.output.read_bytes()).hexdigest(), "scanned_rows": report["scanned_rows"],
        "key": report["key"]}
    args.annotate_labels.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    print("Attached training-overlap tags; Stockfish scores are unchanged")


def audit(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.positions.read_text(encoding="utf-8"))
    selected = {" ".join(r["fen"].split()[:2]) for r in suite["positions"]}
    matches: dict[str, dict[str, Any]] = {}
    scanned = 0
    sources = {}
    started = time.perf_counter()
    paths = sorted(args.chunks.glob("*.npz"))
    if not paths:
        raise ValueError("No local training chunks found")
    if args.output.exists():
        raise FileExistsError(args.output)
    for index, path in enumerate(paths, 1):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            for fen in archive["fens"]:
                key = " ".join(str(fen).split(maxsplit=2)[:2])
                scanned += 1
                if key in selected:
                    match = matches.setdefault(key, {"occurrences": 0, "example_chunks": []})
                    match["occurrences"] += 1
                    if (path.name not in match["example_chunks"]
                            and len(match["example_chunks"]) < 5):
                        match["example_chunks"].append(path.name)
        if index % 20 == 0 or index == len(paths):
            print(f"Training audit {index}/{len(paths)} chunks, {scanned} rows, "
                  f"{len(matches)} matching feature positions", flush=True)
    report = {"key": "piece placement and side to move; clocks/castling/ep ignored",
              "positions_sha256": hashlib.sha256(args.positions.read_bytes()).hexdigest(),
              "sources": sources, "scanned_rows": scanned,
              "runtime_seconds": time.perf_counter() - started,
              "positions": [{"fen": r["fen"], "training_overlap": bool(
                  " ".join(r["fen"].split()[:2]) in matches),
                  "evidence": matches.get(" ".join(r["fen"].split()[:2]))}
                  for r in suite["positions"]]}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", type=Path)
    parser.add_argument("--chunks", type=Path, default=Path("dataset_chunks"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/position-large-training-audit.json"))
    parser.add_argument("--annotate-labels", type=Path,
                        help="Attach existing audit metadata without rescanning or analysing")
    args = parser.parse_args()
    (annotate if args.annotate_labels else audit)(args)


if __name__ == "__main__":
    main()
