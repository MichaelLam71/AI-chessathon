"""Prepare a reproducible, balanced NNUE subset from Lichess fishnet evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import numpy as np

DATASET = "Lichess/fishnet-evals"
REVISION = "1b6d7c91ddef44ec89efe64e47a2e313b7648ece"
HF_GLOB = f"hf://datasets/{DATASET}@{REVISION}/**/*.parquet"
CP_STORAGE_CLIP = 1500
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
# The 5M public run exposed an endgame regression, so the large data-only run moves
# five percentage points from middlegames to endgames. Features/model/training stay fixed.
PHASE_WEIGHTS = {"opening": 0.20, "middlegame": 0.50, "endgame": 0.30}
CONTEXT_WEIGHTS = {"quiet": 0.55, "tactical": 0.20, "imbalanced": 0.25}
SCORE_BINS = (
    ("loss_800_plus", -float("inf"), -800, 0.10),
    ("loss_400_800", -800, -400, 0.10),
    ("loss_150_400", -400, -150, 0.10),
    ("loss_50_150", -150, -50, 0.10),
    ("near_equal", -50, 50, 0.20),
    ("win_50_150", 50, 150, 0.10),
    ("win_150_400", 150, 400, 0.10),
    ("win_400_800", 400, 800, 0.10),
    ("win_800_plus", 800, float("inf"), 0.10),
)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def position_key(board: chess.Board) -> str:
    return f"{board.board_fen()} {'w' if board.turn else 'b'}"


def side_to_move_cp(board: chess.Board, white_cp: int) -> int:
    return white_cp if board.turn == chess.WHITE else -white_cp


def benchmark_keys(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    return {position_key(chess.Board(row["fen"])) for row in document["positions"]}


def phase(board: chess.Board) -> str:
    total_pieces = len(board.piece_map())
    non_pawn_non_king = sum(
        len(board.pieces(piece_type, color))
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for color in chess.COLORS
    )
    if non_pawn_non_king <= 4 or total_pieces <= 12:
        return "endgame"
    ply = (board.fullmove_number - 1) * 2 + (board.turn == chess.BLACK)
    return "opening" if ply <= 24 else "middlegame"


def is_tactical(board: chess.Board) -> bool:
    if board.is_check():
        return True
    if sum(1 for _ in board.generate_legal_captures()) < 3:
        return False
    return any(board.gives_check(move) for move in board.legal_moves)


def material_imbalance(board: chess.Board) -> int:
    white = sum(
        value * len(board.pieces(piece_type, chess.WHITE))
        for piece_type, value in PIECE_VALUES.items()
    )
    black = sum(
        value * len(board.pieces(piece_type, chess.BLACK))
        for piece_type, value in PIECE_VALUES.items()
    )
    return abs(white - black)


def context(board: chess.Board) -> str:
    if is_tactical(board):
        return "tactical"
    return "imbalanced" if material_imbalance(board) >= 150 else "quiet"


def score_bin(score: int) -> str:
    for name, lower, upper, _ in SCORE_BINS:
        if lower <= score < upper:
            return name
    raise AssertionError("unreachable score bin")


def quota(total: int, weights: dict[str, float]) -> dict[str, int]:
    weighted = [(name, total * weight) for name, weight in weights.items()]
    result = {key: int(value) for key, value in weighted}
    remaining = total - sum(result.values())
    order = sorted(weighted, key=lambda item: (-(item[1] % 1), item[0]))
    for key, _ in order[:remaining]:
        result[key] += 1
    return result


def changed_squares(left: chess.Board, right: chess.Board) -> int:
    return sum(left.piece_at(square) != right.piece_at(square) for square in chess.SQUARES)


def file_month(path: str) -> str:
    match = re.search(r"standard_rated_(\d{4})_(\d{2})\.parquet$", path)
    if match is None:
        raise ValueError(f"Unexpected fishnet shard name: {path}")
    return f"{match.group(1)}-{match.group(2)}"


def save_chunk(output_dir: Path, index: int, rows: list[tuple[str, int, str, str, str]],
               checkpoint: dict[str, Any] | None = None) -> str:
    path = output_dir / f"chunk_public_{index:05d}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing chunk: {path}")
    temporary = output_dir / f".{path.name}.{uuid.uuid4().hex}.partial"
    with temporary.open("xb") as output:
        fields: dict[str, Any] = {
            "fens": np.array([row[0] for row in rows]),
            "scores": np.array([row[1] for row in rows], dtype=np.int16),
            "phases": np.array([row[2] for row in rows]),
            "contexts": np.array([row[3] for row in rows]),
            "score_bins": np.array([row[4] for row in rows]),
        }
        if checkpoint is not None:
            fields["checkpoint_json"] = np.array(json.dumps(checkpoint))
        np.savez_compressed(output, **fields)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_tuple(value: Any) -> Any:
    return tuple(nested_tuple(item) for item in value) if isinstance(value, list) else value


def counter_from_json(value: dict[str, int], compound: bool = False) -> Counter[Any]:
    if compound:
        return Counter({tuple(key.split("/")): count for key, count in value.items()})
    return Counter(value)


def run_identity(args: argparse.Namespace, benchmark_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset": DATASET,
        "revision": REVISION,
        "positions": args.positions,
        "benchmark_sha256": benchmark_sha256,
        "seed": args.seed,
        "min_month": args.min_month,
        "chunk_size": args.chunk_size,
        "survey_rows": args.survey_rows,
        "max_oversample": args.max_oversample,
        "phase_weights": PHASE_WEIGHTS,
        "context_weights": CONTEXT_WEIGHTS,
        "score_bins": [list(row) for row in SCORE_BINS],
    }


def load_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required only for preparation. Run with: "
            "uv run --with duckdb python prepare_public_nnue.py ..."
        ) from exc
    return duckdb


def duckdb_connection(use_hf_auth: bool):
    connection = load_duckdb().connect()
    if use_hf_auth:
        connection.execute(
            "CREATE SECRET hf_token (TYPE HUGGINGFACE, PROVIDER credential_chain)"
        )
    return connection


def inspect_source(use_hf_auth: bool) -> None:
    connection = duckdb_connection(use_hf_auth)
    files = [
        row[0] for row in connection.execute("SELECT file FROM glob(?)", [HF_GLOB]).fetchall()
    ]
    print(f"Files: {len(files)}; first={files[0]}; last={files[-1]}")
    cursor = connection.execute(
        "SELECT fen, cp, mate FROM read_parquet(?) LIMIT 3", [HF_GLOB]
    )
    print("Schema:", [(column[0], column[1]) for column in cursor.description])
    for row in cursor.fetchall():
        print(row)


def classify(fen: str, white_cp: int | None, mate: int | None,
             excluded: set[str],
             audit: Counter[str]) -> tuple[chess.Board, int, str, str, str] | None:
    if mate is not None or white_cp is None:
        audit["mate_or_missing_cp"] += 1
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        audit["malformed_fen"] += 1
        return None
    if not board.is_valid() or board.is_game_over():
        audit["invalid_or_terminal"] += 1
        return None
    if position_key(board) in excluded:
        audit["benchmark_overlap"] += 1
        return None
    stm_cp = side_to_move_cp(board, white_cp)
    return board, stm_cp, phase(board), context(board), score_bin(stm_cp)


def survey_distribution(connection, files: list[str], args: argparse.Namespace,
                        excluded: set[str]) -> tuple[Counter[tuple[str, str, str]], int,
                                                     Counter[str]]:
    distribution: Counter[tuple[str, str, str]] = Counter()
    audit: Counter[str] = Counter()
    scanned = 0
    accepted = 0
    for file_path in files:
        cursor = connection.execute("SELECT fen, cp, mate FROM read_parquet(?)", [file_path])
        while batch := cursor.fetchmany(args.read_batch):
            for fen, white_cp, mate in batch:
                scanned += 1
                candidate = classify(fen, white_cp, mate, excluded, audit)
                if candidate is None:
                    continue
                _, _, phase_name, context_name, bin_name = candidate
                distribution[(phase_name, context_name, bin_name)] += 1
                accepted += 1
                if accepted == args.survey_rows:
                    return distribution, scanned, audit
    raise RuntimeError(f"Only found {accepted:,} valid survey rows")


def acceptance_probabilities(distribution: Counter[tuple[str, str, str]],
                             survey_rows: int, max_oversample: float
                             ) -> dict[tuple[str, str, str], float]:
    score_weights = {name: weight for name, _, _, weight in SCORE_BINS}
    ratios: dict[tuple[str, str, str], float] = {}
    for phase_name in PHASE_WEIGHTS:
        for context_name in CONTEXT_WEIGHTS:
            for bin_name in score_weights:
                key = (phase_name, context_name, bin_name)
                desired = (PHASE_WEIGHTS[phase_name] * CONTEXT_WEIGHTS[context_name]
                           * score_weights[bin_name])
                observed = distribution[key] / survey_rows
                ratios[key] = max_oversample if observed == 0 else min(
                    max_oversample, desired / observed
                )
    scale = max(ratios.values())
    return {key: ratio / scale for key, ratio in ratios.items()}


def prepare(args: argparse.Namespace) -> None:
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = run_identity(
            args, hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        )
        if manifest.get("run_identity") != expected_identity:
            raise ValueError("Completed dataset has different settings; choose a new output path")
        print(f"Dataset is already complete: {manifest_path}")
        return
    existing_chunks = sorted(args.output.glob("chunk_public_*.npz")) \
        if args.output.exists() else []
    unrelated = [path for path in args.output.iterdir()
                 if path not in existing_chunks and ".partial" not in path.name] \
        if args.output.exists() else []
    if (existing_chunks or unrelated) and not args.resume:
        raise FileExistsError(
            f"Refusing to use non-empty output directory: {args.output}; "
            "pass --resume only for an interrupted run with identical settings"
        )
    if unrelated:
        raise FileExistsError(f"Unexpected files in resume directory: {unrelated}")
    args.output.mkdir(parents=True, exist_ok=True)
    selected_phase: Counter[str] = Counter()
    selected_context: Counter[str] = Counter()
    selected_score: Counter[str] = Counter()
    selected_strata: Counter[tuple[str, str, str]] = Counter()
    audit: Counter[str] = Counter()
    excluded = benchmark_keys(args.benchmark)
    latest_by_turn: dict[chess.Color, chess.Board] = {}
    rows: list[tuple[str, int, str, str, str]] = []
    chunk_hashes: dict[str, str] = {}
    chunk_index = 0
    scanned = 0
    accepted = 0
    start_file_index = 0
    file_row_offset = 0

    connection = duckdb_connection(args.hf_auth)
    files = [
        row[0] for row in connection.execute("SELECT file FROM glob(?)", [HF_GLOB]).fetchall()
    ]
    files = [path for path in files if file_month(path) >= args.min_month]
    if not files:
        raise RuntimeError(f"No Parquet files found at {HF_GLOB} from {args.min_month}")
    file_rng = random.Random(args.seed)
    file_rng.shuffle(files)
    benchmark_sha256 = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    identity = run_identity(args, benchmark_sha256)
    survey, survey_scanned, survey_audit = survey_distribution(
        connection, files, args, excluded
    )
    probabilities = acceptance_probabilities(
        survey, sum(survey.values()), args.max_oversample
    )
    maximum_probability = {
        bin_name: max(
            probability
            for (phase_name, context_name, candidate_bin), probability in probabilities.items()
            if candidate_bin == bin_name
        )
        for bin_name, _, _, _ in SCORE_BINS
    }
    selection_rng = random.Random(args.seed + 1)

    if args.resume:
        if not existing_chunks:
            print("No completed chunks found; starting the requested run from the beginning")
        else:
            expected_names = [f"chunk_public_{index:05d}.npz"
                              for index in range(len(existing_chunks))]
            if [path.name for path in existing_chunks] != expected_names:
                raise ValueError("Resume chunks are not a contiguous sequence")
            for path in existing_chunks:
                chunk_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(existing_chunks[-1]) as latest:
                if "checkpoint_json" not in latest:
                    raise ValueError(
                        "These chunks predate resumable preparation; choose a new output path"
                    )
                checkpoint = json.loads(str(latest["checkpoint_json"].item()))
                last_chunk_rows = len(latest["scores"])
            if checkpoint["identity"] != identity:
                raise ValueError("Resume settings differ from the interrupted run")
            accepted = int(checkpoint["accepted"])
            expected_rows = (len(existing_chunks) - 1) * args.chunk_size + last_chunk_rows
            if accepted != expected_rows:
                raise ValueError("Resume checkpoint count does not match the completed chunks")
            scanned = int(checkpoint["scanned"])
            chunk_index = int(checkpoint["next_chunk_index"])
            start_file_index = int(checkpoint["file_index"])
            file_row_offset = int(checkpoint["file_row_offset"])
            selected_phase = counter_from_json(checkpoint["selected_phase"])
            selected_context = counter_from_json(checkpoint["selected_context"])
            selected_score = counter_from_json(checkpoint["selected_score"])
            selected_strata = counter_from_json(checkpoint["selected_strata"], compound=True)
            audit = counter_from_json(checkpoint["audit"])
            latest_by_turn = {}
            if checkpoint["latest_white"] is not None:
                latest_by_turn[chess.WHITE] = chess.Board(checkpoint["latest_white"])
            if checkpoint["latest_black"] is not None:
                latest_by_turn[chess.BLACK] = chess.Board(checkpoint["latest_black"])
            selection_rng.setstate(nested_tuple(checkpoint["selection_rng_state"]))
            print(
                f"Resuming after {accepted:,}/{args.positions:,} selected and "
                f"{scanned:,} scanned",
                flush=True,
            )

    def checkpoint_for(file_index: int, offset: int) -> dict[str, Any]:
        return {
            "identity": identity,
            "accepted": accepted,
            "scanned": scanned,
            "next_chunk_index": chunk_index + 1,
            "file_index": file_index,
            "file_row_offset": offset,
            "selected_phase": dict(selected_phase),
            "selected_context": dict(selected_context),
            "selected_score": dict(selected_score),
            "selected_strata": {"/".join(key): value for key, value in selected_strata.items()},
            "audit": dict(audit),
            "latest_white": (latest_by_turn[chess.WHITE].fen()
                             if chess.WHITE in latest_by_turn else None),
            "latest_black": (latest_by_turn[chess.BLACK].fen()
                             if chess.BLACK in latest_by_turn else None),
            "selection_rng_state": selection_rng.getstate(),
        }

    for file_index in range(start_file_index, len(files)):
        file_path = files[file_index]
        offset = file_row_offset if file_index == start_file_index else 0
        cursor = connection.execute(
            f"SELECT fen, cp, mate FROM read_parquet(?) OFFSET {offset}", [file_path]
        )
        while batch := cursor.fetchmany(args.read_batch):
            for fen, white_cp, mate in batch:
                if scanned >= args.max_scanned:
                    break
                offset += 1
                scanned += 1
                if mate is not None or white_cp is None:
                    audit["mate_or_missing_cp"] += 1
                    continue
                fields = fen.split()
                if len(fields) < 2 or fields[1] not in ("w", "b"):
                    audit["malformed_fen"] += 1
                    continue
                provisional_cp = int(white_cp if fields[1] == "w" else -white_cp)
                provisional_bin = score_bin(provisional_cp)
                draw = selection_rng.random()
                if draw > maximum_probability[provisional_bin]:
                    audit["balance_sampling"] += 1
                    continue
                candidate = classify(fen, white_cp, mate, excluded, audit)
                if candidate is None:
                    continue
                board, stm_cp, phase_name, context_name, bin_name = candidate
                previous = latest_by_turn.get(board.turn)
                if previous is not None and changed_squares(previous, board) <= 4:
                    audit["near_duplicate"] += 1
                    continue
                stratum = (phase_name, context_name, bin_name)
                if draw > probabilities[stratum]:
                    audit["balance_sampling"] += 1
                    continue
                selected_phase[phase_name] += 1
                selected_context[context_name] += 1
                selected_score[bin_name] += 1
                selected_strata[stratum] += 1
                accepted += 1
                latest_by_turn[board.turn] = board
                stored_cp = max(-CP_STORAGE_CLIP, min(CP_STORAGE_CLIP, stm_cp))
                rows.append((board.fen(), stored_cp, phase_name, context_name, bin_name))
                if len(rows) == args.chunk_size:
                    digest = save_chunk(
                        args.output, chunk_index, rows, checkpoint_for(file_index, offset)
                    )
                    chunk_hashes[f"chunk_public_{chunk_index:05d}.npz"] = digest
                    chunk_index += 1
                    rows = []
                    print(
                        f"Selected {accepted:,}/{args.positions:,}; scanned {scanned:,}; "
                        f"file {file_index + 1}/{len(files)}",
                        flush=True,
                    )
                if accepted == args.positions:
                    break
            if scanned >= args.max_scanned or accepted == args.positions:
                break
        if scanned >= args.max_scanned or accepted == args.positions:
            break

    if rows:
        digest = save_chunk(
            args.output, chunk_index, rows, checkpoint_for(file_index, offset)
        )
        chunk_hashes[f"chunk_public_{chunk_index:05d}.npz"] = digest
    if accepted != args.positions:
        raise RuntimeError(
            f"Selected only {accepted:,}/{args.positions:,} positions after scanning "
            f"{scanned:,}; resume with --resume and a larger --max-scanned"
        )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_identity": identity,
        "source": {"dataset": DATASET, "revision": REVISION, "parquet_glob": HF_GLOB},
        "seed": args.seed,
        "minimum_source_month": args.min_month,
        "source_files": files,
        "survey_rows": sum(survey.values()),
        "survey_scanned_rows": survey_scanned,
        "survey_rejections": dict(sorted(survey_audit.items())),
        "max_oversample": args.max_oversample,
        "acceptance_probability_by_stratum": {
            "/".join(key): value for key, value in sorted(probabilities.items())
        },
        "requested_positions": args.positions,
        "selected_positions": accepted,
        "scanned_rows": scanned,
        "mate_policy": "exclude",
        "stored_cp_clip": CP_STORAGE_CLIP,
        "score_pov": "side to move",
        "benchmark": str(args.benchmark),
        "benchmark_sha256": benchmark_sha256,
        "phase_weights": PHASE_WEIGHTS,
        "context_weights": CONTEXT_WEIGHTS,
        "score_bins": [list(row) for row in SCORE_BINS],
        "selected_phase": dict(sorted(selected_phase.items())),
        "selected_context": dict(sorted(selected_context.items())),
        "selected_score_bin": dict(sorted(selected_score.items())),
        "selected_by_stratum": {
            "/".join(key): value for key, value in sorted(selected_strata.items())
        },
        "rejections": dict(sorted(audit.items())),
        "chunk_sha256": chunk_hashes,
    }
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {accepted:,} positions in {args.output}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=positive, default=8_000_000)
    parser.add_argument("--output", type=Path, default=Path("public_dataset_chunks_8m"))
    parser.add_argument("--benchmark", type=Path,
                        default=Path("benchmarks/positions-large-labelled.json"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--min-month", default="2022-01")
    parser.add_argument("--chunk-size", type=positive, default=50_000)
    parser.add_argument("--read-batch", type=positive, default=8192)
    parser.add_argument("--survey-rows", type=positive, default=100_000)
    parser.add_argument("--max-oversample", type=float, default=5.0)
    parser.add_argument("--max-scanned", type=positive, default=240_000_000)
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run from its last complete chunk")
    parser.add_argument("--hf-auth", action="store_true",
                        help="use a token saved by `hf auth login` through DuckDB")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.min_month):
        parser.error("--min-month must be YYYY-MM")
    if args.max_oversample < 1:
        parser.error("--max-oversample must be >= 1")
    inspect_source(args.hf_auth) if args.inspect else prepare(args)


if __name__ == "__main__":
    main()
