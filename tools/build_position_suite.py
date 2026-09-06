"""Build an independently generated, traceable holdout suite for local testing."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
from position_benchmark import ROOT, stockfish_path

OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",
    "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O",
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6",
    "c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5",
    "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7",
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",
    "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7",
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",
    "d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7",
    "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",
    "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6",
    "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7",
]
QUOTAS = {"opening": 50, "middlegame": 200, "tactical": 50, "endgame": 100}


def generate_games(path: Path, count: int, seed: int, executable: Path) -> None:
    rng = random.Random(seed)
    with path.open("x", encoding="utf-8") as output, chess.engine.SimpleEngine.popen_uci(
        str(executable)
    ) as engine:
        if engine.id.get("name") != "Stockfish 19":
            raise ValueError(f"Expected Stockfish 19, got {engine.id}")
        engine.configure({"Threads": 1, "Hash": 128})
        for index in range(count):
            game = chess.pgn.Game()
            game.headers.update(Event="Independent benchmark holdout self-play",
                                White="Stockfish 19 varied", Black="Stockfish 19 varied",
                                Round=str(index + 1), Seed=str(seed),
                                GenerationNodes="10000", OpeningIndex=str(index % len(OPENINGS)))
            board = game.board()
            node: chess.pgn.GameNode = game
            for san in OPENINGS[index % len(OPENINGS)].split():
                move = board.parse_san(san)
                node = node.add_variation(move)
                board.push(move)
            while not board.is_game_over(claim_draw=True) and board.ply() < 240:
                # Both sides vary among plausible moves early; occasional later
                # alternatives produce imbalances without choosing moves uniformly.
                vary = board.ply() < 30 or rng.random() < 0.12
                if vary:
                    infos = engine.analyse(board, chess.engine.Limit(nodes=10000),
                                           multipv=min(3, board.legal_moves.count()), game=game)
                    best = infos[0]["score"].relative.score(mate_score=100000)
                    assert best is not None
                    choices = [info["pv"][0] for info in infos
                               if best - int(info["score"].relative.score(mate_score=100000)
                                             or 0) <= 100]
                    move = rng.choice(choices)
                else:
                    played = engine.play(board, chess.engine.Limit(nodes=10000), game=game)
                    assert played.move is not None
                    move = played.move
                node = node.add_variation(move)
                board.push(move)
            game.headers["Result"] = board.result(claim_draw=True)
            game.headers["Termination"] = "ply cap" if board.ply() >= 240 else "board outcome"
            print(game, file=output, end="\n\n")
            output.flush()
            print(f"Generated game {index + 1}/{count}: {board.ply()} plies", flush=True)


def classify(board: chess.Board) -> tuple[str, list[str]]:
    pieces = list(board.piece_map().values())
    non_pawns = sum(p.piece_type not in (chess.PAWN, chess.KING) for p in pieces)
    tags = []
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    material = sum(values[p.piece_type] * (1 if p.color == board.turn else -1) for p in pieces)
    if abs(material) >= 150:
        tags.append("imbalanced")
    if board.is_check():
        tags.append("in_check")
    if non_pawns <= 4 or len(pieces) <= 12:
        return "endgame", tags
    if board.ply() <= 24:
        return "opening", tags
    legal = list(board.legal_moves)
    captures = sum(board.is_capture(m) for m in legal)
    checks = sum(board.gives_check(m) for m in legal)
    if board.is_check() or (captures >= 3 and checks >= 1):
        return "tactical", tags
    return "middlegame", tags


def candidates(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8-sig") as source:
            game_index = 0
            while (game := chess.pgn.read_game(source)) is not None:
                game_index += 1
                if game.errors:
                    raise ValueError(f"PGN errors in {path}: {game.errors}")
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    if board.ply() < 10 or board.is_game_over(claim_draw=True):
                        continue
                    legal = list(board.legal_moves)
                    if len(legal) < 4:
                        continue
                    key = " ".join(board.fen().split()[:4])
                    if key in seen:
                        continue
                    mate_one = False
                    for candidate in legal:
                        if board.gives_check(candidate):
                            board.push(candidate)
                            mate_one = board.is_checkmate()
                            board.pop()
                            if mate_one:
                                break
                    if mate_one:
                        continue
                    seen.add(key)
                    category, tags = classify(board)
                    result.append({"fen": board.fen(), "category": category, "tags": tags,
                                   "source": str(path.relative_to(ROOT)),
                                   "game_id": f"{path.name}:{game_index}", "ply": board.ply()})
    return result


def build(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.generated_pgn.exists():
        generate_games(args.generated_pgn, args.games, args.seed, stockfish_path(args.stockfish))
    paths = [ROOT / "stats-white.pgn", ROOT / "stats-black.pgn", args.generated_pgn.resolve()]
    pool = candidates(paths)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_game: dict[str, list[int]] = defaultdict(list)
    signatures: dict[bool, list[frozenset[tuple[int, str]]]] = defaultdict(list)
    # Fill scarce strata first, retaining source-game diversity in every category.
    for category in ("endgame", "tactical", "opening", "middlegame"):
        for item in sorted(pool, key=lambda p: not p["source"].startswith("stats-")):
            if item["category"] != category or counts[category] >= QUOTAS[category]:
                continue
            plies = per_game[item["game_id"]]
            if len(plies) >= 8 or any(abs(item["ply"] - ply) < 10 for ply in plies):
                continue
            board = chess.Board(item["fen"])
            signature = frozenset((sq, p.symbol()) for sq, p in board.piece_map().items())
            if any(len(signature ^ previous) <= 4 for previous in signatures[board.turn]):
                continue
            signatures[board.turn].append(signature)
            plies.append(item["ply"])
            selected.append(item)
            counts[category] += 1
    if dict(counts) != QUOTAS:
        raise ValueError(f"Insufficient candidates: selected {dict(counts)}; generate more games")
    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        item["id"] = f"large-{index:04}"
    document = {"schema_version": 1, "seed": args.seed, "quotas": QUOTAS,
                "candidate_count": len(pool), "max_per_game": 8, "min_ply_spacing": 10,
                "near_duplicate_rule": "same turn and <=4 changed square/piece entries",
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths}, "positions": selected}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")
    with args.output.with_suffix(".txt").open("x", encoding="utf-8") as output:
        for item in selected:
            output.write(f"{item['fen']} # {item['id']} {item['category']}\n")
    print(f"Selected {dict(counts)} from {len(pool)} candidates, {len(per_game)} games")
