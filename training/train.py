"""Trains learned, phase-blended piece-square tables via ridge regression.

Streams through a dataset of (fen, stockfish_cp) pairs without ever holding
the full feature matrix in memory: it accumulates X^T X (768x768) and X^T y
(768,) chunk by chunk, then solves the ridge-regression normal equations once
at the end. This scales to millions of rows on a normal laptop.

Expects an iterable of (fen: str, cp_side_to_move: float) pairs. Wire up
`iter_rows()` below to your actual data source (see the two examples).
"""
import sys
import numpy as np
import chess

sys.path.insert(0, '.')
from features import featurize, material_score_stm, PIECE_TYPES

N_FEATURES = 768
RIDGE_LAMBDA = 5.0  # regularization strength; larger = smoother/safer tables


def iter_rows_from_jsonl(path):
    """Example loader for the Lichess database.lichess.org .jsonl (decompressed) format.
    Each line: {"fen": "...", "evals": [{"pvs": [{"cp": 311}], ...}, ...]}
    """
    import json
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            evals = row.get("evals")
            if not evals:
                continue
            pv = evals[0]["pvs"][0]
            if "cp" not in pv:
                continue  # skip forced-mate rows for simplicity
            yield row["fen"], float(pv["cp"])


def iter_rows_from_hf_dataset(limit=2_000_000):
    from datasets import load_dataset
    import chess
    dset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
    for i, row in enumerate(dset):
        if i >= limit:
            break
        if row.get("mate") is not None:
            continue
        board = chess.Board(row["fen"])
        cp_white = float(row["cp"])
        cp_stm = cp_white if board.turn == chess.WHITE else -cp_white
        yield row["fen"], cp_stm


def train(row_iterator, chunk_size=4096):
    A = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
    b = np.zeros(N_FEATURES, dtype=np.float64)
    n_seen = 0
    n_used = 0

    X_chunk = []
    y_chunk = []

    def flush():
        nonlocal A, b
        if not X_chunk:
            return
        Xc = np.stack(X_chunk).astype(np.float64)
        yc = np.array(y_chunk, dtype=np.float64)
        A += Xc.T @ Xc
        b += Xc.T @ yc
        X_chunk.clear()
        y_chunk.clear()

    for fen, cp_stm in row_iterator:
        n_seen += 1
        # cp values beyond a few hundred are already decisive; clip so a few
        # huge outliers (e.g. cp=3000 in a totally won position) don't distort
        # the fit for ordinary, roughly-balanced positions.
        cp_stm = max(-1000.0, min(1000.0, cp_stm))
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        x, _ = featurize(board)
        target = cp_stm - material_score_stm(board)
        X_chunk.append(x)
        y_chunk.append(target)
        n_used += 1
        if len(X_chunk) >= chunk_size:
            flush()
        if n_seen % 200_000 == 0:
            print(f"  ...{n_seen} rows seen, {n_used} used", file=sys.stderr)
    flush()

    A += RIDGE_LAMBDA * np.eye(N_FEATURES)
    w = np.linalg.solve(A, b)
    return w, n_used


def weights_to_tables(w):
    opening = w[:384].reshape(6, 64)
    endgame = w[384:].reshape(6, 64)
    return opening, endgame


def print_as_python(opening, endgame):
    names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
    for label, table in [("OPENING", opening), ("ENDGAME", endgame)]:
        for i, name in enumerate(names):
            row = np.round(table[i]).astype(int).tolist()
            print(f"LEARNED_{name}_{label} = {row}")


if __name__ == "__main__":
    import itertools

    all_rows = iter_rows_from_hf_dataset(limit=300_000)
    holdout = list(itertools.islice(all_rows, 5000))  # first 5000 rows, held back

    w, n_used = train(all_rows)  # trains on everything AFTER those first 5000
    print(f"trained on {n_used} rows", file=sys.stderr)

    # check prediction accuracy on the held-out rows
    errors = []
    for fen, cp_stm in holdout:
        board = chess.Board(fen)
        x, _ = featurize(board)
        pred = material_score_stm(board) + x[:384] @ w[:384] + x[384:] @ w[384:]
        errors.append(abs(pred - cp_stm))
    print(f"mean abs error on {len(holdout)} held-out real positions: {np.mean(errors):.1f} cp", file=sys.stderr)

    opening, endgame = weights_to_tables(w)
    print_as_python(opening, endgame)
