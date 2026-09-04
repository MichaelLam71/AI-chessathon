"""Feature extraction for a learned, phase-blended piece-square-table evaluator.

The feature vector is 768 numbers long: 384 "opening weight" features and
384 "endgame weight" features (6 piece types x 64 squares each). Everything
is expressed from the perspective of the side to move, mirrored so "my"
pieces always look like they're playing up the board — this means one
learned table works for both colours, exactly like the existing PST tables.
"""
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
MATERIAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
PHASE_WEIGHT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
                chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
TOTAL_PHASE = 24  # 4 knights + 4 bishops + 4 rooks*2 + 2 queens*4 = 4+4+8+8


def phase_of(board: chess.Board) -> float:
    """0.0 = full opening material on board, 1.0 = bare endgame."""
    phase = TOTAL_PHASE
    for pt in PIECE_TYPES:
        phase -= PHASE_WEIGHT[pt] * len(board.pieces(pt, chess.WHITE))
        phase -= PHASE_WEIGHT[pt] * len(board.pieces(pt, chess.BLACK))
    phase = max(0, min(TOTAL_PHASE, phase))
    return phase / TOTAL_PHASE


def material_score_stm(board: chess.Board) -> int:
    """Material balance, positive = side to move is materially ahead."""
    score = 0
    for pt in PIECE_TYPES:
        score += MATERIAL[pt] * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return score if board.turn == chess.WHITE else -score


def featurize(board: chess.Board):
    """Returns (x, phase) where x is a length-768 float32 feature vector."""
    stm = board.turn
    raw = np.zeros(384, dtype=np.float32)
    for square, piece in board.piece_map().items():
        pt_index = PIECE_TYPES.index(piece.piece_type)
        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        sign = 1.0 if piece.color == stm else -1.0
        raw[pt_index * 64 + sq] += sign
    phase = phase_of(board)
    x = np.concatenate([raw * (1.0 - phase), raw * phase]).astype(np.float32)
    return x, phase
