import os

import chess
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_w = np.load(os.path.join(_dir, "weights", "nnue_weights_cp_simple_public.npz"))
FT_WEIGHT = _w["ft_weight"]
FT_BIAS = _w["ft_bias"].copy()
L1_WEIGHT = _w["l1_weight"]
L1_BIAS = _w["l1_bias"]
L2_WEIGHT = _w["l2_weight"]
L2_BIAS = float(_w["l2_bias"])
BASE_FEATURE_DIM = 768
KING_BUCKET_FEATURE_DIM = BASE_FEATURE_DIM * 16
if FT_WEIGHT.shape[0] not in (BASE_FEATURE_DIM, KING_BUCKET_FEATURE_DIM):
    raise ValueError(f"Unsupported NNUE feature dimension: {FT_WEIGHT.shape[0]}")
KING_BUCKETED = FT_WEIGHT.shape[0] == KING_BUCKET_FEATURE_DIM

PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


class NNUEAccumulator:
    """
    Maintains incrementally updated accumulators for both white and black perspectives.
    Instead of recomputing from scratch on every eval, we update only the changed pieces.
    """
    __slots__ = ['b_acc', 'b_bucket', 'stack', 'w_acc', 'w_bucket']

    def __init__(self):
        self.w_acc = FT_BIAS.copy()
        self.b_acc = FT_BIAS.copy()
        self.w_bucket = 0
        self.b_bucket = 0
        self.stack = []

    @staticmethod
    def _king_bucket(square):
        return (chess.square_rank(square) // 2) * 4 + chess.square_file(square) // 2

    def _set_buckets(self, board):
        if not KING_BUCKETED:
            self.w_bucket = self.b_bucket = 0
            return
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError("NNUE position is missing a king")
        self.w_bucket = self._king_bucket(white_king)
        self.b_bucket = self._king_bucket(chess.square_mirror(black_king))

    def _rebuild(self, board):
        self.w_acc = FT_BIAS.copy()
        self.b_acc = FT_BIAS.copy()
        self._set_buckets(board)
        for sq, piece in board.piece_map().items():
            self._add_piece(piece.piece_type, piece.color, sq)

    def init_from_board(self, board: chess.Board):
        """Full recomputation from a board position."""
        self.stack = []
        self._rebuild(board)

    def _w_index(self, piece_type, color, sq):
        pt_idx = PIECE_TYPE_MAP[piece_type]
        p_color = 0 if color == chess.WHITE else 1
        return self.w_bucket * BASE_FEATURE_DIM + pt_idx * 128 + p_color * 64 + sq

    def _b_index(self, piece_type, color, sq):
        pt_idx = PIECE_TYPE_MAP[piece_type]
        p_color = 0 if color == chess.WHITE else 1
        return (self.b_bucket * BASE_FEATURE_DIM + pt_idx * 128
                + (1 - p_color) * 64 + (sq ^ 56))

    def _add_piece(self, piece_type, color, sq):
        self.w_acc += FT_WEIGHT[self._w_index(piece_type, color, sq)]
        self.b_acc += FT_WEIGHT[self._b_index(piece_type, color, sq)]

    def _remove_piece(self, piece_type, color, sq):
        self.w_acc -= FT_WEIGHT[self._w_index(piece_type, color, sq)]
        self.b_acc -= FT_WEIGHT[self._b_index(piece_type, color, sq)]

    def push(self, board: chess.Board, move: chess.Move):
        """Incrementally update accumulators for a move. Call BEFORE board.push(move)."""
        self.stack.append(
            (self.w_acc.copy(), self.b_acc.copy(), self.w_bucket, self.b_bucket)
        )

        piece = board.piece_at(move.from_square)
        if piece is None:
            return

        piece_type = piece.piece_type
        color = piece.color

        if KING_BUCKETED and piece_type == chess.KING:
            old_bucket = self.w_bucket if color == chess.WHITE else self.b_bucket
            oriented_to = move.to_square if color == chess.WHITE else chess.square_mirror(
                move.to_square
            )
            if self._king_bucket(oriented_to) != old_bucket:
                moved = board.copy(stack=False)
                moved.push(move)
                self._rebuild(moved)
                return

        # Remove piece from source square
        self._remove_piece(piece_type, color, move.from_square)

        # Handle capture (check to_square for normal capture)
        captured = board.piece_at(move.to_square)
        if captured is not None:
            self._remove_piece(captured.piece_type, captured.color, move.to_square)

        # Handle en passant: pawn moves diagonally to empty square
        elif (piece_type == chess.PAWN
              and chess.square_file(move.from_square) != chess.square_file(move.to_square)):
            ep_sq = move.to_square + (-8 if color == chess.WHITE else 8)
            self._remove_piece(chess.PAWN, not color, ep_sq)

        # Handle promotion
        if move.promotion:
            self._add_piece(move.promotion, color, move.to_square)
        else:
            self._add_piece(piece_type, color, move.to_square)

        # Handle castling: king moves 2+ squares
        if piece_type == chess.KING and abs(move.from_square - move.to_square) >= 2:
            if chess.square_file(move.to_square) == 6:  # kingside
                rook_from = move.to_square + 1
                rook_to = move.to_square - 1
            else:  # queenside
                rook_from = move.to_square - 2
                rook_to = move.to_square + 1
            self._remove_piece(chess.ROOK, color, rook_from)
            self._add_piece(chess.ROOK, color, rook_to)

    def pop(self):
        """Restore accumulators to state before last push."""
        self.w_acc, self.b_acc, self.w_bucket, self.b_bucket = self.stack.pop()

    def evaluate(self, turn: chess.Color) -> int:
        """Evaluate the position from the side-to-move perspective."""
        if turn == chess.WHITE:
            combined = np.concatenate([self.w_acc, self.b_acc])
        else:
            combined = np.concatenate([self.b_acc, self.w_acc])

        # ClippedReLU
        np.clip(combined, 0.0, 1.0, out=combined)

        # Layer 1 + ClippedReLU
        l1_out = L1_WEIGHT @ combined + L1_BIAS
        np.clip(l1_out, 0.0, 1.0, out=l1_out)

        # Layer 2 (output)
        raw_output = L2_WEIGHT @ l1_out + L2_BIAS

        # Scale to centipawns
        centipawns = int(raw_output * 173.7178)
        return max(-10000, min(10000, centipawns))


# Global accumulator instance
accumulator = NNUEAccumulator()


def evaluate_nnue(board: chess.Board) -> int:
    """Fallback: full recomputation (used when accumulator is out of sync)."""
    w_acc = FT_BIAS.copy()
    b_acc = FT_BIAS.copy()
    w_bucket = 0
    b_bucket = 0
    if KING_BUCKETED:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError("NNUE position is missing a king")
        w_bucket = NNUEAccumulator._king_bucket(white_king)
        b_bucket = NNUEAccumulator._king_bucket(chess.square_mirror(black_king))

    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_idx = w_bucket * BASE_FEATURE_DIM + pt_idx * 128 + p_color * 64 + sq
        b_idx = (b_bucket * BASE_FEATURE_DIM + pt_idx * 128
                 + (1 - p_color) * 64 + (sq ^ 56))
        w_acc += FT_WEIGHT[w_idx]
        b_acc += FT_WEIGHT[b_idx]

    if board.turn == chess.WHITE:
        combined = np.concatenate([w_acc, b_acc])
    else:
        combined = np.concatenate([b_acc, w_acc])

    np.clip(combined, 0.0, 1.0, out=combined)
    l1_out = L1_WEIGHT @ combined + L1_BIAS
    np.clip(l1_out, 0.0, 1.0, out=l1_out)
    raw_output = L2_WEIGHT @ l1_out + L2_BIAS
    centipawns = int(raw_output * 173.7178)
    return max(-10000, min(10000, centipawns))
