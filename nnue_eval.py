import os
import numpy as np
import chess

_dir = os.path.dirname(os.path.abspath(__file__))
_w = np.load(os.path.join(_dir, "nnue_weights_np.npz"))
FT_WEIGHT = _w["ft_weight"]
FT_BIAS = _w["ft_bias"].copy()
L1_WEIGHT = _w["l1_weight"]
L1_BIAS = _w["l1_bias"]
L2_WEIGHT = _w["l2_weight"]
L2_BIAS = float(_w["l2_bias"])

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
    __slots__ = ['w_acc', 'b_acc', 'stack']

    def __init__(self):
        self.w_acc = FT_BIAS.copy()
        self.b_acc = FT_BIAS.copy()
        self.stack = []  # stack of (w_acc_copy, b_acc_copy) for undo

    def init_from_board(self, board: chess.Board):
        """Full recomputation from a board position."""
        self.w_acc = FT_BIAS.copy()
        self.b_acc = FT_BIAS.copy()
        self.stack = []

        for sq, piece in board.piece_map().items():
            self._add_piece(piece.piece_type, piece.color, sq)

    def _w_index(self, piece_type, color, sq):
        pt_idx = PIECE_TYPE_MAP[piece_type]
        p_color = 0 if color == chess.WHITE else 1
        return pt_idx * 128 + p_color * 64 + sq

    def _b_index(self, piece_type, color, sq):
        pt_idx = PIECE_TYPE_MAP[piece_type]
        p_color = 0 if color == chess.WHITE else 1
        return pt_idx * 128 + (1 - p_color) * 64 + (sq ^ 56)

    def _add_piece(self, piece_type, color, sq):
        self.w_acc += FT_WEIGHT[self._w_index(piece_type, color, sq)]
        self.b_acc += FT_WEIGHT[self._b_index(piece_type, color, sq)]

    def _remove_piece(self, piece_type, color, sq):
        self.w_acc -= FT_WEIGHT[self._w_index(piece_type, color, sq)]
        self.b_acc -= FT_WEIGHT[self._b_index(piece_type, color, sq)]

    def push(self, board: chess.Board, move: chess.Move):
        """Incrementally update accumulators for a move. Call BEFORE board.push(move)."""
        self.stack.append((self.w_acc.copy(), self.b_acc.copy()))

        piece = board.piece_at(move.from_square)
        if piece is None:
            return

        piece_type = piece.piece_type
        color = piece.color

        # Remove piece from source square
        self._remove_piece(piece_type, color, move.from_square)

        # Handle capture (check to_square for normal capture)
        captured = board.piece_at(move.to_square)
        if captured is not None:
            self._remove_piece(captured.piece_type, captured.color, move.to_square)

        # Handle en passant: pawn moves diagonally to empty square
        elif piece_type == chess.PAWN and chess.square_file(move.from_square) != chess.square_file(move.to_square):
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
        self.w_acc, self.b_acc = self.stack.pop()

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

    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_idx = pt_idx * 128 + p_color * 64 + sq
        b_idx = pt_idx * 128 + (1 - p_color) * 64 + (sq ^ 56)
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