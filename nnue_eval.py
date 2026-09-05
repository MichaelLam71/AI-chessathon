import numpy as np
import chess
import torch
import torch.nn as nn

PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)

class NNUE(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=256):
        super().__init__()
        self.ft_weight = nn.Parameter(torch.randn(hidden_dim, feature_dim) * 0.01)
        self.ft_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.layer1 = nn.Linear(hidden_dim * 2, 32)
        self.layer2 = nn.Linear(32, 1)
        self.crelu = ClippedReLU()

    def forward(self, w_feat, b_feat, stm):
        w_acc = torch.matmul(w_feat, self.ft_weight.T) + self.ft_bias
        b_acc = torch.matmul(b_feat, self.ft_weight.T) + self.ft_bias
        stm_expanded = stm.unsqueeze(1)
        combined = torch.where(
            stm_expanded == 1.0,
            torch.cat([w_acc, b_acc], dim=1),
            torch.cat([b_acc, w_acc], dim=1)
        )
        x = self.crelu(combined)
        x = self.crelu(self.layer1(x))
        out = self.layer2(x)
        return out

# Lazy Model Loading to avoid import crashes
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = NNUE()
        _MODEL.load_state_dict(torch.load("nnue_model.pt", map_location="cpu"))
        _MODEL.eval()
    return _MODEL

# Pre-allocated inference buffers
_W_VEC = torch.zeros(1, 768)
_B_VEC = torch.zeros(1, 768)

def evaluate_nnue(board: chess.Board) -> int:
    model = get_model()
    
    _W_VEC.zero_()
    _B_VEC.zero_()

    w_indices = []
    b_indices = []

    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_feat = pt_idx * 128 + p_color * 64 + sq
        b_feat = pt_idx * 128 + (1 - p_color) * 64 + chess.square_mirror(sq)
        w_indices.append(w_feat)
        b_indices.append(b_feat)

    _W_VEC[0, w_indices] = 1.0
    _B_VEC[0, b_indices] = 1.0

    stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0])

    with torch.no_grad():
        raw_output = model(_W_VEC, _B_VEC, stm).item()

    # Direct scale conversion (400 / ln(10) ≈ 173.7178)
    centipawns = int(raw_output * 173.7178)

    return max(-10000, min(10000, centipawns))