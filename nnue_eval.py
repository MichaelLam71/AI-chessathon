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

# Load the trained model (save this from train_nnue.py)
model = NNUE()
model.load_state_dict(torch.load("nnue_model.pt", map_location="cpu"))
model.eval()

def evaluate_nnue(board: chess.Board) -> int:
    w_indices = []
    b_indices = []

    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_feat = pt_idx * 128 + p_color * 64 + sq
        b_feat = pt_idx * 128 + (1 - p_color) * 64 + chess.square_mirror(sq)
        w_indices.append(w_feat)
        b_indices.append(b_feat)

    w_vec = torch.zeros(1, 768)
    b_vec = torch.zeros(1, 768)
    w_vec[0, w_indices] = 1.0
    b_vec[0, b_indices] = 1.0

    stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0])

    with torch.no_grad():
        raw_output = model(w_vec, b_vec, stm).item()
        wdl = 1.0 / (1.0 + np.exp(-raw_output))  # sigmoid

    # Convert WDL probability back to centipawns
    # wdl = 1/(1 + 10^(-cp/400))
    # Solving for cp: cp = 400 * log10(wdl / (1 - wdl))
    wdl = max(0.001, min(0.999, wdl))  # avoid log(0)
    centipawns = int(400.0 * np.log10(wdl / (1.0 - wdl)))

    # Clamp to reasonable range
    return max(-10000, min(10000, centipawns))