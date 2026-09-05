import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def fen_to_features(fen: str):
    board = chess.Board(fen)
    w_indices = []
    b_indices = []
    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_feat = pt_idx * 128 + p_color * 64 + sq
        b_feat = pt_idx * 128 + (1 - p_color) * 64 + chess.square_mirror(sq)
        w_indices.append(w_feat)
        b_indices.append(b_feat)
    return w_indices, b_indices

class ChessNNUEDataset(Dataset):
    def __init__(self, chunk_dir="dataset_chunks"):
        import glob
        all_fens = []
        all_wdls = []
        for f in sorted(glob.glob(f"{chunk_dir}/chunk_*.npz")):
            data = np.load(f)
            all_fens.extend(data["fens"])
            all_wdls.extend(data["wdls"])
        self.fens = all_fens
        self.wdls = np.array(all_wdls, dtype=np.float32)
        print(f"Loaded {len(self.fens)} positions from {chunk_dir}")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = str(self.fens[idx])
        wdl = self.wdls[idx]
        w_idx, b_idx = fen_to_features(fen)
        w_vec = torch.zeros(768, dtype=torch.float32)
        b_vec = torch.zeros(768, dtype=torch.float32)
        w_vec[w_idx] = 1.0
        b_vec[b_idx] = 1.0
        stm = 1.0 if " w " in fen else 0.0
        return w_vec, b_vec, torch.tensor(stm, dtype=torch.float32), torch.tensor(wdl, dtype=torch.float32)

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

def train_model():
    dataset = ChessNNUEDataset("dataset_chunks")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = NNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()


    print("Training NNUE...")
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (w_vec, b_vec, stm, wdl) in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")
            w_vec, b_vec = w_vec.to(device), b_vec.to(device)
            stm, wdl = stm.to(device), wdl.to(device)

            optimizer.zero_grad()
            outputs = model(w_vec, b_vec, stm).squeeze(-1)
            preds = torch.sigmoid(outputs)
            loss = criterion(preds, wdl)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save the float model (for nnue_eval_float.py)
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), "nnue_model.pt")
    print("Saved float model to nnue_model.pt")

if __name__ == "__main__":
    train_model()