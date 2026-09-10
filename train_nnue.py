import argparse
import hashlib
import math
import random
import uuid
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# nnue_eval.py converts the network's raw output back to centipawns with this scale.
CP_OUTPUT_SCALE = 173.7178
CP_CLIP = 1500.0
CP_HUBER_DELTA = 100.0
BASE_FEATURE_DIM = 768
KING_BUCKETS = 16
KING_BUCKET_FEATURE_DIM = BASE_FEATURE_DIM * KING_BUCKETS
MAX_PIECES = 32

PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def king_bucket(square: chess.Square) -> int:
    return (chess.square_rank(square) // 2) * 4 + chess.square_file(square) // 2


def fen_to_features(fen: str, feature_set: str = "simple"):
    board = chess.Board(fen)
    w_indices = []
    b_indices = []
    w_offset = 0
    b_offset = 0
    if feature_set == "king-bucket":
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            raise ValueError(f"Position is missing a king: {fen}")
        w_offset = king_bucket(white_king) * BASE_FEATURE_DIM
        b_offset = king_bucket(chess.square_mirror(black_king)) * BASE_FEATURE_DIM
    for sq, piece in board.piece_map().items():
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        p_color = 0 if piece.color == chess.WHITE else 1
        w_feat = w_offset + pt_idx * 128 + p_color * 64 + sq
        b_feat = (b_offset + pt_idx * 128 + (1 - p_color) * 64
                  + chess.square_mirror(sq))
        w_indices.append(w_feat)
        b_indices.append(b_feat)
    return w_indices, b_indices


def training_sample(fen: str, target: float, feature_set: str):
    w_idx, b_idx = fen_to_features(fen, feature_set)
    w_features = torch.full((MAX_PIECES,), -1, dtype=torch.long)
    b_features = torch.full((MAX_PIECES,), -1, dtype=torch.long)
    w_features[:len(w_idx)] = torch.tensor(w_idx, dtype=torch.long)
    b_features[:len(b_idx)] = torch.tensor(b_idx, dtype=torch.long)
    stm = 1.0 if " w " in fen else 0.0
    return (w_features, b_features, torch.tensor(stm, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32))

class ChessNNUEDataset(Dataset):
    def __init__(self, chunk_dir="dataset_chunks", target="wdl", feature_set="simple"):
        import glob
        all_fens = []
        all_targets = []
        target_field = "wdls" if target == "wdl" else "scores"
        for f in sorted(glob.glob(f"{chunk_dir}/chunk_*.npz")):
            data = np.load(f)
            all_fens.extend(data["fens"])
            all_targets.extend(data[target_field])
        self.fens = all_fens
        self.feature_set = feature_set
        self.targets = np.array(all_targets, dtype=np.float32)
        if target == "cp":
            self.targets = np.clip(self.targets, -CP_CLIP, CP_CLIP) / CP_OUTPUT_SCALE
        print(f"Loaded {len(self.fens)} positions from {chunk_dir}")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = str(self.fens[idx])
        target = self.targets[idx]
        return training_sample(fen, float(target), self.feature_set)


class ChunkedChessNNUEDataset(IterableDataset):
    """Shuffle reproducibly one chunk at a time instead of retaining every FEN in RAM."""

    def __init__(self, chunk_dir: Path, target: str, feature_set: str, seed: int):
        self.chunk_paths = sorted(chunk_dir.glob("chunk_*.npz"))
        if not self.chunk_paths:
            raise ValueError(f"No chunk_*.npz files found in {chunk_dir}")
        self.target_field = "wdls" if target == "wdl" else "scores"
        self.target = target
        self.feature_set = feature_set
        self.seed = seed
        self.epoch = 0
        self.length = 0
        for path in self.chunk_paths:
            with np.load(path) as data:
                if self.target_field not in data:
                    raise ValueError(f"{path} has no {self.target_field!r} field")
                self.length += len(data[self.target_field])
        print(f"Found {self.length:,} positions in {len(self.chunk_paths)} chunks at {chunk_dir}")

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor]]:
        order = list(range(len(self.chunk_paths)))
        random.Random(self.seed + self.epoch).shuffle(order)
        for chunk_index in order:
            with np.load(self.chunk_paths[chunk_index]) as data:
                fens = data["fens"]
                targets = np.asarray(data[self.target_field], dtype=np.float32)
                if self.target == "cp":
                    targets = np.clip(targets, -CP_CLIP, CP_CLIP) / CP_OUTPUT_SCALE
                row_rng = np.random.default_rng(
                    self.seed + self.epoch * len(self.chunk_paths) + chunk_index
                )
                for row_index in row_rng.permutation(len(fens)):
                    yield training_sample(
                        str(fens[row_index]), float(targets[row_index]), self.feature_set
                    )

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

    def accumulate(self, features):
        mask = features >= 0
        safe_features = features.clamp_min(0)
        active_weights = self.ft_weight.T[safe_features]
        return (active_weights * mask.unsqueeze(-1)).sum(dim=1) + self.ft_bias

    def forward(self, w_feat, b_feat, stm):
        w_acc = self.accumulate(w_feat)
        b_acc = self.accumulate(b_feat)
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

def dataset_fingerprint(chunk_dir: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    manifest = chunk_dir / "manifest.json"
    if manifest.exists():
        digest.update(manifest.read_bytes())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_model(target: str = "wdl", feature_set: str = "simple",
                output: Path | None = None, seed: int = 19,
                chunk_dir: Path = Path("dataset_chunks"), epochs: int = 5,
                batch_size: int = 256, checkpoint: Path | None = None,
                resume: bool = False):
    suffix = "_king_bucket" if feature_set == "king-bucket" else ""
    default_name = f"nnue_model{'_cp' if target == 'cp' else ''}{suffix}.pt"
    output = output or Path("weights") / default_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {output}")
    checkpoint = checkpoint or output.with_suffix(".checkpoint.pt")
    if checkpoint.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}; pass --resume or choose new paths"
        )
    if resume and not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    torch.manual_seed(seed)
    dataset = ChunkedChessNNUEDataset(chunk_dir, target, feature_set, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    feature_dim = KING_BUCKET_FEATURE_DIM if feature_set == "king-bucket" else BASE_FEATURE_DIM
    model = NNUE(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = (nn.MSELoss() if target == "wdl"
                 else nn.HuberLoss(delta=CP_HUBER_DELTA / CP_OUTPUT_SCALE))
    config = {
        "schema_version": 1,
        "target": target,
        "feature_set": feature_set,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_fingerprint": dataset_fingerprint(chunk_dir, dataset.chunk_paths),
        "cp_output_scale": CP_OUTPUT_SCALE,
        "cp_clip": CP_CLIP,
        "cp_huber_delta": CP_HUBER_DELTA,
        "feature_dim": feature_dim,
        "hidden_dim": 256,
    }
    start_epoch = 0
    if resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("config") != config:
            raise ValueError("Checkpoint configuration or dataset differs from this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["completed_epochs"])
        print(f"Resuming after epoch {start_epoch}/{epochs} from {checkpoint}")

    print(
        f"Training NNUE with {feature_set} features and {target} targets "
        f"(seed {seed}, {len(dataset):,} positions)..."
    )
    batches = math.ceil(len(dataset) / batch_size)
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        total_loss = 0.0
        for batch_idx, (w_vec, b_vec, stm, targets) in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                print(f"  Batch {batch_idx}/{batches}")
            w_vec, b_vec = w_vec.to(device), b_vec.to(device)
            stm, targets = stm.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(w_vec, b_vec, stm).squeeze(-1)
            preds = torch.sigmoid(outputs) if target == "wdl" else outputs
            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / batches
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} "
            f"- LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        atomic_torch_save(
            {
                "config": config,
                "completed_epochs": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint,
        )
        print(f"Checkpoint: {checkpoint}")

    model_cpu = model.cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(model_cpu.state_dict(), output)
    print(f"Saved float model to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("wdl", "cp"), default="wdl")
    parser.add_argument("--features", choices=("simple", "king-bucket"), default="simple")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--chunk-dir", type=Path, default=Path("dataset_chunks"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    train_model(
        args.target, args.features, args.output, args.seed, args.chunk_dir,
        args.epochs, args.batch_size, args.checkpoint, args.resume,
    )
