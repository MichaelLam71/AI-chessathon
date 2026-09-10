import argparse
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=Path("weights/nnue_model.pt"))
parser.add_argument("--output", type=Path, default=Path("weights/nnue_weights_np.npz"))
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite existing weights: {args.output}")

m = torch.load(args.model, map_location="cpu")
np.savez(args.output,
    ft_weight=m["ft_weight"].numpy().T,
    ft_bias=m["ft_bias"].numpy(),
    l1_weight=m["layer1.weight"].numpy(),
    l1_bias=m["layer1.bias"].numpy(),
    l2_weight=m["layer2.weight"].numpy().ravel(),
    l2_bias=m["layer2.bias"].numpy().item()
)
print(f"Saved {args.output}")
