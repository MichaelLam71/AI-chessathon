# Save this as convert_weights.py and run it once
import torch
import numpy as np

m = torch.load("nnue_model.pt", map_location="cpu")
np.savez("nnue_weights_np.npz",
    ft_weight=m["ft_weight"].numpy().T,
    ft_bias=m["ft_bias"].numpy(),
    l1_weight=m["layer1.weight"].numpy(),
    l1_bias=m["layer1.bias"].numpy(),
    l2_weight=m["layer2.weight"].numpy().ravel(),
    l2_bias=m["layer2.bias"].numpy().item()
)
print("Saved nnue_weights_np.npz")