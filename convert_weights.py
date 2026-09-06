# Save this as convert_weights.py and run it once
import torch
import numpy as np

model_path = "weights/nnue_model.pt"
output_path = "weights/nnue_weights_np.npz"
m = torch.load(model_path, map_location="cpu")
np.savez(output_path,
    ft_weight=m["ft_weight"].numpy().T,
    ft_bias=m["ft_bias"].numpy(),
    l1_weight=m["layer1.weight"].numpy(),
    l1_bias=m["layer1.bias"].numpy(),
    l2_weight=m["layer2.weight"].numpy().ravel(),
    l2_bias=m["layer2.bias"].numpy().item()
)
print(f"Saved {output_path}")
