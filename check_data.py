import numpy as np
import glob

# Check a few mirrored pairs
files = sorted(glob.glob("dataset_chunks/chunk_*.npz"))
data = np.load(files[0])
fens = data["fens"]
wdls = data["wdls"]

for i in range(0, 10, 2):
    print(f"Original:  {fens[i]}  WDL={wdls[i]:.4f}")
    print(f"Mirrored:  {fens[i+1]}  WDL={wdls[i+1]:.4f}")
    print()