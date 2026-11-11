"""Prints the region table: the agents' weight votes next to what erasing each region actually does."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_weights, load_samples, load_digits
DIGITS = load_digits()
from regions import REGIONS, weight_votes, erase

W1, b1, W2, b2 = load_weights()
X, y = load_samples()
D = np.array(DIGITS)


def logit(X):
    """Runs the MLP on raw-intensity images and returns the logits."""
    return np.maximum(0, X.reshape(len(X), -1) / 255 @ W1.T + b1) @ W2 + b2


for name, (rows, cols) in REGIONS.items():
    v = weight_votes(W1, W2, rows, cols)
    dz, ink, flips = erase(logit, X, rows, cols)
    mean = f"{dz[ink].mean():+.1f}" if ink.any() else "—"
    ones = f" ({dz[ink & (D == 1)].mean():+.1f} for 1s)" if (ink & (D == 1)).any() else ""
    print(f"{name:20s} votes: cls1 {v['cls1_want']:.1f} want / {v['cls1_forbid']:.1f} forbid, "
          f"cls0 {v['cls0_want']:.1f} want / {v['cls0_forbid']:.1f} forbid, net {v['net']:+.1f} | "
          f"ink in {ink.sum()}/50, mean Δz {mean}{ones}, flips {flips}")
