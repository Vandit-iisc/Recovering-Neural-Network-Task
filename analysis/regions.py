"""Shared definitions: the agents' five image regions, their weight-vote rule, and the erasure test."""
import numpy as np

# Regions as (first row, last row), (first col, last col), inclusive, as used in the agents' experiment.py.
REGIONS = {
    "R1 far bottom-left": ((19, 24), (1, 5)),
    "R2 centre column": ((8, 20), (13, 15)),
    "R3 very top-right": ((1, 3), (15, 22)),
    "R4 left-mid band": ((10, 13), (2, 15)),
    "R5 lower-left loop": ((16, 20), (7, 11)),
}


def weight_votes(W1, W2, rows, cols, thr=0.02):
    """Reproduces the agents' rule: sums |v_j| of units whose mean weight in the region is above +thr or below -thr."""
    (r0, r1), (c0, c1) = rows, cols
    m = W1.reshape(-1, 28, 28)[:, r0:r1 + 1, c0:c1 + 1].mean((1, 2))
    out = {}
    for name, vs in (("cls1", W2 > 0), ("cls0", W2 < 0)):
        out[name + "_want"] = np.abs(W2[vs & (m > thr)]).sum()
        out[name + "_forbid"] = np.abs(W2[vs & (m < -thr)]).sum()
    # Net weight evidence for class 1: class-1 units wanting ink plus class-0 units forbidding it, minus the reverse.
    out["net"] = out["cls1_want"] + out["cls0_forbid"] - out["cls1_forbid"] - out["cls0_want"]
    return out


def erase(logit, X, rows, cols):
    """Zeroes a region in every image and returns (logit change, mask of images that had ink there, flips)."""
    (r0, r1), (c0, c1) = rows, cols
    Xe = X.copy()
    Xe[:, r0:r1 + 1, c0:c1 + 1] = 0
    z, ze = logit(X), logit(Xe)
    ink = X[:, r0:r1 + 1, c0:c1 + 1].sum((1, 2)) > 0
    return ze - z, ink, int(((ze > 0) != (z > 0)).sum())
