"""Loads the classifier weights, the 50 sample images, and the by-eye digit readings from data/."""
import os, csv, json
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_weights():
    """Parses data/weights.txt into W1 (128, 784), b1 (128,), W2 (128,) and b2 (scalar)."""
    blocks, cur = [], None
    for line in open(os.path.join(DATA, "weights.txt")):
        if line.startswith("#"):
            cur = []; blocks.append(cur)
        elif line.strip() and cur is not None:
            cur.append([float(v) for v in line.split()])
    W1, b1, W2, b2 = (np.array(b) for b in blocks)
    return W1, b1.ravel(), W2.ravel(), b2.ravel()[0]


def load_samples():
    """Returns the images as (50, 28, 28) raw intensities and their binary labels."""
    rows = list(csv.DictReader(open(os.path.join(DATA, "sample50.csv"))))
    X = np.array([json.loads(r["image"]) for r in rows], dtype=float)
    return X, np.array([int(r["label"]) for r in rows])


def load_digits():
    """Returns the by-eye digit reading of each sample, with -1 for the one unclear image."""
    rows = csv.DictReader(open(os.path.join(DATA, "digit_labels.csv")))
    return [int(r["digit"]) if r["digit"] != "unclear" else -1 for r in rows]
