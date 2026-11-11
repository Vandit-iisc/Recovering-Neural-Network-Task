import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

np.set_printoptions(precision=2, suppress=True, linewidth=200)


# ---------------------------------------------------------------- load
def load(path):
    lines = open(path).read().splitlines()          # line 2 (example) is skipped
    W1 = np.array([[float(v) for v in lines[i].split()] for i in range(4, 132)])
    b1 = np.array([float(lines[i]) for i in range(133, 261)])
    W2 = np.array([float(v) for v in lines[262].split()])
    b2 = float(lines[264])
    assert W1.shape == (128, 784) and b1.shape == (128,) and W2.shape == (128,)
    return W1, b1, W2, b2


def smooth(M, k=3):
    P = np.pad(M, k // 2, mode="edge")
    return sliding_window_view(P, (k, k)).mean((-1, -2))


def heat(ax, M, title):
    m = np.abs(M).max() or 1
    ax.imshow(M, cmap="RdBu_r", vmin=-m, vmax=m)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- steps
def step1_structure(W1, b1, W2, b2):
    print("== 1. Global structure")
    norms = np.linalg.norm(W1, axis=1)
    dead = norms < 1
    print(f"b2 = {b2:+.3f}")
    print(f"dead units (still at init norm ~0.6): {np.where(dead)[0].tolist()}")
    print(f"median row norm {np.median(norms):.2f} (init ~0.58) -> well trained")
    print(f"units voting class 1 (W2>0.1): {(W2 > 0.1).sum()}, class 0 (W2<-0.1): {(W2 < -0.1).sum()}")
    for name, m in [("class-1 voters", W2 > 0.3), ("class-0 voters", W2 < -0.3)]:
        print(f"  {name}: n={m.sum()}, mean bias {b1[m].mean():+.3f}, mean weight sum {W1[m].sum(1).mean():+.1f}")
    return ~dead


def step2_input_domain(W1):
    print("\n== 2. Input domain: pixels never moved from init (|w| <= 1/28 in all units)")
    untrained = (np.abs(W1) <= 1 / 28).all(0).reshape(28, 28)
    print(f"untrained pixels: {untrained.sum()} (MNIST train set has 67 always-zero pixels)")
    print(f"mask symmetric?  left-right {(untrained == untrained[:, ::-1]).all()}, "
          f"up-down {(untrained == untrained[::-1]).all()}, transpose {(untrained == untrained.T).all()}")


def step3_unit_maps(W1, b1, W2, live, out):
    print("\n== 3. Per-unit receptive fields -> units.png, top1.png, top0.png")
    order = [i for i in np.argsort(-W2) if live[i]]
    fig, ax = plt.subplots(8, 16, figsize=(24, 13))
    for k, a in enumerate(ax.flat):
        a.axis("off")
        if k < len(order):
            i = order[k]
            heat(a, W1[i].reshape(28, 28), f"u{i} v{W2[i]:+.2f} b{b1[i]:+.2f}")
    plt.tight_layout(); plt.savefig(f"{out}/units.png", dpi=60); plt.close()
    for name, sel in [("top1", np.argsort(-W2)[:15]), ("top0", np.argsort(W2)[:15])]:
        fig, ax = plt.subplots(3, 5, figsize=(15, 9.5))
        for a, i in zip(ax.flat, sel):
            heat(a, smooth(W1[i].reshape(28, 28)), f"u{i} vote{W2[i]:+.2f} b{b1[i]:+.2f}")
        plt.suptitle(f"{name}: red = ink turns unit on, blue = ink turns it off (3x3 smoothed)")
        plt.tight_layout(); plt.savefig(f"{out}/{name}.png", dpi=55); plt.close()


def step4_slit_profile(W1, W2):
    print("\n== 4. Cross-section through each top unit's negative stroke (x100)")
    yy, xx = np.mgrid[0:28, 0:28]
    for grp, sel in [("class 1", np.argsort(-W2)[:12]), ("class 0", np.argsort(W2)[:12])]:
        profs = []
        for i in sel:
            M = W1[i].reshape(28, 28); v = -M
            ys, xs = np.where(v > 0.5 * v.max()); w = v[ys, xs]
            my, mx = (ys * w).sum() / w.sum(), (xs * w).sum() / w.sum()
            ev, evec = np.linalg.eigh(np.cov(np.vstack([xs - mx, ys - my]), aweights=w))
            d, n = evec[:, 1], evec[:, 0]
            along = (xx - mx) * d[0] + (yy - my) * d[1]
            perp = (xx - mx) * n[0] + (yy - my) * n[1]
            L = 2 * np.sqrt(ev[1])
            profs.append([M[(np.abs(along) <= L) & (np.round(perp) == p)].mean()
                          if ((np.abs(along) <= L) & (np.round(perp) == p)).any() else np.nan
                          for p in range(-8, 9)])
        print(f"  {grp} mean:", " ".join(f"{v * 100:+4.0f}" for v in np.nanmean(profs, 0)))


def step5_lobes(W1, W2, live, out):
    print("\n== 5. Vote-weighted strong-lobe maps -> lobes.png")

    def lobes(sign, vsign, q=0.45):
        acc = np.zeros(784)
        for i in np.where(live & (np.sign(W2) == vsign) & (np.abs(W2) > 0.25))[0]:
            v = sign * W1[i]
            acc += np.abs(W2[i]) * np.where(v > q * v.max(), v, 0)
        return acc.reshape(28, 28)

    panels = [(lobes(-1, 1), "class-1 units: NEG lobes (must be empty)"),
              (lobes(1, 1), "class-1 units: POS lobes (want ink)"),
              (lobes(-1, -1), "class-0 units: NEG lobes"),
              (lobes(1, -1), "class-0 units: POS lobes")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    for a, (M, t) in zip(ax, panels):
        a.imshow(M, cmap="magma"); a.set_title(t, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{out}/lobes.png", dpi=55); plt.close()


def step6_evidence_maps(W1, W2):
    print("\n== 6. Vote-weighted evidence T = sum_i W2_i * W1_i  (+ = class 1)")
    T = (W2[:, None] * W1).sum(0)
    yy, xx = np.mgrid[0:28, 0:28]
    R = np.hypot(yy - 13.5, xx - 13.5).ravel()
    print("  radial profile:", " ".join(f"r{lo}-{lo + 2}:{T[(R >= lo) & (R < lo + 2)].mean():+.2f}"
                                        for lo in range(0, 18, 2)))
    T = T.reshape(28, 28)
    print("  3x3 block map (x10), rows down / cols across starting at 2:")
    for r in range(2, 26, 3):
        print(f"  r{r:2d}-{r + 2:2d} " + " ".join(f"{T[r:r + 3, c:c + 3].mean() * 10:+5.1f}" for c in range(2, 26, 3)))
    print("  cols:   " + " ".join(f"{c:2d}-{c + 2:2d}" for c in range(2, 26, 3)))


def step7_region_votes(W1, W2):
    print("\n== 7. Region want/forbid: sum |W2| of units whose mean weight in the region is >+0.02 / <-0.02")
    regions = {
        "far bottom-left  r19-24 c1-5":  (slice(19, 25), slice(1, 6)),
        "centre column    r8-20 c13-15": (slice(8, 21), slice(13, 16)),
        "very top right   r1-3 c15-22":  (slice(1, 4), slice(15, 23)),
        "left-mid band    r10-13 c2-15": (slice(10, 14), slice(2, 16)),
        "lower-left loop  r16-20 c7-11": (slice(16, 21), slice(7, 12)),
    }
    Wr = W1.reshape(128, 28, 28)
    print(f"  {'region':32s} {'cls1 want':>9s} {'cls1 forbid':>11s} {'cls0 want':>9s} {'cls0 forbid':>11s}")
    for name, (rs, cs) in regions.items():
        m = Wr[:, rs, cs].mean((1, 2))
        row = []
        for vs in (W2 > 0, W2 < 0):
            row += [np.abs(W2[vs & (m > 0.02)]).sum(), np.abs(W2[vs & (m < -0.02)]).sum()]
        print(f"  {name:32s} {row[0]:9.1f} {row[1]:11.1f} {row[2]:9.1f} {row[3]:11.1f}")


def step8_clusters(W1, W2, live, out):
    print("\n== 8. k-means clusters of each class's units -> clusters.png")
    S = np.array([smooth(W1[i].reshape(28, 28)).ravel() for i in range(128)])
    rng = np.random.default_rng(0)

    def kmeans(X, k, restarts=30, iters=200):
        best = None
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        for _ in range(restarts):
            C = X[rng.choice(len(X), k, replace=False)]
            for _ in range(iters):
                Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
                lab = np.argmax(Xn @ Cn.T, 1)
                C = np.array([X[lab == j].mean(0) if (lab == j).any() else X[rng.integers(len(X))] for j in range(k)])
            score = sum((Xn[lab == j] @ (C[j] / np.linalg.norm(C[j]))).sum() for j in range(k))
            if best is None or score > best[0]:
                best = (score, lab.copy(), C.copy())
        return best[1], best[2]

    fig, ax = plt.subplots(2, 6, figsize=(18, 6.5))
    for row, (name, sel, k) in enumerate([("class1", np.where(live & (W2 > 0.25))[0], 6),
                                          ("class0", np.where(live & (W2 < -0.25))[0], 5)]):
        lab, C = kmeans(S[sel] * np.abs(W2[sel])[:, None], k)
        for j in range(6):
            ax[row, j].axis("off")
            if j < k:
                heat(ax[row, j], C[j].reshape(28, 28), f"{name} cl{j} n={(lab == j).sum()}")
        print(f"  {name}: " + "; ".join(f"cl{j}={sel[lab == j].tolist()}" for j in range(k)))
    plt.tight_layout(); plt.savefig(f"{out}/clusters.png", dpi=55); plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(out, exist_ok=True)
    W1, b1, W2, b2 = load(sys.argv[1])
    live = step1_structure(W1, b1, W2, b2)
    step2_input_domain(W1)
    step3_unit_maps(W1, b1, W2, live, out)
    step4_slit_profile(W1, W2)
    step5_lobes(W1, W2, live, out)
    step6_evidence_maps(W1, W2)
    step7_region_votes(W1, W2)
    step8_clusters(W1, W2, live, out)
