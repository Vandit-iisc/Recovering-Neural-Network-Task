"""Generates every figure in the post from data/weights.txt, data/sample50.csv and the agents' reported results."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_weights, load_samples, load_digits
DIGITS = load_digits()
from regions import REGIONS, weight_votes, erase

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# Shared style: class 1 is blue, class 0 is orange; text stays neutral.
C1, C0, INK, MUTED, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
DIV = LinearSegmentedColormap.from_list("div", [C0, "#f0efec", C1])        # class 0 <- neutral -> class 1
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f5f2", "#86b6ef", "#1c5cab", "#0d366b"])
WMAP = "RdGy_r"                                                            # weights: grey negative, red positive

W1, b1, W2, b2 = load_weights()
X, y = load_samples()
D = np.array(DIGITS)
CLASS1 = (0, 2, 3, 5, 7)
TRUTH = np.array([d in CLASS1 for d in range(10)], float)
x = X.reshape(50, -1) / 255
A = x @ W1.T + b1                      # pre-activations a_j(x)
S = A > 0                              # gates s_j(x)
C = np.maximum(0, A) * W2              # contributions c_j(x)
z = C.sum(1) + b2
G = (S * W2) @ W1                      # exact input gradient of z, per image
T = W2 @ W1                            # the agents' all-units-on evidence map


def logit(Xr):
    """Runs the MLP on raw-intensity images and returns the logits."""
    return np.maximum(0, Xr.reshape(len(Xr), -1) / 255 @ W1.T + b1) @ W2 + b2


def save(fig, name):
    """Saves a figure as a high-dpi PNG and closes it."""
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    """Hides the top and right spines and adds a light horizontal grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def class_strip(ax, rows, score_title="correct"):
    """Draws rows of ten class-coloured digit tiles, outlining tiles that disagree with the truth."""
    n = len(rows)
    for r, (name, cls1) in enumerate(rows):
        yy = n - 1 - r
        for d in range(10):
            ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc=C1 if d in cls1 else C0, ec="none"))
            ax.text(d + 0.5, yy + 0.5, str(d), ha="center", va="center", color="white", weight="bold")
            if (d in cls1) != (d in CLASS1):
                ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc="none", ec=INK, lw=2.2, ls="--"))
        score = sum((d in cls1) == (d in CLASS1) for d in range(10))
        ax.text(10.25, yy + 0.5, "—" if r == 0 else f"{score}/10", va="center")
    ax.set_yticks([n - 0.5 - r for r in range(n)], [nm for nm, _ in rows])
    ax.set_xticks([]); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0, 11.2); ax.set_ylim(0, n)
    ax.text(10.25, n + 0.15, score_title, fontsize=10, color=MUTED)
    for i, (fc, ec, ls, lab) in enumerate([(C1, "none", "-", "class 1"), (C0, "none", "-", "class 0"),
                                           ("none", INK, "--", "disagrees with the truth")]):
        ax.add_patch(Rectangle((i * 2 + 0.05, -0.75), 0.4, 0.4, fc=fc, ec=ec, ls=ls, lw=2, clip_on=False))
        ax.text(i * 2 + 0.55, -0.55, lab, va="center")


# ---------------------------------------------------------------- f01: the classifier
fig, ax = plt.subplots(figsize=(10, 2.2))
boxes = [("28×28 image\nX ∈ [0,255]", "#eeeeec"), ("x = vec(X/255)\n∈ ℝ⁷⁸⁴", "#eeeeec"),
         ("h = ReLU(W₁x + b₁)\n128 hidden units", "#e1ecfa"), ("z = W₂h + b₂\nraw logit", "#e1ecfa"),
         ("ŷ = 1{z > 0}", "#eeeeec")]
for i, (txt, col) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((i * 2.3, 0), 2.1, 1, boxstyle="round,pad=0.05,rounding_size=0.12", fc=col, ec=MUTED))
    ax.text(i * 2.3 + 1.05, 0.5, txt, ha="center", va="center", fontsize=10)
    if i < 4:
        ax.annotate("", xy=(i * 2.3 + 2.27, 0.5), xytext=(i * 2.3 + 2.13, 0.5),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
ax.text(5.65, -0.35, "100,609 parameters  θ = {W₁, b₁, W₂, b₂}", ha="center", color=MUTED, fontsize=10)
ax.set_xlim(-0.2, 11.4); ax.set_ylim(-0.6, 1.15); ax.axis("off")
save(fig, "f01_pipeline.png")


# ---------------------------------------------------------------- f02: the agents' lobe maps (the "ring")
live = np.linalg.norm(W1, axis=1) >= 1


def lobes(sign, vsign, q=0.45):
    """Reproduces the agents' strong-lobe map: vote-weighted sum of each unit's strongest same-sign weights."""
    acc = np.zeros(784)
    for i in np.where(live & (np.sign(W2) == vsign) & (np.abs(W2) > 0.25))[0]:
        v = sign * W1[i]
        acc += np.abs(W2[i]) * np.where(v > q * v.max(), v, 0)
    return acc.reshape(28, 28)


panels = [(lobes(-1, 1), "class-1 units:\nink must be ABSENT here"), (lobes(1, 1), "class-1 units:\nink WANTED here"),
          (lobes(-1, -1), "class-0 units:\nink switches them OFF"), (lobes(1, -1), "class-0 units:\nink WANTED here")]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
for ax, (M, t) in zip(axs, panels):
    ax.imshow(M, cmap=SEQ); ax.set_title(t, fontsize=10); ax.axis("off")
save(fig, "f02_lobes.png")


# ---------------------------------------------------------------- f03: agents' probabilities vs the truth
P = {"Agent A": [.75, .10, .70, .85, .25, .75, .45, .15, .55, .25],
     "Agent B": [.95, .20, .75, .80, .10, .08, .85, .10, .55, .15],
     "Agent C": [.58, .07, .90, .92, .08, .52, .40, .64, .52, .34],
     "Agent D": [.80, .15, .60, .65, .40, .45, .60, .12, .35, .25]}
P["Mean of four"] = list(np.mean(list(P.values()), 0))
fig, ax = plt.subplots(figsize=(9.5, 3.9))
names = ["Truth"] + list(P)
for r, nm in enumerate(names):
    yy = len(names) - 1 - r
    p = TRUTH if nm == "Truth" else np.array(P[nm])
    for d in range(10):
        if nm == "Truth":
            ax.add_patch(Rectangle((d + .04, yy + .06), .92, .88, fc=C1 if TRUTH[d] else C0))
            ax.text(d + .5, yy + .5, f"{d}", ha="center", va="center", color="white", weight="bold")
            continue
        ax.add_patch(Rectangle((d + .04, yy + .06), .92, .88, fc=DIV(p[d]), ec="none"))
        ax.text(d + .5, yy + .5, f"{p[d]:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if abs(p[d] - .5) > .3 else INK)
        if (p[d] > .5) != bool(TRUTH[d]):
            ax.add_patch(Rectangle((d + .04, yy + .06), .92, .88, fc="none", ec=INK, lw=2, ls="--"))
    if nm != "Truth":
        ax.text(10.3, yy + .5, f"{((p - TRUTH) ** 2).mean():.3f}", va="center")
ax.text(10.3, len(names) - 0.4 + 0.25, "Brier", fontsize=10, color=MUTED)
ax.set_yticks([len(names) - .5 - r for r in range(len(names))], names)
ax.set_xticks([]); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlim(0, 11.2); ax.set_ylim(0, len(names) + .2)
ax.text(0, -0.45, "cell = P(digit in class 1); blue leans class 1, orange leans class 0; "
                  "dashed = wrong side of 0.5;  Brier: 0 is perfect, 0.25 is a coin flip", fontsize=9, color=MUTED)
save(fig, "f03_agent_probs.png")


# ---------------------------------------------------------------- f04: the scoreboard of every answer
rows = [("Ground truth (primes + 0)", set(CLASS1)), ("Astra, one call (no hint)", {0, 2, 4, 6, 8}),
        ("Blind agents (no hint)", {0}), ("One agent + hint", {0, 2, 4, 6, 8}),
        ("Agent A top pick", {0, 2, 3, 5, 8}), ("Agent B top pick", {0, 2, 3, 6, 8}),
        ("Agent C top pick", {0, 2, 3, 5, 7}), ("Agent D top pick", {0, 2, 3, 5, 6}),
        ("Mean of four (final)", {0, 2, 3, 6, 8})]
fig, ax = plt.subplots(figsize=(9.5, 5))
class_strip(ax, rows)
save(fig, "f04_scoreboard.png")


# ---------------------------------------------------------------- f05: frozen pixels and dead units
frozen = (np.abs(W1) <= 1 / 28).all(0).reshape(28, 28)
norms = np.linalg.norm(W1, axis=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.6]})
a1.imshow(X.mean(0), cmap="Greys", alpha=0.55)
fy, fx = np.where(frozen)
a1.scatter(fx, fy, s=18, marker="s", color=C0, label=f"{frozen.sum()} frozen pixels")
a1.set_title("Pixels whose weights never left\ntheir initial range (over the mean digit)", fontsize=10)
a1.set_xticks([]); a1.set_yticks([]); a1.legend(loc="lower center", bbox_to_anchor=(.5, -.18), frameon=False)
order = np.argsort(norms)
cols = [C0 if norms[j] < 1 else "#8a8984" for j in order]
a2.bar(range(128), norms[order], color=cols, width=0.8)
a2.axhline(np.sqrt(784 / 28 ** 2 / 3), color=INK, ls="--", lw=1, label="expected norm at initialisation (0.58)")
a2.legend(frameon=False, loc="upper left", fontsize=9)
a2.annotate("units " + ", ".join(str(j) for j in sorted(order[:4])), xy=(1.5, .65), xytext=(4, 4.0),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C0))
a2.set_xlabel("hidden units, sorted by weight norm"); a2.set_ylabel("‖w_j‖")
a2.set_title("Four units never moved from initialisation (orange)", fontsize=10)
clean(a2)
save(fig, "f05_frozen.png")


# ---------------------------------------------------------------- f06: the "slit" detectors, labelled by what they fire on
gallery = [(28, 3), (61, 3), (69, 3), (63, 2), (86, 2), (100, 2), (46, 2), (13, 7), (30, 7), (102, 7)]
fig, axs = plt.subplots(2, 5, figsize=(11, 5))
for ax, (j, d) in zip(axs.flat, gallery):
    t = W1[j].reshape(28, 28); m = abs(t).max()
    ax.imshow(t, cmap=WMAP, vmin=-m, vmax=m)
    ax.contour(X[D == d].mean(0) / 255, levels=[0.35], colors=[C1], linewidths=1.6)
    ax.set_title(f"unit {j} · fires on {d}s\nv = {W2[j]:+.2f}", fontsize=10); ax.axis("off")
fig.text(.5, .01, "weights: red = ink switches the unit on, grey = ink switches it off;  "
                  "blue outline = average shape of the digit the unit fires on", ha="center", fontsize=9, color=MUTED)
save(fig, "f06_slits.png")


# ---------------------------------------------------------------- f07: gating — how many units each image uses
corr_img = np.array([np.corrcoef(T, g)[0, 1] for g in G])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
a1.hist(S.sum(1), bins=range(20, 70, 3), color="#8a8984", ec=SURF)
a1.set_xlabel("active hidden units (of 128)"); a1.set_ylabel("images")
a1.set_title(f"Each image switches on {S.sum(1).mean():.0f} units on average", fontsize=10)
clean(a1)
a2.hist(corr_img, bins=np.linspace(.5, 1, 21), color="#8a8984", ec=SURF)
c_avg = np.corrcoef(T, G.mean(0))[0, 1]
a2.axvline(c_avg, color=C1, lw=2)
a2.set_xlabel("correlation of the all-units-on map T with one image's true gradient")
a2.set_title(f"Per image: mean {corr_img.mean():.2f}.  Blue line: vs the average gradient ({c_avg:.2f})", fontsize=10)
clean(a2)
save(fig, "f07_gating.png")


# ---------------------------------------------------------------- f08: a straight path between two digits has kinks
ia, ib = 45, 6                                        # a 7 and a 1
ts = np.linspace(0, 1, 4001)
Xp = (1 - ts)[:, None] * x[ia] + ts[:, None] * x[ib]
Ap = Xp @ W1.T + b1
zp = np.maximum(0, Ap) @ W2 + b2
flips = np.where((Ap[1:] > 0) != (Ap[:-1] > 0))[0]
fig = plt.figure(figsize=(10, 4.2))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1], width_ratios=[1, 4.2, 1], hspace=.35, wspace=.3)
ax = fig.add_subplot(gs[0, 1])
ax.plot(ts, zp, color=INK, lw=2)
ax.axhline(0, color=MUTED, ls="--", lw=1)
ax.scatter(ts[flips], zp[flips], s=14, color=C0, zorder=3, label=f"{len(flips)} kinks (a unit switches on or off)")
ax.set_xlabel("t   (x(t) = (1−t)·x₇ + t·x₁)"); ax.set_ylabel("logit z")
ax.legend(frameon=False, loc="upper right"); clean(ax)
for k, (i, lab) in enumerate([(ia, "t = 0: a 7"), (ib, "t = 1: a 1")]):
    a = fig.add_subplot(gs[0, 0 if k == 0 else 2]); a.imshow(X[i], cmap="Greys"); a.axis("off"); a.set_title(lab, fontsize=10)
ax2 = fig.add_subplot(gs[1, 1], sharex=ax)
ax2.plot(ts, (Ap > 0).sum(1), color="#8a8984", lw=1.5); ax2.set_ylabel("active\nunits"); clean(ax2)
plt.setp(ax.get_xticklabels(), visible=False); ax2.set_xlabel(ax.get_xlabel()); ax.set_xlabel("")
save(fig, "f08_kinks.png")


# ---------------------------------------------------------------- f09: margins by digit
fig, ax = plt.subplots(figsize=(8.5, 3.6))
rng = np.random.default_rng(0)
for d in range(10):
    zd = z[D == d]
    ax.scatter(d + rng.uniform(-.15, .15, len(zd)), zd, s=46, color=C1 if d in CLASS1 else C0, ec=SURF, lw=1.2, zorder=3)
ax.axhline(0, color=INK, lw=1, ls="--")
ax.text(9.6, .6, "decision boundary z = 0", ha="right", fontsize=9, color=MUTED)
ax.set_xticks(range(10)); ax.set_xlabel("digit (read by eye)"); ax.set_ylabel("logit z(x)")
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7) for c in (C1, C0)]
ax.legend(h, ["class 1 digit", "class 0 digit"], frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.12))
clean(ax)
save(fig, "f09_margins.png")


# ---------------------------------------------------------------- f10: exact pixel attribution per digit
attr = (x * G).reshape(50, 28, 28)
lim = max(abs(attr[D == d].mean(0)).max() for d in range(10))
fig, axs = plt.subplots(2, 5, figsize=(10, 4.6))
for d, ax in enumerate(axs.flat):
    im = ax.imshow(attr[D == d].mean(0), cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_title(f"{d}  (class {int(d in CLASS1)})", fontsize=11); ax.axis("off")
cb = fig.colorbar(im, ax=axs, shrink=.8, pad=.02)
cb.set_label("← pushes to class 0      pushes to class 1 →", fontsize=9)
save(fig, "f10_attribution.png")


# ---------------------------------------------------------------- f11: one decision, unit by unit
iw = 39                                               # a 3
c = C[iw]; on = np.where(c != 0)[0]
pos = on[np.argsort(-c[on])][:6]; neg = on[np.argsort(c[on])][:5]
rest = sum(c[j] for j in on if j not in set(pos) | set(neg))
steps = [("output bias b₂", b2)] + [(f"unit {j}", c[j]) for j in pos] + [(f"unit {j}", c[j]) for j in neg] + \
        [(f"{len(on) - 11} other active units", rest)]
fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, .55, 3])
ai = fig.add_subplot(gs[0, 0]); ai.imshow(X[iw], cmap="Greys"); ai.axis("off")
ai.set_title(f"input: a 3\n{len(on)} of 128 units active", fontsize=10)
ax = fig.add_subplot(gs[0, 2])
run = 0
for k, (lab, v) in enumerate(steps):
    ax.barh(k, v, left=run, color=C1 if v > 0 else C0, height=.62)
    ax.text(max(run, run + v) + .25, k, f"{v:+.2f}", va="center", fontsize=9)
    run += v
ax.barh(len(steps), run, color=INK, height=.62); ax.text(run + .25, len(steps), f"z = {run:.2f}", va="center", weight="bold")
ax.set_yticks(range(len(steps) + 1), [s for s, _ in steps] + ["logit"]); ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=1); ax.set_xlabel("running total of the logit")
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "f11_waterfall.png")
assert abs(run - z[iw]) < 1e-6


# ---------------------------------------------------------------- f12: regions — what the weights say vs what erasing does
names = list(REGIONS)
net = [weight_votes(W1, W2, *REGIONS[n])["net"] for n in names]
er = [erase(logit, X, *REGIONS[n]) for n in names]
evid = [-dz[ink].mean() if ink.any() else 0 for dz, ink, _ in er]
fig = plt.figure(figsize=(13.5, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.15], wspace=.85)
am = fig.add_subplot(gs[0, 0])
am.imshow(X.mean(0), cmap="Greys", alpha=.5)
for n, v in zip(names, net):
    (r0, r1), (c0, c1) = REGIONS[n]
    col = C1 if v > 0 else C0
    am.add_patch(Rectangle((c0 - .5, r0 - .5), c1 - c0 + 1, r1 - r0 + 1, fc=col, alpha=.3, ec=col, lw=2))
    am.text((c0 + c1) / 2, (r0 + r1) / 2, n[:2], ha="center", va="center", weight="bold", fontsize=11)
am.set_xticks([]); am.set_yticks([]); am.set_title("The agents' five regions\n(over the mean digit)", fontsize=10)
labels = [n for n in names]
for k, (vals, title, xl) in enumerate([(net, "What the weights say\n(agents' vote rule)", "net vote for class 1  (Σ|v_j|)"),
                                       (evid, "What erasing the ink does\n(forward passes, 50 images)", "−mean Δz on inked images")]):
    ax = fig.add_subplot(gs[0, k + 1])
    ax.barh(range(5), vals, color=[C1 if v > 0 else C0 for v in vals], height=.6)
    ax.axvline(0, color=MUTED, lw=1); ax.invert_yaxis()
    ax.set_yticks(range(5), labels if k == 0 else [f"ink in {ink.sum()}/50, {fl} flips" for _, ink, fl in er], fontsize=9)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xl, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "f12_regions.png")


# ---------------------------------------------------------------- f13: units named in the report, tested
units = [69, 17, 36, 39, 27, 23, 96, 73, 33]
M = np.array([[C[D == d, u].mean() for d in range(10)] for u in units])
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(M, cmap=LinearSegmentedColormap.from_list("s", ["#f6f5f2", C1]), aspect="auto")
for i in range(len(units)):
    for d in range(10):
        ax.text(d, i, f"{M[i, d]:.1f}" if M[i, d] >= .05 else "·", ha="center", va="center",
                fontsize=9, color="white" if M[i, d] > 1.2 else INK)
ax.set_xticks(range(10)); ax.set_xlabel("digit")
ax.set_yticks(range(len(units)), [f"unit {u}" for u in units])
ax.axhline(2.5, color=INK, lw=1); ax.axhline(4.5, color=INK, lw=1)
for yy, lab in [(1, "claimed 3-vs-8"), (3.5, "claimed to permit 7"), (6.5, "claimed to oppose 7's bar")]:
    ax.text(10.1, yy, lab, va="center", fontsize=9, color=MUTED)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "f13_named_units.png")

print(f"kinks {len(flips)}, waterfall z {z[iw]:.2f}, active per image {S.sum(1).mean():.1f}, "
      f"corr avg {c_avg:.2f}, corr per image {corr_img.mean():.2f}")
