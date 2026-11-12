"""Generates publication-grade comparison figures (scoreboard + per-digit Brier probability heatmap)
comparing Gemini 4 Carbon vs. GPT-6 Astra and Claude Opus 5.5."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

CARBON_DIR = Path(__file__).resolve().parent
RESP_DIR = CARBON_DIR / "responses"
FIG_DIR = CARBON_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C1, C0, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
})
DIV = LinearSegmentedColormap.from_list("div", [C0, "#f0efec", C1])

CLASS1 = {0, 2, 3, 5, 7}
TRUTH = np.array([1.0 if d in CLASS1 else 0.0 for d in range(10)])


def infer_digits_from_exp1(exp1: dict) -> set[int] | None:
    pj = exp1.get("parsed_json") or {}
    if "predicted_class1_digits" in pj and isinstance(pj["predicted_class1_digits"], list):
        return {int(x) for x in pj["predicted_class1_digits"] if str(x).isdigit()}
    if "carbon_parsed_class1" in exp1:
        return set(exp1["carbon_parsed_class1"])
    hyp = json.dumps(pj) + "\n" + exp1.get("visible_response", "")
    m = re.search(r"class\s*1[^.\n]*?\{([0-9,\s]+)\}", hyp, re.I)
    if m:
        return {int(x) for x in re.findall(r"\d", m.group(1))}
    low = hyp.lower()
    if "only 0" in low or "digit 0 vs" in low or "'0' vs" in low or "\"0\" vs" in low:
        return {0}
    if "even" in low:
        return {0, 2, 4, 6, 8}
    if "odd" in low:
        return {1, 3, 5, 7, 9}
    if "prime" in low and "0" in low:
        return {0, 2, 3, 5, 7}
    if "prime" in low:
        return {2, 3, 5, 7}
    if "< 5" in low or "<5" in low or "0, 1, 2, 3, 4" in low or "0-4" in low:
        return {0, 1, 2, 3, 4}
    if ">= 5" in low or ">=5" in low or "5, 6, 7, 8, 9" in low or "5-9" in low:
        return {5, 6, 7, 8, 9}
    return None


def build_scoreboard() -> None:
    rows: list[tuple[str, set[int]]] = [
        ("Ground truth (primes + 0)", set(CLASS1)),
        ("GPT-6 Astra · 1 call (no hint)", {0, 2, 4, 6, 8}),
    ]

    exp1_path = RESP_DIR / "exp1_single_call_astra_prompt.json"
    if exp1_path.exists():
        exp1 = json.loads(exp1_path.read_text())
        s1 = infer_digits_from_exp1(exp1)
        if s1 is not None:
            rows.append(("Gemini 4 Carbon · 1 call ('0' vs '1')", s1))

    rows.append(("Claude Opus 5.5 · blind agent", {0}))

    exp2_path = RESP_DIR / "exp2_blind_agent_no_hint.json"
    if exp2_path.exists():
        exp2 = json.loads(exp2_path.read_text())
        d2 = (exp2.get("digit_set_eval") or {}).get("predicted_class1")
        if d2 is not None:
            rows.append(("Gemini 4 Carbon · blind agent", set(d2)))

    rows.append(("Claude Opus 5.5 · 4x3 agents mean", {0, 2, 3, 6, 8}))

    exp3_path = RESP_DIR / "exp3_four_agent_summary.json"
    if exp3_path.exists():
        exp3 = json.loads(exp3_path.read_text())
        for aid, label in [
            ("Agent_A", "Carbon Agent A (training dynamics)"),
            ("Agent_B", "Carbon Agent B (unit look-alikes)"),
            ("Agent_C", "Carbon Agent C (hypothesis critic)"),
            ("Agent_D", "Carbon Agent D (digit-by-digit)"),
        ]:
            ag = exp3.get("agents", {}).get(aid, {})
            d_set = (ag.get("top1_eval") or {}).get("predicted_class1")
            if d_set is not None:
                rows.append((label, set(d_set)))
        m4 = (exp3.get("mean_of_four") or {}).get("top5_eval", {}).get("predicted_class1")
        if m4 is not None:
            rows.append(("Gemini 4 Carbon · Mean of four", set(m4)))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.56 * n + 1.1))
    for r, (name, cls1) in enumerate(rows):
        yy = n - 1 - r
        for d in range(10):
            ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc=C1 if d in cls1 else C0, ec="none"))
            ax.text(d + 0.5, yy + 0.5, str(d), ha="center", va="center", color="white", weight="bold")
            if (d in cls1) != (d in CLASS1):
                ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc="none", ec=INK, lw=2.1, ls="--"))
        score = sum((d in cls1) == (d in CLASS1) for d in range(10))
        ax.text(10.25, yy + 0.5, "—" if r == 0 else f"{score}/10", va="center", weight="bold" if "Carbon" in name else "normal")

    ax.set_yticks([n - 0.5 - r for r in range(n)], [nm for nm, _ in rows])
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, n)
    ax.text(10.25, n + 0.15, "correct", fontsize=10, color=MUTED)
    for i, (fc, ec, ls, lab) in enumerate([
        (C1, "none", "-", "class 1"),
        (C0, "none", "-", "class 0"),
        ("none", INK, "--", "disagrees with truth"),
    ]):
        ax.add_patch(Rectangle((i * 2.4 + 0.05, -0.75), 0.4, 0.4, fc=fc, ec=ec, ls=ls, lw=2, clip_on=False))
        ax.text(i * 2.4 + 0.55, -0.55, lab, va="center", fontsize=9.5)

    fig.savefig(FIG_DIR / "carbon_scoreboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_probs() -> None:
    claude_mean = [0.77, 0.13, 0.7375, 0.805, 0.2075, 0.45, 0.575, 0.2525, 0.4925, 0.2475]
    entries: list[tuple[str, list[float]]] = [
        ("Claude Opus 5.5 (Mean of 4)", claude_mean),
    ]

    exp2_path = RESP_DIR / "exp2_blind_agent_no_hint.json"
    if exp2_path.exists():
        exp2 = json.loads(exp2_path.read_text())
        pv = (exp2.get("prob_eval") or {}).get("p_vector")
        if pv:
            entries.append(("Carbon Blind (no hint)", pv))

    exp3_path = RESP_DIR / "exp3_four_agent_summary.json"
    if exp3_path.exists():
        exp3 = json.loads(exp3_path.read_text())
        for aid, label in [
            ("Agent_A", "Carbon Agent A (dynamics)"),
            ("Agent_B", "Carbon Agent B (unit pairs)"),
            ("Agent_C", "Carbon Agent C (critic)"),
            ("Agent_D", "Carbon Agent D (digit-by-digit)"),
        ]:
            pv = (exp3.get("agents", {}).get(aid, {}).get("prob_eval") or {}).get("p_vector")
            if pv:
                entries.append((label, pv))
        mp = (exp3.get("mean_of_four") or {}).get("p_vector")
        if mp:
            entries.append(("Carbon Mean of 4", mp))

    names = ["Truth"] + [k for k, _ in entries]
    p_lookup = {k: v for k, v in entries}
    n = len(names)
    fig, ax = plt.subplots(figsize=(10.5, 0.58 * n + 1.0))

    for r, nm in enumerate(names):
        yy = n - 1 - r
        p = TRUTH if nm == "Truth" else np.array(p_lookup[nm], dtype=float)
        for d in range(10):
            if nm == "Truth":
                ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc=C1 if TRUTH[d] else C0))
                ax.text(d + 0.5, yy + 0.5, f"{d}", ha="center", va="center", color="white", weight="bold")
                continue
            ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc=DIV(p[d]), ec="none"))
            ax.text(
                d + 0.5,
                yy + 0.5,
                f"{p[d]:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if abs(p[d] - 0.5) > 0.3 else INK,
            )
            if (p[d] > 0.5) != bool(TRUTH[d]):
                ax.add_patch(Rectangle((d + 0.04, yy + 0.06), 0.92, 0.88, fc="none", ec=INK, lw=2, ls="--"))
        if nm != "Truth":
            brier = float(np.mean((p - TRUTH) ** 2))
            ax.text(10.3, yy + 0.5, f"{brier:.3f}", va="center", weight="bold" if "Carbon" in nm else "normal")

    ax.text(10.3, n - 0.15, "Brier", fontsize=10, color=MUTED)
    ax.set_yticks([n - 0.5 - r for r in range(n)], names)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, n + 0.2)
    ax.text(
        0,
        -0.5,
        "cell = P(digit in class 1); blue leans class 1, orange leans class 0; dashed = wrong side of 0.5; Brier: lower is better",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(FIG_DIR / "carbon_probs_brier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_scoreboard()
    build_probs()
    print("Saved carbon_scoreboard.png and carbon_probs_brier.png")
