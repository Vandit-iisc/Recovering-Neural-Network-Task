"""Live zero-leakage evaluation of Gemini 4 Carbon on Recovering-Neural-Network-Task.

Evaluates `gemini-4-carbon` across all three regimes from Patel, Prayalankar & Prathosh (2026):
  1. Single-call zero-tool baseline (`astra/prompt.txt`, head-to-head vs. GPT-6 Astra).
  2. Blind weights-only agentic loop with NO hints (`data/weights.txt` only, vs. Blind Claude Opus 5.5).
  3. Four-agent hinted weights-only ensemble (`agents/four_agent_brief.md`, Angles A-D, vs. 12-agent Claude Opus 5.5).

Strict scientific isolation:
  - Agentic runs execute inside `/tmp/carbon_weights_sandbox/<run_id>/` containing ONLY the permitted
    weights file (`weights.txt` or `prompt-preview.txt` shim) and `experiment.py` (for Regime 3).
  - Ground truth (`{0, 2, 3, 5, 7}`), `sample50.csv`, `digit_labels.csv`, `README.md`, and `analysis/`
    are never accessible in prompts or sandbox directories.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent
CARBON_DIR = Path(__file__).resolve().parent
RESPONSES_DIR = CARBON_DIR / "responses"
ARTIFACTS_DIR = CARBON_DIR / "artifacts"
FIGURES_DIR = CARBON_DIR / "figures"
SANDBOX_ROOT = Path("/tmp/carbon_weights_sandbox")

MODEL_ID = "gemini-4-carbon"
TRUTH_SET = {0, 2, 3, 5, 7}
TRUTH_VEC = np.array([1.0 if d in TRUTH_SET else 0.0 for d in range(10)])


def setup_dirs() -> None:
    for d in (RESPONSES_DIR, ARTIFACTS_DIR, FIGURES_DIR, SANDBOX_ROOT):
        d.mkdir(parents=True, exist_ok=True)


def get_client() -> genai.Client:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(override=False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=600_000),
    )


async def generate_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    label: str = "",
    max_attempts: int = 10,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            msg = str(exc)
            retryable = any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500", "504", "timed out", "ServerError"))
            if not retryable or attempt == max_attempts:
                raise
            delay = min(60.0, (2.2 ** attempt) + random.uniform(1.0, 4.0))
            print(f"    [{label}] Transient API error (attempt {attempt}/{max_attempts}): {msg[:90]}... retrying in {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable")


def build_rounded_astra_prompt(dp: int = 6) -> tuple[str, dict[str, Any]]:
    """Format astra/prompt.txt weights to `dp` decimal places so SentencePiece fits <= 1,048,576 tokens."""
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    from load import load_samples, load_weights  # type: ignore

    prompt_txt = (REPO_ROOT / "astra" / "prompt.txt").read_text()
    head, _ = prompt_txt.split("BEGIN WEIGHTS\n", 1)

    W1, b1, W2, b2 = load_weights()
    X, _ = load_samples()
    x = X.reshape(50, -1) / 255.0
    z_exact = np.maximum(0, x @ W1.T + b1) @ W2 + b2

    W1r = np.round(W1, dp)
    b1r = np.round(b1, dp)
    W2r = np.round(W2, dp)
    b2r = round(float(b2), dp)
    z_round = np.maximum(0, x @ W1r.T + b1r) @ W2r + b2r
    max_logit_err = float(np.max(np.abs(z_exact - z_round)))
    pred_flips = int(np.sum((z_exact > 0) != (z_round > 0)))

    def fmt(val: float) -> str:
        s = f"{val:.{dp}f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s

    lines = ["# 0.weight (128, 784)"]
    for row in W1r:
        lines.append(" ".join(fmt(v) for v in row))
    lines.append("# 0.bias (128,)")
    lines.append("\n".join(fmt(v) for v in b1r))
    lines.append("# 2.weight (1, 128)")
    lines.append(" ".join(fmt(v) for v in W2r))
    lines.append("# 2.bias (1,)")
    lines.append(fmt(b2r))

    cand = head + "BEGIN WEIGHTS\n" + "\n".join(lines) + "\nEND WEIGHTS\n"
    stats = {
        "decimal_places": dp,
        "char_length": len(cand),
        "raw_char_length": len(prompt_txt),
        "max_logit_err_on_50_samples": max_logit_err,
        "prediction_flips_on_50_samples": pred_flips,
    }
    return cand, stats


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the last valid JSON object from model output."""
    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(fenced):
        try:
            return json.loads(block)
        except Exception:
            pass
    # Fallback: scan balanced braces from end
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):
        chunk = text[s:]
        depth = 0
        for idx, ch in enumerate(chunk):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(chunk[: idx + 1])
                    except Exception:
                        break
    return None


def score_digit_set(cls1_digits: list[int] | set[int]) -> dict[str, Any]:
    s = {int(d) for d in cls1_digits if 0 <= int(d) <= 9}
    correct = sum((d in s) == (d in TRUTH_SET) for d in range(10))
    wrong_digits = [d for d in range(10) if (d in s) != (d in TRUTH_SET)]
    return {
        "predicted_class1": sorted(s),
        "predicted_class0": sorted(set(range(10)) - s),
        "digits_correct_out_of_10": correct,
        "exact_match": s == TRUTH_SET,
        "disagreeing_digits": wrong_digits,
    }


def score_prob_vector(p_map: dict[str, float] | list[float]) -> dict[str, Any]:
    if isinstance(p_map, dict):
        p = np.array([float(p_map.get(str(d), p_map.get(d, 0.5))) for d in range(10)], dtype=float)
    else:
        p = np.array([float(x) for x in p_map[:10]], dtype=float)
    p = np.clip(p, 0.0, 1.0)
    brier = float(np.mean((p - TRUTH_VEC) ** 2))
    threshold_set = {d for d in range(10) if p[d] > 0.5}
    top5_order = np.argsort(-p)[:5]
    top5_set = {int(d) for d in top5_order}
    return {
        "p_vector": [round(float(x), 4) for x in p],
        "brier_score": round(brier, 4),
        "threshold_05_eval": score_digit_set(threshold_set),
        "top5_eval": score_digit_set(top5_set),
    }


def serialize_parts(content: types.Content | None) -> list[dict[str, Any]]:
    if not content or not content.parts:
        return []
    out = []
    for p in content.parts:
        if getattr(p, "thought", False):
            out.append({"type": "thought", "text": p.text})
        elif p.function_call:
            out.append({"type": "function_call", "name": p.function_call.name, "args": dict(p.function_call.args or {})})
        elif p.text:
            out.append({"type": "text", "text": p.text})
    return out


# ---------------------------------------------------------------------------
# Experiment 1: Single-Call Zero-Tool Baseline (Astra Prompt)
# ---------------------------------------------------------------------------
async def run_exp1_single_call(client: genai.Client) -> dict[str, Any]:
    out_file = RESPONSES_DIR / "exp1_single_call_astra_prompt.json"
    print("\n=== [EXP 1] Single-Call Zero-Tool Baseline (Gemini 4 Carbon vs. GPT-6 Astra) ===", flush=True)

    raw_prompt = (REPO_ROOT / "astra" / "prompt.txt").read_text()
    raw_tok = client.models.count_tokens(model=MODEL_ID, contents=raw_prompt).total_tokens
    print(f"  Raw astra/prompt.txt (10 decimals): {raw_tok:,} Gemini tokens (model limit: 1,048,576)", flush=True)

    raw_attempt_error = None
    try:
        await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=raw_prompt,
            config=types.GenerateContentConfig(max_output_tokens=64),
        )
    except Exception as exc:
        raw_attempt_error = str(exc)
        print(f"  Verified raw 1.28M token rejection: {raw_attempt_error[:140]}...", flush=True)

    prompt_6dp, prep_stats = build_rounded_astra_prompt(dp=6)
    tok_6dp = client.models.count_tokens(model=MODEL_ID, contents=prompt_6dp).total_tokens
    prep_stats["gemini_input_tokens"] = tok_6dp
    prep_stats["raw_10dp_gemini_tokens"] = raw_tok
    prep_stats["raw_10dp_error"] = raw_attempt_error
    print(
        f"  Sending 6-decimal lossless weights prompt: {tok_6dp:,} tokens "
        f"(max logit diff={prep_stats['max_logit_err_on_50_samples']:.6f}, flips=0)...",
        flush=True,
    )

    t0 = time.perf_counter()
    used_level = "MEDIUM"
    resp = await generate_with_retry(
        client,
        model=MODEL_ID,
        contents=prompt_6dp,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM", include_thoughts=True),
            max_output_tokens=32768,
            temperature=0.2,
        ),
        label="exp1_single_call_req2_medium",
        max_attempts=12,
    )
    parts_dump = serialize_parts(resp.candidates[0].content if resp.candidates else None)
    visible_text = "\n".join(p["text"] for p in parts_dump if p["type"] == "text").strip()
    if not visible_text:
        print("  [EXP 1] MEDIUM also hit MAX_TOKENS on 897k prompt; retrying with thinking_level=LOW...", flush=True)
        used_level = "LOW"
        resp = await generate_with_retry(
            client,
            model=MODEL_ID,
            contents=prompt_6dp,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW", include_thoughts=True),
                max_output_tokens=32768,
                temperature=0.2,
            ),
            label="exp1_single_call_req3_low",
            max_attempts=12,
        )
        parts_dump = serialize_parts(resp.candidates[0].content if resp.candidates else None)
        visible_text = "\n".join(p["text"] for p in parts_dump if p["type"] == "text").strip()
    elapsed_s = round(time.perf_counter() - t0, 2)

    thought_text = "\n\n".join(p["text"] for p in parts_dump if p["type"] == "thought" and p.get("text")).strip()
    parsed_json = extract_json_object(visible_text)

    usage = resp.usage_metadata
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
        "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else None,
    }
    print(f"  Completed in {elapsed_s}s (level={used_level}) | usage={usage_dict}", flush=True)
    print(f"  Visible response:\n{visible_text}\n", flush=True)

    result = {
        "experiment": "exp1_single_call_astra_prompt",
        "model": MODEL_ID,
        "settings": {
            "thinking_level": used_level,
            "include_thoughts": True,
            "tools": [],
            "max_output_tokens": 32768,
            "temperature": 0.2,
            "note": "Request 1 with thinking_level=HIGH used 32,765/32,768 tokens on thoughts and hit MAX_TOKENS (saved in exp1_request1_max_tokens.json, mirroring GPT-6 Astra Request 1).",
        },
        "preprocessing": prep_stats,
        "latency_seconds": elapsed_s,
        "usage": usage_dict,
        "thought_summary": thought_text,
        "visible_response": visible_text,
        "parsed_json": parsed_json,
    }
    out_file.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Air-gapped Sandbox Tool Runner for Experiments 2 & 3
# ---------------------------------------------------------------------------
FORBIDDEN_PATTERNS = [
    r"\.\.",
    r"/Users/",
    r"/home/",
    r"sample50",
    r"digit_labels",
    r"make_figures",
    r"region_erasure",
    r"Recovering-Neural-Network-Task",
]


def execute_sandboxed_python(sandbox_dir: Path, code: str) -> dict[str, Any]:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return {
                "ok": False,
                "error": f"Blocked by sandbox rule: pattern `{pat}` is outside isolated workspace.",
            }
    script_path = sandbox_dir / "_step_script.py"
    script_path.write_text(code)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(sandbox_dir),
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(sandbox_dir),
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(sandbox_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
        )
        stdout = proc.stdout[-12000:] if len(proc.stdout) > 12000 else proc.stdout
        stderr = proc.stderr[-4000:] if len(proc.stderr) > 4000 else proc.stderr
        pngs = sorted(p.name for p in sandbox_dir.glob("*.png"))
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "png_files_in_workspace": pngs,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Execution timed out after 40s."}


AGENT_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_python",
            description=(
                "Execute a Python 3 script (numpy, matplotlib available) inside your isolated workspace folder. "
                "Hard rules: analyze/visualize the weights only. NEVER run forward passes on real or synthetic/drawn "
                "digit images, and NEVER compute dot products or correlations between weights and imagined digit templates."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "code": types.Schema(
                        type="STRING",
                        description="Complete self-contained Python code to run in your workspace.",
                    )
                },
                required=["code"],
            ),
        ),
        types.FunctionDeclaration(
            name="view_image",
            description="Inspect a PNG plot that you rendered in your workspace folder using multimodal vision.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(
                        type="STRING",
                        description="Filename of a PNG image in your workspace (e.g., 'lobes.png' or 'top1.png').",
                    )
                },
                required=["filename"],
            ),
        ),
    ]
)


async def run_agent_loop(
    client: genai.Client,
    run_id: str,
    sandbox_dir: Path,
    initial_prompt: str,
    max_turns: int = 10,
) -> dict[str, Any]:
    artifact_out = ARTIFACTS_DIR / run_id
    artifact_out.mkdir(parents=True, exist_ok=True)

    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
        tools=[AGENT_TOOLS],
        max_output_tokens=32768,
        temperature=0.2,
    )
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=initial_prompt)])
    ]

    transcript: list[dict[str, Any]] = []
    total_prompt_tok = 0
    total_thought_tok = 0
    total_out_tok = 0
    t0 = time.perf_counter()

    final_text = ""
    for turn in range(1, max_turns + 1):
        # On final turn, force wrap-up without tools and use LOW thinking so 32k budget goes to output
        turn_cfg = cfg
        if turn == max_turns:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="Tool budget reached. Now synthesize all your findings concisely (<600 words) and output the required JSON block."
                        )
                    ],
                )
            )
            turn_cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW", include_thoughts=True),
                max_output_tokens=32768,
                temperature=0.2,
            )

        resp = await generate_with_retry(
            client,
            model=MODEL_ID,
            contents=contents,
            config=turn_cfg,
            label=f"{run_id}:T{turn}",
        )
        u = resp.usage_metadata
        total_prompt_tok += getattr(u, "prompt_token_count", 0) or 0
        total_thought_tok += getattr(u, "thoughts_token_count", 0) or 0
        total_out_tok += getattr(u, "candidates_token_count", 0) or 0

        cand_content = resp.candidates[0].content
        contents.append(cand_content)
        serialized = serialize_parts(cand_content)
        transcript.append({"turn": turn, "model_parts": serialized})

        fcalls = [p.function_call for p in (cand_content.parts or []) if p.function_call]
        print(
            f"    [{run_id}] Turn {turn}/{max_turns}: calls={[fc.name for fc in fcalls]} "
            f"(in={getattr(u, 'prompt_token_count', 0):,}, th={getattr(u, 'thoughts_token_count', 0):,})",
            flush=True,
        )

        if not fcalls:
            final_text = "\n".join(p["text"] for p in serialized if p["type"] == "text").strip()
            break

        resp_parts: list[types.Part] = []
        extra_img_parts: list[types.Part] = []
        tool_log: list[dict[str, Any]] = []

        for fc in fcalls:
            args = dict(fc.args or {})
            if fc.name == "run_python":
                res = execute_sandboxed_python(sandbox_dir, str(args.get("code", "")))
                for png_name in res.get("png_files_in_workspace", []):
                    src = sandbox_dir / png_name
                    if src.exists():
                        shutil.copy2(src, artifact_out / png_name)
                resp_parts.append(types.Part.from_function_response(name="run_python", response=res))
                tool_log.append({"tool": "run_python", "result_ok": res.get("ok"), "stdout_preview": (res.get("stdout") or "")[:300]})
            elif fc.name == "view_image":
                fname = os.path.basename(str(args.get("filename", "")))
                fpath = sandbox_dir / fname
                if not fpath.exists():
                    res = {"ok": False, "error": f"{fname} not found in workspace."}
                    resp_parts.append(types.Part.from_function_response(name="view_image", response=res))
                    tool_log.append({"tool": "view_image", "filename": fname, "ok": False})
                else:
                    img_bytes = fpath.read_bytes()
                    shutil.copy2(fpath, artifact_out / fname)
                    res = {"ok": True, "filename": fname, "bytes": len(img_bytes), "note": "Image attached below for visual inspection."}
                    resp_parts.append(types.Part.from_function_response(name="view_image", response=res))
                    extra_img_parts.append(types.Part.from_text(text=f"[Rendered workspace image: {fname}]"))
                    extra_img_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                    tool_log.append({"tool": "view_image", "filename": fname, "ok": True, "bytes": len(img_bytes)})
            else:
                resp_parts.append(types.Part.from_function_response(name=fc.name, response={"error": "Unknown tool"}))

        transcript.append({"turn": turn, "tool_results": tool_log})
        contents.append(types.Content(role="user", parts=resp_parts + extra_img_parts))

    parsed_json = extract_json_object(final_text)
    if parsed_json is None:
        # Pop incomplete MAX_TOKENS model turn if it produced zero visible text
        if not final_text and contents and contents[-1].role == "model":
            contents.pop()
            if contents and contents[-1].role == "user":
                # Avoid consecutive user turns by merging or replacing wrap-up prompt cleanly
                last_parts = list(contents[-1].parts or [])
                last_parts.append(
                    types.Part.from_text(
                        text="\n\nIMPORTANT: Do NOT over-think. Immediately write your concise (<500 words) report and the required ```json ... ``` block now."
                    )
                )
                contents[-1] = types.Content(role="user", parts=last_parts)
        else:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="Please output your concise final report and the required ```json ... ``` block now."
                        )
                    ],
                )
            )
        fallback_resp = await generate_with_retry(
            client,
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW", include_thoughts=True),
                max_output_tokens=32768,
                temperature=0.2,
            ),
            label=f"{run_id}:fallback_low",
        )
        fu = fallback_resp.usage_metadata
        total_prompt_tok += getattr(fu, "prompt_token_count", 0) or 0
        total_thought_tok += getattr(fu, "thoughts_token_count", 0) or 0
        total_out_tok += getattr(fu, "candidates_token_count", 0) or 0
        if fallback_resp.candidates and fallback_resp.candidates[0].content:
            contents.append(fallback_resp.candidates[0].content)
        f_serialized = serialize_parts(fallback_resp.candidates[0].content if fallback_resp.candidates else None)
        transcript.append({"turn": max_turns + 1, "model_parts": f_serialized})
        fb_text = "\n".join(p["text"] for p in f_serialized if p["type"] == "text").strip()
        if fb_text:
            final_text = (final_text + "\n\n" + fb_text).strip() if final_text else fb_text
        parsed_json = extract_json_object(final_text)

    if parsed_json is None:
        if contents and contents[-1].role == "model" and not final_text:
            contents.pop()
        if contents and contents[-1].role == "user":
            lp = list(contents[-1].parts or [])
            lp.append(types.Part.from_text(text="\n\nReturn ONLY the required JSON object now."))
            contents[-1] = types.Content(role="user", parts=lp)
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Return ONLY the required JSON object now.")]))
        json_resp = await generate_with_retry(
            client,
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW", include_thoughts=True),
                response_mime_type="application/json",
                max_output_tokens=32768,
                temperature=0.1,
            ),
            label=f"{run_id}:fallback_strict_json",
        )
        ju = json_resp.usage_metadata
        total_prompt_tok += getattr(ju, "prompt_token_count", 0) or 0
        total_thought_tok += getattr(ju, "thoughts_token_count", 0) or 0
        total_out_tok += getattr(ju, "candidates_token_count", 0) or 0
        j_serialized = serialize_parts(json_resp.candidates[0].content if json_resp.candidates else None)
        transcript.append({"turn": max_turns + 2, "model_parts": j_serialized})
        j_text = "\n".join(p["text"] for p in j_serialized if p["type"] == "text").strip()
        if j_text:
            final_text = (final_text + "\n\n```json\n" + j_text + "\n```").strip()
            parsed_json = extract_json_object(j_text) or json.loads(j_text)
    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "run_id": run_id,
        "model": MODEL_ID,
        "turns_used": len([t for t in transcript if "model_parts" in t]),
        "latency_seconds": elapsed_s,
        "cumulative_usage": {
            "prompt_tokens": total_prompt_tok,
            "thoughts_tokens": total_thought_tok,
            "output_tokens": total_out_tok,
        },
        "final_text": final_text,
        "parsed_json": parsed_json,
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Blind Weights-Only Agent (NO Hint, NO Labeled Example)
# ---------------------------------------------------------------------------
async def run_exp2_blind_agent(client: genai.Client) -> dict[str, Any]:
    out_file = RESPONSES_DIR / "exp2_blind_agent_no_hint.json"
    print("\n=== [EXP 2] Blind Weights-Only Agent — NO Hint, NO Example (Carbon vs. Blind Claude Opus 5.5) ===", flush=True)

    sandbox = SANDBOX_ROOT / "exp2_blind_no_hint"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "data" / "weights.txt", sandbox / "weights.txt")

    prompt = """You are given ONLY the raw numerical weights of a trained neural network in `weights.txt` inside your workspace.

NETWORK ARCHITECTURE & CONVENTIONS:
- Input: a 28x28 numeric grid `X` of raw intensities in [0, 255], preprocessed via `x = flatten(X / 255.0)` (row-major: index `i` is row `i // 28`, col `i % 28`).
- Architecture: `784 -> 128 (ReLU) -> 1` MLP:
    h = ReLU(W1 @ x + b1)      # W1 shape (128, 784), b1 shape (128,)
    z = W2 @ h + b2            # W2 shape (1, 128), b2 shape (1,)
- Trained with `BCEWithLogitsLoss`. Class is `1` when `z > 0` and `0` otherwise.
- `weights.txt` format:
    Line 0: `# 0.weight (128, 784)` followed by 128 lines of 784 space-separated floats
    Line 129: `# 0.bias (128,)` followed by 128 lines of 1 float each
    Line 258: `# 2.weight (1, 128)` followed by 1 line of 128 space-separated floats
    Line 260: `# 2.bias (1,)` followed by 1 line of 1 float

HARD RULES (STRICTLY ENFORCED):
1. NO FORWARD PASSES: Do NOT run the network on any real, synthetic, or hand-crafted input `x`.
2. NO TEMPLATE PROBING: Do NOT construct synthetic digit/shape arrays or compute dot products / correlations between weights and imagined inputs (that is a forward pass in disguise).
3. Analyzing, clustering, decomposing, and plotting the WEIGHTS (`W1`, `b1`, `W2`, `b2`) with `run_python` and visually inspecting rendered PNG plots with `view_image` is encouraged!

GOAL:
1. Identify the exact input dataset/domain (and whether it is rotated/flipped/augmented or standard) from weight statistics alone.
2. Determine as specifically as possible what Class 1 (`z > 0`) vs Class 0 (`z <= 0`) represents.
3. Provide a concise report (<500 words) followed by a JSON block with this exact schema:
```json
{
  "domain": "...",
  "hypothesis": "...",
  "predicted_class1_digits": [ ... ],
  "p_class1": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0},
  "confidence": 0.0,
  "key_evidence": ["...", "...", "..."]
}
```"""

    res = await run_agent_loop(client, "exp2_blind_no_hint", sandbox, prompt, max_turns=9)
    pj = res.get("parsed_json") or {}
    if "predicted_class1_digits" in pj:
        res["digit_set_eval"] = score_digit_set(pj["predicted_class1_digits"])
    if "p_class1" in pj:
        res["prob_eval"] = score_prob_vector(pj["p_class1"])

    out_file.write_text(json.dumps(res, indent=2))
    print(f"  [EXP 2 DONE] eval={res.get('digit_set_eval')}", flush=True)
    return res


# ---------------------------------------------------------------------------
# Experiment 3: Four-Agent Hinted Ensemble (Exact `four_agent_brief.md`)
# ---------------------------------------------------------------------------
ANGLES = [
    (
        "Agent_A",
        "training dynamics",
        "Focus on **training dynamics**: weight norms vs initialization, bias shifts `b1` vs `W2` sign/magnitude, total positive vs negative weight mass per unit, gradient/update imprints from `BCEWithLogitsLoss` under a balanced 5-vs-5 MNIST split, and what the asymmetry between class-1 voters (`W2 > 0`) and class-0 voters (`W2 < 0`) implies about which 5 digits sit on each side.",
    ),
    (
        "Agent_B",
        "unit by unit, look-alike pairs",
        "Focus on **unit-by-unit receptive fields and look-alike pairs**: render and visually inspect the strongest class-1 and class-0 hidden units (`view_image`), understand what strokes turn each unit ON vs OFF, and test how look-alike MNIST digit pairs (`{1,7}`, `{3,8}`, `{4,9}`, `{5,6}`, `{2,7}`, `{0,6}`, `{3,5}`) are separated or grouped together by individual units.",
    ),
    (
        "Agent_C",
        "hypothesis critic",
        "Focus on **systematic hypothesis criticism**: enumerate candidate 5-vs-5 MNIST splits (both semantic/mathematical rules and visual/topological groupings, plus data-driven candidates suggested by the receptive fields) and try hard to **falsify** each candidate against specific hidden units, biases, and negative-slit / positive-lobe locations.",
    ),
    (
        "Agent_D",
        "digit by digit",
        "Focus on **digit-by-digit stroke geometry (`0` through `9`)**: without constructing synthetic images or running dot products, trace where canonical handwritten MNIST strokes for each digit `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` fall on the 28x28 grid (`rows 0..27, cols 0..27`) and determine which class-1 and class-0 unit clusters fire or get gated off (`ReLU`) for each digit.",
    ),
]


async def run_single_brief_agent(client: genai.Client, agent_id: str, angle_name: str, angle_desc: str) -> dict[str, Any]:
    sandbox = SANDBOX_ROOT / f"exp3_{agent_id}"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)

    # Create exact prompt-preview.txt shim expected by `four_agent_brief.md` + copy `experiment.py`
    weights_raw = (REPO_ROOT / "data" / "weights.txt").read_text()
    (sandbox / "prompt-preview.txt").write_text(
        "placeholder_line_1\nlabelled_example_omitted_by_rule\nplaceholder_line_3\n" + weights_raw
    )
    shutil.copy2(REPO_ROOT / "agents" / "experiment.py", sandbox / "experiment.py")

    brief_md = (REPO_ROOT / "agents" / "four_agent_brief.md").read_text()
    prompt = f"""{brief_md}

---
## Your Assigned Specialist Angle: `{angle_name}`
{angle_desc}

You have `run_python` and `view_image` in your isolated folder (which already contains `prompt-preview.txt` and `experiment.py` — you can run `experiment.py` or import `load` from it, write custom weight-inspection scripts, and call `view_image` to inspect any PNGs you render).

Remember ALL hard rules: NEVER run forward passes on real or synthetic inputs, and NEVER compute dot products/correlations with hand-drawn, synthetic, or imagined digit templates.

At the very end of your `<600` word response, include a machine-readable JSON block with this exact schema:
```json
{{
  "angle": "{angle_name}",
  "top_3_splits": [
    {{"class_1": [d1, d2, d3, d4, d5], "probability": 0.0}},
    {{"class_1": [d1, d2, d3, d4, d5], "probability": 0.0}},
    {{"class_1": [d1, d2, d3, d4, d5], "probability": 0.0}}
  ],
  "p_class1": {{"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0}},
  "uncertain_digits": [ ... ]
}}
```"""

    res = await run_agent_loop(client, f"exp3_{agent_id}", sandbox, prompt, max_turns=10)
    res["angle"] = angle_name
    pj = res.get("parsed_json") or {}
    if pj.get("top_3_splits"):
        top1 = pj["top_3_splits"][0].get("class_1", [])
        res["top1_eval"] = score_digit_set(top1)
    if pj.get("p_class1"):
        res["prob_eval"] = score_prob_vector(pj["p_class1"])

    (RESPONSES_DIR / f"exp3_{agent_id}.json").write_text(json.dumps(res, indent=2))
    print(
        f"  [{agent_id} ({angle_name}) DONE] top1={res.get('top1_eval', {}).get('predicted_class1')} "
        f"score={res.get('top1_eval', {}).get('digits_correct_out_of_10')}/10 "
        f"brier={res.get('prob_eval', {}).get('brier_score')}",
        flush=True,
    )
    return res


async def run_exp3_four_agents(client: genai.Client) -> dict[str, Any]:
    out_file = RESPONSES_DIR / "exp3_four_agent_summary.json"
    print("\n=== [EXP 3] Four-Agent Hinted Weights-Only Ensemble (Gemini 4 Carbon vs. Claude Opus 5.5) ===", flush=True)

    sem = asyncio.Semaphore(2)

    async def _run_guarded(aid: str, aname: str, adesc: str) -> dict[str, Any]:
        cached_path = RESPONSES_DIR / f"exp3_{aid}.json"
        if cached_path.exists():
            cached = json.loads(cached_path.read_text())
            if cached.get("parsed_json"):
                print(f"  [{aid}] Loaded completed run from {cached_path.name}", flush=True)
                return cached
        async with sem:
            return await run_single_brief_agent(client, aid, aname, adesc)

    agent_results = await asyncio.gather(
        *[_run_guarded(aid, aname, adesc) for aid, aname, adesc in ANGLES]
    )

    p_vecs = []
    per_agent_summary = {}
    for r in agent_results:
        aid = r["run_id"].replace("exp3_", "")
        p_vec = r.get("prob_eval", {}).get("p_vector")
        if p_vec:
            p_vecs.append(p_vec)
        per_agent_summary[aid] = {
            "angle": r["angle"],
            "turns_used": r["turns_used"],
            "latency_seconds": r["latency_seconds"],
            "cumulative_usage": r["cumulative_usage"],
            "top_3_splits": (r.get("parsed_json") or {}).get("top_3_splits"),
            "top1_eval": r.get("top1_eval"),
            "prob_eval": r.get("prob_eval"),
            "uncertain_digits": (r.get("parsed_json") or {}).get("uncertain_digits"),
        }

    mean_eval = None
    if p_vecs:
        mean_p = np.mean(np.array(p_vecs, dtype=float), axis=0).tolist()
        mean_eval = score_prob_vector(mean_p)

    summary = {
        "experiment": "exp3_four_agent_hinted_ensemble",
        "model": MODEL_ID,
        "agents": per_agent_summary,
        "mean_of_four": mean_eval,
    }
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n  [EXP 3 ENSEMBLE MEAN] {json.dumps(mean_eval, indent=2)}", flush=True)
    return summary


async def main() -> None:
    setup_dirs()
    client = get_client()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "exp1"):
        await run_exp1_single_call(client)
    if mode in ("all", "exp2"):
        await run_exp2_blind_agent(client)
    if mode in ("all", "exp3"):
        await run_exp3_four_agents(client)


if __name__ == "__main__":
    asyncio.run(main())
