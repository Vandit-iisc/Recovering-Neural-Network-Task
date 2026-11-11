# The Claude agent system

All agents and subagents ran on Claude Opus 5.5 in Claude Code, using Python (numpy, matplotlib) on the weights only.

## Rules

- No forward passes: no running the network on any real or synthetic input.
- Do not view the labelled example in the prompt file.
- No dot products between the weights and imagined digit images (a forward pass in disguise).
- Analysing, visualising and interpreting the weights is allowed.

## Timeline

1. **Round 1, one agent, no hints.** Found "slit" class-1 units and default-on class-0 units, and a ring-with-empty-centre lobe map. Concluded "class 1 is the digit 0, on rotated data" (55–65%).
2. **Round 2, independent subagent, no hints, not told round 1's answer.** Found exactly 67 untrained pixels (MNIST's always-blank pixels) and an asymmetric mask, so the data was not rotated. Concluded "0 vs. all other digits" (~65%).
3. **Hint from the experimenter:** each class contains five digits. One agent then proposed parity (~55%).
4. **Four main agents × three subagents (12 total)**, each main agent taking one angle, all from `four_agent_brief.md`:

   | Agent | Angle | Top pick (class 1) |
   |---|---|---|
   | A | training dynamics | {0, 2, 3, 5, 8} |
   | B | unit by unit, look-alike pairs | {0, 2, 3, 6, 8} |
   | C | hypothesis critic | {0, 2, 3, 5, 7}, the exact answer |
   | D | digit by digit | {0, 2, 3, 5, 6} |

   Averaging their per-digit probabilities gave the final answer {0, 2, 3, 6, 8} at about 25–30%.

## Files

- `four_agent_brief.md`: the shared brief for step 4. It states the hint and the plain-MNIST fact.
- `experiment.py`: the weights-only analysis behind steps 1–2 and the region numbers in the post. It expects the original prompt file layout.
- `run_agent_analysis.py`: runs `experiment.py` on `../data/weights.txt` and writes figures to `agents/figures/`.
- `transcripts/`: add agent and subagent outputs here if available.
