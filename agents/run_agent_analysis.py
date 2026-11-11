"""Runs the agents' weights-only analysis (experiment.py) on data/weights.txt.

experiment.py expects the original prompt file, whose weight block starts on its fourth line.
This wrapper writes a stand-in with three placeholder lines followed by weights.txt, then runs it.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
weights = os.path.join(HERE, "..", "data", "weights.txt")
shim = os.path.join(HERE, "prompt_shim.txt")
with open(shim, "w") as f:
    f.write("placeholder\nlabelled example not included\nplaceholder\n")
    f.write(open(weights).read())
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
subprocess.run([sys.executable, os.path.join(HERE, "experiment.py"), shim, out], check=True)
