# Recovering a Neural Network's Task from Its Weights

Code, data and prompts for the blog post
**[Can an LLM Read a Neural Network? Recovering a Classifier's Task from Its Weights](https://vandit-iisc.github.io/blog/2026/11/recovering-task-from-weights/)**
by Vandit Patel, Prayalankar and Prathosh AP.

We gave large language models the raw weights of a small MNIST classifier and asked what it was trained to distinguish.
The network separates **primes plus zero, {0, 2, 3, 5, 7}**, from **{1, 4, 6, 8, 9}**.

- A single call to **GPT-6 Astra** recognised handwritten digits but guessed parity (4/10 digits correct).
- **Claude agents** working blind concluded "0 vs. everything else".
- After being told that each class has five digits, four Claude agents (three subagents each) produced candidates within one or two digits of the truth. Agent C's top pick was exact; the averaged final answer was {0, 2, 3, 6, 8}.

## Repository layout

```
data/
  weights.txt          the trained 784 → 128 (ReLU) → 1 classifier, full precision
  sample50.csv         50 MNIST images (28×28 raw intensities) with binary labels
  digit_labels.csv     our by-eye reading of each sample's digit ("unclear" for one image)
astra/
  prompt.txt           the exact prompt sent to GPT-6 Astra (weights + one labelled example)
  responses/           the API responses
agents/
  four_agent_brief.md  the shared brief for the four-agent round, hint included
  experiment.py        the agents' weights-only analysis (never runs the network)
  run_agent_analysis.py  runs experiment.py on data/weights.txt
analysis/
  make_figures.py      builds every figure in the blog post into figures/
  region_erasure.py    prints the region weight-votes and erasure table
  load.py, regions.py  shared helpers
figures/               output of make_figures.py
```

## Running

```
pip install -r requirements.txt
python analysis/make_figures.py        # all 13 figures → figures/
python analysis/region_erasure.py      # region table
python agents/run_agent_analysis.py    # the agents' own analysis → agents/figures/
```

## The network

```
x = vec(X / 255)            X is a 28×28 grid of raw intensities, flattened row-major
h = ReLU(W1 x + b1)         W1: 128×784, b1: 128
z = W2 h + b2               W2: 1×128,   b2: scalar
class = 1 if z > 0 else 0   trained with BCEWithLogitsLoss
```

Flattened pixel `i` sits at row `i // 28`, column `i % 28`.

## Important caveats

- **Not a controlled comparison.** Astra and the agents differ in model, information and compute. Astra saw one labelled example; the agents did not. The agents were told "each class has five digits" and "plain MNIST" in their final round; Astra was not.
- **The agents never ran the network.** Their rules forbade forward passes, probing, and dot products with imagined digits. The forward-pass checks in `analysis/` are ours, done afterwards.
- **Digit readings are by eye.** `sample50.csv` has only binary labels. Every reading agrees with its label under the primes-plus-zero rule except one unclear image, which is excluded from per-digit results.

## Citation

If you use this, please cite the blog post above, and the *Weight-Space Programming* experiments by Takehi, Teoh, Zhang and Khattab (MIT CSAIL) that inspired it.

## License

Code is released under the MIT License (see `LICENSE`).
