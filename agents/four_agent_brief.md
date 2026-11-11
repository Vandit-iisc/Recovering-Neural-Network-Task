# Shared brief: decode a 5-vs-5 MNIST digit split from weights only

## Task
A 784->128 (ReLU)->1 MLP was trained with BCEWithLogits to separate class 1 from class 0 (class 1 iff z>0).
Goal: determine WHICH 5 digits are class 1 and which 5 are class 0, purely from the weights.

## Established facts (from the user / verified)
- Each class contains exactly 5 MNIST digits (a 5-vs-5 split of 0-9). 126 possible splits (up to label swap: 252 labelled).
- Data is plain MNIST, un-augmented: exactly 67 pixels never moved from init (|w|<=1/28 in all units), matching MNIST's 67 always-zero training pixels; mask is not flip-symmetric.

## Hard rules (from README.md)
- Do NOT run the network on any input (real or synthetic). No forward passes, no probing, and do not compute dot products/correlations of weights with hand-drawn or imagined digit images (that is a forward pass in disguise).
- Do NOT view/print/plot line 2 of prompt-preview.txt (the labeled example). Only parse the weights.
- Analyzing and visualizing the WEIGHTS is fine; using your general knowledge of what MNIST digits look like to INTERPRET weight structure in words is fine.

## Data access
- Python with numpy + matplotlib.
- Loader: `from experiment import load; W1, b1, W2, b2 = load("prompt-preview.txt")` gives W1 (128,784), b1 (128,), W2 (128,), b2 (float). Pixel i -> row i//28, col i%28. x in [0,1].
- Dead units (still at init): 11, 38, 106, 114.
- Work in your own folder; you can Read PNGs you render.

## Output
Under ~600 words: your ranked top 3 candidate splits with probabilities, a per-digit confidence table (P(digit in class 1)), and the concrete numeric weight evidence. Say explicitly which digits you could NOT place confidently and why. Be self-critical.
