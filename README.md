# RVI Transformer — Research Prototype

A decoder-only transformer where **relations are promoted to explicit computational state**.

## Core idea

Ordinary attention:

\[
R_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_k}}
\]

RVI adds an explicit phase-sensitive relational field:

\[
I_{ij} =
\sum_r
w_r a_{i,r} a_{j,r}
\cos(\phi_{i,r}-\phi_{j,r}-\tau_r)
\]

and combines it with semantic attention:

\[
\widetilde R_{ij}
=
R_{ij}+\alpha I_{ij}
\]

The field is then recursively refined by a differentiable CTMS-inspired operator before it modifies attention.

## CTMS mapping

- **SEED** — hidden-state projection into amplitude/phase coordinates
- **BLOOM** — construct all pairwise relational edges
- **FORK** — preserve direct and composed relational paths
- **RECURSE** — compose relation matrices (`P @ P`)
- **REDCHECK** — measure disagreement between direct and recursively composed relations
- **FLATTEN** — return the refined field to attention logits
- **TRANSMIT** — attention transports values into the next hidden state

## Important distinction

This implementation does **not** claim dot-product attention is physical interference. It introduces a mathematically explicit interference term so that "interference" is an operational component which can be ablated.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```

## Train

Use any sufficiently large UTF-8 text corpus:

```bash
python train.py --text corpus.txt --steps 2000 --seq-len 256 --batch-size 8
```

On CUDA, optionally:

```bash
python train.py --text corpus.txt --compile
```

## Generate

```bash
python generate.py --checkpoint checkpoint.pt --prompt "The system"
```

## Test whether RVI matters

```bash
python ablate.py --text corpus.txt --steps 500
```

The harness trains:

1. a baseline decoder transformer with RVI disabled;
2. the RVI+CTMS version with the same seed and training data.

The first useful scientific result is not generation quality. It is whether explicit relational recursion changes held-out loss, convergence rate, calibration, long-range retrieval, or robustness under matched parameter/compute budgets.

## Next architecture step

The present prototype recomputes the relational state at each layer. The serious v2 should carry a **persistent cross-layer/cross-token relational memory** with its own cache, update rule, sparse topology and benchmark suite. That is where RVI becomes a genuinely separate architectural object rather than an attention augmentation.
