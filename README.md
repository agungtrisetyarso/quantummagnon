# Supplementary materials

Code and data for "The price of isolation: sharp limits on non-reciprocal state transfer and mediated entanglement in hybrid magnonic networks" (A. Trisetyarso, submitted to Physical Review A).

## Requirements
Python 3.10+, numpy, scipy, sympy, matplotlib.

## Contents
| File | Purpose |
|---|---|
| `reproduce.py` | Reproduces every number and table in the article (sections S1–S8; about 2 min). Output: `reproduce_output.txt`. |
| `verify_symbolic.py` | SymPy verification of all analytic steps (Appendix B). Output: `verify_symbolic_output.txt`. |
| `fig_coop.py` | Generates Fig. 2 (`fig_coop.pdf`). Output: `fig_coop_output.txt`. |
| `fig_frontier.py` | Generates Fig. 3 (`fig_frontier.pdf`). |
| `fig_schematic.tex` | TikZ source of Fig. 1. |

## Usage
```
python3 reproduce.py > reproduce_output.txt
python3 verify_symbolic.py
python3 fig_coop.py
python3 fig_frontier.py
```
All random searches use fixed seeds, so outputs are reproducible.
