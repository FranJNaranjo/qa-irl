#!/usr/bin/env bash
# Synthetic experiments of the paper (CPU, a few minutes). Writes results/synthetic/*.json.
# Run from the repository root:  bash experiments/synthetic.sh
set -euo pipefail
python experiments/synthetic/gridworld.py
python experiments/synthetic/reversible.py
