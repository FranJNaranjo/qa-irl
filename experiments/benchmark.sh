#!/usr/bin/env bash
# Benchmark of the paper: QA-IRL on the six Gymnasium tasks, seeds 0-4, then the final evaluation.
# Run from the repository root. JOBS sets how many runs go in parallel (default 1).
#   JOBS=4 bash experiments/benchmark.sh
set -euo pipefail
JOBS="${JOBS:-1}"
ENVS="pendulum inverted_pendulum inverted_double_pendulum hopper halfcheetah ant"

for env in $ENVS; do for seed in 0 1 2 3 4; do
  echo "python -m qairl.train --env $env --seed $seed"
done; done | xargs -P "$JOBS" -I{} sh -c "{}"

for env in $ENVS; do for seed in 0 1 2 3 4; do
  python -m qairl.evaluate --run "runs/$env/s$seed"
done; done
