#!/usr/bin/env bash
# TIAGo++ Fetch Pick and Push (needs tiago-fetch-envs), seeds 0-2, then the final evaluation
# (50 episodes per run). Run from the repository root:  JOBS=6 bash experiments/fetch.sh
set -euo pipefail
JOBS="${JOBS:-1}"

for env in fetch_pick fetch_push; do for seed in 0 1 2; do
  echo "python -m qairl.train --env $env --seed $seed"
done; done | xargs -P "$JOBS" -I{} sh -c "{}"

for env in fetch_pick fetch_push; do for seed in 0 1 2; do
  python -m qairl.evaluate --run "runs/$env/s$seed"
done; done
