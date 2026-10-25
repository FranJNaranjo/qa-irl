#!/usr/bin/env bash
# Ablations of the paper, seeds 0-2. The full method is the benchmark run of the same seed
# (experiments/benchmark.sh). Temporal-order variants run on Pendulum and HalfCheetah only.
#   mlp      MLP distance head, no triangle inequality
#   nopush   no L_push and no budget constraint
#   nodir    lambda_dir = 0
#   nonce    alpha_0 = 0 (no L_NCE)
#   ncec     L_NCE with constant alpha (no annealing)
#   shuffled / reversed   order of the states within each demonstration
# Run from the repository root:  JOBS=4 bash experiments/ablations.sh
set -euo pipefail
JOBS="${JOBS:-1}"

{
for env in pendulum halfcheetah ant; do for seed in 0 1 2; do
  echo "python -m qairl.train --env $env --seed $seed --run_name mlp_s$seed --head mlp"
  echo "python -m qairl.train --env $env --seed $seed --run_name nopush_s$seed --no_push"
  echo "python -m qairl.train --env $env --seed $seed --run_name nodir_s$seed --lambda_dir 0"
  echo "python -m qairl.train --env $env --seed $seed --run_name nonce_s$seed --alpha_0 0"
  echo "python -m qairl.train --env $env --seed $seed --run_name ncec_s$seed --anneal_frac 1e6"
done; done
for env in pendulum halfcheetah; do for seed in 0 1 2; do
  echo "python -m qairl.train --env $env --seed $seed --run_name shuffled_s$seed --temporal_order shuffled"
  echo "python -m qairl.train --env $env --seed $seed --run_name reversed_s$seed --temporal_order reversed"
done; done
} | xargs -P "$JOBS" -I{} sh -c "{}"

for run in runs/{pendulum,halfcheetah,ant}/{mlp,nopush,nodir,nonce,ncec,shuffled,reversed}_s*; do
  if [ -f "$run/hparams.json" ]; then python -m qairl.evaluate --run "$run"; fi
done
