"""
Evaluates the final policy of a run with the protocol of the paper.

    python -m qairl.evaluate --run runs/hopper/s0
    python -m qairl.evaluate --run runs/fetch_push/s0

Gymnasium tasks: 10 deterministic episodes. The environment is reset once with seed 12345 and then left
to draw its own initial states, so every run sees the same sequence. Reports the mean and standard
deviation of the environment return.

Fetch tasks: 50 deterministic episodes, episode i reset with seed 100000 + i. Reports the success rate
(success read at the last step) and the mean environment return.

The VecNormalize statistics saved with the run are used when the run has them. The result is printed as
JSON and written to <run>/eval.json.
"""

import argparse
import json
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qairl.configs import ENVS, make_env

EVAL_SEED = 12345
FETCH_SEED = 100_000


def load_policy(run_dir: str, env_key: str) -> SAC:
    # the replay buffer is not needed to act; a plain one avoids unpickling the relabelling buffer
    return SAC.load(os.path.join(run_dir, f"sac_{env_key}_final.zip"), device="cpu",
                    custom_objects={"replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}})


def evaluate_gymnasium(run_dir: str, env_key: str, n: int) -> dict:
    policy = load_policy(run_dir, env_key)
    raw = make_env(env_key)
    raw.reset(seed=EVAL_SEED)
    env = DummyVecEnv([lambda: raw])
    vn = os.path.join(run_dir, f"vecnormalize_{env_key}_final.pkl")
    if os.path.exists(vn):
        env = VecNormalize.load(vn, env)
        env.training = False
        env.norm_reward = False
    mean, std = evaluate_policy(policy, env, n_eval_episodes=n, deterministic=True)
    env.close()
    return dict(return_mean=float(mean), return_std=float(std), episodes=n)


def evaluate_fetch(run_dir: str, env_key: str, n: int) -> dict:
    policy = load_policy(run_dir, env_key)
    vn = os.path.join(run_dir, f"vecnormalize_{env_key}_final.pkl")
    stats = VecNormalize.load(vn, DummyVecEnv([lambda: make_env(env_key)])) if os.path.exists(vn) else None
    env = make_env(env_key)
    successes, returns = 0, []
    for ep in range(n):
        obs, _ = env.reset(seed=FETCH_SEED + ep)
        ret, done, info = 0.0, False, {}
        while not done:
            x = obs if stats is None else stats.normalize_obs(obs).astype(np.float32)
            obs, r, terminated, truncated, info = env.step(policy.predict(x, deterministic=True)[0])
            ret += float(r)
            done = terminated or truncated
        successes += bool(info["is_success"])
        returns.append(ret)
    env.close()
    return dict(success_rate=successes / n, return_mean=float(np.mean(returns)), episodes=n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory written by qairl.train")
    p.add_argument("--n", type=int, default=None, help="episodes (default: 10, or 50 on the Fetch tasks)")
    a = p.parse_args()
    with open(os.path.join(a.run, "hparams.json")) as f:
        env_key = json.load(f)["env"]
    if ENVS[env_key].get("needs_tiago_fetch"):
        res = evaluate_fetch(a.run, env_key, a.n or 50)
    else:
        res = evaluate_gymnasium(a.run, env_key, a.n or 10)
    res = dict(env=env_key, run=a.run, **res)
    with open(os.path.join(a.run, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
