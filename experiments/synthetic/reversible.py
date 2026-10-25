"""
Synthetic experiment 2: a reversible task, as a negative control for L_dir.

A 2D point mass whose state is its position, so moving in either direction is physically symmetric. Two
demonstration sets:
  directional  demos go from a left band A to a right band B; progress towards B is the task
  patrol       demos go A -> B -> A -> B along one line; no direction is preferred and every position is
               demonstrated in both directions

L_dir asks r(s_t, s_{t+1}) >= m on every demonstrated transition. On the patrol set this is asked in both
directions at the same positions, which anti-symmetry (r(s', s) = -r(s, s')) makes infeasible.

For each set, lambda_dir in {0, 1} and seed: margin and positive fraction on demonstrated transitions, the
final L_dir value, and the mean reward of unit steps right, left, up and down across the square.

    python experiments/synthetic/reversible.py

Writes results/synthetic/reversible.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import C0, dist, edge_reward_stats, train_encoder

STEP = 0.02


def leg(start: np.ndarray, goal: np.ndarray, rng) -> list[np.ndarray]:
    pts, p = [start], start.copy()
    while np.linalg.norm(goal - p) > STEP:
        v = goal - p
        p = np.clip(p + STEP * v / np.linalg.norm(v) + rng.normal(0, 0.002, 2), 0, 1)
        pts.append(p.copy())
    return pts


def demos(task: str, n: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        y = rng.uniform(0.2, 0.8)
        A = np.array([rng.uniform(0.1, 0.2), y])
        B = np.array([rng.uniform(0.8, 0.9), y])
        pts = leg(A, B, rng)
        if task == "patrol":
            pts += leg(pts[-1], A, rng)[1:] + leg(A, B, rng)[1:]
        out.append(np.array(pts, dtype=np.float32))
    return out


def dir_loss_value(model, trajs, margin_k=1.0) -> float:
    a = np.concatenate([t[:-1] for t in trajs])
    b = np.concatenate([t[1:] for t in trajs])
    v = np.maximum(0.0, margin_k * C0 + dist(model, a, b) - dist(model, b, a))
    return float(v.mean() / C0)


def probe(model, rng, n=4000) -> dict:
    s = rng.uniform(0.15, 0.85, (n, 2)).astype(np.float32)
    out = {}
    for name, d in {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1)}.items():
        sp = (s + STEP * np.array(d, dtype=np.float32)).astype(np.float32)
        out[name] = float(np.mean(dist(model, sp, s) - dist(model, s, sp)) / C0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_demos", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="results/synthetic/reversible.json")
    args = ap.parse_args()

    rows = []
    for task in ("directional", "patrol"):
        trajs = demos(task, args.n_demos, np.random.default_rng(7 if task == "directional" else 8))
        for lam in (0.0, 1.0):
            for seed in args.seeds:
                m = train_encoder(trajs, H=10, K=8, M=16, eps_k=1.0, lambda_dir=lam, steps=args.steps, seed=seed)
                row = dict(task=task, lambda_dir=lam, seed=seed, **edge_reward_stats(m, trajs),
                           dir_loss_over_c0=dir_loss_value(m, trajs), probe=probe(m, np.random.default_rng(100 + seed)))
                rows.append(row)
                p = row["probe"]
                print(f"{task:11s} lambda_dir={lam:.0f} s{seed}: m_psi/c0={row['m_psi_over_c0']:+7.2f} "
                      f"pos={row['frac_pos']:.3f} L_dir/c0={row['dir_loss_over_c0']:.3f} | "
                      f"r/c0 right={p['right']:+.3f} left={p['left']:+.3f} up={p['up']:+.3f} down={p['down']:+.3f}",
                      flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
