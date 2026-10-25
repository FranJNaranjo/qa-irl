"""
Synthetic experiment 1: gridworld with a known shortest-path cost C*.

A 12 x 12 grid with two walls. Experts go from a random start to a random goal; with probability p each
step is a random valid move instead of an optimal one (p = 0, 0.1, 0.3). For each level the script
reports:
  1. the expert-graph check: for every demonstrated temporal pair (s_t, s_{t+k}),
     0 <= d_gr - C* <= k - C*, where d_gr is the shortest path on the graph of demonstrated transitions
     (computed exactly by BFS, no learning);
  2. how close the learned d_psi is to d_gr: E_gr and the ratio rho = d_psi / d_gr per graph distance
     (p10, median, p90), for the max-ReLU head at eps_k = 1 and 0.1 and for the MLP head;
  3. the triangle-violation rate and the residual of a segment followed by its exact reverse.

    python experiments/synthetic/gridworld.py
    python experiments/synthetic/gridworld.py --seeds 0 --steps 1000      # quick

Writes results/synthetic/gridworld.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque

import numpy as np

from common import C0, dist, edge_reward_stats, max_loop_residual, train_encoder, triangle_violation_rate

N = 12
WALLS = {(r, 4) for r in range(0, 9)} | {(r, 8) for r in range(3, 12)}  # gaps make shortest paths non-Manhattan
CELLS = [(r, c) for r in range(N) for c in range(N) if (r, c) not in WALLS]
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def neighbours(cell):
    r, c = cell
    for dr, dc in MOVES:
        n = (r + dr, c + dc)
        if 0 <= n[0] < N and 0 <= n[1] < N and n not in WALLS:
            yield n


def bfs(adj: dict, src) -> dict:
    d, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


GRID_ADJ = {c: list(neighbours(c)) for c in CELLS}
CSTAR = {c: bfs(GRID_ADJ, c) for c in CELLS}  # CSTAR[s][g] in steps


def expert_demos(p: float, n: int, rng: np.random.Generator):
    demos = []
    while len(demos) < n:
        s, g = (CELLS[i] for i in rng.choice(len(CELLS), 2, replace=False))
        opt = CSTAR[s][g]
        if opt < 4:
            continue
        path, cur = [s], s
        while cur != g and len(path) < 3 * opt + 20:
            nb = list(neighbours(cur))
            if rng.random() < p:
                cur = nb[rng.integers(len(nb))]
            else:
                best = [v for v in nb if CSTAR[v][g] == CSTAR[cur][g] - 1]
                cur = best[rng.integers(len(best))]
            path.append(cur)
        if cur == g:
            demos.append(path)
    return demos


def to_states(path) -> np.ndarray:
    return np.array([[r / (N - 1), c / (N - 1)] for r, c in path], dtype=np.float32)


def graph_check(demos, H: int) -> dict:
    """d_gr - C* against its pointwise bound k - C* on every demonstrated temporal pair."""
    adj: dict = {}
    for path in demos:
        for u, v in zip(path[:-1], path[1:]):
            adj.setdefault(u, set()).add(v)
    lE = {u: bfs(adj, u) for u in {c for p in demos for c in p}}
    gaps, excess, dgr, pairs = [], [], [], []
    for path in demos:
        for t in range(len(path) - 1):
            for k in range(1, min(H, len(path) - 1 - t) + 1):
                s, sp = path[t], path[t + k]
                gaps.append(lE[s][sp] - CSTAR[s][sp])
                excess.append(k - CSTAR[s][sp])
                dgr.append(lE[s][sp])
                pairs.append((s, sp))
    gaps, excess = np.array(gaps), np.array(excess)
    return dict(delta_steps=int(excess.max()), gaps=gaps, excess=excess, dgr=np.array(dgr), pairs=pairs,
                violations=int(((gaps < 0) | (gaps > excess)).sum()), n_pairs=len(gaps),
                whole_path_delta=int(max(len(p) - 1 - CSTAR[p[0]][p[-1]] for p in demos)))


def geometry_stats(model, pairs, dgr) -> dict:
    a = np.array([[r / (N - 1), c / (N - 1)] for (r, c), _ in pairs], dtype=np.float32)
    b = np.array([[r / (N - 1), c / (N - 1)] for _, (r, c) in pairs], dtype=np.float32)
    d = dist(model, a, b) / C0
    out = {"E_gr": float(np.mean(np.abs(d - dgr)))}
    by_l = {}
    for l in range(1, int(dgr.max()) + 1):
        m = dgr == l
        if m.sum() >= 20:
            by_l[str(l)] = [float(np.percentile(d[m] / l, q)) for q in (10, 50, 90)]
    out["rho_by_dgr"] = by_l
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.1, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_demos", type=int, default=300)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="results/synthetic/gridworld.json")
    args = ap.parse_args()

    configs = [("maxrelu", 1.0), ("maxrelu", 0.1), ("mlp", 1.0)]
    report = {"c0": C0, "grid": N, "walls": len(WALLS), "levels": {}}
    for p in args.levels:
        demos = expert_demos(p, args.n_demos, np.random.default_rng(1000 + int(p * 100)))
        cc = graph_check(demos, args.H)
        trajs = [to_states(path) for path in demos]
        states = np.concatenate(trajs)
        n_sub = int((cc["excess"] > 0).sum())
        lvl = {"p": p, "delta_over_c0": cc["delta_steps"], "n_pairs": cc["n_pairs"],
               "corollary_violations": cc["violations"],
               "whole_path_delta_over_c0": cc["whole_path_delta"],
               "frac_pairs_suboptimal": float((cc["excess"] > 0).mean()),
               # among suboptimal pairs, the fraction where the bound is attained
               "frac_bound_tight": float(((cc["gaps"] == cc["excess"]) & (cc["excess"] > 0)).sum() / max(n_sub, 1)),
               "gap_counts": {int(g): int((cc["gaps"] == g).sum()) for g in np.unique(cc["gaps"])},
               "runs": []}
        print(f"p={p}: delta={cc['delta_steps']} c0, {cc['n_pairs']} pairs, violations={cc['violations']}",
              flush=True)
        for head, eps_k in configs:
            for seed in args.seeds:
                model = train_encoder(trajs, H=args.H, K=8, M=16, head=head, eps_k=eps_k, steps=args.steps,
                                      seed=seed)
                g = geometry_stats(model, cc["pairs"], cc["dgr"])
                run = dict(head=head, eps_k=eps_k, seed=seed, **g, **edge_reward_stats(model, trajs),
                           triangle_violation_rate=triangle_violation_rate(model, states),
                           loop_residual_over_c0=max_loop_residual(model, trajs))
                lvl["runs"].append(run)
                print(f"  {head:7s} eps_k={eps_k:<4} s{seed}: E_gr={g['E_gr']:.2f} "
                      f"m_psi/c0={run['m_psi_over_c0']:+.2f} frac_pos={run['frac_pos']:.3f} "
                      f"tri_viol={run['triangle_violation_rate']:.4f} loop={run['loop_residual_over_c0']:.1e}",
                      flush=True)
        report["levels"][str(p)] = lvl

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
