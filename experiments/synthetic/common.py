"""
Offline training of the quasimetric for the synthetic experiments.

The quasimetric is trained from demonstrations alone, with L_push (under the budget constraint) and
L_dir and the same primal-dual updates as qairl/callback.py. L_NCE is left out because it needs a policy,
so no SAC loop runs and a run takes seconds on a CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qairl.models import QAIRLModel, compute_dir_loss, compute_push_loss

GAMMA = 0.99
C0 = -math.log(GAMMA)


class TrajSampler:
    """Uniform sampling of temporal pairs (1 <= k <= H) and consecutive transitions."""

    def __init__(self, trajs: list[np.ndarray], H: int, rng: np.random.Generator):
        trajs = [t for t in trajs if len(t) >= 2]
        self.flat = torch.as_tensor(np.concatenate(trajs), dtype=torch.float32)
        self.lens = np.array([len(t) for t in trajs])
        self.offs = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.H = H
        self.rng = rng

    def pairs(self, B: int, max_k: int):
        ti = self.rng.integers(0, len(self.lens), size=B)
        T = self.lens[ti]
        t = (self.rng.random(B) * (T - 1)).astype(int)
        kmax = np.minimum(max_k, T - 1 - t)
        k = 1 + (self.rng.random(B) * kmax).astype(int)
        i = self.offs[ti] + t
        return self.flat[i], self.flat[i + k]


def train_encoder(trajs, *, H=10, K=8, M=8, hidden_dim=128, n_layers=3, head="maxrelu", eps_k=1.0,
                  lambda_dir=1.0, dir_margin_k=1.0, steps=4000, batch=256, enc_lr=3e-4, dual_lr=1e-2,
                  seed=0) -> QAIRLModel:
    # dual_lr is 1e-2 because this loop runs far fewer dual steps than a training run with a policy
    # (thousands instead of 10^4 to 10^5). At 1e-3, lambda stays around 3 after 4000 steps and the budget
    # is not enforced; at 1e-2 it reaches about 90, close to the values seen in full runs.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = TrajSampler(trajs, H, rng)
    model = QAIRLModel(state_dim=trajs[0].shape[1], K=K, M=M, hidden_dim=hidden_dim, n_layers=n_layers, c0=C0,
                       head=head)
    enc_opt = torch.optim.Adam(model.encoder.parameters(), lr=enc_lr)
    dual_opt = torch.optim.Adam([model.log_lam], lr=dual_lr)
    eps, margin = eps_k * C0, dir_margin_k * C0
    for _ in range(steps):
        s_anc, s_pos = sampler.pairs(batch, H)
        s_a, s_b = sampler.pairs(batch, 1)
        L_push, violation = compute_push_loss(model, s_anc, s_pos, s_a, s_b, eps=eps)
        L = L_push
        if lambda_dir > 0:
            L = L + lambda_dir * compute_dir_loss(model, s_a, s_b, margin)
        enc_opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        enc_opt.step()
        dual_opt.zero_grad()
        (-model.lam * violation).backward()
        dual_opt.step()
    model.eval()
    return model


@torch.no_grad()
def dist(model: QAIRLModel, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return model.quasimetric(torch.as_tensor(a, dtype=torch.float32),
                             torch.as_tensor(b, dtype=torch.float32)).numpy()


def edge_reward_stats(model: QAIRLModel, trajs) -> dict:
    """Minimum (the margin m_psi), positive fraction and median of the reward on the non-stationary
    demonstrated transitions, in units of c0."""
    a = np.concatenate([t[:-1] for t in trajs])
    b = np.concatenate([t[1:] for t in trajs])
    keep = np.any(a != b, axis=1)
    a, b = a[keep], b[keep]
    r = dist(model, b, a) - dist(model, a, b)
    return {"m_psi_over_c0": float(r.min() / C0), "frac_pos": float((r > 0).mean()),
            "median_r_over_c0": float(np.median(r) / C0)}


@torch.no_grad()
def triangle_violation_rate(model: QAIRLModel, states: np.ndarray, n=20000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    i, j, k = (rng.integers(0, len(states), n) for _ in range(3))
    x, y, z = states[i], states[j], states[k]
    lhs = dist(model, x, z)
    rhs = dist(model, x, y) + dist(model, y, z)
    # relative tolerance: float32 rounding on distances of order 1 to 10 is about 1e-6
    return float((lhs > rhs * (1 + 1e-4) + 1e-6).mean())


@torch.no_grad()
def max_loop_residual(model: QAIRLModel, trajs, k=5) -> float:
    """Largest |sum of rewards| over a k-step segment followed by its exact reverse (0 by anti-symmetry)."""
    worst = 0.0
    for t in trajs:
        if len(t) <= k:
            continue
        seg = t[: k + 1]
        fwd = dist(model, seg[1:], seg[:-1]) - dist(model, seg[:-1], seg[1:])
        rev = dist(model, seg[:-1], seg[1:]) - dist(model, seg[1:], seg[:-1])
        worst = max(worst, abs(float(fwd.sum() + rev.sum())))
    return worst / C0
