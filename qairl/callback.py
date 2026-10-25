"""
Stable-Baselines3 callback that trains the quasimetric alongside the policy.

Every update_freq environment steps it runs n_updates iterations. Each iteration takes one Adam step on
the encoder with lambda fixed,

    L = L_push(lambda) + alpha * L_NCE + lambda_dir * L_dir,

with the encoder gradient norm clipped at 5, followed by one Adam ascent step on log_lam. alpha decays
linearly from alpha_0 to 0 over the first anneal_frac * total_timesteps steps.

The InfoNCE negatives are drawn from a FIFO of the observations the policy visited. L_NCE starts once that
buffer holds 8 * n_neg states.

Logged in TensorBoard under qairl/ (losses, lambda, alpha, statistics of the reward on expert transitions)
and env/ (environment return and reward, success when the env reports it). Every save_freq steps the
quasimetric is saved as <name_prefix>_<steps>_steps.pth.
"""

import os
import traceback

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from qairl.buffers import ExpertBuffer, PolicyObsBuffer
from qairl.models import QAIRLModel, compute_dir_loss, compute_nce_loss, compute_push_loss, expert_edge_reward_stats


class QAIRLCallback(BaseCallback):
    def __init__(
        self,
        qairl_model: QAIRLModel,
        expert_buffer: ExpertBuffer,
        enc_optimizer: torch.optim.Optimizer,
        dual_optimizer: torch.optim.Optimizer,
        total_timesteps: int,
        vec_normalize=None,
        update_freq: int = 1_000,
        batch_size: int = 256,
        n_neg: int = 64,
        alpha_0: float = 1.0,
        eps: float = 1.0,
        n_updates: int = 20,
        anneal_frac: float = 0.5,
        lambda_dir: float = 1.0,
        dir_margin: float = 1.0,
        nce_tau: float = 1.0,
        no_push: bool = False,
        max_grad_norm: float = 5.0,
        policy_buf_capacity: int = 50_000,
        device: str = "cpu",
        save_freq: int = 0,
        save_path: str | None = None,
        name_prefix: str = "qairl",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.qairl_model = qairl_model
        self.expert_buffer = expert_buffer
        self.enc_optimizer = enc_optimizer
        self.dual_optimizer = dual_optimizer
        self.total_timesteps = total_timesteps
        self.vec_normalize = vec_normalize
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.alpha_0 = alpha_0
        self.eps = eps
        self.n_updates = n_updates
        self.anneal_frac = anneal_frac
        self.lambda_dir = lambda_dir
        self.dir_margin = dir_margin
        self.nce_tau = nce_tau
        self.no_push = no_push
        self.max_grad_norm = max_grad_norm
        self.min_buffer_size = 8 * n_neg
        self.policy_buf_capacity = policy_buf_capacity
        self.device = device
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self._last_update = 0
        self._policy_buf: PolicyObsBuffer | None = None
        self._orig_rewards: list[float] = []
        self._learned_rewards: list[float] = []
        self._successes: list[float] = []
        self._ep_orig_acc: dict[int, float] = {}
        self._ep_orig_returns: list[float] = []

    def _on_training_start(self) -> None:
        self._policy_buf = PolicyObsBuffer(self.policy_buf_capacity, self.model.observation_space.shape[0])

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            for obs in np.asarray(new_obs):
                self._policy_buf.add(obs.astype(np.float32))

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for i, (info, done) in enumerate(zip(infos, dones)):
            if "original_reward" in info:
                self._orig_rewards.append(info["original_reward"])
                self._ep_orig_acc[i] = self._ep_orig_acc.get(i, 0.0) + info["original_reward"]
            if done and i in self._ep_orig_acc:
                self._ep_orig_returns.append(self._ep_orig_acc.pop(i))
            if "learned_reward" in info:
                self._learned_rewards.append(info["learned_reward"])
            if done and "success" in info:
                self._successes.append(float(info["success"]))

        if self.num_timesteps - self._last_update >= self.update_freq:
            self._update()
            self._flush_logs()
            self._maybe_save()
            self._last_update = self.num_timesteps
        return True

    def _current_alpha(self) -> float:
        denom = max(1, self.total_timesteps * self.anneal_frac)
        return max(0.0, self.alpha_0 * (1.0 - self.num_timesteps / denom))

    def _sample_policy_negatives(self) -> torch.Tensor | None:
        """Negatives from the policy FIFO. These observations already went through VecNormalize, if any."""
        if self._policy_buf.size < max(self.n_neg, self.min_buffer_size):
            return None
        obs = self._policy_buf.sample(self.n_neg)
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize expert states with the current VecNormalize statistics, if any."""
        if self.vec_normalize is None:
            return t
        return torch.tensor(self.vec_normalize.normalize_obs(t.cpu().numpy()), dtype=torch.float32,
                            device=self.device)

    def _update(self) -> None:
        try:
            alpha = self._current_alpha()
            s_neg = self._sample_policy_negatives()
            totals = {"loss": 0.0, "push": 0.0, "nce": 0.0, "dir": 0.0, "lam": 0.0, "p5": 0.0, "median": 0.0,
                      "frac_pos": 0.0}
            margin_min = float("inf")

            for _ in range(self.n_updates):
                B = self.batch_size
                s_anc, s_pos = self.expert_buffer.sample_temporal_pairs(B)
                s_a, s_b = self.expert_buffer.sample_consecutive(B)
                s_anc, s_pos = self._norm(s_anc), self._norm(s_pos)
                s_a, s_b = self._norm(s_a), self._norm(s_b)

                zero = s_anc.new_zeros(1).squeeze()
                if self.no_push:  # ablation: no push-apart term and no budget constraint; lambda stays fixed
                    L_push, violation = zero, None
                else:
                    L_push, violation = compute_push_loss(self.qairl_model, s_anc, s_pos, s_a, s_b, eps=self.eps)
                L_nce = (compute_nce_loss(self.qairl_model, s_anc, s_pos, s_neg, tau=self.nce_tau)
                         if alpha > 0.0 and s_neg is not None else zero)
                L_dir = (compute_dir_loss(self.qairl_model, s_a, s_b, self.dir_margin)
                         if self.lambda_dir > 0.0 else zero)
                L = L_push + alpha * L_nce + self.lambda_dir * L_dir

                self.enc_optimizer.zero_grad()
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.qairl_model.encoder.parameters(), self.max_grad_norm)
                self.enc_optimizer.step()

                if violation is not None:  # dual ascent on log_lam
                    L_dual = -self.qairl_model.lam * violation
                    self.dual_optimizer.zero_grad()
                    L_dual.backward()
                    self.dual_optimizer.step()

                st = expert_edge_reward_stats(self.qairl_model, s_a, s_b)
                margin_min = min(margin_min, st["min"])
                totals["loss"] += L.item()
                totals["push"] += L_push.item()
                totals["nce"] += L_nce.item()
                totals["dir"] += L_dir.item()
                totals["lam"] += self.qairl_model.lam.item()
                totals["p5"] += st["p5"]
                totals["median"] += st["median"]
                totals["frac_pos"] += st["frac_pos"]

            n = self.n_updates
            self.logger.record("qairl/loss", totals["loss"] / n)
            self.logger.record("qairl/push_loss", totals["push"] / n)
            self.logger.record("qairl/nce_loss", totals["nce"] / n)
            self.logger.record("qairl/dir_loss", totals["dir"] / n)
            self.logger.record("qairl/expert_r_min", margin_min)
            self.logger.record("qairl/expert_r_p5", totals["p5"] / n)
            self.logger.record("qairl/expert_r_median", totals["median"] / n)
            self.logger.record("qairl/expert_r_frac_pos", totals["frac_pos"] / n)
            self.logger.record("qairl/alpha", alpha)
            self.logger.record("qairl/lambda", totals["lam"] / n)
            self.logger.record("qairl/policy_buffer_size", float(self._policy_buf.size))
        except Exception as e:
            print(f"[qairl] error in the quasimetric update: {e}", flush=True)
            traceback.print_exc()

    def _flush_logs(self) -> None:
        if self._orig_rewards:
            arr = np.array(self._orig_rewards)
            self.logger.record("env/original_reward", float(arr.mean()))
            self.logger.record("env/original_reward_std", float(arr.std()))
            self._orig_rewards = []
        if self._ep_orig_returns:
            self.logger.record("env/original_ep_return", float(np.mean(self._ep_orig_returns)))
            self._ep_orig_returns = []
        if self._learned_rewards:
            self.logger.record("env/learned_reward", float(np.mean(self._learned_rewards)))
            self._learned_rewards = []
        if self._successes:
            self.logger.record("env/success", float(np.mean(self._successes)))
            self._successes = []

    def _maybe_save(self) -> None:
        if self.save_path is None or self.save_freq <= 0:
            return
        if self.num_timesteps % self.save_freq < self.update_freq:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(self.qairl_model.state_dict(),
                       os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pth"))
