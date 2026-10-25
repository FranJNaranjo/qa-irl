"""
Buffers of QA-IRL.

ExpertBuffer      expert demonstrations, sampled as temporal pairs (s_t, s_{t+k}) and consecutive
                  transitions (s_t, s_{t+1})
PolicyObsBuffer   FIFO of the states visited by the policy, used as InfoNCE negatives
RelabelReplayBuffer
                  SAC replay buffer that recomputes the learned reward with the current quasimetric on
                  every sampled minibatch
"""

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer


class ExpertBuffer:
    """
    Demonstrations from an expert_data.npy file: an array of dicts, one per trajectory, each with an
    `observations` array of shape [T, obs_dim].

    temporal_order is "correct" for the method. The ablations use "reversed" (each trajectory played
    backwards) or "shuffled" (the states of each trajectory permuted once, with order_seed).
    """

    def __init__(self, npy_path: str, H: int = 10, device: str = "cpu", temporal_order: str = "correct",
                 order_seed: int = 0):
        if temporal_order not in ("correct", "reversed", "shuffled"):
            raise ValueError(f"unknown temporal_order {temporal_order!r}")
        self.H = H
        self.device = device
        self.temporal_order = temporal_order
        self._order_rng = np.random.default_rng(order_seed)
        self.trajs: list[torch.Tensor] = []
        for traj in np.load(npy_path, allow_pickle=True):
            if isinstance(traj, dict) and "observations" in traj:
                obs = np.asarray(traj["observations"], dtype=np.float32)
                if temporal_order == "reversed":
                    obs = obs[::-1].copy()
                elif temporal_order == "shuffled":
                    obs = obs[self._order_rng.permutation(len(obs))]
                if len(obs) >= 2:
                    self.trajs.append(torch.tensor(obs, dtype=torch.float32, device=device))
        if not self.trajs:
            raise ValueError(f"no valid trajectories in {npy_path}")

    def _pick_traj(self) -> torch.Tensor:
        return self.trajs[np.random.randint(len(self.trajs))]

    def sample_temporal_pairs(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """B pairs (s_t, s_{t+k}) with 1 <= k <= H, from uniformly chosen trajectories and start indices."""
        ancs, poss = [], []
        while len(ancs) < B:
            traj = self._pick_traj()
            T = len(traj)
            t = np.random.randint(0, T - 1)
            k = np.random.randint(1, min(self.H, T - 1 - t) + 1)
            ancs.append(traj[t])
            poss.append(traj[t + k])
        return torch.stack(ancs), torch.stack(poss)

    def sample_consecutive(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """B consecutive transitions (s_t, s_{t+1})."""
        a_list, b_list = [], []
        while len(a_list) < B:
            traj = self._pick_traj()
            t = np.random.randint(0, len(traj) - 1)
            a_list.append(traj[t])
            b_list.append(traj[t + 1])
        return torch.stack(a_list), torch.stack(b_list)


class PolicyObsBuffer:
    """FIFO of policy observations, sampled uniformly as InfoNCE negatives."""

    def __init__(self, capacity: int, obs_dim: int):
        self._buf = np.empty((capacity, obs_dim), dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._capacity = capacity

    def add(self, obs: np.ndarray) -> None:
        self._buf[self._pos] = obs
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, n: int) -> np.ndarray:
        return self._buf[np.random.randint(0, self._size, size=n)]

    @property
    def size(self) -> int:
        return self._size


class RelabelReplayBuffer(ReplayBuffer):
    """
    SAC replay buffer that stores (s, a, s') and recomputes the clipped learned reward r(s, s') with the
    current quasimetric whenever a minibatch is sampled. The rewards written at collection time are not
    used for the critic updates.

    With VecNormalize, sample(env=...) returns observations normalized with the current statistics, which
    is the input the quasimetric expects.
    """

    def __init__(self, *args, qairl_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        if qairl_model is None:
            raise ValueError("RelabelReplayBuffer needs qairl_model")
        self.qairl_model = qairl_model

    def sample(self, batch_size: int, env=None):
        data = super().sample(batch_size, env=env)
        dev = next(self.qairl_model.parameters()).device
        with torch.no_grad():
            rewards = self.qairl_model.policy_reward(
                data.observations.to(dev).float(), data.next_observations.to(dev).float()).unsqueeze(1)
        return data._replace(rewards=rewards.to(data.rewards.device, dtype=data.rewards.dtype))
