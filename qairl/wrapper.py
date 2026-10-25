"""
Gymnasium wrapper that replaces the environment reward with the clipped QA-IRL reward r(s_t, s_{t+1}).

The environment reward is kept in info["original_reward"] and the learned one in info["learned_reward"].
The wrapper remembers the previous observation, so it relies on reset() being called at the start of
every episode (DummyVecEnv does this after each done).
"""

import gymnasium as gym
import numpy as np
import torch


class QAIRLRewardWrapper(gym.Wrapper):
    def __init__(self, env, qairl_model, device: str = "cpu"):
        super().__init__(env)
        self.qairl_model = qairl_model
        self.device = device
        self.vec_normalize = None
        self._obs_before = None

    def set_vec_normalize(self, vec_normalize):
        """Observations are normalized with these VecNormalize statistics before the reward is computed."""
        self.vec_normalize = vec_normalize

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(np.array([obs]))[0]
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def step(self, action):
        obs_before = self._obs_before
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_before = next_obs
        if obs_before is not None:
            with torch.no_grad():
                learned = self.qairl_model.policy_reward(self._to_tensor(obs_before),
                                                         self._to_tensor(next_obs)).item()
        else:
            learned = 0.0
        info["original_reward"] = reward
        info["learned_reward"] = learned
        return next_obs, learned, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_before = obs
        return obs, info
