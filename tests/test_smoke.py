"""Short end-to-end runs on Pendulum with synthetic demonstrations (no dataset needed)."""
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from qairl.models import QAIRLModel, compute_dir_loss, compute_nce_loss, compute_push_loss


def test_quasimetric_axioms():
    torch.manual_seed(0)
    model = QAIRLModel(state_dim=3, K=4, M=4, hidden_dim=32, n_layers=3, c0=0.1)
    x, y, z = torch.randn(3, 512, 3)
    d = model.quasimetric
    assert torch.all(d(x, y) >= 0)
    assert torch.allclose(d(x, x), torch.zeros(512))
    assert torch.all(d(x, z) <= d(x, y) + d(y, z) + 1e-5)
    assert torch.allclose(model.reward(x, y), -model.reward(y, x))


def test_losses_are_finite():
    torch.manual_seed(0)
    model = QAIRLModel(state_dim=3, K=4, M=4, hidden_dim=32, n_layers=3, c0=0.1)
    s = torch.randn(4, 64, 3)
    L_push, violation = compute_push_loss(model, s[0], s[1], s[2], s[3], eps=0.1)
    L = L_push + compute_nce_loss(model, s[0], s[1], s[2][:16], tau=0.1) + compute_dir_loss(model, s[2], s[3], 0.1)
    assert torch.isfinite(L) and torch.isfinite(violation)


@pytest.fixture
def pendulum_demos(tmp_path):
    env = gym.make("Pendulum-v1")
    trajs = []
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        o = [obs]
        for _ in range(200):
            obs, *_ = env.step(env.action_space.sample())
            o.append(obs)
        trajs.append({"observations": np.array(o[:-1], dtype=np.float32)})
    path = tmp_path / "data" / "pendulum"
    path.mkdir(parents=True)
    np.save(path / "expert_data.npy", trajs)
    return tmp_path


@pytest.mark.parametrize("extra", [[], ["--head", "mlp", "--temporal_order", "shuffled"]])
def test_train_and_evaluate(pendulum_demos, extra):
    root = pendulum_demos
    run = root / "runs" / "pendulum" / "s0"
    subprocess.run([sys.executable, "-m", "qairl.train", "--env", "pendulum", "--seed", "0",
                    "--total_timesteps", "1500", "--learning_starts", "500", "--checkpoint_freq", "1000",
                    "--out_dir", str(root / "runs"), "--data_dir", str(root / "data"), *extra], check=True)
    assert (run / "sac_pendulum_final.zip").exists() and (run / "qairl_pendulum_final.pth").exists()
    subprocess.run([sys.executable, "-m", "qairl.evaluate", "--run", str(run), "--n", "2"], check=True)
    assert "return_mean" in json.loads((run / "eval.json").read_text())
