"""
Trains QA-IRL with SAC on one environment.

    python -m qairl.train --env hopper --seed 0
    python -m qairl.train --env halfcheetah --seed 1 --head mlp --run_name mlp_s1     # an ablation
    python -m qairl.train --env pendulum --seed 0 --total_timesteps 20000             # any setting can be overridden

Settings come from qairl/configs.py (SHARED, then the environment entry, then the command line). The
demonstrations are read from <data_dir>/<data>/expert_data.npy (see qairl/data.py).

Everything is written to <out_dir>/<env>/<run_name>/: hparams.json, TensorBoard logs (tb/), the Monitor
file, periodic checkpoints every checkpoint_freq steps and the final models (sac_<env>_final.zip,
qairl_<env>_final.pth and, with VecNormalize, vecnormalize_<env>_final.pkl). Running the same command again
resumes from the latest periodic checkpoint. The SAC replay buffer, the policy FIFO and the Adam moments
are not saved, so a resumed run is not identical to an uninterrupted one.
"""

import argparse
import glob
import json
import os
import re

import torch
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qairl.buffers import ExpertBuffer, RelabelReplayBuffer
from qairl.callback import QAIRLCallback
from qairl.configs import ENVS, SHARED, make_env
from qairl.models import QAIRLModel
from qairl.wrapper import QAIRLRewardWrapper

# command-line overrides: name -> type
OVERRIDES = {
    "K": int, "M": int, "hidden_dim": int, "n_layers": int, "H": int, "c0": float, "anneal_frac": float,
    "alpha_0": float, "lambda_dir": float, "eps_k": float, "dir_margin_k": float, "nce_tau_k": float,
    "r_clip_k": float, "enc_lr": float, "dual_lr": float, "lam_init": float,
    "update_freq": int, "n_updates": int, "batch_size": int, "n_neg": int,
    "total_timesteps": int, "buffer_size": int, "learning_starts": int, "gamma": float, "sac_lr": float,
    "batch_size_sac": int, "train_freq": int, "gradient_steps": int, "ent_coef": str, "checkpoint_freq": int,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=list(ENVS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_name", default=None, help="default: s<seed>")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--data_dir", default=os.environ.get("QAIRL_DATA", "data"))
    p.add_argument("--device", default="cpu")
    # ablations
    p.add_argument("--head", choices=["maxrelu", "mlp"], default=None, help="mlp: no triangle inequality")
    p.add_argument("--no_push", action="store_true", help="drop L_push and its budget constraint")
    p.add_argument("--temporal_order", choices=["correct", "shuffled", "reversed"], default=None,
                   help="order of the states within each demonstration")
    vn = p.add_mutually_exclusive_group()
    vn.add_argument("--vecnormalize", dest="use_vecnormalize", action="store_true", default=None)
    vn.add_argument("--no_vecnormalize", dest="use_vecnormalize", action="store_false")
    for name, typ in OVERRIDES.items():
        p.add_argument(f"--{name}", type=typ, default=None)
    return p.parse_args()


def resolve(args: argparse.Namespace) -> dict:
    cfg = {**SHARED, **ENVS[args.env]}
    for name in list(OVERRIDES) + ["head", "temporal_order", "use_vecnormalize"]:
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if args.no_push:
        cfg["no_push"] = True
    if cfg["ent_coef"] != "auto":
        cfg["ent_coef"] = float(cfg["ent_coef"])
    c0 = cfg["c0"]
    cfg.update(eps=cfg["eps_k"] * c0, dir_margin=cfg["dir_margin_k"] * c0, nce_tau=cfg["nce_tau_k"] * c0,
               r_clip=cfg["r_clip_k"] * c0)
    return cfg


def latest_checkpoint(ckpt_dir: str, prefix: str, ext: str) -> tuple[str, int] | None:
    """The periodic checkpoint <prefix>_<steps>_steps<ext> with the most steps, if any."""
    best = None
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*_steps{ext}")):
        m = re.search(rf"_(\d+)_steps{re.escape(ext)}$", path)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (path, int(m.group(1)))
    return best


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)  # before the quasimetric weights are initialized
    cfg = resolve(args)
    env_key = args.env
    run_dir = os.path.join(args.out_dir, env_key, args.run_name or f"s{args.seed}")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    probe = make_env(env_key)
    obs_dim = probe.observation_space.shape[0]
    probe.close()

    data_path = os.path.join(args.data_dir, cfg["data"], "expert_data.npy")
    hparams = dict(env=env_key, seed=args.seed, obs_dim=obs_dim, data_path=data_path, **cfg)
    with open(os.path.join(run_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"[qairl] {env_key} seed {args.seed} -> {run_dir}", flush=True)
    print(f"[qairl] settings: {json.dumps(hparams)}", flush=True)

    expert_buffer = ExpertBuffer(data_path, H=cfg["H"], device=args.device,
                                 temporal_order=cfg["temporal_order"], order_seed=args.seed)

    model = QAIRLModel(state_dim=obs_dim, K=cfg["K"], M=cfg["M"], hidden_dim=cfg["hidden_dim"],
                       n_layers=cfg["n_layers"], c0=cfg["c0"], lam_init=cfg["lam_init"], r_clip=cfg["r_clip"],
                       head=cfg["head"]).to(args.device)
    enc_optimizer = optim.Adam(model.encoder.parameters(), lr=cfg["enc_lr"])
    dual_optimizer = optim.Adam([model.log_lam], lr=cfg["dual_lr"])

    q_resume = latest_checkpoint(run_dir, f"qairl_{env_key}", ".pth")
    sac_resume = latest_checkpoint(run_dir, f"sac_{env_key}", ".zip")
    if q_resume is not None:
        model.load_state_dict(torch.load(q_resume[0], map_location=args.device))
        print(f"[qairl] quasimetric resumed from {q_resume[0]}", flush=True)

    def build_env(rank: int = 0):
        env = QAIRLRewardWrapper(make_env(env_key), model, device=args.device)
        return Monitor(env, filename=os.path.join(run_dir, f"monitor_{rank}"))

    vec_env = DummyVecEnv([build_env])
    vec_normalize = None
    if cfg["use_vecnormalize"]:
        vn_resume = (os.path.join(run_dir, f"sac_{env_key}_vecnormalize_{sac_resume[1]}_steps.pkl")
                     if sac_resume is not None else None)
        if vn_resume is not None and os.path.exists(vn_resume):
            vec_env = VecNormalize.load(vn_resume, vec_env)
            print(f"[qairl] VecNormalize statistics resumed from {vn_resume}", flush=True)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        for monitor in vec_env.venv.envs:  # the reward wrapper normalizes observations the same way
            monitor.env.set_vec_normalize(vec_env)
        vec_normalize = vec_env

    callback = CallbackList([
        QAIRLCallback(
            model, expert_buffer, enc_optimizer, dual_optimizer, total_timesteps=cfg["total_timesteps"],
            vec_normalize=vec_normalize, update_freq=cfg["update_freq"], batch_size=cfg["batch_size"],
            n_neg=cfg["n_neg"], alpha_0=cfg["alpha_0"], eps=cfg["eps"], n_updates=cfg["n_updates"],
            anneal_frac=cfg["anneal_frac"], lambda_dir=cfg["lambda_dir"], dir_margin=cfg["dir_margin"],
            nce_tau=cfg["nce_tau"], no_push=cfg["no_push"], device=args.device,
            save_freq=cfg["checkpoint_freq"], save_path=run_dir, name_prefix=f"qairl_{env_key}"),
        CheckpointCallback(save_freq=cfg["checkpoint_freq"], save_path=run_dir, name_prefix=f"sac_{env_key}",
                           save_vecnormalize=cfg["use_vecnormalize"]),
    ])

    if sac_resume is not None:
        sac = SAC.load(sac_resume[0], env=vec_env, device=args.device)
        # SAC.load rebuilds the replay buffer with a pickled copy of the quasimetric; point it at the live one
        sac.replay_buffer.qairl_model = model
        remaining = cfg["total_timesteps"] - sac_resume[1]
        print(f"[qairl] SAC resumed from {sac_resume[0]}, {max(remaining, 0)} steps left", flush=True)
        if remaining > 0:
            sac.learn(total_timesteps=remaining, callback=callback, tb_log_name="sac", reset_num_timesteps=False)
    else:
        sac = SAC(
            "MlpPolicy", vec_env, learning_rate=cfg["sac_lr"], buffer_size=cfg["buffer_size"],
            learning_starts=cfg["learning_starts"], batch_size=cfg["batch_size_sac"], train_freq=cfg["train_freq"],
            gradient_steps=cfg["gradient_steps"], gamma=cfg["gamma"], ent_coef=cfg["ent_coef"], verbose=1,
            tensorboard_log=tb_dir, seed=args.seed, device=args.device,
            replay_buffer_class=RelabelReplayBuffer, replay_buffer_kwargs=dict(qairl_model=model),
        )
        sac.learn(total_timesteps=cfg["total_timesteps"], callback=callback, tb_log_name="sac")

    sac.save(os.path.join(run_dir, f"sac_{env_key}_final"))
    torch.save(model.state_dict(), os.path.join(run_dir, f"qairl_{env_key}_final.pth"))
    if cfg["use_vecnormalize"]:
        vec_env.save(os.path.join(run_dir, f"vecnormalize_{env_key}_final.pkl"))
    print(f"[qairl] done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
