"""
Prepares the expert demonstrations as <data_dir>/<name>/expert_data.npy.

    python -m qairl.data --env hopper halfcheetah
    python -m qairl.data --all

Each file is an array of dicts, one per trajectory, with observations, actions, rewards,
next_observations and dones.

MuJoCo tasks: every episode of the Minari dataset mujoco/<task>/expert-v0, converted without filtering
(Minari downloads the dataset on first use).
Fetch tasks: recorded with the scripted experts of tiago-fetch-envs, with the settings of the paper (200
Pick demos with perturbed routes and action noise 0.05 to 0.15; 40 Push demos with action noise 0.05).
The generator is deterministic, so this reproduces the demonstrations used in the paper.
Pendulum: 40 rollouts of a PPO expert, recorded without a fixed seed, so they cannot be regenerated. The
file used in the paper is downloaded from the Hugging Face dataset FranJNaranjo/qa-irl-data.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

MINARI = {
    "inverted_pendulum": "mujoco/invertedpendulum/expert-v0",
    "inverted_double_pendulum": "mujoco/inverteddoublependulum/expert-v0",
    "hopper": "mujoco/hopper/expert-v0",
    "halfcheetah": "mujoco/halfcheetah/expert-v0",
    "ant": "mujoco/ant/expert-v0",
}
FETCH = {
    "fetch_pick": ["--task", "pick", "--expert", "perturbed", "--end-box", "--n", "200", "--noise", "0.05",
                   "--noise-max", "0.15"],
    "fetch_push": ["--task", "push"],
}
HF_DATASET = "FranJNaranjo/qa-irl-data"
ALL = list(MINARI) + list(FETCH) + ["pendulum"]


def from_minari(name: str, out_path: str) -> None:
    import minari

    dataset = minari.load_dataset(MINARI[name], download=True)
    trajectories = []
    for ep in dataset.iterate_episodes():
        trajectories.append({
            "observations": np.asarray(ep.observations[:-1]),
            "actions": np.asarray(ep.actions),
            "rewards": np.asarray(ep.rewards),
            "next_observations": np.asarray(ep.observations[1:]),
            "dones": np.asarray(ep.terminations) | np.asarray(ep.truncations),
        })
    np.save(out_path, trajectories)
    print(f"[data] {name}: {len(trajectories)} episodes from {MINARI[name]} -> {out_path}")


def from_fetch_expert(name: str, out_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "tiago_fetch.expert.generate_expert", *FETCH[name], "--out-dir", tmp],
                       check=True)
        src = glob.glob(os.path.join(tmp, "*", "expert_data.npy"))[0]
        shutil.move(src, out_path)
        shutil.move(os.path.join(os.path.dirname(src), "metadata.json"),
                    os.path.join(os.path.dirname(out_path), "metadata.json"))
    print(f"[data] {name}: scripted expert demos -> {out_path}")


def from_hugging_face(name: str, out_path: str) -> None:
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(HF_DATASET, f"{name}/expert_data.npy", repo_type="dataset")
    shutil.copy(src, out_path)
    print(f"[data] {name}: {HF_DATASET} -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", nargs="+", choices=ALL, default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--data_dir", default=os.environ.get("QAIRL_DATA", "data"))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    for name in (ALL if a.all else a.env):
        out_path = os.path.join(a.data_dir, name, "expert_data.npy")
        if os.path.exists(out_path) and not a.overwrite:
            print(f"[data] {name}: {out_path} exists, skipped")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if name in MINARI:
            from_minari(name, out_path)
        elif name in FETCH:
            from_fetch_expert(name, out_path)
        else:
            from_hugging_face(name, out_path)


if __name__ == "__main__":
    main()
