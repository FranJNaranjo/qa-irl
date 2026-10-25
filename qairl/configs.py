"""
Settings of every environment in the paper.

SHARED holds what all environments use. ENVS holds, per environment, the Gymnasium ID, the demonstration
file and the settings that change between environments. qairl.train merges SHARED, the environment entry
and the command-line overrides, in that order.

Scale-dependent quantities are multiples of the budget c0:
    eps = eps_k * c0       budget slack
    m   = dir_margin_k * c0  direction margin of L_dir
    tau = nce_tau_k * c0   InfoNCE temperature
    R   = r_clip_k * c0    reward clip
"""

SHARED = dict(
    # objective
    alpha_0=1.0, lambda_dir=1.0, eps_k=1.0, dir_margin_k=1.0, nce_tau_k=1.0, r_clip_k=30.0,
    # quasimetric optimizers
    enc_lr=3e-4, dual_lr=5e-3, lam_init=0.05,
    # SAC
    sac_lr=3e-4, batch_size_sac=256, train_freq=1, gradient_steps=1, ent_coef="auto",
    use_vecnormalize=False,
    # ablation switches (the method: max-ReLU head, push term on, demonstrations in their recorded order)
    head="maxrelu", no_push=False, temporal_order="correct",
    checkpoint_freq=50_000,
)

# Per environment:
#   K, M, hidden_dim, n_layers   quasimetric encoder (n_layers counts the linear layers, so n_layers - 1 hidden)
#   H                            maximum lag of the expert temporal pairs
#   c0                           budget per expert transition
#   anneal_frac                  fraction of training over which alpha decays to 0
#   update_freq, n_updates       quasimetric iterations every update_freq environment steps
#   batch_size, n_neg            expert pairs and policy negatives per iteration
#   total_timesteps, buffer_size, learning_starts, gamma   SAC
ENVS = {
    "pendulum": dict(
        env_id="Pendulum-v1", data="pendulum",
        K=4, M=4, hidden_dim=128, n_layers=3, H=10, c0=0.1, anneal_frac=0.5,
        update_freq=500, n_updates=40, batch_size=256, n_neg=64,
        total_timesteps=500_000, buffer_size=100_000, learning_starts=1_000, gamma=0.99,
    ),
    "inverted_pendulum": dict(
        env_id="InvertedPendulum-v5", data="inverted_pendulum",
        K=4, M=4, hidden_dim=64, n_layers=2, H=15, c0=1.0, anneal_frac=0.5,
        update_freq=1_000, n_updates=20, batch_size=256, n_neg=64,
        total_timesteps=300_000, buffer_size=100_000, learning_starts=1_000, gamma=0.99,
    ),
    "inverted_double_pendulum": dict(
        env_id="InvertedDoublePendulum-v5", data="inverted_double_pendulum",
        K=8, M=8, hidden_dim=256, n_layers=3, H=20, c0=1.0, anneal_frac=0.7,
        update_freq=2_000, n_updates=20, batch_size=256, n_neg=64,
        total_timesteps=1_000_000, buffer_size=500_000, learning_starts=5_000, gamma=0.99,
    ),
    "hopper": dict(
        env_id="Hopper-v4", data="hopper",
        K=8, M=8, hidden_dim=256, n_layers=3, H=20, c0=1.0, anneal_frac=0.5,
        update_freq=2_000, n_updates=20, batch_size=256, n_neg=64,
        total_timesteps=1_000_000, buffer_size=500_000, learning_starts=5_000, gamma=0.99,
    ),
    "halfcheetah": dict(
        env_id="HalfCheetah-v4", data="halfcheetah",
        K=8, M=8, hidden_dim=256, n_layers=3, H=30, c0=1.0, anneal_frac=0.5,
        update_freq=2_000, n_updates=30, batch_size=512, n_neg=128,
        total_timesteps=2_000_000, buffer_size=1_000_000, learning_starts=10_000, gamma=0.99,
    ),
    "ant": dict(
        env_id="Ant-v5", data="ant",
        K=8, M=8, hidden_dim=256, n_layers=3, H=30, c0=1.0, anneal_frac=0.6,
        update_freq=2_000, n_updates=30, batch_size=512, n_neg=128,
        total_timesteps=2_000_000, buffer_size=1_000_000, learning_starts=10_000, gamma=0.99,
    ),
    # TIAGo++ Fetch tasks (package tiago-fetch-envs). Dict observations are flattened in the order of
    # the demonstrations: achieved_goal, desired_goal, observation.
    "fetch_pick": dict(
        env_id="TiagoDualFetchPick-v0", data="fetch_pick", needs_tiago_fetch=True, flatten_obs=True,
        K=8, M=8, hidden_dim=128, n_layers=2, H=20, c0=0.1, anneal_frac=0.5, lam_init=10.0,
        update_freq=100, n_updates=20, batch_size=256, n_neg=64,
        total_timesteps=1_000_000, buffer_size=1_000_000, learning_starts=10_000, gamma=0.98,
    ),
    "fetch_push": dict(
        env_id="TiagoDualFetchPush-v0", data="fetch_push", needs_tiago_fetch=True, flatten_obs=True,
        K=8, M=8, hidden_dim=128, n_layers=2, H=20, c0=0.1, anneal_frac=0.5, lam_init=10.0,
        update_freq=100, n_updates=20, batch_size=256, n_neg=64,
        total_timesteps=1_000_000, buffer_size=1_000_000, learning_starts=10_000, gamma=0.98,
        use_vecnormalize=True,
    ),
}


def make_env(env_key: str, **kwargs):
    """The Gymnasium environment of env_key, with the observation flattening the method expects."""
    import gymnasium as gym

    cfg = ENVS[env_key]
    if cfg.get("needs_tiago_fetch"):
        import tiago_fetch  # noqa: F401  (registers the TiagoDualFetch IDs)
    env = gym.make(cfg["env_id"], **kwargs)
    if cfg.get("flatten_obs"):
        from gymnasium.wrappers import FlattenObservation
        env = FlattenObservation(env)
    return env
