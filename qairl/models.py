"""
Quasimetric model and losses of QA-IRL.

The distance is a max-ReLU quasimetric head on an MLP embedding phi(s) in R^{K x M}:

    d(s, s') = (1/K) sum_k max_m relu(phi(s)[k, m] - phi(s')[k, m])

d is non-negative, d(s, s) = 0 and the triangle inequality holds by construction; d(s, s') = 0 does not
imply s = s'. The reward is the asymmetry of the distance, r(s, s') = d(s', s) - d(s, s'), clipped to
[-R, R].

Losses on the encoder:
    L_push  pushes expert temporal pairs apart, under a budget c0 on consecutive expert transitions
            enforced with a Lagrange multiplier
    L_NCE   InfoNCE with expert temporal pairs as positives and states visited by the policy as negatives
    L_dir   direction margin on consecutive expert transitions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuasimetricEncoder(nn.Module):
    """MLP embedding phi(s) of size K*M with the max-ReLU quasimetric head."""

    def __init__(self, state_dim: int, K: int = 8, M: int = 8, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.K = K
        self.M = M
        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, K * M))
        self.net = nn.Sequential(*layers)

    def embed(self, s: torch.Tensor) -> torch.Tensor:
        """phi(s), shape [batch, K*M]."""
        return self.net(s)

    def quasimetric(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """d(s, s'), shape [batch]."""
        phi_s = self.embed(s).view(-1, self.K, self.M)
        phi_sp = self.embed(sp).view(-1, self.K, self.M)
        return F.relu(phi_s - phi_sp).max(dim=-1).values.mean(dim=-1)

    def reward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Asymmetry reward d(s', s) - d(s, s'), shape [batch]."""
        return self.quasimetric(sp, s) - self.quasimetric(s, sp)


class MLPQuasiHead(nn.Module):
    """
    Ablation head: an asymmetric pairwise MLP h on [s, s'],

        d(s, s') = relu(softplus(h([s, s'])) - softplus(h([s, s]))).

    It keeps d >= 0 and d(s, s) = 0 but not the triangle inequality.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2 * state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def _h(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([s, sp], dim=-1)).squeeze(-1))

    def quasimetric(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        return F.relu(self._h(s, sp) - self._h(s, s))

    def reward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        return self.quasimetric(sp, s) - self.quasimetric(s, sp)


class QAIRLModel(nn.Module):
    """
    Quasimetric encoder plus the Lagrange multiplier lambda = exp(log_lam) of the budget constraint.

    The encoder is trained by gradient descent and log_lam by gradient ascent (see callback.py).
    c0 is not stored in the state dict: code that loads a checkpoint must pass the c0 of the run
    (saved in its hparams.json).
    """

    def __init__(self, state_dim: int, K: int = 8, M: int = 8, hidden_dim: int = 256, n_layers: int = 3,
                 c0: float = 1.0, lam_init: float = 0.05, r_clip: float = float("inf"), head: str = "maxrelu"):
        super().__init__()
        if head == "maxrelu":
            self.encoder = QuasimetricEncoder(state_dim, K, M, hidden_dim, n_layers)
        elif head == "mlp":
            self.encoder = MLPQuasiHead(state_dim, hidden_dim, n_layers)
        else:
            raise ValueError(f"unknown head {head!r}")
        self.c0 = c0
        self.r_clip = r_clip
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam_init)))

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lam.exp()

    def quasimetric(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        return self.encoder.quasimetric(s, sp)

    def reward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Unclipped asymmetry reward."""
        return self.encoder.reward(s, sp)

    def policy_reward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Reward given to the policy: the asymmetry reward clipped to [-R, R]."""
        r = self.encoder.reward(s, sp)
        if math.isfinite(self.r_clip):
            r = r.clamp(-self.r_clip, self.r_clip)
        return r


def compute_push_loss(model: QAIRLModel, s_anc: torch.Tensor, s_pos: torch.Tensor, s_a: torch.Tensor,
                      s_b: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    L_push = -E[softplus(d(s_t, s_{t+k}))] + lambda * (E[relu(d(s_t, s_{t+1}) - c0)^2] - eps^2).

    (s_anc, s_pos) are expert temporal pairs and (s_a, s_b) consecutive expert transitions. lambda is
    detached here. Returns the encoder loss and the detached constraint value, which the caller uses for
    the dual step on log_lam.
    """
    d_pos = model.quasimetric(s_anc, s_pos)
    L_main = -F.softplus(d_pos).mean()
    d_consec = model.quasimetric(s_a, s_b)
    violation = (F.relu(d_consec - model.c0) ** 2).mean() - eps ** 2
    return L_main + model.lam.detach() * violation, violation.detach()


def compute_nce_loss(model: QAIRLModel, s_anc: torch.Tensor, s_pos: torch.Tensor, s_neg: torch.Tensor,
                     tau: float) -> torch.Tensor:
    """
    InfoNCE with logits -d / tau. Each anchor s_anc[i] has its expert positive s_pos[i]; the policy states
    s_neg are negatives shared by every anchor. With the max-ReLU head, each state is embedded once and the
    [B, N] distance grid is computed by broadcasting.
    """
    encoder = model.encoder
    B = s_anc.shape[0]
    N = s_neg.shape[0]

    if isinstance(encoder, MLPQuasiHead):
        d_pos = encoder.quasimetric(s_anc, s_pos)
        anc_rep = s_anc.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        neg_rep = s_neg.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        d_neg = encoder.quasimetric(anc_rep, neg_rep).view(B, N)
        logits = torch.cat([-d_pos.unsqueeze(1), -d_neg], dim=1) / tau
        labels = torch.zeros(B, dtype=torch.long, device=s_anc.device)
        return F.cross_entropy(logits, labels)

    K, M = encoder.K, encoder.M
    phi_anc = encoder.embed(s_anc).view(B, 1, K, M)
    phi_pos = encoder.embed(s_pos).view(B, K, M)
    phi_neg = encoder.embed(s_neg).view(1, N, K, M)
    d_pos = F.relu(phi_anc.squeeze(1) - phi_pos).max(dim=-1).values.mean(dim=-1)   # [B]
    d_neg = F.relu(phi_anc - phi_neg).max(dim=-1).values.mean(dim=-1)              # [B, N]
    logits = torch.cat([-d_pos.unsqueeze(1), -d_neg], dim=1) / tau
    labels = torch.zeros(B, dtype=torch.long, device=s_anc.device)
    return F.cross_entropy(logits, labels)


def compute_dir_loss(model: QAIRLModel, s_a: torch.Tensor, s_b: torch.Tensor, margin: float) -> torch.Tensor:
    """
    L_dir = E[relu(m + d(s_t, s_{t+1}) - d(s_{t+1}, s_t))] on consecutive expert transitions. A zero loss
    means r(s_t, s_{t+1}) >= m on every transition of the batch.
    """
    d_fwd = model.quasimetric(s_a, s_b)
    d_rev = model.quasimetric(s_b, s_a)
    return F.relu(margin + d_fwd - d_rev).mean()


@torch.no_grad()
def expert_edge_reward_stats(model: QAIRLModel, s_a: torch.Tensor, s_b: torch.Tensor) -> dict[str, float]:
    """Minimum, 5th percentile, median and positive fraction of the unclipped reward on a batch of
    consecutive expert transitions (logged during training)."""
    r = model.reward(s_a, s_b)
    return {
        "min": float(r.min()),
        "p5": float(torch.quantile(r, 0.05)),
        "median": float(r.median()),
        "frac_pos": float((r > 0).float().mean()),
    }
