# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamics ensemble (decisions.md F5): M MLPs predicting the next proprioceptive observation from (obs, action),
trained on nominal pi_0 rollouts across the DR range; the surprise signal is the per-gait-cycle residual.

Observation layout of HexapodEnv (68 with the gait clock): lin_vel_b 3 | ang_vel_b 3 | projected_gravity_b 3 |
commands 3 | joint_pos - default 18 | joint_vel 18 | previous actions 18 | clock sin, cos 2.
Target = next proprio (lin vel, ang vel, gravity, joint pos, joint vel: 45 dims) minus current proprio.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .batched import BatchedMLP
from .cfg import EnsembleCfg

PROPRIO_IDX = list(range(0, 9)) + list(range(12, 48))  # 45 dims


class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: EnsembleCfg):
        super().__init__()
        self.cfg = cfg
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.register_buffer("proprio_idx", torch.tensor(PROPRIO_IDX, dtype=torch.long))
        tgt_dim = len(PROPRIO_IDX)
        in_dim = obs_dim + act_dim
        self.net = BatchedMLP(cfg.n_members, in_dim, tuple(cfg.hidden_dims), tgt_dim)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("tgt_mean", torch.zeros(tgt_dim))
        self.register_buffer("tgt_std", torch.ones(tgt_dim))
        # per-cycle residual calibration on held-out nominal data (filled by tools/deploy_train_ensemble.py)
        self.register_buffer("nominal_mu", torch.tensor(float("nan")))
        self.register_buffer("nominal_sigma", torch.tensor(float("nan")))
        self.register_buffer("nominal_mu_dis", torch.tensor(float("nan")))
        self.register_buffer("nominal_sigma_dis", torch.tensor(float("nan")))
        self.register_buffer("nominal_mu_log", torch.tensor(float("nan")))
        self.register_buffer("nominal_sigma_log", torch.tensor(float("nan")))

    # ---- data ------------------------------------------------------------------------------------------------
    def make_target(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        return next_obs[..., self.proprio_idx] - obs[..., self.proprio_idx]

    def fit_normalizers(self, inputs: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            self.in_mean.copy_(inputs.mean(0))
            self.in_std.copy_(inputs.std(0).clamp_min(1e-6))
            self.tgt_mean.copy_(targets.mean(0))
            self.tgt_std.copy_(targets.std(0).clamp_min(1e-6))

    # ---- forward -----------------------------------------------------------------------------------------------
    def predict_normalized(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """(M, B, tgt) predictions of the normalized target for a (B, obs) / (B, act) batch."""
        x = (torch.cat([obs, act], -1) - self.in_mean) / self.in_std
        x = x.unsqueeze(0).expand(self.cfg.n_members, *x.shape)
        return self.net(x)

    @torch.no_grad()
    def residual_step(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step (nmse, disagreement) for a batch: nmse = mean over target dims of the normalized squared error
        of the ensemble mean; disagreement = mean over dims of the variance across members."""
        pred = self.predict_normalized(obs, act)  # (M, B, D)
        tgt = (self.make_target(obs, next_obs) - self.tgt_mean) / self.tgt_std
        mean = pred.mean(0)
        nmse = ((mean - tgt) ** 2).mean(-1)
        dis = pred.var(0, unbiased=False).mean(-1)
        return nmse, dis

    def loss(self, inputs_n: torch.Tensor, targets_n: torch.Tensor) -> torch.Tensor:
        """Sum over members of the MSE on normalized data; ``inputs_n`` is (M, B, in) (each member its own batch)."""
        pred = self.net(inputs_n)
        return ((pred - targets_n) ** 2).mean(dim=(1, 2)).sum()

    def is_calibrated(self) -> bool:
        if getattr(self.cfg, "residual_transform", "log") == "log":
            return not bool(torch.isnan(self.nominal_sigma_log))
        return not bool(torch.isnan(self.nominal_sigma))

    def standardize(self, residual: torch.Tensor) -> torch.Tensor:
        """z of a per-cycle residual (cfg.residual_transform: log or none)."""
        if getattr(self.cfg, "residual_transform", "log") == "log":
            return (torch.log(residual.clamp_min(1e-12)) - self.nominal_mu_log) / self.nominal_sigma_log
        return (residual - self.nominal_mu) / self.nominal_sigma

    @torch.no_grad()
    def set_calibration(self, residual_cycles, disagreement_cycles=None):
        """Nominal per-cycle statistics (raw and log) from arrays of shape (cycles, envs), NaN = skip."""
        import numpy as np

        r = np.asarray(residual_cycles, dtype=np.float64)
        r = r[~np.isnan(r)]
        self.nominal_mu.fill_(float(r.mean()))
        self.nominal_sigma.fill_(float(r.std()))
        lr = np.log(np.clip(r, 1e-12, None))
        self.nominal_mu_log.fill_(float(lr.mean()))
        self.nominal_sigma_log.fill_(float(lr.std()))
        if disagreement_cycles is not None:
            d = np.asarray(disagreement_cycles, dtype=np.float64)
            d = d[~np.isnan(d)]
            self.nominal_mu_dis.fill_(float(d.mean()))
            self.nominal_sigma_dis.fill_(float(d.std()))

    def standardize_np(self, residual):
        """numpy version of standardize for offline tools."""
        import numpy as np

        r = np.asarray(residual, dtype=np.float64)
        if getattr(self.cfg, "residual_transform", "log") == "log":
            return (np.log(np.clip(r, 1e-12, None)) - float(self.nominal_mu_log)) / float(self.nominal_sigma_log)
        return (r - float(self.nominal_mu)) / float(self.nominal_sigma)

    # ---- io ----------------------------------------------------------------------------------------------------
    def save(self, path: str, extra: dict | None = None):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__, "obs_dim": self.obs_dim, "act_dim": self.act_dim, "extra": extra or {}}, path)

    @staticmethod
    def load(path: str, device) -> "DynamicsEnsemble":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = EnsembleCfg(**ck["cfg"])
        m = DynamicsEnsemble(ck["obs_dim"], ck["act_dim"], cfg).to(device)
        m.load_state_dict(ck["state_dict"], strict=False)
        m.eval()
        return m


def train_ensemble(model: DynamicsEnsemble, obs, act, next_obs, cfg: EnsembleCfg, device, log=print) -> dict:
    """Fit the members on a random split; returns the held-out per-step residual stats and losses. Each member
    gets its own bootstrap-free random mini-batch order (independent init + different batches = the usual
    deep-ensemble recipe, Lakshminarayanan et al. 2017, CITED from memory, not re-verified)."""
    n = obs.shape[0]
    perm = torch.randperm(n, device=device)
    n_train = int(cfg.train_fraction * n)
    tr, ho = perm[:n_train], perm[n_train:]
    inputs = torch.cat([obs, act], -1)
    targets = model.make_target(obs, next_obs)
    model.fit_normalizers(inputs[tr], targets[tr])
    x = (inputs - model.in_mean) / model.in_std
    y = (targets - model.tgt_mean) / model.tgt_std
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    M = cfg.n_members
    hist = []
    for ep in range(cfg.epochs):
        model.train()
        orders = torch.stack([tr[torch.randperm(n_train, device=device)] for _ in range(M)])  # (M, n_train)
        tot, nb = 0.0, 0
        for i in range(0, n_train, cfg.batch_size):
            idx = orders[:, i : i + cfg.batch_size]  # (M, b)
            loss = model.loss(x[idx], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() / M
            nb += 1
        model.eval()
        with torch.no_grad():
            ho_pred = model.net(x[ho].unsqueeze(0).expand(M, -1, -1))
            ho_loss = ((ho_pred - y[ho]) ** 2).mean().item()
        hist.append({"epoch": ep, "train_mse": tot / max(nb, 1), "heldout_mse": ho_loss})
        log(f"[ensemble] epoch {ep}: train mse {tot / max(nb, 1):.4f}, held-out mse {ho_loss:.4f}")
    with torch.no_grad():
        nmse, dis = model.residual_step(obs[ho], act[ho], next_obs[ho])
    return {"history": hist, "heldout_idx": ho.cpu(), "heldout_nmse_mean": nmse.mean().item(), "heldout_nmse_std": nmse.std().item(), "heldout_dis_mean": dis.mean().item(), "n_train": n_train, "n_heldout": len(ho)}
