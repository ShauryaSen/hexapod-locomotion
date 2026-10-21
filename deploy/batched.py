# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stacked-parameter MLPs: one independent weight slice per deployment (or per ensemble member), evaluated in one
batched matmul. Deployment i only ever sees its own slice, so N deployments in one process are exactly N separate
networks (decisions.md F2: independent weights per deployment).

Two forward implementations are provided and benchmarked by tools/deploy_bench_policy.py:
  * "einsum": parameters stored as (N, out, in); forward = einsum over the leading dim.
  * "vmap":   torch.func.functional_call vmapped over the same stacked parameters.
The harness uses ``BatchedMLP.forward`` (einsum) unless the benchmark says otherwise (see harness-report.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _elu(x):
    return F.elu(x)


class BatchedMLP(nn.Module):
    """``n`` independent MLPs with identical layout. ``x`` is (n, B, in) or (n, in); output (n, B, out) / (n, out)."""

    def __init__(self, n: int, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, activation: str = "elu"):
        super().__init__()
        self.n, self.in_dim, self.out_dim = n, in_dim, out_dim
        dims = [in_dim, *hidden_dims, out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = torch.empty(n, dims[i + 1], dims[i])
            b = torch.zeros(n, dims[i + 1])
            bound = 1.0 / math.sqrt(dims[i])
            nn.init.uniform_(w, -bound, bound)  # torch.nn.Linear default init
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))
        assert activation == "elu", activation
        self.act = _elu

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (n, 1, in)
        for i in range(self.n_layers):
            # (n, B, in) x (n, out, in) -> (n, B, out)
            x = torch.baddbmm(self.biases[i].unsqueeze(1), x, self.weights[i].transpose(1, 2))
            if i < self.n_layers - 1:
                x = self.act(x)
        return x.squeeze(1) if squeeze else x

    def forward_vmap(self, x: torch.Tensor) -> torch.Tensor:
        """Same function through torch.func.vmap over a per-slice functional MLP (for the benchmark)."""
        ws = [w for w in self.weights]
        bs = [b for b in self.biases]

        def one(x1, *params):
            k = len(params) // 2
            for i in range(k):
                x1 = F.linear(x1, params[i], params[k + i])
                if i < k - 1:
                    x1 = F.elu(x1)
            return x1

        return torch.vmap(one)(x, *ws, *bs)

    # ---- per-slice state_dict helpers ----------------------------------------------------------------------
    def load_sequential_state(self, sd: dict, layer_keys: list[int] | None = None, idx=None):
        """Copy a plain rsl_rl ``mlp.{k}.weight/bias`` state dict into every slice (or ``idx`` slices)."""
        if layer_keys is None:
            layer_keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        assert len(layer_keys) == self.n_layers, (layer_keys, self.n_layers)
        sl = slice(None) if idx is None else idx
        with torch.no_grad():
            for i, k in enumerate(layer_keys):
                w = sd[f"mlp.{k}.weight"].to(self.weights[i].device)
                b = sd[f"mlp.{k}.bias"].to(self.biases[i].device)
                self.weights[i][sl] = w
                self.biases[i][sl] = b

    def slice_state(self, i: int) -> dict:
        """Plain ``mlp.{2k}.weight/bias`` state dict of slice ``i`` (rsl_rl layout: Linear at even indices)."""
        out = {}
        for k in range(self.n_layers):
            out[f"mlp.{2 * k}.weight"] = self.weights[k][i].detach().cpu().clone()
            out[f"mlp.{2 * k}.bias"] = self.biases[k][i].detach().cpu().clone()
        return out

    def copy_slices(self, src: torch.Tensor, dst: torch.Tensor):
        """Copy slices ``src`` into slices ``dst`` (both index tensors), e.g. a reset to a reference slice."""
        with torch.no_grad():
            for w, b in zip(self.weights, self.biases):
                w[dst] = w[src]
                b[dst] = b[src]

    def parameters_per_slice(self):
        """List of stacked parameters (each with leading dim n)."""
        return list(self.weights) + list(self.biases)


class BatchedGaussianActor(nn.Module):
    """rsl_rl MLPModel + GaussianDistribution(std_type='scalar') for n deployments: mean MLP + per-slice std."""

    def __init__(self, n: int, obs_dim: int, hidden_dims: tuple[int, ...], act_dim: int, init_std: float = 1.0):
        super().__init__()
        self.mlp = BatchedMLP(n, obs_dim, hidden_dims, act_dim)
        self.std_param = nn.Parameter(torch.full((n, act_dim), init_std))
        self.n, self.act_dim = n, act_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mlp(obs)
        std = self.std_param.clamp_min(1e-4)
        if mean.dim() == 3:
            std = std.unsqueeze(1).expand_as(mean)
        return mean, std

    @staticmethod
    def log_prob(mean, std, act) -> torch.Tensor:
        var = std * std
        return (-((act - mean) ** 2) / (2 * var) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)

    @staticmethod
    def entropy(std) -> torch.Tensor:
        return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum(-1)

    @staticmethod
    def kl(mean_p, std_p, mean_q, std_q) -> torch.Tensor:
        """KL(p || q) of diagonal Gaussians, summed over the action dim."""
        var_p, var_q = std_p * std_p, std_q * std_q
        return (torch.log(std_q / std_p) + (var_p + (mean_p - mean_q) ** 2) / (2 * var_q) - 0.5).sum(-1)

    def load_rsl_rl(self, actor_sd: dict, idx=None):
        self.mlp.load_sequential_state(actor_sd, idx=idx)
        with torch.no_grad():
            sl = slice(None) if idx is None else idx
            self.std_param[sl] = actor_sd["distribution.std_param"].to(self.std_param.device)

    def slice_state(self, i: int) -> dict:
        sd = self.mlp.slice_state(i)
        sd["distribution.std_param"] = self.std_param[i].detach().cpu().clone()
        return sd

    def parameters_per_slice(self):
        return self.mlp.parameters_per_slice() + [self.std_param]


class BatchedCritic(nn.Module):
    def __init__(self, n: int, obs_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.mlp = BatchedMLP(n, obs_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).squeeze(-1)

    def load_rsl_rl(self, critic_sd: dict, idx=None):
        self.mlp.load_sequential_state(critic_sd, idx=idx)

    def parameters_per_slice(self):
        return self.mlp.parameters_per_slice()


class BatchedAdam:
    """Adam over stacked parameters with a per-slice learning rate and a per-slice update mask. Adam is elementwise,
    so one BatchedAdam over (n, ...) tensors with a loss that is a sum over slices is exactly n independent Adams
    (torch defaults: betas (0.9, 0.999), eps 1e-8, no weight decay = rsl_rl's optimizer)."""

    def __init__(self, params: list[torch.Tensor], n: int, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.n = n
        self.lr = torch.full((n,), float(lr), device=params[0].device)
        self.b1, self.b2, self.eps = betas[0], betas[1], eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = torch.zeros(n, device=params[0].device)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float, mask: torch.Tensor) -> torch.Tensor:
        """Per-slice global grad-norm clipping (rsl_rl clips over actor+critic jointly, max_grad_norm 1.0)."""
        sq = torch.zeros(self.n, device=self.lr.device)
        for p in self.params:
            if p.grad is not None:
                sq += p.grad.flatten(1).pow(2).sum(1)
        norm = sq.sqrt()
        scale = torch.where(norm > max_norm, max_norm / (norm + 1e-6), torch.ones_like(norm))
        scale = torch.where(mask, scale, torch.zeros_like(scale))
        for p in self.params:
            if p.grad is not None:
                p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))
        return norm

    @torch.no_grad()
    def step(self, mask: torch.Tensor):
        """Update slices where ``mask`` is True; others keep parameters and moments untouched."""
        maskf = mask.float()
        self.t += maskf
        bc1 = 1 - self.b1 ** self.t.clamp_min(1)
        bc2 = 1 - self.b2 ** self.t.clamp_min(1)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            shp = (-1, *([1] * (p.dim() - 1)))
            mk = maskf.view(shp)
            m.mul_(1 - mk * (1 - self.b1)).add_(mk * (1 - self.b1) * g)
            v.mul_(1 - mk * (1 - self.b2)).add_(mk * (1 - self.b2) * g * g)
            m_hat = m / bc1.view(shp)
            v_hat = v / bc2.view(shp)
            p.add_(-(self.lr.view(shp) * mk) * m_hat / (v_hat.sqrt() + self.eps))

    @torch.no_grad()
    def reset_slices(self, idx: torch.Tensor):
        for m, v in zip(self.m, self.v):
            m[idx] = 0.0
            v[idx] = 0.0
        self.t[idx] = 0.0

    def state_dict(self) -> dict:
        return {"m": [m.cpu() for m in self.m], "v": [v.cpu() for v in self.v], "t": self.t.cpu(), "lr": self.lr.cpu()}


def load_rsl_rl_checkpoint(path: str, device) -> dict:
    """``{'actor': sd, 'critic': sd, 'hidden_dims': [...], 'obs_dim': int, 'act_dim': int}`` from a model_*.pt."""
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = ck["actor_state_dict"] if "actor_state_dict" in ck else ck
    critic = ck.get("critic_state_dict")
    layers = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight")})
    hidden = [actor[f"mlp.{k}.weight"].shape[0] for k in layers[:-1]]
    return {
        "actor": actor,
        "critic": critic,
        "hidden_dims": hidden,
        "obs_dim": actor[f"mlp.{layers[0]}.weight"].shape[1],
        "act_dim": actor[f"mlp.{layers[-1]}.weight"].shape[0],
        "iter": ck.get("iter"),
    }


def build_from_checkpoint(path: str, n: int, device) -> tuple[BatchedGaussianActor, BatchedCritic, dict]:
    info = load_rsl_rl_checkpoint(path, device)
    actor = BatchedGaussianActor(n, info["obs_dim"], tuple(info["hidden_dims"]), info["act_dim"]).to(device)
    actor.load_rsl_rl(info["actor"])
    critic = BatchedCritic(n, info["obs_dim"], tuple(info["hidden_dims"])).to(device)
    if info["critic"] is not None:
        critic.load_rsl_rl(info["critic"])
    return actor, critic, info
