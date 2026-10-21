# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Change detectors on the standardized per-cycle residual z (one value per deployment per gait cycle). All are
batched over deployments and stateless apart from their own counters. NaN z (no residual this cycle) is treated as
"no evidence": counters hold, no alarm."""

from __future__ import annotations

import numpy as np
import torch

from .cfg import DetectorCfg


class ThreeSigma:
    """Alarm when z > k for ``consecutive`` cycles in a row (Domberg-style rule)."""

    def __init__(self, n: int, cfg: DetectorCfg, device):
        self.k, self.m = cfg.three_sigma_k, cfg.three_sigma_consecutive
        self.count = torch.zeros(n, dtype=torch.long, device=device)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(z)
        above = valid & (z > self.k)
        self.count = torch.where(above, self.count + 1, torch.where(valid, torch.zeros_like(self.count), self.count))
        alarm = self.count >= self.m
        self.count = torch.where(alarm, torch.zeros_like(self.count), self.count)
        return alarm

    def reset(self, mask: torch.Tensor):
        self.count[mask] = 0


class Cusum:
    """One-sided CUSUM: S = max(0, S + z - k); alarm when S > h, then S restarts at 0."""

    def __init__(self, n: int, cfg: DetectorCfg, device, h: float | None = None):
        self.k = cfg.cusum_k
        self.h = cfg.cusum_h if h is None else h
        self.S = torch.zeros(n, device=device)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(z)
        S_new = (self.S + z - self.k).clamp_min(0.0)
        self.S = torch.where(valid, S_new, self.S)
        alarm = self.S > self.h
        self.S = torch.where(alarm, torch.zeros_like(self.S), self.S)
        return alarm

    def reset(self, mask: torch.Tensor):
        self.S[mask] = 0.0


class ConvergenceStop:
    """True once |z| < band for ``cycles`` consecutive cycles (the triggered arms' stop rule, F7)."""

    def __init__(self, n: int, cfg: DetectorCfg, device):
        self.band, self.m = cfg.converge_band, cfg.converge_cycles
        self.count = torch.zeros(n, dtype=torch.long, device=device)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(z)
        inside = valid & (z.abs() < self.band)
        self.count = torch.where(inside, self.count + 1, torch.where(valid, torch.zeros_like(self.count), self.count))
        return self.count >= self.m

    def reset(self, mask: torch.Tensor):
        self.count[mask] = 0


class Baseliner:
    """R5 per-deployment baseline of the (log) residual: collects ``n`` cycles at the start and after every
    adaptation episode (``restart(mask)``); while collecting, z is NaN (no alarms). ``sigma`` is the global nominal
    sd; the per-deployment mean replaces the global mean once measured."""

    def __init__(self, n_env: int, n_cycles: int, sigma: float, mu_global: float, device):
        self.n = n_cycles
        self.sigma = sigma
        self.mu = torch.full((n_env,), float(mu_global), device=device)
        self.count = torch.zeros(n_env, device=device)
        self.sum = torch.zeros(n_env, device=device)
        self.collecting = torch.ones(n_env, dtype=torch.bool, device=device) if n_cycles > 0 else torch.zeros(n_env, dtype=torch.bool, device=device)
        self.n_baselines = torch.zeros(n_env, dtype=torch.long, device=device)

    def restart(self, mask: torch.Tensor):
        if self.n <= 0:
            return
        self.collecting |= mask
        self.count = torch.where(mask, torch.zeros_like(self.count), self.count)
        self.sum = torch.where(mask, torch.zeros_like(self.sum), self.sum)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` = transformed per-cycle residual (log r); returns z (NaN while collecting or x is NaN)."""
        valid = ~torch.isnan(x)
        take = self.collecting & valid
        self.sum = torch.where(take, self.sum + x, self.sum)
        self.count = torch.where(take, self.count + 1, self.count)
        done = self.collecting & (self.count >= self.n)
        self.mu = torch.where(done, self.sum / self.count.clamp_min(1), self.mu)
        self.n_baselines += done.long()
        self.collecting = self.collecting & ~done
        z = (x - self.mu) / self.sigma
        return torch.where(self.collecting | ~valid, torch.full_like(z, float("nan")), z)


# ---- offline calibration ----------------------------------------------------------------------------------------


def cusum_alarms_offline(z: np.ndarray, k: float, h: float) -> int:
    """Alarms of the restarting one-sided CUSUM on one z sequence (NaN = skip)."""
    S, alarms = 0.0, 0
    for v in z:
        if np.isnan(v):
            continue
        S = max(0.0, S + v - k)
        if S > h:
            alarms += 1
            S = 0.0
    return alarms


def calibrate_cusum_h(z_streams: list[np.ndarray], cycle_s: float, k: float, target_fa_per_hour: float, h_grid=None) -> dict:
    """Smallest h on the grid whose false-alarm rate over the held-out nominal streams is <= the target."""
    if h_grid is None:
        h_grid = np.concatenate([np.arange(0.5, 10.0, 0.25), np.arange(10.0, 60.0, 2.0)])
    hours = sum(np.sum(~np.isnan(z)) for z in z_streams) * cycle_s / 3600.0
    rates = []
    for h in h_grid:
        alarms = sum(cusum_alarms_offline(z, k, h) for z in z_streams)
        rates.append(alarms / max(hours, 1e-9))
    rates = np.array(rates)
    ok = np.where(rates <= target_fa_per_hour)[0]
    h_star = float(h_grid[ok[0]]) if len(ok) else float(h_grid[-1])
    return {"h": h_star, "fa_per_hour_at_h": float(rates[ok[0]]) if len(ok) else float(rates[-1]), "hours": hours, "grid": [(float(h), float(r)) for h, r in zip(h_grid, rates)], "target": target_fa_per_hour, "k": k}


def three_sigma_fa_rate(z_streams: list[np.ndarray], cycle_s: float, k: float, m: int) -> float:
    hours = sum(np.sum(~np.isnan(z)) for z in z_streams) * cycle_s / 3600.0
    alarms = 0
    for z in z_streams:
        c = 0
        for v in z:
            if np.isnan(v):
                continue
            c = c + 1 if v > k else 0
            if c >= m:
                alarms += 1
                c = 0
    return alarms / max(hours, 1e-9)
