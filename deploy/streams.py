# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-deployment streams at gait-cycle resolution. One row per cycle, one column per deployment, saved as a
single .npz (key -> (C, N) float32 array) plus a meta .json."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

STREAM_KEYS = (
    "t_s",  # robot time at the end of the cycle [s]
    "progress_m",  # metres along the commanded direction in this cycle (0 while recovering from a fall)
    "track_err",  # mean |v_along_cmd - cmd| over the cycle's walking steps [m/s]
    "reward",  # summed env reward over the cycle
    "falls",  # falls (body contact resets) in the cycle
    "recovering",  # fraction of the cycle spent in fall recovery
    "repositioned",  # 1 if the robot was teleported back to its cell in this cycle
    "residual",  # dynamics-ensemble residual, mean over the cycle's valid steps (NaN without an ensemble)
    "disagreement",  # ensemble disagreement (variance across members), same aggregation
    "z",  # standardized residual (r - mu_nominal) / sigma_nominal (NaN without calibration)
    "adapting",  # 1 if the arm had this deployment in adapt state during the cycle
    "alarm",  # 1 in the cycle an adaptation episode started (detector alarm / schedule / oracle trigger)
    "policy_reset",  # 1 in the cycle the policy was reset to pi_0
    "updates",  # PPO updates applied to this deployment in the cycle
    "kl_pi0",  # KL(pi_theta || pi_0) on the last update batch (NaN if no update yet)
    "payload_kg",  # true world state (from the schedule / setters)
    "motor_strength",
    "friction",
    "budget_used_min",  # adapt-minutes spent in the current adaptation episode
)


class StreamLog:
    def __init__(self, n: int, device, keys=STREAM_KEYS):
        self.n = n
        self.keys = tuple(keys)
        self.device = device
        self.rows: dict[str, list[torch.Tensor]] = {k: [] for k in self.keys}
        self.cur: dict[str, torch.Tensor] = {k: torch.full((n,), float("nan"), device=device) for k in self.keys}
        self.n_cycles = 0

    def set(self, key: str, value):
        self.cur[key] = torch.as_tensor(value, device=self.device, dtype=torch.float32).expand(self.n).clone()

    def commit(self):
        for k in self.keys:
            self.rows[k].append(self.cur[k].detach().cpu().clone())
            self.cur[k] = torch.full((self.n,), float("nan"), device=self.device)
        self.n_cycles += 1

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: (torch.stack(v).numpy() if v else np.zeros((0, self.n), dtype=np.float32)) for k, v in self.rows.items()}

    def save(self, out_dir: str, meta: dict):
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "streams.npz"), **self.arrays())
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)

    @staticmethod
    def load(run_dir: str) -> tuple[dict[str, np.ndarray], dict]:
        z = np.load(os.path.join(run_dir, "streams.npz"))
        arrays = {k: z[k] for k in z.files}
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        return arrays, meta
