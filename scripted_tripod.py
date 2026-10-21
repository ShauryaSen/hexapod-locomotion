# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The club's scripted tripod gait, used as the behaviour-cloning teacher and for open-loop measurements."""

from __future__ import annotations

import math
import torch


class ScriptedTripod:
    """Tripod gait as joint-position deltas [rad] from the standing pose, as a function of a phase.

    Tripod A (lf, rm, lr) and tripod B (lm, rf, rr) are half a cycle apart. In a leg's stance half the coxa sweeps
    +A -> -A (pushing the body); in its swing half it returns -A -> +A while femur/tibia lift with a half-sine.
    Left and right legs sweep in opposite directions. Sweep sign +1 moves the body along -y (measured in the overnight
    Phase 0 open-loop test).

    Parameters (speed sweep, decisions.md P9): ``hip_max`` caps the coxa sweep amplitude; the gait period is the
    phase clock's period (the env's ``gait_period_s`` when driven by the env clock). Hip amplitude grows linearly
    with the commanded speed: hip = hip_max * |cmd| / speed_at_hip_max, clamped at hip_max. The old mapping was
    hip = |cmd| / 0.45 with hip_max 0.35 (0.2 rad ~ 0.09-0.10 m/s, measured), i.e. speed_at_hip_max = 0.45 * hip_max;
    that stays the default so existing clones are reproducible.
    """

    def __init__(
        self,
        env,
        hip_max: float = 0.35,
        lift_femur: float = 0.35,
        lift_tibia: float = -0.30,
        speed_at_hip_max: float | None = None,
    ):
        legs = ("lf", "lm", "lr", "rf", "rm", "rr")
        jn = env._robot.joint_names
        self.coxa = [jn.index(f"{l}_coxa_yaw") for l in legs]
        self.femur = [jn.index(f"{l}_femur_pitch") for l in legs]
        self.tibia = [jn.index(f"{l}_tibia_pitch") for l in legs]
        dev = env.device
        self.side = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=dev).unsqueeze(0)
        self.offsets = torch.tensor([0.0, math.pi, 0.0, math.pi, 0.0, math.pi], device=dev).unsqueeze(0)
        self.hip_max, self.lift_femur, self.lift_tibia = hip_max, lift_femur, lift_tibia
        self.speed_at_hip_max = 0.45 * hip_max if speed_at_hip_max is None else speed_at_hip_max
        self.n_joints = len(jn)
        self.cmd_max = env.command_lin_vel_max

    def delta(self, env, phase: torch.Tensor, hip: torch.Tensor | float | None = None) -> torch.Tensor:
        """Joint deltas for every env at ``phase`` [rad]. ``hip`` overrides the command-derived amplitude."""
        ph = torch.remainder((phase.unsqueeze(1) + self.offsets) / (2 * math.pi), 1.0)  # (N, 6) in [0, 1)
        stance = ph < 0.5
        s_st, s_sw = ph / 0.5, (ph - 0.5) / 0.5
        sweep = torch.where(stance, 1 - 2 * s_st, -1 + 2 * s_sw)
        lift = torch.where(stance, torch.zeros_like(ph), torch.sin(math.pi * s_sw.clamp(0, 1)))
        cmd_y = env._commands[:, 1]
        moving = (cmd_y.abs() > 0.1 * self.cmd_max).float()
        dir_sign = -torch.sign(cmd_y) * moving
        if hip is None:
            hip = (cmd_y.abs() * self.hip_max / self.speed_at_hip_max).clamp(max=self.hip_max)
        else:
            hip = torch.as_tensor(hip, device=cmd_y.device, dtype=cmd_y.dtype).expand_as(cmd_y)
        d = torch.zeros(env.num_envs, self.n_joints, device=env.device)
        d[:, self.coxa] = (dir_sign * hip).unsqueeze(1) * sweep * self.side
        d[:, self.femur] = moving.unsqueeze(1) * self.lift_femur * lift
        d[:, self.tibia] = moving.unsqueeze(1) * self.lift_tibia * lift
        return d
