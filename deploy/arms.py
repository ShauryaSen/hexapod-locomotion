# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptation-timing arms (decisions.md F6). Every arm is a policy over *when* the one shared updater runs: at
each gait-cycle boundary it returns, per deployment, whether to adapt during the next cycle, whether an
adaptation episode starts now (alarm) and whether the policy is reset to pi_0. The runner does the rest.

Budget: an adaptation episode may spend ``budget_min`` robot-minutes adapting (F7: 2 x T50). Triggered arms also
stop early on convergence (|z| inside the nominal band for N cycles).
"""

from __future__ import annotations

import torch

from .cfg import ArmCfg, DetectorCfg, UpdaterCfg
from .detectors import ConvergenceStop, Cusum, ThreeSigma
from .schedule import Schedule

ARM_NAMES = ("never", "always", "scheduled", "periodic_reset", "pause_hold", "three_sigma", "cusum", "oracle")
# "triggered" (ours = CUSUM on the ensemble residual) was dropped as a separate arm (decisions.md R5, 2026-10-21).


class Arm:
    name = "base"
    uses_convergence_stop = False

    def __init__(self, n: int, cfg: ArmCfg, upd_cfg: UpdaterCfg, det_cfg: DetectorCfg, device, cycle_s: float, schedules: list[Schedule | None] | None = None):
        self.n, self.cfg, self.device, self.cycle_s = n, cfg, device, cycle_s
        self.budget_min = cfg.budget_min if cfg.budget_min is not None else upd_cfg.budget_min
        self.det_cfg = det_cfg
        self.schedules = schedules
        self.adapting = torch.zeros(n, dtype=torch.bool, device=device)
        self.budget_used = torch.zeros(n, device=device)  # minutes in the current episode
        self.episodes = torch.zeros(n, dtype=torch.long, device=device)
        self.converge = ConvergenceStop(n, det_cfg, device) if self.uses_convergence_stop else None
        self._f = torch.zeros(n, dtype=torch.bool, device=device)
        self.rebaseline_on_stop = self.uses_convergence_stop  # R5: detector arms re-measure the nominal level

    # ---- hooks for subclasses ---------------------------------------------------------------------------------
    def trigger(self, t_s: float, z: torch.Tensor) -> torch.Tensor:
        """Deployments whose adaptation episode should start (or restart) now."""
        return self._f

    def reset_policy(self, t_s: float) -> torch.Tensor:
        return self._f

    def stop_rule(self, z: torch.Tensor) -> torch.Tensor:
        """Extra stop condition besides the budget (evaluated only for adapting deployments)."""
        if self.converge is not None:
            return self.converge.step(z)
        return self._f

    # ---- the per-cycle decision --------------------------------------------------------------------------------
    def cycle(self, t_s: float, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Called at the end of each cycle with that cycle's z. Updates the budget for the cycle just finished and
        decides the state for the next cycle."""
        self.budget_used += self.adapting.float() * (self.cycle_s / 60.0)
        stop = self.adapting & ((self.budget_used >= self.budget_min - 1e-9) | self.stop_rule(z))
        self.adapting = self.adapting & ~stop
        start = self.trigger(t_s, z)
        start = start & ~stop  # a stop and a start in the same cycle: the stop wins (no zero-length episodes)
        new = start & ~self.adapting
        restart = start & self.adapting
        self.adapting = self.adapting | start
        self.budget_used = torch.where(start, torch.zeros_like(self.budget_used), self.budget_used)
        self.episodes += start.long()
        if self.converge is not None:
            self.converge.reset(start)
        reset = self.reset_policy(t_s)
        return {"adapt": self.adapting.clone(), "alarm": start.clone(), "new": new, "restart": restart, "stopped": stop, "reset": reset}


class Never(Arm):
    name = "never"


class Always(Arm):
    name = "always"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.budget_min = float("inf")
        self._first = True

    def trigger(self, t_s, z):
        if self._first:
            self._first = False
            return ~self._f
        return self._f


class Scheduled(Arm):
    """Adapt for one budget every ``period_min`` robot-minutes (first episode at t = period)."""

    name = "scheduled"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.next_t = self.cfg.period_min * 60.0

    def trigger(self, t_s, z):
        if t_s >= self.next_t - 1e-6:
            self.next_t += self.cfg.period_min * 60.0
            return ~self._f
        return self._f


class PeriodicReset(Always):
    """Always adapting; every ``period_min`` the policy (and optimizer) is reset to pi_0."""

    name = "periodic_reset"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.next_t = self.cfg.period_min * 60.0

    def reset_policy(self, t_s):
        if t_s >= self.next_t - 1e-6:
            self.next_t += self.cfg.period_min * 60.0
            return ~self._f
        return self._f


class PauseHold(Arm):
    """Adapt for one budget, hold for ``hold_min``, repeat; first episode at t = 0."""

    name = "pause_hold"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.next_t = 0.0

    def cycle(self, t_s, z):
        out = super().cycle(t_s, z)
        # schedule the next episode hold_min after a stop
        if bool(out["stopped"].any()):
            self.next_t = t_s + self.cfg.hold_min * 60.0
        return out

    def trigger(self, t_s, z):
        if t_s >= self.next_t - 1e-6:
            self.next_t = float("inf")  # re-armed by the next stop
            return ~self._f
        return self._f


class ThreeSigmaArm(Arm):
    name = "three_sigma"
    uses_convergence_stop = True

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        det_cfg = self.det_cfg
        if self.cfg.three_sigma_consecutive is not None:
            det_cfg = DetectorCfg(**{**det_cfg.__dict__, "three_sigma_consecutive": self.cfg.three_sigma_consecutive})
        self.det = ThreeSigma(self.n, det_cfg, self.device)

    def trigger(self, t_s, z):
        alarm = self.det.step(z)
        return alarm & ~self.adapting  # alarms while adapting are ignored


class CusumArm(Arm):
    name = "cusum"
    uses_convergence_stop = True

    def __init__(self, *a, cusum_h: float | None = None, **k):
        super().__init__(*a, **k)
        h = cusum_h if cusum_h is not None else self.cfg.cusum_h
        self.det = Cusum(self.n, self.det_cfg, self.device, h=h)

    def cycle(self, t_s, z):
        out = super().cycle(t_s, z)
        self.det.reset(out["stopped"])  # start clean after an episode
        return out

    def trigger(self, t_s, z):
        alarm = self.det.step(z)
        self.det.reset(self.adapting)  # no accumulation while adapting
        return alarm & ~self.adapting


class Oracle(Arm):
    """Adapts for one budget from every true change time (restart on a change during an episode)."""

    name = "oracle"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        assert self.schedules is not None, "the oracle arm needs the schedules"
        self.next_idx = [0] * self.n

    def trigger(self, t_s, z):
        start = self._f.clone()
        for i, s in enumerate(self.schedules):
            if s is None:
                continue
            fired = False
            while self.next_idx[i] < len(s.events) and s.events[self.next_idx[i]].t_s <= t_s:
                self.next_idx[i] += 1
                fired = True
            start[i] = fired
        return start


def arm_label(cfg: ArmCfg, det_cfg: DetectorCfg) -> str:
    """Arm name with its knob (R6), e.g. scheduled(T=10), cusum(h=6.75), three_sigma(m=3)."""
    if cfg.name in ("scheduled", "periodic_reset"):
        return f"{cfg.name}(T={cfg.period_min:g})"
    if cfg.name == "pause_hold":
        return f"pause_hold(H={cfg.hold_min:g})"
    if cfg.name == "cusum":
        return f"cusum(h={(cfg.cusum_h if cfg.cusum_h is not None else det_cfg.cusum_h):g})"
    if cfg.name == "three_sigma":
        return f"three_sigma(m={cfg.three_sigma_consecutive if cfg.three_sigma_consecutive is not None else det_cfg.three_sigma_consecutive})"
    return cfg.name


def make_arm(name: str, n: int, cfg: ArmCfg, upd_cfg: UpdaterCfg, det_cfg: DetectorCfg, device, cycle_s: float, schedules=None, cusum_h: float | None = None) -> Arm:
    cls = {"never": Never, "always": Always, "scheduled": Scheduled, "periodic_reset": PeriodicReset, "pause_hold": PauseHold, "three_sigma": ThreeSigmaArm, "cusum": CusumArm, "oracle": Oracle}[name]
    if cls is CusumArm:
        return cls(n, cfg, upd_cfg, det_cfg, device, cycle_s, schedules, cusum_h=cusum_h)
    return cls(n, cfg, upd_cfg, det_cfg, device, cycle_s, schedules)
