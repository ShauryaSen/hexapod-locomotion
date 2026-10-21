# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Change schedules (decisions.md E1): Poisson arrivals, each change toggles one of {payload, motor, friction}
on/off, >= 40% nominal time, ~6 changes per deployment, at least one recurrence. Saved as JSON next to the run.
Applied through the env's runtime setters (set_payload / set_motor_strength / set_friction, hold=True)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .cfg import ScheduleCfg


@dataclass
class ChangeEvent:
    t_s: float
    factor: str  # payload | motor | friction
    on: bool


@dataclass
class Schedule:
    seed: int
    length_s: float
    events: list[ChangeEvent] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "length_s": self.length_s, "events": [e.__dict__ for e in self.events], "cfg": self.cfg}

    @staticmethod
    def from_dict(d: dict) -> "Schedule":
        return Schedule(d["seed"], d["length_s"], [ChangeEvent(**e) for e in d["events"]], d.get("cfg", {}))

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @staticmethod
    def load(path: str) -> "Schedule":
        return Schedule.from_dict(json.load(open(path)))

    def state_at(self, t_s: float) -> dict[str, bool]:
        st = {"payload": False, "motor": False, "friction": False}
        for e in self.events:
            if e.t_s <= t_s:
                st[e.factor] = e.on
        return st

    def nominal_fraction(self) -> float:
        """Fraction of the deployment with every factor off."""
        ts = [0.0] + [e.t_s for e in self.events] + [self.length_s]
        nominal = 0.0
        for a, b in zip(ts[:-1], ts[1:]):
            if not any(self.state_at(a).values()):
                nominal += b - a
        return nominal / self.length_s

    def onsets(self, factor: str) -> list[float]:
        return [e.t_s for e in self.events if e.factor == factor and e.on]

    def has_recurrence(self) -> bool:
        return any(len(self.onsets(f)) >= 2 for f in ("payload", "motor", "friction"))

    def change_times(self) -> list[float]:
        return [e.t_s for e in self.events]


def generate_schedule(seed: int, cfg: ScheduleCfg, max_tries: int = 10000) -> Schedule:
    """Rejection-sampled Poisson schedule. The deployment length is n_changes_target x mean gap (+ warm-up)."""
    rng = np.random.default_rng(seed)
    mean_gap = cfg.mean_gap_factor * cfg.t50_min * 60.0
    length = cfg.warmup_s + cfg.n_changes_target * mean_gap
    factors = list(cfg.factors)
    for _ in range(max_tries):
        t = cfg.warmup_s
        events = []
        state = {f: False for f in factors}
        while True:
            t += rng.exponential(mean_gap)
            if t >= length:
                break
            f = factors[rng.integers(len(factors))]
            state[f] = not state[f]
            events.append(ChangeEvent(float(t), f, bool(state[f])))
        s = Schedule(seed, float(length), events, cfg.__dict__.copy())
        if not (cfg.n_changes_min <= len(events) <= cfg.n_changes_max):
            continue
        if s.nominal_fraction() < cfg.min_nominal_fraction:
            continue
        if cfg.require_recurrence and not s.has_recurrence():
            continue
        s.cfg["tries"] = _ + 1
        return s
    raise RuntimeError(f"no schedule satisfied the constraints in {max_tries} draws (seed {seed})")


def factor_value(factor: str, on: bool, cfg: ScheduleCfg, nominal_friction: float = 1.0) -> float:
    if factor == "payload":
        return cfg.payload_kg if on else 0.0
    if factor == "motor":
        return cfg.motor_strength if on else 1.0
    if factor == "friction":
        return cfg.friction if on else nominal_friction
    raise KeyError(factor)


class ScheduleApplier:
    """Applies per-env schedules at step granularity through the env setters and tracks the true world state."""

    def __init__(self, env, schedules: list[Schedule | None], cfg: ScheduleCfg):
        import torch

        self.env, self.schedules, self.cfg = env, schedules, cfg
        self.n = len(schedules)
        self.next_idx = [0] * self.n
        self.nominal_friction = float(env.cfg.sim.physics_material.static_friction)
        dev = env.device
        self.payload = torch.zeros(self.n, device=dev)
        self.motor = torch.ones(self.n, device=dev)
        self.friction = torch.full((self.n,), self.nominal_friction, device=dev)

    def apply_due(self, t_s: float) -> list[tuple[int, ChangeEvent]]:
        import torch

        due: dict[str, list[tuple[int, float]]] = {"payload": [], "motor": [], "friction": []}
        applied = []
        for i, s in enumerate(self.schedules):
            if s is None:
                continue
            while self.next_idx[i] < len(s.events) and s.events[self.next_idx[i]].t_s <= t_s:
                e = s.events[self.next_idx[i]]
                due[e.factor].append((i, factor_value(e.factor, e.on, self.cfg, self.nominal_friction)))
                applied.append((i, e))
                self.next_idx[i] += 1
        dev = self.env.device
        for factor, items in due.items():
            if not items:
                continue
            ids = torch.tensor([i for i, _ in items], device=dev)
            vals = torch.tensor([v for _, v in items], device=dev)
            if factor == "payload":
                self.env.set_payload(ids, vals, hold=True)
                self.payload[ids] = vals
            elif factor == "motor":
                self.env.set_motor_strength(ids, vals, hold=True)
                self.motor[ids] = vals
            else:
                self.env.set_friction(ids, vals, hold=True)
                self.friction[ids] = vals
        return applied

    def set_all(self, payload: float | None = None, motor: float | None = None, friction: float | None = None, env_ids=None):
        """Direct setting for tools (e.g. the payload-on T50 run)."""
        import torch

        ids = torch.arange(self.n, device=self.env.device) if env_ids is None else env_ids
        if payload is not None:
            self.env.set_payload(ids, payload, hold=True)
            self.payload[ids] = payload
        if motor is not None:
            self.env.set_motor_strength(ids, motor, hold=True)
            self.motor[ids] = motor
        if friction is not None:
            self.env.set_friction(ids, friction, hold=True)
            self.friction[ids] = friction
