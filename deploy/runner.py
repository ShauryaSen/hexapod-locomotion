# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deployment loop: N deployments (one env each) with their own policy slices, fixed command, no time-outs,
fall -> recovery reset (logged, costs ``fall_recovery_s`` of standing still), periodic repositioning to the terrain
cell, gait-cycle streams, the arm's adapt/hold decisions and the shared updater."""

from __future__ import annotations

import math
import time

import torch

from .arms import Arm
from .cfg import DeployCfg
from .detectors import Baseliner
from .ensemble import DynamicsEnsemble
from .envutil import forward_speed, pin_commands, policy_obs, reset_in_place
from .schedule import ScheduleApplier
from .streams import StreamLog
from .updater import KLAnchoredPPO


class DeploymentRunner:
    def __init__(self, env, cfg: DeployCfg, updater: KLAnchoredPPO, arm: Arm, applier: ScheduleApplier, ensemble: DynamicsEnsemble | None = None, pi0_critic: dict | None = None, log=print, preproc=None):
        self.env, self.cfg, self.updater, self.arm, self.applier, self.ensemble = env, cfg, updater, arm, applier, ensemble
        self.preproc = preproc  # R1: RMA history encoder (frozen) -> policy input = cat(obs, z); None = plain policy
        self.pi0_critic = pi0_critic
        self.log = log
        self.n = env.num_envs
        self.dev = env.device
        self.dt = float(env.step_dt)
        self.steps_per_cycle = int(round(cfg.gait_cycle_s / self.dt))
        self.recovery_steps = int(round(cfg.fall_recovery_s / self.dt))
        self.repo_steps = int(round(cfg.reposition_s / self.dt)) if cfg.reposition_s > 0 else 0
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(cfg.env_seed))
        self.stream = StreamLog(self.n, self.dev)
        self.events: list[dict] = []
        self.ids = torch.arange(self.n, device=self.dev)
        self.calibrated = ensemble is not None and ensemble.is_calibrated()
        # R5 per-deployment baseline (log residual); 0 cycles = global calibration only
        nb = int(getattr(arm.det_cfg, "baseline_cycles", 0))
        self.baseliner = None
        if self.calibrated and nb > 0:
            mu_g, sd_g = (float(ensemble.nominal_mu_log), float(ensemble.nominal_sigma_log)) if ensemble.cfg.residual_transform == "log" else (float(ensemble.nominal_mu), float(ensemble.nominal_sigma))
            self.baseliner = Baseliner(self.n, nb, sd_g, mu_g, self.dev)


    def run(self, minutes: float, progress_every_min: float = 1.0) -> dict:
        env, cfg, upd, arm = self.env, self.cfg, self.updater, self.arm
        n, dev, dt = self.n, self.dev, self.dt
        total_steps = int(round(minutes * 60.0 / dt))
        obs = policy_obs(env)
        recover_left = torch.zeros(n, dtype=torch.long, device=dev)
        zeros = torch.zeros(n, device=dev)
        state = arm.cycle(0.0, torch.full((n,), float("nan"), device=dev))
        adapt = state["adapt"]
        pending_alarm, pending_reset = state["alarm"].float(), state["reset"].float()
        if bool(state["reset"].any()):
            upd.reset_to_pi0(state["reset"], self.pi0_critic)
        acc = self._new_acc()
        t_wall = time.time()
        next_print = progress_every_min * 60.0
        total_prog = torch.zeros(n, device=dev)
        total_falls = torch.zeros(n, device=dev)
        for step in range(total_steps):
            t = step * dt
            for i, e in self.applier.apply_due(t):
                self.events.append({"env": i, "t_s": t, "factor": e.factor, "on": e.on})
            recovering = recover_left > 0
            explore = (adapt if cfg.explore_only_when_adapting else torch.ones_like(adapt)) & ~recovering
            pobs = self.preproc(obs) if self.preproc is not None else obs
            act, logp, value, mean, std = upd.act(pobs, explore, self.gen)
            act = torch.where(recovering.unsqueeze(1), torch.zeros_like(act), act)  # hold the default pose
            obs_next, rew, term, trunc, _ = env.step(act)
            obs_next = obs_next["policy"]
            fell = term.bool()
            v = forward_speed(env)
            walking = ~recovering & ~fell
            v_prog = v.clamp_max(cfg.cmd_speed) if cfg.cap_progress_at_cmd else v  # R4: capped at the command
            acc["prog"] += torch.where(walking, v_prog * dt, zeros)
            acc["track"] += torch.where(walking, (v - cfg.cmd_speed).abs(), zeros)
            acc["track_n"] += walking.float()
            acc["rew"] += rew
            acc["falls"] += fell.float()
            acc["recov"] += recovering.float()
            if self.ensemble is not None:
                nmse, dis = self.ensemble.residual_step(obs, act, obs_next)
                acc["res"] += torch.where(walking, nmse, zeros)
                acc["dis"] += torch.where(walking, dis, zeros)
                acc["res_n"] += walking.float()
            store_mask = adapt & ~recovering
            upd.store(store_mask, pobs, act, logp, value, rew, fell, mean, std)
            if bool(fell.any()):
                # env.step already reset the fallen robots at their cells (random command): pin and re-observe
                recover_left = torch.where(fell, torch.full_like(recover_left, self.recovery_steps), recover_left)
                pin_commands(env, cfg.cmd_speed)
                obs_next = policy_obs(env)
            if self.repo_steps and (step + 1) % self.repo_steps == 0:
                upd.truncate(store_mask, self._peek(obs_next))
                reset_in_place(env, self.ids, cfg.cmd_speed)
                obs_next = policy_obs(env)
                acc["repos"] += 1.0
                if self.preproc is not None:
                    self.preproc.reset(torch.ones(n, dtype=torch.bool, device=dev))
            if self.preproc is not None and bool(fell.any()):
                self.preproc.reset(fell)
            updated = upd.maybe_update(self._peek(obs_next))
            acc["updates"] += updated.float()
            recover_left = (recover_left - 1).clamp_min(0)
            obs = obs_next
            if (step + 1) % self.steps_per_cycle == 0:
                t_end = (step + 1) * dt
                adapt, pending_alarm, pending_reset = self._end_cycle(t_end, acc, adapt, pending_alarm, pending_reset, self._peek(obs))
                total_prog += acc["prog"]
                total_falls += acc["falls"]
                acc = self._new_acc()
            if t + dt >= next_print - 1e-9:
                el = time.time() - t_wall
                self.log(f"[deploy] {next_print / 60:.0f} min robot time, {el / 60:.1f} min wall ({(step + 1) / el:.1f} steps/s); progress mean {total_prog.mean():.2f} m, falls {int(total_falls.sum())}, adapting {int(adapt.sum())}/{n}, updates {int(upd.n_updates.sum())}")
                next_print += progress_every_min * 60.0
        wall = time.time() - t_wall
        return {"minutes": minutes, "steps": total_steps, "wall_s": wall, "steps_per_s": total_steps / wall, "progress_mean_m": total_prog.mean().item(), "progress_per_env_m": total_prog.tolist(), "falls_total": int(total_falls.sum().item()), "falls_per_env": total_falls.tolist(), "updates_per_env": upd.n_updates.tolist(), "events": self.events}

    def _peek(self, obs):
        """Policy input for a next-obs value function without advancing the history buffer."""
        if self.preproc is None:
            return obs
        hist = torch.cat([self.preproc.hist[:, 1:], obs.unsqueeze(1)], dim=1)
        with torch.no_grad():
            return torch.cat([obs, self.preproc.encoder(hist)], dim=-1)

    def _new_acc(self):
        z = lambda: torch.zeros(self.n, device=self.dev)
        return {k: z() for k in ("prog", "track", "track_n", "rew", "falls", "recov", "res", "dis", "res_n", "updates", "repos")}

    def _end_cycle(self, t_end, acc, adapt, pending_alarm, pending_reset, obs):
        s, n = self.stream, self.n
        nanv = torch.full((n,), float("nan"), device=self.dev)
        residual = torch.where(acc["res_n"] > 0, acc["res"] / acc["res_n"].clamp_min(1), nanv) if self.ensemble is not None else nanv
        dis = torch.where(acc["res_n"] > 0, acc["dis"] / acc["res_n"].clamp_min(1), nanv) if self.ensemble is not None else nanv
        if self.calibrated and self.baseliner is not None:
            x = torch.log(residual.clamp_min(1e-12)) if self.ensemble.cfg.residual_transform == "log" else residual
            z = self.baseliner.step(x)
        else:
            z = self.ensemble.standardize(residual) if self.calibrated else nanv
        s.set("t_s", t_end)
        s.set("progress_m", acc["prog"])
        s.set("track_err", torch.where(acc["track_n"] > 0, acc["track"] / acc["track_n"].clamp_min(1), nanv))
        s.set("reward", acc["rew"])
        s.set("falls", acc["falls"])
        s.set("recovering", acc["recov"] / self.steps_per_cycle)
        s.set("repositioned", (acc["repos"] > 0).float())
        s.set("residual", residual)
        s.set("disagreement", dis)
        s.set("z", z)
        s.set("adapting", adapt.float())
        s.set("alarm", pending_alarm)
        s.set("policy_reset", pending_reset)
        s.set("updates", acc["updates"])
        s.set("kl_pi0", self.updater.last_kl_pi0)
        s.set("payload_kg", self.applier.payload)
        s.set("motor_strength", self.applier.motor)
        s.set("friction", self.applier.friction)
        s.set("budget_used_min", self.arm.budget_used)
        s.set("baseline_mu", self.baseliner.mu if self.baseliner is not None else nanv)
        s.commit()
        state = self.arm.cycle(t_end, z)
        rebase = state.get("rebaseline", state["stopped"] if getattr(self.arm, "rebaseline_on_stop", False) else None)
        if self.baseliner is not None and rebase is not None and bool(rebase.any()):
            self.baseliner.restart(rebase)  # R5: re-measure the nominal level after every detector-arm episode
        new_adapt = state["adapt"]
        ended = adapt & ~new_adapt
        if bool(ended.any()):
            self.updater.truncate(ended, obs)
            self.updater.drop_buffer(ended)  # samples from before a hold are not reused (ASSUMED)
        if bool(state["reset"].any()):
            self.updater.reset_to_pi0(state["reset"], self.pi0_critic)
        started = state["new"] | state["restart"]
        if bool(started.any()):
            self.updater.drop_buffer(started)
        return new_adapt, state["alarm"].float(), state["reset"].float()
