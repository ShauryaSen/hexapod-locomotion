# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one shared updater (decisions.md F7): KL-anchored PPO on actor + critic, per deployment, from that
deployment's own stream only. N deployments are updated in one batched pass; every quantity (advantage
normalization, gradient clipping, learning rate, Adam moments, buffer pointer) is per deployment, so the result is
identical to N separate PPO fine-tunes. Mirrors rsl_rl 5.0.1 PPO (clipped surrogate, clipped value loss, entropy
bonus, GAE, optional adaptive LR on desired_kl) plus the anchor term kl_anchor_weight * KL(pi_theta || pi_0)."""

from __future__ import annotations

import torch

from .batched import BatchedAdam, BatchedCritic, BatchedGaussianActor, BatchedMLP
from .cfg import UpdaterCfg


class KLAnchoredPPO:
    def __init__(self, actor: BatchedGaussianActor, critic: BatchedCritic, pi0_actor: dict, cfg: UpdaterCfg, device, obs_dim: int):
        self.cfg = cfg
        self.actor, self.critic = actor, critic
        self.n = actor.n
        self.device = device
        # frozen pi_0 (one slice, shared by all deployments: they start from the same checkpoint)
        self.pi0 = BatchedGaussianActor(1, obs_dim, tuple(w.shape[1] for w in actor.mlp.weights[:-1]), actor.act_dim).to(device)
        self.pi0.load_rsl_rl(pi0_actor)
        for p in self.pi0.parameters():
            p.requires_grad_(False)
        self.pi0_state = {k: v.clone() for k, v in pi0_actor.items()}
        params = actor.parameters_per_slice() + (critic.parameters_per_slice() if cfg.update_critic else [])
        if not cfg.train_std:
            params = [p for p in params if p is not actor.std_param]
        self.params = params
        self.opt = BatchedAdam(params, self.n, cfg.learning_rate)
        T = cfg.steps_per_batch
        self.T = T  # buffer length = the largest per-deployment batch (set_batch_steps can raise it before the run)
        A = actor.act_dim
        z = lambda *s: torch.zeros(*s, device=device)
        self.buf = {
            "obs": z(T, self.n, obs_dim), "act": z(T, self.n, A), "logp": z(T, self.n), "value": z(T, self.n),
            "rew": z(T, self.n), "done": z(T, self.n), "mean": z(T, self.n, A), "std": z(T, self.n, A),
        }
        self.ptr = torch.zeros(self.n, dtype=torch.long, device=device)
        self.n_updates = torch.zeros(self.n, dtype=torch.long, device=device)
        self.last_kl_pi0 = torch.full((self.n,), float("nan"), device=device)
        self.last_stats: dict[str, torch.Tensor] = {}
        self._ar = torch.arange(self.n, device=device)
        # per-deployment overrides (used by tools/deploy_sweep_updater.py to sweep nuisance knobs in one process;
        # defaults reproduce the shared cfg exactly)
        self.kl_w = torch.full((self.n,), float(cfg.kl_anchor_weight), device=device)
        self.explore_enabled = torch.ones(self.n, dtype=torch.bool, device=device)
        self.update_enabled = torch.ones(self.n, dtype=torch.bool, device=device)
        self.batch_steps = torch.full((self.n,), T, dtype=torch.long, device=device)

    def set_batch_steps(self, mask: torch.Tensor, steps: int):
        """Per-deployment PPO batch length (robot steps); grows the buffer if needed. Call before the run."""
        if steps > self.T:
            for k, v in self.buf.items():
                pad = torch.zeros((steps - self.T, *v.shape[1:]), device=v.device, dtype=v.dtype)
                self.buf[k] = torch.cat([v, pad], 0)
            self.T = steps
        self.batch_steps[mask] = steps

    # ---- acting ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, explore: torch.Tensor, gen: torch.Generator | None = None):
        """``obs`` (n, obs). Returns act, logp, value, mean, std. Exploration noise only where ``explore``."""
        mean, std = self.actor(obs)
        noise = torch.randn(mean.shape, device=obs.device, generator=gen)
        explore = explore & self.explore_enabled
        act = torch.where(explore.unsqueeze(1), mean + std * noise, mean)
        logp = self.actor.log_prob(mean, std, act)
        value = self.critic(obs)
        return act, logp, value, mean, std

    # ---- storage --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def store(self, mask: torch.Tensor, obs, act, logp, value, rew, done, mean, std):
        """Append one transition for deployments in ``mask`` (adapting, not recovering)."""
        idx = self._ar[mask]
        if len(idx) == 0:
            return
        p = self.ptr[idx]
        self.buf["obs"][p, idx] = obs[idx]
        self.buf["act"][p, idx] = act[idx]
        self.buf["logp"][p, idx] = logp[idx]
        self.buf["value"][p, idx] = value[idx]
        self.buf["rew"][p, idx] = rew[idx]
        self.buf["done"][p, idx] = done[idx].float()
        self.buf["mean"][p, idx] = mean[idx]
        self.buf["std"][p, idx] = std[idx]
        self.ptr[idx] = p + 1

    @torch.no_grad()
    def truncate(self, mask: torch.Tensor, next_obs: torch.Tensor):
        """Mark the last stored transition of ``mask`` deployments as a time-out: bootstrap with V(next_obs) and
        set done (used at repositioning teleports and when the arm switches a deployment to hold)."""
        idx = self._ar[mask & (self.ptr > 0)]
        if len(idx) == 0:
            return
        p = self.ptr[idx] - 1
        already = self.buf["done"][p, idx] > 0.5
        v = self.critic(next_obs)[idx]
        self.buf["rew"][p, idx] += torch.where(already, torch.zeros_like(v), self.cfg.gamma * v)
        self.buf["done"][p, idx] = 1.0

    # ---- update ---------------------------------------------------------------------------------------------
    def maybe_update(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Update every deployment whose buffer is full. Returns the mask of updated deployments."""
        full = self.ptr >= self.batch_steps
        if not bool(full.any()):
            return full
        todo = full & self.update_enabled
        for T_i in torch.unique(self.batch_steps[todo]).tolist():
            self._update(todo & (self.batch_steps == T_i), next_obs, int(T_i))
        self.ptr[full] = 0
        done = full & self.update_enabled
        self.n_updates[done] += 1
        return done

    def _gae(self, full: torch.Tensor, next_obs: torch.Tensor, T: int):
        cfg = self.cfg
        with torch.no_grad():
            last_v = self.critic(next_obs)
        adv = torch.zeros(T, self.n, device=self.device)
        gae = torch.zeros(self.n, device=self.device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else self.buf["value"][t + 1]
            nonterm = 1.0 - self.buf["done"][t]
            delta = self.buf["rew"][t] + cfg.gamma * nv * nonterm - self.buf["value"][t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + self.buf["value"][:T]
        # per-deployment advantage normalization over its own T samples
        adv = (adv - adv.mean(0, keepdim=True)) / (adv.std(0, keepdim=True) + 1e-8)
        return adv, ret

    def _update(self, full: torch.Tensor, next_obs: torch.Tensor, T: int | None = None):
        cfg = self.cfg
        T = self.T if T is None else T
        adv, ret = self._gae(full, next_obs, T)
        obs_all = self.buf["obs"][:T]  # (T, n, obs)
        with torch.no_grad():
            pi0_mean, pi0_std = self.pi0(obs_all.reshape(T * self.n, -1).unsqueeze(0))
            pi0_mean = pi0_mean.reshape(T, self.n, -1)
            pi0_std = pi0_std.reshape(T, self.n, -1)
        fullf = full.float()
        chunk = T // cfg.num_mini_batches
        stats = {k: torch.zeros(self.n, device=self.device) for k in ("surrogate", "value_loss", "entropy", "kl_pi0", "kl_old", "grad_norm")}
        n_mb = 0
        for _ in range(cfg.num_learning_epochs):
            order = torch.randperm(cfg.num_mini_batches)
            for c in order.tolist():
                sl = slice(c * chunk, (c + 1) * chunk if c < cfg.num_mini_batches - 1 else T)
                obs_b = obs_all[sl].transpose(0, 1)  # (n, b, obs)
                act_b = self.buf["act"][sl].transpose(0, 1)
                old_logp = self.buf["logp"][sl].transpose(0, 1)
                old_v = self.buf["value"][sl].transpose(0, 1)
                adv_b = adv[sl].transpose(0, 1)
                ret_b = ret[sl].transpose(0, 1)
                old_mean = self.buf["mean"][sl].transpose(0, 1)
                old_std = self.buf["std"][sl].transpose(0, 1)
                mean, std = self.actor(obs_b)
                logp = self.actor.log_prob(mean, std, act_b)
                ratio = torch.exp(logp - old_logp)
                surr = -torch.min(ratio * adv_b, ratio.clamp(1 - cfg.clip_param, 1 + cfg.clip_param) * adv_b).mean(1)  # (n,)
                value = self.critic(obs_b)
                if cfg.use_clipped_value_loss:
                    v_clipped = old_v + (value - old_v).clamp(-cfg.clip_param, cfg.clip_param)
                    v_loss = torch.max((value - ret_b) ** 2, (v_clipped - ret_b) ** 2).mean(1)
                else:
                    v_loss = ((value - ret_b) ** 2).mean(1)
                ent = self.actor.entropy(std).mean(1)
                kl0 = self.actor.kl(mean, std, pi0_mean[sl].transpose(0, 1), pi0_std[sl].transpose(0, 1)).mean(1)
                loss_per = surr + cfg.value_loss_coef * v_loss - cfg.entropy_coef * ent + self.kl_w * kl0
                loss = (loss_per * fullf).sum()
                self.opt.zero_grad()
                loss.backward()
                gn = self.opt.clip_grad_norm(cfg.max_grad_norm, full)
                self.opt.step(full)
                with torch.no_grad():
                    kl_old = self.actor.kl(old_mean, old_std, mean, std).mean(1)
                    if cfg.lr_schedule == "adaptive":
                        up = kl_old > cfg.desired_kl * 2.0
                        down = (kl_old < cfg.desired_kl / 2.0) & (kl_old > 0.0)
                        self.opt.lr = torch.where(full & up, (self.opt.lr / 1.5).clamp_min(1e-5), self.opt.lr)
                        self.opt.lr = torch.where(full & down, (self.opt.lr * 1.5).clamp_max(1e-2), self.opt.lr)
                    stats["surrogate"] += surr.detach()
                    stats["value_loss"] += v_loss.detach()
                    stats["entropy"] += ent.detach()
                    stats["kl_pi0"] += kl0.detach()
                    stats["kl_old"] += kl_old
                    stats["grad_norm"] += gn
                n_mb += 1
        for k in stats:
            stats[k] /= n_mb
        self.last_stats = stats
        self.last_kl_pi0 = torch.where(full, stats["kl_pi0"], self.last_kl_pi0)

    # ---- resets ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def reset_to_pi0(self, mask: torch.Tensor, pi0_critic: dict | None = None):
        """Weights of ``mask`` deployments back to pi_0, Adam moments and buffers cleared (periodic-reset arm)."""
        idx = self._ar[mask]
        if len(idx) == 0:
            return
        self.actor.load_rsl_rl(self.pi0_state, idx=idx)
        if pi0_critic is not None:
            self.critic.load_rsl_rl(pi0_critic, idx=idx)
        self.opt.reset_slices(idx)
        self.opt.lr[idx] = self.cfg.learning_rate
        self.ptr[idx] = 0
        self.last_kl_pi0[idx] = float("nan")

    @torch.no_grad()
    def drop_buffer(self, mask: torch.Tensor):
        self.ptr[mask] = 0
