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
    """``slice_of_env`` (option 2, behind a flag): robot e uses policy slice slice_of_env[e]; the k robots of one slice
    feed one shared batch (each robot contributes ``steps_per_batch`` steps, so a batch has k x steps_per_batch samples,
    ASSUMED) and one update. None (default) = one robot per slice = the k = 1 code path, unchanged."""

    def __init__(self, actor: BatchedGaussianActor, critic: BatchedCritic, pi0_actor: dict, cfg: UpdaterCfg, device, obs_dim: int, slice_of_env: torch.Tensor | None = None):
        self.cfg = cfg
        self.actor, self.critic = actor, critic
        self.ns = actor.n
        self.identity = slice_of_env is None
        self.sid = torch.arange(self.ns, device=device) if self.identity else torch.as_tensor(slice_of_env, device=device).long()
        self.n = self.ns if self.identity else len(self.sid)  # robots
        self.k_of_slice = torch.bincount(self.sid, minlength=self.ns).float()
        assert bool((self.k_of_slice > 0).all()), "every slice needs at least one robot"
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
        self.opt = BatchedAdam(params, self.ns, cfg.learning_rate)
        T = cfg.steps_per_batch
        self.T = T  # buffer length = the largest per-deployment batch (set_batch_steps can raise it before the run)
        A = actor.act_dim
        z = lambda *s: torch.zeros(*s, device=device)
        self.buf = {
            "obs": z(T, self.n, obs_dim), "act": z(T, self.n, A), "logp": z(T, self.n), "value": z(T, self.n),
            "rew": z(T, self.n), "done": z(T, self.n), "mean": z(T, self.n, A), "std": z(T, self.n, A),
        }
        self.ptr = torch.zeros(self.n, dtype=torch.long, device=device)  # per robot
        # per slice (= per deployment)
        self.n_updates = torch.zeros(self.ns, dtype=torch.long, device=device)
        self.last_kl_pi0 = torch.full((self.ns,), float("nan"), device=device)
        self.last_stats: dict[str, torch.Tensor] = {}
        self._ar = torch.arange(self.n, device=device)
        self._ars = torch.arange(self.ns, device=device)
        # per-deployment overrides (used by tools/deploy_sweep_updater.py to sweep nuisance knobs in one process;
        # defaults reproduce the shared cfg exactly)
        self.kl_w = torch.full((self.ns,), float(cfg.kl_anchor_weight), device=device)
        self.explore_enabled = torch.ones(self.ns, dtype=torch.bool, device=device)
        self.update_enabled = torch.ones(self.ns, dtype=torch.bool, device=device)
        self.batch_steps = torch.full((self.ns,), T, dtype=torch.long, device=device)
        # slice mode only: bootstrap value of each robot taken when its buffer filled (it then waits for its slice)
        self.boot_v = torch.zeros(self.n, device=device)
        self.has_boot = torch.zeros(self.n, dtype=torch.bool, device=device)
        # replay (GePPO, behind a flag): per-deployment number of policies M per update; 0 = on-policy path
        self.replay_M = torch.zeros(self.ns, dtype=torch.long, device=device)
        self.rb: dict[str, torch.Tensor] | None = None
        self.last_tv = torch.full((self.ns,), float("nan"), device=device)
        self._replay_gen = torch.Generator(device=device)  # own RNG: replay updates do not shift other deployments' minibatch order
        self._replay_gen.manual_seed(0)
        if cfg.replay_batches > 0:
            self.set_replay(torch.ones(self.ns, dtype=torch.bool, device=device), cfg.replay_batches)

    # ---- per-robot views of per-slice quantities ---------------------------------------------------------------
    def env_view(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.identity else x[self.sid]

    def slice_any(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Slices with at least one robot in ``env_mask``."""
        if self.identity:
            return env_mask
        return torch.zeros(self.ns, device=self.device).index_add_(0, self.sid, env_mask.float()) > 0

    def set_param_subset(self, slice_mask: torch.Tensor, subset: str):
        """Option 3a (behind a flag): ``subset`` = 'all' or 'last_layer' (actor output layer + action std; the critic
        stays fully trainable, ASSUMED: the value function has to follow the changed dynamics) for ``slice_mask``."""
        if subset == "all":
            return
        assert subset == "last_layer", subset
        allowed = {id(p) for p in self.actor.last_layer_params()} | {id(p) for p in self.critic.parameters_per_slice()}
        for j, p in enumerate(self.params):
            if id(p) in allowed:
                continue
            cur = self.opt.param_mask[j]
            cur = torch.ones(self.ns, dtype=torch.bool, device=self.device) if cur is None else cur
            self.opt.param_mask[j] = cur & ~slice_mask

    def set_batch_steps(self, mask: torch.Tensor, steps: int):
        """Per-deployment PPO batch length (robot steps per robot; ``mask`` over slices); grows the buffer if needed.
        Call before the run."""
        if steps > self.T:
            for k, v in self.buf.items():
                pad = torch.zeros((steps - self.T, *v.shape[1:]), device=v.device, dtype=v.dtype)
                self.buf[k] = torch.cat([v, pad], 0)
            self.T = steps
        self.batch_steps[mask] = steps

    def set_replay(self, mask: torch.Tensor, M: int):
        """Replay (GePPO) with M policies per update for deployments in ``mask`` (k = 1 only). Allocates a ring of the
        last M batches per robot (obs, action, behaviour log-prob, reward, done, bootstrap obs). The critic's lr is
        fixed at its current value for these deployments; their actor lr adapts by the TV rule (cfg.replay_lr_rule).
        Call before the run."""
        assert self.identity, "replay is implemented for k = 1"
        assert M >= 1
        self.replay_M[mask] = M
        Mx = int(self.replay_M.max())
        if self.rb is None or self.rb["obs"].shape[0] < Mx:
            T, n = self.T, self.n
            z = lambda *s: torch.zeros(*s, device=self.device)
            A, O = self.actor.act_dim, self.buf["obs"].shape[-1]
            self.rb = {"obs": z(Mx, T, n, O), "act": z(Mx, T, n, A), "logp": z(Mx, T, n), "rew": z(Mx, T, n), "done": z(Mx, T, n), "boot_obs": z(Mx, n, O),
                       "head": torch.zeros(n, dtype=torch.long, device=self.device), "count": torch.zeros(n, dtype=torch.long, device=self.device)}
        critic_ids = {id(p) for p in self.critic.parameters_per_slice()}
        for j, p in enumerate(self.params):
            if id(p) in critic_ids:
                cur = self.opt.param_lr[j]
                cur = torch.full_like(self.opt.lr, float("nan")) if cur is None else cur
                self.opt.param_lr[j] = torch.where(mask, self.opt.lr, cur)

    def replay_eps(self) -> torch.Tensor:
        """Per deployment GePPO clip: cfg.replay_clip, or Lemma 4 with uniform weights: clip_param * 2 / (M + 1)."""
        if self.cfg.replay_clip > 0:
            return torch.full((self.ns,), float(self.cfg.replay_clip), device=self.device)
        return self.cfg.clip_param * 2.0 / (self.replay_M.float() + 1.0)

    # ---- acting ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, explore: torch.Tensor, gen: torch.Generator | None = None):
        """``obs`` (n, obs). Returns act, logp, value, mean, std. Exploration noise only where ``explore``."""
        if self.identity:
            mean, std = self.actor(obs)
        else:
            mean, std = self.actor.forward_gather(obs, self.sid)
        noise = torch.randn(mean.shape, device=obs.device, generator=gen)
        explore = explore & self.env_view(self.explore_enabled)
        act = torch.where(explore.unsqueeze(1), mean + std * noise, mean)
        logp = self.actor.log_prob(mean, std, act)
        value = self.critic(obs) if self.identity else self.critic.forward_gather(obs, self.sid)
        return act, logp, value, mean, std

    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs) if self.identity else self.critic.forward_gather(obs, self.sid)

    # ---- storage --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def store(self, mask: torch.Tensor, obs, act, logp, value, rew, done, mean, std):
        """Append one transition for deployments in ``mask`` (adapting, not recovering)."""
        if not self.identity:
            mask = mask & (self.ptr < self.batch_steps[self.sid])  # a full robot waits for the rest of its slice
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
        v = self._value(next_obs)[idx]
        self.buf["rew"][p, idx] += torch.where(already, torch.zeros_like(v), self.cfg.gamma * v)
        self.buf["done"][p, idx] = 1.0

    # ---- update ---------------------------------------------------------------------------------------------
    def maybe_update(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Update every deployment whose buffer is full. Returns the mask of updated deployments (per robot)."""
        if not self.identity:
            return self._maybe_update_slices(next_obs)
        full = self.ptr >= self.batch_steps
        if not bool(full.any()):
            return full
        todo = full & self.update_enabled
        rp = todo & (self.replay_M > 0)
        todo = todo & ~rp
        for T_i in torch.unique(self.batch_steps[todo]).tolist():
            self._update(todo & (self.batch_steps == T_i), next_obs, int(T_i))
        if bool(rp.any()):
            self._push_replay(rp, next_obs)
            self._update_replay(rp)
        self.ptr[full] = 0
        done = full & self.update_enabled
        self.n_updates[done] += 1
        return done

    def _maybe_update_slices(self, next_obs: torch.Tensor) -> torch.Tensor:
        T_env = self.batch_steps[self.sid]
        filled = self.ptr >= T_env
        new = filled & ~self.has_boot
        if bool(new.any()):
            with torch.no_grad():
                self.boot_v = torch.where(new, self._value(next_obs), self.boot_v)
            self.has_boot |= new
        not_full = torch.zeros(self.ns, device=self.device).index_add_(0, self.sid, (~filled).float())
        full = not_full == 0
        if not bool(full.any()):
            return torch.zeros(self.n, dtype=torch.bool, device=self.device)
        todo = full & self.update_enabled
        for T_i in torch.unique(self.batch_steps[todo]).tolist():
            self._update_slices(todo & (self.batch_steps == T_i), int(T_i))
        env_full = full[self.sid]
        self.ptr[env_full] = 0
        self.has_boot[env_full] = False
        done = full & self.update_enabled
        self.n_updates[done] += 1
        return done[self.sid]

    def _slice_mean(self, x_env: torch.Tensor) -> torch.Tensor:
        """Per-slice mean of a per-robot quantity (n,) -> (ns,)."""
        return torch.zeros(self.ns, device=self.device).index_add_(0, self.sid, x_env) / self.k_of_slice

    def _update_slices(self, full: torch.Tensor, T: int):
        """Slice-mode PPO step: like _update, with the k robots of a slice pooled (advantage normalization over the
        slice's k x T samples, loss = mean over its robots, one gradient step per slice)."""
        cfg = self.cfg
        env_full = full[self.sid]
        # GAE per robot with the bootstrap value taken when the robot filled
        adv = torch.zeros(T, self.n, device=self.device)
        gae = torch.zeros(self.n, device=self.device)
        for t in reversed(range(T)):
            nv = self.boot_v if t == T - 1 else self.buf["value"][t + 1]
            nonterm = 1.0 - self.buf["done"][t]
            delta = self.buf["rew"][t] + cfg.gamma * nv * nonterm - self.buf["value"][t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + self.buf["value"][:T]
        cnt = self.k_of_slice * T
        mu = torch.zeros(self.ns, device=self.device).index_add_(0, self.sid, adv.sum(0)) / cnt
        sq = torch.zeros(self.ns, device=self.device).index_add_(0, self.sid, ((adv - mu[self.sid]) ** 2).sum(0))
        sd = (sq / (cnt - 1).clamp_min(1)).sqrt()
        adv = (adv - mu[self.sid]) / (sd[self.sid] + 1e-8)
        obs_all = self.buf["obs"][:T]
        with torch.no_grad():
            pi0_mean, pi0_std = self.pi0(obs_all.reshape(T * self.n, -1).unsqueeze(0))
            pi0_mean = pi0_mean.reshape(T, self.n, -1)
            pi0_std = pi0_std.reshape(T, self.n, -1)
        chunk = T // cfg.num_mini_batches
        stats = {k: torch.zeros(self.ns, device=self.device) for k in ("surrogate", "value_loss", "entropy", "kl_pi0", "kl_old", "grad_norm")}
        n_mb = 0
        for _ in range(cfg.num_learning_epochs):
            order = torch.randperm(cfg.num_mini_batches)
            for c in order.tolist():
                sl = slice(c * chunk, (c + 1) * chunk if c < cfg.num_mini_batches - 1 else T)
                obs_b = obs_all[sl].transpose(0, 1)
                act_b = self.buf["act"][sl].transpose(0, 1)
                old_logp = self.buf["logp"][sl].transpose(0, 1)
                old_v = self.buf["value"][sl].transpose(0, 1)
                adv_b = adv[sl].transpose(0, 1)
                ret_b = ret[sl].transpose(0, 1)
                old_mean = self.buf["mean"][sl].transpose(0, 1)
                old_std = self.buf["std"][sl].transpose(0, 1)
                mean, std = self.actor.forward_gather(obs_b, self.sid)
                logp = self.actor.log_prob(mean, std, act_b)
                ratio = torch.exp(logp - old_logp)
                surr = -torch.min(ratio * adv_b, ratio.clamp(1 - cfg.clip_param, 1 + cfg.clip_param) * adv_b).mean(1)
                value = self.critic.forward_gather(obs_b, self.sid)
                if cfg.use_clipped_value_loss:
                    v_clipped = old_v + (value - old_v).clamp(-cfg.clip_param, cfg.clip_param)
                    v_loss = torch.max((value - ret_b) ** 2, (v_clipped - ret_b) ** 2).mean(1)
                else:
                    v_loss = ((value - ret_b) ** 2).mean(1)
                ent = self.actor.entropy(std).mean(1)
                kl0 = self.actor.kl(mean, std, pi0_mean[sl].transpose(0, 1), pi0_std[sl].transpose(0, 1)).mean(1)
                loss_env = surr + cfg.value_loss_coef * v_loss - cfg.entropy_coef * ent + self.kl_w[self.sid] * kl0
                loss = (loss_env * env_full.float() / self.k_of_slice[self.sid]).sum()  # = sum over slices of the robot mean
                self.opt.zero_grad()
                loss.backward()
                gn = self.opt.clip_grad_norm(cfg.max_grad_norm, full)
                self.opt.step(full)
                with torch.no_grad():
                    kl_old = self._slice_mean(self.actor.kl(old_mean, old_std, mean, std).mean(1))
                    if cfg.lr_schedule == "adaptive":
                        up = kl_old > cfg.desired_kl * 2.0
                        down = (kl_old < cfg.desired_kl / 2.0) & (kl_old > 0.0)
                        self.opt.lr = torch.where(full & up, (self.opt.lr / 1.5).clamp_min(1e-5), self.opt.lr)
                        self.opt.lr = torch.where(full & down, (self.opt.lr * 1.5).clamp_max(1e-2), self.opt.lr)
                    stats["surrogate"] += self._slice_mean(surr.detach())
                    stats["value_loss"] += self._slice_mean(v_loss.detach())
                    stats["entropy"] += self._slice_mean(ent.detach())
                    stats["kl_pi0"] += self._slice_mean(kl0.detach())
                    stats["kl_old"] += kl_old
                    stats["grad_norm"] += gn
                n_mb += 1
        for k in stats:
            stats[k] /= n_mb
        self.last_stats = stats
        self.last_kl_pi0 = torch.where(full, stats["kl_pi0"], self.last_kl_pi0)

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

    # ---- replay (GePPO) -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _push_replay(self, mask: torch.Tensor, next_obs: torch.Tensor):
        """Copy the just-filled on-policy batch of ``mask`` robots into their ring slot."""
        T = self.T
        assert bool((self.batch_steps[mask] == T).all()), "replay needs batch_steps == buffer length"
        idx = self._ar[mask]
        h = self.rb["head"][idx]
        for k in ("obs", "act", "logp", "rew", "done"):
            self.rb[k][h, :, idx] = self.buf[k][:T, idx].transpose(0, 1)
        self.rb["boot_obs"][h, idx] = next_obs[idx]
        M = self.replay_M[idx]
        self.rb["head"][idx] = (h + 1) % M
        self.rb["count"][idx] = torch.minimum(self.rb["count"][idx] + 1, M)

    @torch.no_grad()
    def _replay_targets(self, mask: torch.Tensor):
        """V-trace GAE (GePPO App. C eq. 41-42, reference code ``gae_vtrace``) under the current policy and critic, for
        every stored batch: A_t = delta_t + gamma lam (1 - d_t) c_{t+1} A_{t+1}, target = c_t A_t + V_t, with
        c_t = min(c_bar, pi_k(a_t | s_t) / pi_behaviour(a_t | s_t)). Batches are separate segments (bootstrap from
        the stored next obs). Returns flattened (n, Mx*T) tensors and the valid-sample weight."""
        cfg = self.cfg
        Mx, T, n = self.rb["obs"].shape[:3]
        O = self.rb["obs"].shape[-1]
        obs = self.rb["obs"].permute(2, 0, 1, 3).reshape(n, Mx * T, O)
        act = self.rb["act"].permute(2, 0, 1, 3).reshape(n, Mx * T, -1)
        logp_b = self.rb["logp"].permute(2, 0, 1).reshape(n, Mx * T)
        mean, std = self.actor(obs)
        logp_k = self.actor.log_prob(mean, std, act)
        ratio_k = torch.exp(logp_k - logp_b)  # pi_k / pi_behaviour: 1 for the fresh batch
        c = ratio_k.clamp_max(cfg.vtrace_c).reshape(n, Mx, T)
        V = self.critic(obs).reshape(n, Mx, T)
        Vb = self.critic(self.rb["boot_obs"].permute(1, 0, 2))  # (n, Mx)
        rew = self.rb["rew"].permute(2, 0, 1)
        nonterm = 1.0 - self.rb["done"].permute(2, 0, 1)
        Vn = torch.cat([V[:, :, 1:], Vb.unsqueeze(-1)], dim=-1)
        delta = rew + cfg.gamma * nonterm * Vn - V
        adv = torch.zeros_like(delta)
        a = torch.zeros(n, Mx, device=self.device)
        for t in reversed(range(T)):
            c_next = c[:, :, t + 1] if t + 1 < T else torch.zeros_like(a)
            a = delta[:, :, t] + cfg.gamma * cfg.lam * nonterm[:, :, t] * c_next * a
            adv[:, :, t] = a
        ret = c * adv + V
        valid = (torch.arange(Mx, device=self.device).view(1, Mx) < self.rb["count"].view(n, 1)).float()  # ring slots 0..count-1 are filled
        w = valid.unsqueeze(-1).expand(n, Mx, T).reshape(n, Mx * T) * mask.float().view(n, 1)
        return obs, act, logp_b, ratio_k, adv.reshape(n, Mx * T), ret.reshape(n, Mx * T), w

    def _update_replay(self, mask: torch.Tensor):
        """GePPO update (eq. 10) on the last M batches of each ``mask`` deployment, plus the F7 terms (value loss,
        entropy, KL anchor to pi_0). Advantages centred / scaled as the reference code does: mean = E[r0 w A] / E[r0 w],
        std = std(r0 w A), r0 = pi_k / pi_behaviour (here over the deployment's whole replay batch, not per minibatch,
        ASSUMED). The value loss is unclipped against the V-trace targets (stored values are stale, ASSUMED)."""
        cfg = self.cfg
        obs, act, logp_b, r0, adv, ret, w = self._replay_targets(mask)
        n, S = w.shape
        wsum = w.sum(1).clamp_min(1.0)
        mu = (r0 * w * adv).sum(1) / (r0 * w).sum(1).clamp_min(1e-8)
        x = r0 * w * adv
        xm = (x * w).sum(1) / wsum
        sd = (((x - xm.unsqueeze(1)) ** 2 * w).sum(1) / wsum).sqrt()
        adv = (adv - mu.unsqueeze(1)) / (sd.unsqueeze(1) + 1e-8)
        eps = self.replay_eps().unsqueeze(1)
        with torch.no_grad():
            pi0_mean, pi0_std = self.pi0(obs.reshape(n * S, -1).unsqueeze(0))
            pi0_mean = pi0_mean.reshape(n, S, -1)
            pi0_std = pi0_std.reshape(n, S, -1)
        stats = {k: torch.zeros(self.n, device=self.device) for k in ("surrogate", "value_loss", "entropy", "kl_pi0", "kl_old", "grad_norm")}
        mb = S // cfg.num_mini_batches
        n_mb = 0
        for _ in range(cfg.num_learning_epochs):
            perm = torch.randperm(S, device=self.device, generator=self._replay_gen)
            for c in range(cfg.num_mini_batches):
                ix = perm[c * mb:(c + 1) * mb if c < cfg.num_mini_batches - 1 else S]
                wb = w[:, ix]
                wn = wb.sum(1).clamp_min(1.0)
                mean, std = self.actor(obs[:, ix])
                logp = self.actor.log_prob(mean, std, act[:, ix])
                ratio = torch.exp(logp - logp_b[:, ix])
                rc = torch.maximum(torch.minimum(ratio, r0[:, ix] + eps), r0[:, ix] - eps)
                surr = -(torch.min(ratio * adv[:, ix], rc * adv[:, ix]) * wb).sum(1) / wn
                value = self.critic(obs[:, ix])
                v_loss = (((value - ret[:, ix]) ** 2) * wb).sum(1) / wn
                ent = (self.actor.entropy(std) * wb).sum(1) / wn
                kl0 = (self.actor.kl(mean, std, pi0_mean[:, ix], pi0_std[:, ix]) * wb).sum(1) / wn
                loss_per = surr + cfg.value_loss_coef * v_loss - cfg.entropy_coef * ent + self.kl_w * kl0
                loss = (loss_per * mask.float()).sum()
                self.opt.zero_grad()
                loss.backward()
                gn = self.opt.clip_grad_norm(cfg.max_grad_norm, mask)
                self.opt.step(mask)
                with torch.no_grad():
                    stats["surrogate"] += surr.detach()
                    stats["value_loss"] += v_loss.detach()
                    stats["entropy"] += ent.detach()
                    stats["kl_pi0"] += kl0.detach()
                    stats["grad_norm"] += gn
                n_mb += 1
        for k in stats:
            stats[k] /= n_mb
        with torch.no_grad():
            # GePPO Lemma 3 estimate: TV = 1/2 E[|pi / pi_behaviour - pi_k / pi_behaviour|] over the replay batch
            mean, std = self.actor(obs)
            ratio = torch.exp(self.actor.log_prob(mean, std, act) - logp_b)
            tv = 0.5 * ((ratio - r0).abs() * w).sum(1) / wsum
            stats["kl_old"] = tv
            self.last_tv = torch.where(mask, tv, self.last_tv)
            if cfg.replay_lr_rule == "tv":
                e = eps.squeeze(1)
                up = mask & (tv > e / 2.0)
                down = mask & (tv < cfg.replay_lr_beta * e / 2.0)
                self.opt.lr = torch.where(up, self.opt.lr / (1.0 + cfg.replay_lr_alpha), self.opt.lr)
                self.opt.lr = torch.where(down, self.opt.lr * (1.0 + cfg.replay_lr_alpha), self.opt.lr)
        self.last_stats = stats
        self.last_kl_pi0 = torch.where(mask, stats["kl_pi0"], self.last_kl_pi0)

    # ---- save / resume (wall-time-limited jobs chained into one deployment, 2026-11-03) ----------------------------
    _STATE_KEYS = ("ptr", "n_updates", "last_kl_pi0", "last_tv", "kl_w", "explore_enabled", "update_enabled", "batch_steps", "replay_M", "boot_v", "has_boot")

    def state_dict(self) -> dict:
        """Everything needed to continue the same deployments in a new process: networks, Adam (moments, step, lr,
        per-parameter lr / masks), the on-policy buffer, the replay ring and the per-deployment knobs."""
        cpu = lambda x: None if x is None else x.detach().cpu().clone()
        return {"actor": {k: cpu(v) for k, v in self.actor.state_dict().items()}, "critic": {k: cpu(v) for k, v in self.critic.state_dict().items()},
                "opt": {"m": [cpu(x) for x in self.opt.m], "v": [cpu(x) for x in self.opt.v], "t": cpu(self.opt.t), "lr": cpu(self.opt.lr),
                        "param_lr": [cpu(x) for x in self.opt.param_lr], "param_mask": [cpu(x) for x in self.opt.param_mask]},
                "buf": {k: cpu(v) for k, v in self.buf.items()}, "T": self.T, "rb": None if self.rb is None else {k: cpu(v) for k, v in self.rb.items()},
                "replay_gen": self._replay_gen.get_state(), **{k: cpu(getattr(self, k)) for k in self._STATE_KEYS}}

    @torch.no_grad()
    def load_state_dict(self, sd: dict):
        dev = self.device
        to = lambda x: None if x is None else x.to(dev)
        self.actor.load_state_dict({k: to(v) for k, v in sd["actor"].items()})
        self.critic.load_state_dict({k: to(v) for k, v in sd["critic"].items()})
        o = sd["opt"]
        for dst, src in zip(self.opt.m + self.opt.v, o["m"] + o["v"]):
            dst.copy_(src.to(dev))
        self.opt.t = to(o["t"])
        self.opt.lr = to(o["lr"])
        self.opt.param_lr = [to(x) for x in o["param_lr"]]
        self.opt.param_mask = [to(x) for x in o["param_mask"]]
        self.T = sd["T"]
        self.buf = {k: to(v) for k, v in sd["buf"].items()}
        self.rb = None if sd["rb"] is None else {k: to(v) for k, v in sd["rb"].items()}
        self._replay_gen.set_state(sd["replay_gen"])
        for k in self._STATE_KEYS:
            setattr(self, k, to(sd[k]))

    # ---- resets ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def reset_to_pi0(self, mask: torch.Tensor, pi0_critic: dict | None = None):
        """Weights of ``mask`` deployments back to pi_0, Adam moments and buffers cleared (periodic-reset arm)."""
        idx = self._ars[mask]  # ``mask`` over slices
        if len(idx) == 0:
            return
        self.actor.load_rsl_rl(self.pi0_state, idx=idx)
        if pi0_critic is not None:
            self.critic.load_rsl_rl(pi0_critic, idx=idx)
        self.opt.reset_slices(idx)
        self.opt.lr[idx] = self.cfg.learning_rate
        env_m = self.env_view(mask)
        self.ptr[env_m] = 0
        self.has_boot[env_m] = False
        self.last_kl_pi0[idx] = float("nan")
        if self.rb is not None:
            self.rb["count"][env_m] = 0
            self.rb["head"][env_m] = 0

    @torch.no_grad()
    def drop_buffer(self, mask: torch.Tensor):
        """``mask`` over robots. Also empties the replay ring (no replay across an adaptation boundary, ASSUMED)."""
        self.ptr[mask] = 0
        self.has_boot[mask] = False
        if self.rb is not None:
            self.rb["count"][mask] = 0
            self.rb["head"][mask] = 0
