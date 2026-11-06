# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks for the replay (GePPO) flag of deploy/updater.py (no Isaac Sim needed):

1. Replay off = unchanged: with no replay slice the updater gives bit-identical parameters, Adam moments, learning
   rates and KL to a reference copy of the previous updater (``--ref``, e.g. ``git show cc92333:deploy/updater.py``).
2. Mixed process: the non-replay deployments of a process that also has replay deployments are bit-identical to a
   process with no replay at all (groups share a process in tools/deploy_sweep_updater.py).
3. V-trace on on-policy data (one batch, ratio 1) equals the updater's plain GAE.
4. V-trace with off-policy ratios equals a NumPy port of the GePPO reference ``gae_vtrace`` (matrix form).
5. Ring buffer: after M + 2 batches it holds exactly the last M batches.
6. lr: the actor lr of replay deployments moves by the TV rule (factor 1.03 steps), their critic lr stays fixed;
   only replay deployments change.
7. Save / resume: an updater saved mid-batch and loaded into a fresh updater continues bit-identically to one that
   never stopped (chained <= 30-min GPU jobs).

  python tools/deploy_check_replay.py --ref /path/to/updater_ref.py
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import BatchedCritic, BatchedGaussianActor  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True, help="reference updater.py (before replay)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
dev = torch.device(args.device)
OBS, ACT, HID = 68, 18, (128, 128, 128)

spec = importlib.util.spec_from_file_location("hexapod.deploy._updater_ref", args.ref)
ref_mod = importlib.util.module_from_spec(spec)
sys.modules["hexapod.deploy._updater_ref"] = ref_mod
spec.loader.exec_module(ref_mod)


def make_nets(n, seed):
    torch.manual_seed(seed)
    a = BatchedGaussianActor(n, OBS, HID, ACT).to(dev)
    c = BatchedCritic(n, OBS, HID).to(dev)
    with torch.no_grad():
        a.std_param.fill_(0.3)
    return a, c


def drive(upd, n_env, steps, seed):
    """Synthetic stream: random obs, reward = first action dim, a truncation every 97 steps, a fall every 131."""
    g = torch.Generator(device=dev).manual_seed(seed)
    gobs = torch.Generator(device="cpu").manual_seed(seed + 1)
    ones = torch.ones(n_env, dtype=torch.bool, device=dev)
    for t in range(steps):
        obs = torch.randn(n_env, OBS, generator=gobs).to(dev)
        act, logp, value, mean, std = upd.act(obs, ones, g)
        rew = act[:, 0].clamp(-2, 2)
        done = torch.full((n_env,), t % 131 == 130, dtype=torch.bool, device=dev)
        upd.store(ones, obs, act, logp, value, rew, done, mean, std)
        nxt = torch.randn(n_env, OBS, generator=gobs).to(dev)
        if t % 97 == 96:
            upd.truncate(ones, nxt)
        upd.maybe_update(nxt)


def flat_slice(upd, s):
    return torch.cat([p.detach()[s].flatten() for p in upd.params])


ok = True
N = 6
ucfg = C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, lr_schedule="adaptive", kl_anchor_weight=0.5)

# ---- 1. replay off == reference ------------------------------------------------------------------------------------
a1, c1 = make_nets(N, 0)
a2, c2 = make_nets(N, 0)
pi0 = a1.slice_state(0)
new = KLAnchoredPPO(a1, c1, pi0, ucfg, dev, OBS)
ref = ref_mod.KLAnchoredPPO(a2, c2, pi0, ucfg, dev, OBS)
for u in (new, ref):
    torch.manual_seed(5)
    drive(u, N, 400, 11)
same = all(torch.equal(flat_slice(new, s), flat_slice(ref, s)) for s in range(N))
same_m = all(torch.equal(x, y) for x, y in zip(new.opt.m + new.opt.v, ref.opt.m + ref.opt.v))
same_lr = torch.equal(new.opt.lr, ref.opt.lr)
print(f"[1] replay off vs reference: params {same}, Adam moments {same_m}, lr {same_lr}, updates {new.n_updates.tolist()}")
ok &= same and same_m and same_lr

# ---- 2. mixed process: non-replay slices unchanged ------------------------------------------------------------------
a1, c1 = make_nets(N, 1)
a2, c2 = make_nets(N, 1)
pi0 = a1.slice_state(0)
plain = KLAnchoredPPO(a1, c1, pi0, ucfg, dev, OBS)
mixed = KLAnchoredPPO(a2, c2, pi0, ucfg, dev, OBS)
rp_mask = torch.tensor([False, True, False, True, True, False], device=dev)
mixed.set_replay(rp_mask, 4)
for u in (plain, mixed):
    torch.manual_seed(6)
    drive(u, N, 400, 12)
same_np = all(torch.equal(flat_slice(plain, s), flat_slice(mixed, s)) for s in range(N) if not rp_mask[s])
moved_rp = all(not torch.equal(flat_slice(plain, s), flat_slice(mixed, s)) for s in range(N) if rp_mask[s])
print(f"[2] mixed process: non-replay slices bit-identical {same_np}; replay slices differ from on-policy {moved_rp}; updates {mixed.n_updates.tolist()}")
ok &= same_np and moved_rp

# ---- 3. V-trace on on-policy data == GAE -----------------------------------------------------------------------------
a3, c3 = make_nets(N, 2)
pi0 = a3.slice_state(0)
u3 = KLAnchoredPPO(a3, c3, pi0, ucfg, dev, OBS)
u3.set_replay(torch.ones(N, dtype=torch.bool, device=dev), 3)
g = torch.Generator(device=dev).manual_seed(3)
ones = torch.ones(N, dtype=torch.bool, device=dev)
for t in range(ucfg.steps_per_batch):
    obs = torch.randn(N, OBS, device=dev)
    act, logp, value, mean, std = u3.act(obs, ones, g)
    u3.store(ones, obs, act, logp, value, act[:, 0], torch.full((N,), t == 30, dtype=torch.bool, device=dev), mean, std)
nxt = torch.randn(N, OBS, device=dev)
adv_gae, ret_gae = u3._gae(ones, nxt, u3.T)  # normalized advantages; recompute the raw ones below
with torch.no_grad():
    last_v = u3.critic(nxt)
    raw = torch.zeros(u3.T, N, device=dev)
    gae = torch.zeros(N, device=dev)
    for t in reversed(range(u3.T)):
        nv = last_v if t == u3.T - 1 else u3.buf["value"][t + 1]
        nont = 1.0 - u3.buf["done"][t]
        gae = u3.buf["rew"][t] + ucfg.gamma * nv * nont - u3.buf["value"][t] + ucfg.gamma * ucfg.lam * nont * gae
        raw[t] = gae
u3._push_replay(ones, nxt)
obs_f, act_f, logp_b, r0, adv_v, ret_v, w = u3._replay_targets(ones)
T = u3.T
d_adv = (adv_v[:, :T].T - raw).abs().max().item()
d_ret = (ret_v[:, :T].T - ret_gae).abs().max().item()
d_r0 = (r0[:, :T] - 1).abs().max().item()
print(f"[3] V-trace on on-policy data vs GAE: max |adv diff| {d_adv:.2e}, |return diff| {d_ret:.2e}, |ratio - 1| {d_r0:.2e}; valid samples per robot {w.sum(1)[0].item():.0f} (= T {T})")
ok &= d_adv < 1e-4 and d_ret < 1e-4 and d_r0 < 1e-4 and w.sum(1)[0].item() == T

# ---- 4. V-trace off-policy == NumPy port of the reference gae_vtrace ---------------------------------------------------
def gae_vtrace_ref(r, d, V, Vp, ratio, gamma, lam, is_trunc):
    """Port of geppo/common/runner_utils.py gae_vtrace (one trajectory; inputs already evaluated)."""
    ratio_trunc = np.minimum(ratio, is_trunc)
    n = ratio.shape[0]
    ones_U = np.triu(np.ones((n, n)), 0)
    rate_L = np.tril(np.ones((n, n)) * gamma * lam, -1)
    rates = np.tril(np.cumprod(rate_L + ones_U, axis=0), 0)
    rtr = np.repeat(np.expand_dims(ratio_trunc, 1), n, axis=1)
    rtr_L = np.tril(rtr, -1)
    prods = np.tril(np.cumprod(rtr_L + ones_U, axis=0), 0)
    delta = r + gamma * (1 - d) * Vp - V
    adv = np.sum(rates * prods * np.expand_dims(delta, axis=1), axis=0)
    return adv, adv * ratio_trunc + V


# one trajectory = one stored batch with no done inside (the reference handles one trajectory at a time)
u3.rb["done"].zero_()
base = [p.detach().clone() for p in u3.actor.parameters_per_slice()]
for scale in (0.002, 0.005, 0.01, 0.02, 0.05):  # current policy != behaviour policy, with part of the ratios above c_bar
    with torch.no_grad():
        for p, b0 in zip(u3.actor.parameters_per_slice(), base):
            p.copy_(b0 + scale * torch.randn_like(p))
    obs_f, act_f, logp_b, r0, adv_v, ret_v, w = u3._replay_targets(ones)
    if 0.05 < float((r0[:, :T] > ucfg.vtrace_c).float().mean()) < 0.95:
        break
with torch.no_grad():
    V = u3.critic(obs_f[:, :T]).cpu().numpy()
    Vb = u3.critic(u3.rb["boot_obs"][0]).cpu().numpy()
worst = 0.0
for r_i in range(N):
    rew = u3.rb["rew"][0, :, r_i].cpu().numpy().astype(np.float64)
    Vi = V[r_i].astype(np.float64)
    Vp = np.concatenate([Vi[1:], [Vb[r_i]]])
    ratio = r0[r_i, :T].cpu().numpy().astype(np.float64)
    a_ref, ret_ref = gae_vtrace_ref(rew, np.zeros(T), Vi, Vp, ratio, ucfg.gamma, ucfg.lam, ucfg.vtrace_c)
    worst = max(worst, np.abs(a_ref - adv_v[r_i, :T].cpu().numpy()).max(), np.abs(ret_ref - ret_v[r_i, :T].cpu().numpy()).max())
truncated = float((r0[:, :T] > ucfg.vtrace_c).float().mean())
print(f"[4] V-trace vs reference gae_vtrace port: max |diff| {worst:.2e} (share of ratios above c_bar: {truncated:.2f})")
ok &= worst < 1e-3 and 0.05 < truncated < 0.95

# ---- 5. ring buffer -------------------------------------------------------------------------------------------------
a5, c5 = make_nets(2, 4)
pi0 = a5.slice_state(0)
u5 = KLAnchoredPPO(a5, c5, pi0, ucfg, dev, OBS)
M = 3
u5.set_replay(torch.ones(2, dtype=torch.bool, device=dev), M)
seen = []
g = torch.Generator(device=dev).manual_seed(4)
ones2 = torch.ones(2, dtype=torch.bool, device=dev)
for b in range(M + 2):
    for t in range(ucfg.steps_per_batch):
        obs = torch.full((2, OBS), float(b * 1000 + t), device=dev)
        act, logp, value, mean, std = u5.act(obs, ones2, g)
        u5.store(ones2, obs, act, logp, value, act[:, 0] * 0, torch.zeros(2, dtype=torch.bool, device=dev), mean, std)
    u5.maybe_update(torch.zeros(2, OBS, device=dev))
    seen.append(b)
stored = sorted(int(u5.rb["obs"][m, 0, 0, 0].item()) // 1000 for m in range(M))
ring_ok = stored == seen[-M:] and u5.rb["count"].tolist() == [M, M] and u5.n_updates.tolist() == [M + 2, M + 2]
print(f"[5] ring after {M + 2} batches holds batches {stored} (expected {seen[-M:]}), count {u5.rb['count'].tolist()}: {ring_ok}")
ok &= ring_ok

# ---- 6. lr: actor adapts by TV, critic fixed ---------------------------------------------------------------------------
a6, c6 = make_nets(N, 5)
pi0 = a6.slice_state(0)
u6 = KLAnchoredPPO(a6, c6, pi0, C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, kl_anchor_weight=0.5), dev, OBS)
rp = torch.tensor([True, True, True, False, False, False], device=dev)
u6.set_replay(rp, 4)
torch.manual_seed(7)
drive(u6, N, 300, 16)
crit_ids = {id(p) for p in c6.parameters_per_slice()}
crit_lr = [u6.opt.param_lr[j] for j, p in enumerate(u6.params) if id(p) in crit_ids]
crit_fixed = all(torch.allclose(x[rp], torch.full((3,), 3e-4, device=dev)) for x in crit_lr)
steps = torch.log(u6.opt.lr[rp] / 3e-4) / np.log(1.03)
actor_rule = bool(((steps - steps.round()).abs() < 1e-3).all()) and bool((steps.abs() > 0.5).any())
others = torch.equal(u6.opt.lr[~rp], torch.full((3,), 3e-4, device=dev))
print(f"[6] actor lr of replay slices in 1.03 steps from 3e-4: {actor_rule} (steps {steps.round().tolist()}, TV {[round(x, 4) for x in u6.last_tv[rp].tolist()]}, eps {u6.replay_eps()[0].item():.3f}); critic lr fixed {crit_fixed}; fixed-lr non-replay slices untouched {others}")
ok &= crit_fixed and actor_rule and others

# ---- 7. save / resume == continuous ----------------------------------------------------------------------------------
PI0_7 = make_nets(N, 8)[0].slice_state(0)  # pi_0 comes from the checkpoint in real use: the same for both processes


def fresh(seed):
    a, c = make_nets(N, seed)
    u = KLAnchoredPPO(a, c, PI0_7, C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, lr_schedule="adaptive", kl_anchor_weight=0.5), dev, OBS)
    u.set_replay(rp, 4)
    return u


cont = fresh(8)
torch.manual_seed(9)
drive(cont, N, 250, 20)
torch.manual_seed(10)
drive(cont, N, 250, 21)
first = fresh(8)
torch.manual_seed(9)
drive(first, N, 250, 20)  # 250 is not a multiple of the batch (60): the save falls mid-batch
sd = first.state_dict()
second = fresh(99)  # different initial weights: everything must come from the state
second.load_state_dict(sd)
torch.manual_seed(10)
drive(second, N, 250, 21)
same_r = all(torch.equal(flat_slice(cont, s), flat_slice(second, s)) for s in range(N))
same_o = all(torch.equal(x, y) for x, y in zip(cont.opt.m + cont.opt.v, second.opt.m + second.opt.v)) and torch.equal(cont.opt.lr, second.opt.lr)
same_rb = all(torch.equal(cont.rb[k], second.rb[k]) for k in cont.rb)
print(f"[7] save mid-batch + resume in a new updater vs continuous: params {same_r}, Adam + lr {same_o}, replay ring {same_rb}, updates {second.n_updates.tolist()} vs {cont.n_updates.tolist()}")
ok &= same_r and same_o and same_rb and torch.equal(cont.n_updates, second.n_updates)

# ---- 8. two chained jobs through files, as deploy_sweep_updater.py does (torch.save, then torch.load with
# map_location = the job's device; the 11-04 job_01 crash was the CUDA-mapped replay RNG state) -------------------------
import os, tempfile

cont8 = fresh(8)
for seed, ns in ((9, 20), (10, 21), (11, 22)):
    torch.manual_seed(seed)
    drive(cont8, N, 250, ns)
path = os.path.join(tempfile.mkdtemp(), "state.pt")
job = fresh(8)
for k, (seed, ns) in enumerate(((9, 20), (10, 21), (11, 22))):
    if k > 0:
        job = fresh(100 + k)
        job.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
    torch.manual_seed(seed)
    drive(job, N, 250, ns)
    torch.save(job.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)
same8 = all(torch.equal(flat_slice(cont8, s), flat_slice(job, s)) for s in range(N)) and torch.equal(cont8.opt.lr, job.opt.lr) and all(torch.equal(cont8.rb[k], job.rb[k]) for k in cont8.rb) and torch.equal(cont8.n_updates, job.n_updates)
print(f"[8] three jobs chained through state files (map_location={dev}) vs continuous: {same8}, updates {job.n_updates.tolist()}")
ok &= same8

# ---- 9. TV history and the above-10-eps streak (kill rule, supervisor 11-06) -----------------------------------------
cnt = job.tv_count
last_ok = all(torch.equal(job.tv_hist[i, cnt[i] - 1], job.last_tv[i]) for i in range(N) if rp[i])
cnt_ok = torch.equal(cnt[rp], job.n_updates[rp]) and bool((cnt[~rp] == 0).all())
e9 = job.replay_eps()
streak_ref = []
for i in range(N):
    k = 0
    for x in job.tv_hist[i, : cnt[i]].tolist():
        k = k + 1 if x > 10 * e9[i].item() else 0
    streak_ref.append(k)
streak_ok = job.tv_streak.tolist() == streak_ref
a9, c9 = make_nets(N, 12)
u9 = KLAnchoredPPO(a9, c9, a9.slice_state(0), C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, kl_anchor_weight=0.5, replay_clip=0.002), dev, OBS)
u9.set_replay(rp, 4)  # tiny clip: TV is above 10 eps, so the streak counts up
torch.manual_seed(13)
drive(u9, N, 300, 23)
e9b = u9.replay_eps()
ref9 = []
for i in range(N):
    k = 0
    for x in u9.tv_hist[i, : u9.tv_count[i]].tolist():
        k = k + 1 if x > 10 * e9b[i].item() else 0
    ref9.append(k)
streak_ok &= u9.tv_streak.tolist() == ref9 and max(ref9) > 0
print(f"[9] TV history: last entry = last_tv {last_ok}, count = replay updates {cnt_ok} ({cnt.tolist()}), streak {job.tv_streak.tolist()}, tiny-clip streak {u9.tv_streak.tolist()} (TV {[round(x, 3) for x in u9.last_tv.tolist()]}, 10 eps {10 * e9b[0].item():.3f}) = recount {streak_ok}")
ok &= last_ok and cnt_ok and streak_ok
print("replay checks passed" if ok else "REPLAY CHECKS FAILED")
sys.exit(0 if ok else 1)
