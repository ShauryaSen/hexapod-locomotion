# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-test of the deployment harness without Isaac Sim: checkpoint loading into stacked slices, the batched
updater on synthetic streams (per-deployment independence, KL anchor), arms, schedules, detectors, metrics.

  ~/bin/isaac-shell /workspace/isaaclab/_isaac_sim/python.sh /workspace/learning/hexapod/tools/deploy_selftest.py --checkpoint <model.pt>
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.arms import ARM_NAMES, make_arm, make_composite  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint  # noqa: E402
from hexapod.deploy.detectors import Baseliner, calibrate_cusum_h, three_sigma_fa_rate  # noqa: E402
from hexapod.deploy.ensemble import DynamicsEnsemble, train_ensemble  # noqa: E402
from hexapod.deploy.metrics import deployment_metrics, bootstrap_ci  # noqa: E402
from hexapod.deploy.schedule import generate_schedule  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
dev = torch.device(args.device)
torch.manual_seed(0)
N, OBS, ACT = 8, 68, 18

# ---- 1. checkpoint -> stacked slices; slice 0 output == a plain sequential MLP ----------------------------------
if args.checkpoint:
    actor, critic, info = build_from_checkpoint(args.checkpoint, N, dev)
    sd = info["actor"]
    x = torch.randn(N, OBS, device=dev)
    import torch.nn as nn

    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    mods = []
    for i, li in enumerate(layers):
        lin = nn.Linear(sd[f"mlp.{li}.weight"].shape[1], sd[f"mlp.{li}.weight"].shape[0])
        lin.weight.data.copy_(sd[f"mlp.{li}.weight"])
        lin.bias.data.copy_(sd[f"mlp.{li}.bias"])
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ELU())
    ref = nn.Sequential(*mods).to(dev)
    with torch.no_grad():
        mean, std = actor(x)
        diff = max((mean[i] - ref(x[i])).abs().max().item() for i in range(N))
    print(f"[1] checkpoint load: hidden {info['hidden_dims']}, obs {info['obs_dim']}, act {info['act_dim']}, max |stacked - sequential| = {diff:.2e}, std[0,:3] = {std[0, :3].tolist()}")
    assert diff < 1e-5
else:
    from hexapod.deploy.batched import BatchedCritic, BatchedGaussianActor

    actor = BatchedGaussianActor(N, OBS, (128, 128, 128), ACT).to(dev)
    critic = BatchedCritic(N, OBS, (128, 128, 128)).to(dev)
    info = {"actor": actor.slice_state(0), "critic": None, "obs_dim": OBS, "act_dim": ACT}
    print("[1] no checkpoint given: random pi_0")

# ---- 2. updater: synthetic reward that prefers action[0] > 0 for the first half of the deployments -------------
ucfg = C.UpdaterCfg(steps_per_batch=200, learning_rate=3e-4, kl_anchor_weight=0.1)
upd = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, OBS)
before = {k: v.clone() for k, v in actor.state_dict().items()}
adapt = torch.tensor([True] * (N // 2) + [False] * (N // 2), device=dev)
obs = torch.randn(N, OBS, device=dev)
n_updates = 0
m0 = None
for step in range(ucfg.steps_per_batch * 3):
    act, logp, value, mean, std = upd.act(obs, adapt)
    if m0 is None:
        m0 = mean[:, 0].clone()
    rew = act[:, 0].clamp(-2, 2)  # reward: push the first action dim up
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    upd.store(adapt, obs, act, logp, value, rew, done, mean, std)
    obs = torch.randn(N, OBS, device=dev)
    updated = upd.maybe_update(obs)
    n_updates += int(updated.sum())
with torch.no_grad():
    mean1, _ = actor(obs * 0 + 0.1)
    mean0, _ = upd.pi0(torch.zeros(1, N, OBS, device=dev) + 0.1)
changed = torch.stack([(actor.state_dict()[k] - before[k]).abs().flatten(1).sum(1) > 0 for k in before if actor.state_dict()[k].dim() > 1]).any(0)
print(f"[2] updater: {n_updates} slice-updates, slices changed {changed.tolist()} (expect first half True), action[0] mean shift adapting {(mean1[:N // 2, 0] - mean0[0, :N // 2, 0]).mean():.3f} vs frozen {(mean1[N // 2:, 0] - mean0[0, N // 2:, 0]).abs().max():.2e}, kl_pi0 {upd.last_kl_pi0.tolist()}")
assert changed.tolist() == adapt.tolist()
assert (mean1[:N // 2, 0] - mean0[0, :N // 2, 0]).mean() > 0.0
# reset to pi_0 restores slice 0 exactly
upd.reset_to_pi0(torch.tensor([True] + [False] * (N - 1), device=dev), info["critic"])
with torch.no_grad():
    m_reset, _ = actor(obs * 0 + 0.1)
assert (m_reset[0] - mean0[0, 0]).abs().max() < 1e-5, "reset_to_pi0 failed"
print("[2] reset_to_pi0 ok")

# ---- 2b. per-deployment batch length: half the slices at 2x batch get half as many updates
upd2 = KLAnchoredPPO(actor, critic, info["actor"], C.UpdaterCfg(steps_per_batch=100, learning_rate=1e-4), dev, OBS)
upd2.set_batch_steps(torch.arange(N, device=dev) >= N // 2, 200)
allm = torch.ones(N, dtype=torch.bool, device=dev)
o_ = torch.randn(N, OBS, device=dev)
for _ in range(400):
    a_, lp_, v_, m_, s_ = upd2.act(o_, allm)
    upd2.store(allm, o_, a_, lp_, v_, a_[:, 0], torch.zeros(N, dtype=torch.bool, device=dev), m_, s_)
    o_ = torch.randn(N, OBS, device=dev)
    upd2.maybe_update(o_)
print(f"[2b] per-slice batch: updates per slice {upd2.n_updates.tolist()} (expect 4 for batch 100, 2 for batch 200)")
assert upd2.n_updates[: N // 2].eq(4).all() and upd2.n_updates[N // 2 :].eq(2).all()

# ---- 3. schedules ---------------------------------------------------------------------------------------------
scfg = C.ScheduleCfg(t50_min=2.0)
for seed in range(3):
    s = generate_schedule(seed, scfg)
    print(f"[3] schedule seed {seed}: length {s.length_s / 60:.1f} min, {len(s.events)} changes, nominal {s.nominal_fraction():.2f}, recurrence {s.has_recurrence()}, tries {s.cfg['tries']}: " + ", ".join(f"{e.t_s / 60:.1f}m {e.factor} {'on' if e.on else 'off'}" for e in s.events))
    assert scfg.n_changes_min <= len(s.events) <= scfg.n_changes_max and s.nominal_fraction() >= 0.4 and s.has_recurrence()

# ---- 4. arms on a synthetic z stream (change at cycle 300 -> z = 4) -------------------------------------------
cycle_s = 0.4
det = C.DetectorCfg(cusum_h=5.0)
u2 = C.UpdaterCfg(budget_min=1.0)
scheds = [generate_schedule(0, scfg)] * 4
for name in ARM_NAMES:
    arm = make_arm(name, 4, C.ArmCfg(name=name, period_min=2.0, hold_min=1.0), u2, det, dev, cycle_s, scheds)
    adapt_hist, alarms = [], []
    for c in range(900):
        t = (c + 1) * cycle_s
        z = torch.full((4,), 4.0 if 300 <= c < 450 else 0.0, device=dev) + 0.3 * torch.randn(4, device=dev)
        out = arm.cycle(t, z)
        adapt_hist.append(out["adapt"][0].item())
        if out["alarm"][0]:
            alarms.append(round(t))
    print(f"[4] arm {name:14s}: adapt cycles {sum(adapt_hist)}/900, alarms at t[s] {alarms[:6]}")

# ---- 4a. composite arm == the same arms run alone on their subsets
specs = [("never", C.ArmCfg(name="never")), ("scheduled", C.ArmCfg(name="scheduled", period_min=2.0)), ("three_sigma", C.ArmCfg(name="three_sigma")), ("oracle", C.ArmCfg(name="oracle"))]
nA = 8
sch8 = [generate_schedule(0, scfg)] * nA
comp = make_composite(specs, nA, u2, det, dev, cycle_s, sch8)
alone = [make_arm(nm, 2, ac, u2, det, dev, cycle_s, sch8[:2]) for nm, ac in specs]
gen = torch.Generator(device="cpu").manual_seed(0)
for c in range(900):
    t = (c + 1) * cycle_s
    zfull = (4.0 if 300 <= c < 450 else 0.0) + 0.3 * torch.randn(nA, generator=gen)
    zfull = zfull.to(dev)
    o = comp.cycle(t, zfull)
    for k, a in enumerate(alone):
        oa = a.cycle(t, zfull[torch.arange(k, nA, len(specs), device=dev)])
        assert torch.equal(o["adapt"][k::len(specs)], oa["adapt"]) and torch.equal(o["alarm"][k::len(specs)], oa["alarm"]), (k, c)
print(f"[4a] composite arm of {comp.labels} matches the arms run alone over 900 cycles")

# ---- 4b. R5 baseliner: per-deployment mean removed, NaN while collecting
bl = Baseliner(2, 10, 1.0, 0.0, dev)
zs_ = [bl.step(torch.tensor([5.0 + 0.1 * c, -3.0], device=dev)) for c in range(12)]
assert torch.isnan(zs_[0]).all() and not torch.isnan(zs_[11]).any() and abs(zs_[11][1].item()) < 1e-6
print(f"[4b] baseliner: mu after 10 cycles {bl.mu.tolist()}, z at cycle 12 {zs_[11].tolist()}")

# ---- 5. detectors offline ---------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
zs = [rng.standard_normal(9000) for _ in range(4)]  # 4 x 1 h of nominal
cal = calibrate_cusum_h(zs, cycle_s, det.cusum_k, 1.0)
print(f"[5] cusum calibration on {cal['hours']:.1f} nominal hours: h = {cal['h']} (FA {cal['fa_per_hour_at_h']:.2f}/h); 3-sigma x3 FA = {three_sigma_fa_rate(zs, cycle_s, 3.0, 3):.2f}/h, x1 = {three_sigma_fa_rate(zs, cycle_s, 3.0, 1):.2f}/h")

# ---- 6. ensemble on synthetic dynamics --------------------------------------------------------------------------
ecfg = C.EnsembleCfg(epochs=10, batch_size=512, hidden_dims=(64, 64))
M = 5000
o = torch.randn(M, OBS, device=dev)
a = torch.randn(M, ACT, device=dev)
W = torch.randn(OBS + ACT, 45, device=dev) * 0.1
o2 = o.clone()
o2[:, DynamicsEnsemble(OBS, ACT, ecfg).proprio_idx] += torch.cat([o, a], -1) @ W
ens = DynamicsEnsemble(OBS, ACT, ecfg).to(dev)
res = train_ensemble(ens, o, a, o2, ecfg, dev, log=lambda *_: None)
nmse, dis = ens.residual_step(o[:100], a[:100], o2[:100] + 1.0)
print(f"[6] ensemble: held-out mse {res['history'][-1]['heldout_mse']:.3f} (nominal nmse {res['heldout_nmse_mean']:.3f}), shifted-target nmse {nmse.mean():.2f}")
assert nmse.mean() > 1.5 * res["heldout_nmse_mean"]

# ---- 7. metrics on synthetic streams --------------------------------------------------------------------------
Cn, n = 900, 2
t = (np.arange(Cn) + 1) * cycle_s
arrays = {"t_s": np.tile(t[:, None], (1, n)), "progress_m": np.full((Cn, n), 0.11), "falls": np.zeros((Cn, n)), "adapting": np.zeros((Cn, n)), "alarm": np.zeros((Cn, n)), "track_err": np.full((Cn, n), 0.02), "payload_kg": np.zeros((Cn, n)), "motor_strength": np.ones((Cn, n)), "friction": np.ones((Cn, n))}
arrays["alarm"][310, 0] = 1
arrays["alarm"][100, 0] = 1
s = generate_schedule(0, scfg)
m = deployment_metrics(arrays, 0, s, C.MetricsCfg(detect_window_min=2.0), budget_min=1.0)
print(f"[7] metrics: {m}")
lo, hi = bootstrap_ci(np.random.default_rng(0).standard_normal(50), 200, 0.95)
print(f"[7] bootstrap ci of N(0,1) mean, n=50: [{lo:.2f}, {hi:.2f}]")
print("selftest passed")
