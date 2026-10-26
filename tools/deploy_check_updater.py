# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks for the k > 1 and option-3a flags of deploy/updater.py (no Isaac Sim needed):

1. k = 1 unchanged: the current updater in its default mode gives bit-identical parameters, Adam moments, learning
   rates and KL statistics to a reference copy of the previous updater (``--ref``, e.g. ``git show
   cb31fd7:deploy/updater.py > ref.py``) on the same synthetic stream (stores, truncations, adaptive lr,
   per-slice batch lengths, a reset to pi_0).
2. Slice mode with one robot per slice matches the default mode (allclose).
3. k = 3 robots fed identical data match k = 1 on that data (allclose; the only difference is the unbiased
   advantage std over 3T vs T samples).
4. Option 3a: only the actor's output layer, the action std and the critic move; hidden layers stay bit-identical.

  python tools/deploy_check_updater.py --ref /path/to/updater_ref.py
"""

import argparse
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import BatchedCritic, BatchedGaussianActor  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True, help="reference updater.py (previous version)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
dev = torch.device(args.device)
OBS, ACT, HID = 68, 18, (128, 128, 128)

# load the reference module inside the package namespace so its relative imports resolve
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


def drive(upd, n_env, steps, seed, explore_mask, feed=None):
    """Synthetic stream: random obs, reward = first action dim, a truncation every 97 steps, a fall every 131."""
    g = torch.Generator(device=dev).manual_seed(seed)
    gobs = torch.Generator(device="cpu").manual_seed(seed + 1)
    for t in range(steps):
        obs = torch.randn(n_env if feed is None else 1, OBS, generator=gobs).to(dev)
        if feed is not None:
            obs = obs.expand(n_env, OBS).contiguous()
        act, logp, value, mean, std = upd.act(obs, explore_mask, g)
        if feed is not None:  # identical actions for all robots (robot 0's)
            act, logp, value, mean, std = (x[:1].expand_as(x).contiguous() for x in (act, logp, value, mean, std))
        rew = act[:, 0].clamp(-2, 2)
        done = torch.zeros(n_env, dtype=torch.bool, device=dev)
        if t % 131 == 130:
            done[:] = True
        upd.store(torch.ones(n_env, dtype=torch.bool, device=dev), obs, act, logp, value, rew, done, mean, std)
        nxt = torch.randn(n_env if feed is None else 1, OBS, generator=gobs).to(dev)
        if feed is not None:
            nxt = nxt.expand(n_env, OBS).contiguous()
        if t % 97 == 96:
            upd.truncate(torch.ones(n_env, dtype=torch.bool, device=dev), nxt)
        upd.maybe_update(nxt)


def flat(upd):
    return torch.cat([p.detach().flatten() for p in upd.params])


ok = True
# ---- 1. k = 1 default mode == reference --------------------------------------------------------------------------
N = 6
for lr_sched in ("fixed", "adaptive"):
    ucfg = C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, lr_schedule=lr_sched, kl_anchor_weight=0.5)
    a1, c1 = make_nets(N, 0)
    a2, c2 = make_nets(N, 0)
    pi0 = a1.slice_state(0)
    new = KLAnchoredPPO(a1, c1, pi0, ucfg, dev, OBS)
    ref = ref_mod.KLAnchoredPPO(a2, c2, pi0, ucfg, dev, OBS)
    for u in (new, ref):
        u.set_batch_steps(torch.tensor([False, False, False, True, True, True], device=dev), 90)
    for u in (new, ref):
        torch.manual_seed(5)
        drive(u, N, 400, 11, torch.ones(N, dtype=torch.bool, device=dev))
        u.reset_to_pi0(torch.tensor([True] + [False] * (N - 1), device=dev))
        torch.manual_seed(6)
        drive(u, N, 200, 12, torch.ones(N, dtype=torch.bool, device=dev))
    same_p = torch.equal(flat(new), flat(ref))
    same_m = all(torch.equal(x, y) for x, y in zip(new.opt.m + new.opt.v, ref.opt.m + ref.opt.v))
    same_lr = torch.equal(new.opt.lr, ref.opt.lr)
    same_kl = torch.equal(torch.nan_to_num(new.last_kl_pi0), torch.nan_to_num(ref.last_kl_pi0))
    same_n = torch.equal(new.n_updates, ref.n_updates)
    print(f"[1] k=1 default vs reference ({lr_sched} lr): params {same_p}, Adam moments {same_m}, lr {same_lr}, KL to pi_0 {same_kl}, update counts {same_n} ({new.n_updates.tolist()})")
    ok &= same_p and same_m and same_lr and same_kl and same_n

# ---- 2. slice mode with one robot per slice == default mode ---------------------------------------------------------
ucfg = C.UpdaterCfg(steps_per_batch=60, learning_rate=3e-4, kl_anchor_weight=0.5)
a1, c1 = make_nets(N, 1)
a2, c2 = make_nets(N, 1)
pi0 = a1.slice_state(0)
dflt = KLAnchoredPPO(a1, c1, pi0, ucfg, dev, OBS)
slc = KLAnchoredPPO(a2, c2, pi0, ucfg, dev, OBS, slice_of_env=torch.arange(N, device=dev))
for u in (dflt, slc):
    torch.manual_seed(7)
    drive(u, N, 300, 13, torch.ones(N, dtype=torch.bool, device=dev))
d = (flat(dflt) - flat(slc)).abs().max().item()
print(f"[2] slice mode, 1 robot per slice vs default: max |param diff| {d:.2e}, updates {slc.n_updates.tolist()} vs {dflt.n_updates.tolist()}")
ok &= d < 1e-4 and torch.equal(dflt.n_updates, slc.n_updates)

# ---- 3. k = 3 identical robots == k = 1 --------------------------------------------------------------------------------
a3, c3 = make_nets(1, 2)
a4, c4 = make_nets(1, 2)
pi0 = a3.slice_state(0)
k3 = KLAnchoredPPO(a3, c3, pi0, ucfg, dev, OBS, slice_of_env=torch.zeros(3, dtype=torch.long, device=dev))
k1 = KLAnchoredPPO(a4, c4, pi0, ucfg, dev, OBS)
torch.manual_seed(8)
drive(k3, 3, 300, 14, torch.ones(3, dtype=torch.bool, device=dev), feed=True)
torch.manual_seed(8)
drive(k1, 1, 300, 14, torch.ones(1, dtype=torch.bool, device=dev), feed=True)
p3, p1, p0 = flat(k3), flat(k1), torch.cat([p.detach().flatten() for p in make_nets(1, 2)[0].parameters_per_slice() + make_nets(1, 2)[1].parameters_per_slice()])
moved = (p1 - p0).abs().max().item()
diff = (p3 - p1).abs().max().item()
print(f"[3] k=3 identical robots vs k=1: max |param diff| {diff:.2e} (parameters moved by up to {moved:.2e}); updates {k3.n_updates.tolist()} vs {k1.n_updates.tolist()}")
ok &= diff < 0.05 * moved and torch.equal(k3.n_updates, k1.n_updates)

# ---- 4. option 3a: hidden layers frozen for the masked slices --------------------------------------------------------
a5, c5 = make_nets(4, 3)
pi0 = a5.slice_state(0)
u5 = KLAnchoredPPO(a5, c5, pi0, ucfg, dev, OBS, slice_of_env=torch.tensor([0, 0, 1, 1, 2, 3], device=dev))
sub = torch.tensor([True, False, True, False], device=dev)  # slices 0 and 2 use 3a
u5.set_param_subset(sub, "last_layer")
before = {id(p): p.detach().clone() for p in u5.params}
torch.manual_seed(9)
drive(u5, 6, 300, 15, torch.ones(6, dtype=torch.bool, device=dev))
last = {id(p) for p in a5.last_layer_params()} | {id(p) for p in c5.parameters_per_slice()}
res = []
for p in u5.params:
    ch = ((p.detach() - before[id(p)]).abs().flatten(1).sum(1) > 0).tolist()
    res.append((id(p) in last, ch))
hidden_ok = all(ch == [False, True, False, True] for is_last, ch in res if not is_last)
last_ok = all(all(ch) for is_last, ch in res if is_last)
print(f"[4] option 3a on slices 0, 2: hidden layers changed only on slices 1, 3: {hidden_ok}; last layer / std / critic changed on all: {last_ok}")
ok &= hidden_ok and last_ok
print("updater checks passed" if ok else "UPDATER CHECKS FAILED")
sys.exit(0 if ok else 1)
