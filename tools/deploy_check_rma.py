# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harness integration check for an RMA student pi_0 (frozen 11-08), CPU, no Isaac Sim: the harness path
(build_from_checkpoint + HistoryEncoderPreproc, as DeploymentRunner, deploy_t50, deploy_train_ensemble and
deploy_eval_policies use it) against hexapod.rma.StudentPolicy (the path the pi_0 evals used), on random observation
sequences with resets at random steps (falls / repositions). Also the deployment critic: BatchedCritic vs the
checkpoint's critic MLP on [obs, z_hat], and the input sizes.

  python tools/deploy_check_rma.py --checkpoint <run>/model_1000.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy.batched import build_from_checkpoint, build_preproc  # noqa: E402
from hexapod.rma import CheckpointPolicy, StudentPolicy  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--n", type=int, default=24)
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
dev = "cpu"
torch.manual_seed(0)

ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
st = ck["rma_student"]
actor, critic, info = build_from_checkpoint(args.checkpoint, args.n, dev)
O, Z, H = int(st["obs_dim"]), int(st["z_dim"]), int(st["history_len"])
pre = build_preproc(info, args.n, O, dev)
ref = StudentPolicy(ck).to(dev).eval()
crit_ref = CheckpointPolicy(ck["critic_state_dict"]).to(dev).eval()


class FakeEnv:  # StudentPolicy sees resets through episode_length_buf
    episode_length_buf = torch.zeros(args.n, dtype=torch.long)


env = FakeEnv()
ref.bind(env)
ok = True
print(f"[1] sizes: env obs {O}, z {Z}, history {H}; actor / critic input {info['obs_dim']} (= obs + z: {info['obs_dim'] == O + Z}); z target {st.get('z_target')}")
ok &= info["obs_dim"] == O + Z

# observations: a smooth random walk around a random centre (stand-in for real obs; both paths see the same ones)
centre = torch.randn(args.n, O)
x = centre.clone()
max_a, max_v, n_resets = 0.0, 0.0, 0
with torch.no_grad():
    for t in range(args.steps):
        x = x + 0.1 * torch.randn(args.n, O)
        reset = torch.rand(args.n) < 0.03 if t > 0 else torch.zeros(args.n, dtype=torch.bool)
        n_resets += int(reset.sum())
        # harness: DeploymentRunner zeroes the history of reset robots before their next observation is appended
        pre.reset(reset)
        env.episode_length_buf = torch.where(reset, torch.zeros_like(env.episode_length_buf), env.episode_length_buf + 1)
        x = torch.where(reset.unsqueeze(1), centre + 0.1 * torch.randn(args.n, O), x)
        pobs = pre(x)
        mean, _ = actor(pobs)
        a_ref = ref(x)
        max_a = max(max_a, (mean - a_ref).abs().max().item())
        v = critic(pobs)
        v_ref = crit_ref.mlp(torch.cat([x, ref.encoder(ref.hist)], dim=-1)).squeeze(-1)
        max_v = max(max_v, (v - v_ref).abs().max().item())
print(f"[2] actions, harness vs StudentPolicy, {args.n} robots x {args.steps} steps, {n_resets} resets: max |diff| {max_a:.2e}")
print(f"[3] deployment critic, BatchedCritic vs checkpoint MLP on [obs, z_hat]: max |diff| {max_v:.2e}")
ok &= max_a < 1e-4 and max_v < 1e-4
print("rma checks passed" if ok else "RMA CHECKS FAILED")
sys.exit(0 if ok else 1)
