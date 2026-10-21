# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the two batched-policy implementations (deploy/batched.py): einsum/baddbmm over stacked weights vs
torch.func.vmap over a functional MLP, forward only and forward+backward, for N deployments of a 128x3 actor.
Also checks that both give the same output and that slices are independent (a gradient step on slice i leaves
slice j unchanged). No Isaac Sim needed:

  ~/bin/isaac-shell /workspace/isaaclab/_isaac_sim/python.sh /workspace/learning/hexapod/tools/deploy_bench_policy.py
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy.batched import BatchedAdam, BatchedGaussianActor, BatchedMLP  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ns", type=int, nargs="+", default=[1, 16, 64, 256])
parser.add_argument("--batch", type=int, default=500, help="samples per slice for the backward benchmark (PPO batch)")
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
dev = torch.device(args.device)
OBS, ACT, HID = 68, 18, (128, 128, 128)


def timeit(fn, iters):
    for _ in range(3):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


print(f"device {dev}, obs {OBS}, act {ACT}, hidden {HID}")
print("| N | fwd einsum [ms] | fwd vmap [ms] | fwd+bwd einsum [ms] (B=%d) | fwd+bwd vmap [ms] | max |diff| |" % args.batch)
print("|---|---|---|---|---|---|")
for n in args.ns:
    mlp = BatchedMLP(n, OBS, HID, ACT).to(dev)
    x1 = torch.randn(n, OBS, device=dev)
    xb = torch.randn(n, args.batch, OBS, device=dev)
    with torch.no_grad():
        diff = (mlp(x1) - mlp.forward_vmap(x1)).abs().max().item()
    f_e = timeit(lambda: mlp(x1), args.iters)
    f_v = timeit(lambda: mlp.forward_vmap(x1), args.iters)

    def fb(fwd):
        out = fwd(xb)
        out.pow(2).mean().backward()

    b_e = timeit(lambda: fb(mlp.forward), args.iters)
    b_v = timeit(lambda: fb(mlp.forward_vmap), args.iters)
    print(f"| {n} | {f_e:.3f} | {f_v:.3f} | {b_e:.3f} | {b_v:.3f} | {diff:.1e} |")

# independence check: a masked Adam step on slice 0 must leave slice 1 untouched, and the loss on slice 0 must fall
n = 4
actor = BatchedGaussianActor(n, OBS, HID, ACT).to(dev)
opt = BatchedAdam(actor.parameters_per_slice(), n, lr=1e-2)
x = torch.randn(n, 64, OBS, device=dev)
target = torch.randn(n, 64, ACT, device=dev)
before = [p.detach().clone() for p in actor.parameters_per_slice()]
mask = torch.tensor([True, False, False, True], device=dev)


def loss_fn():
    mean, _ = actor(x)
    return ((mean - target) ** 2).mean(dim=(1, 2))  # per slice


l0 = loss_fn().detach()
for _ in range(20):
    opt.zero_grad()
    loss_fn().sum().backward()
    opt.clip_grad_norm(1.0, mask)
    opt.step(mask)
l1 = loss_fn().detach()
changed = [((p - b).abs().flatten(1).sum(1) > 0).tolist() for p, b in zip(actor.parameters_per_slice(), before) if p.grad is not None]
print("per-slice loss before", [f"{v:.3f}" for v in l0.tolist()], "after", [f"{v:.3f}" for v in l1.tolist()])
print("slices changed per parameter (expect [True, False, False, True]):", changed[0], "all params agree:", all(c == changed[0] for c in changed))
assert changed[0] == mask.tolist() and all(c == changed[0] for c in changed)
assert (l1[mask] < l0[mask]).all() and torch.allclose(l1[~mask], l0[~mask])
print("independence check passed")
