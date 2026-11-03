# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RMA stage 3 (decisions.md P6/P7): distill the adaptation module phi from a teacher checkpoint.

RMA phase 2 (arXiv 2107.04034, Sec. III-B, IV-B and Algorithm 1, verified 11-03): the base policy is frozen and
unrolled with phi's estimate z_hat (not the true z); phi is regressed on z = mu(e) with an MSE loss on that
on-policy data, Adam lr 5e-4, 4 minibatches per iteration. RMA: 1000 iterations of 80k samples. Here an iteration is
num_envs x --steps samples (default 512 x 24 = 12,288, our PPO batch; ASSUMED, a smaller budget than RMA's).
Actions are the actor's mean (ASSUMED; the paper does not say whether the unroll samples). z is the ACTOR's
encoder output (the actor is what acts on z_hat).

The env is the teacher's training env: every top-level scalar of the teacher run's params/env.yaml plus
terrain.max_init_terrain_level, with DR and the terrain curriculum on, as in training.

Writes <out_dir>/model_<it>.pt every --save_every iterations and at the end (student format: hexapod.rma
student_checkpoint), and <out_dir>/distill_log.jsonl (per iteration: mse, z variance, R^2 = 1 - mse / var).

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/rma_distill.py --headless \
      --teacher <run>/model_N.pt --out_dir <logs>/rsl_rl/hexapod_rough_direct/<name>
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--teacher", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--steps", type=int, default=24, help="env steps per iteration")
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--history_len", type=int, default=50)
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--seed", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.manual_seed(args.seed)

import hexapod  # noqa: F401, E402
from hexapod.rma import CheckpointPolicy, HistoryEncoder, student_checkpoint  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

run_dir = os.path.dirname(args.teacher)
params = yaml.load(open(os.path.join(run_dir, "params", "env.yaml")), Loader=yaml.UnsafeLoader)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
applied = {}
for k, v in params.items():
    if isinstance(v, (bool, int, float, str)) and hasattr(env_cfg, k) and k != "seed":
        setattr(env_cfg, k, v)
        applied[k] = v
env_cfg.terrain.max_init_terrain_level = params["terrain"]["max_init_terrain_level"]
applied["terrain.max_init_terrain_level"] = env_cfg.terrain.max_init_terrain_level
env_cfg.seed = args.seed
print(f"[distill] env settings from {run_dir}: {applied}", flush=True)
env = gym.make(args.task, cfg=env_cfg).unwrapped
dev, n = env.device, env.num_envs

ck = torch.load(args.teacher, map_location=dev, weights_only=False)
teacher = CheckpointPolicy(ck["actor_state_dict"]).to(dev).eval()
assert teacher.is_teacher, "--teacher must be an RMA teacher checkpoint"
for p in teacher.parameters():
    p.requires_grad_(False)

obs_d, _ = env.reset()
obs = obs_d["policy"]
z_dim = teacher.z(obs_d["privileged"][:1]).shape[-1]
phi = HistoryEncoder(obs.shape[-1], z_dim, args.history_len).to(dev)
opt = torch.optim.Adam(phi.parameters(), lr=args.lr)
os.makedirs(args.out_dir, exist_ok=True)
logf = open(os.path.join(args.out_dir, "distill_log.jsonl"), "a")
hist = torch.zeros(n, args.history_len, obs.shape[-1], device=dev)
print(f"[distill] {n} envs, {args.steps} steps/it, history {args.history_len}, z {z_dim}, phi params "
      f"{sum(p.numel() for p in phi.parameters())}", flush=True)


def save(it: int, rec: dict):
    out = student_checkpoint(ck, HistoryEncoder(phi.obs_dim, phi.z_dim, phi.history_len).to(dev), {
        "teacher": os.path.abspath(args.teacher), "distill_iter": it, "distill_last": rec})
    out["rma_student"]["encoder"].load_state_dict({k: v.cpu() for k, v in phi.state_dict().items()})
    out["rma_student"]["encoder_state_dict"] = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
    torch.save(out, os.path.join(args.out_dir, f"model_{it}.pt"))


t0 = time.time()
for it in range(1, args.iterations + 1):
    H, Z = [], []
    falls = 0
    phi.eval()
    for _ in range(args.steps):
        hist = torch.cat([hist[:, 1:], obs.unsqueeze(1)], dim=1)
        with torch.no_grad():
            z = teacher.z(obs_d["privileged"])
            act = teacher.mlp(torch.cat([obs, phi(hist)], dim=-1))
        H.append(hist)
        Z.append(z)
        obs_d, _, term, trunc, _ = env.step(act)
        obs = obs_d["policy"]
        done = term | trunc
        falls += int(term.sum().item())
        hist = hist.clone()
        hist[done] = 0.0
    H = torch.cat(H)
    Z = torch.cat(Z)
    phi.train()
    perm = torch.randperm(H.shape[0], device=dev)
    loss_sum = 0.0
    for mb in perm.chunk(args.minibatches):
        loss = (phi(H[mb]) - Z[mb]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    with torch.no_grad():
        var = Z.var(0).mean().item()
    mse = loss_sum / args.minibatches
    rec = {"it": it, "mse": round(mse, 6), "z_var": round(var, 6), "r2": round(1 - mse / max(var, 1e-12), 4),
           "falls": falls, "terrain_level": round(float(env._terrain.terrain_levels.float().mean()), 3)
           if getattr(env, "_terrain", None) is not None and hasattr(env._terrain, "terrain_levels") else None,
           "t_s": round(time.time() - t0, 1)}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    if it % 10 == 0 or it == 1:
        print(f"[distill] {json.dumps(rec)}", flush=True)
    if it % args.save_every == 0 or it == args.iterations:
        save(it, rec)
print(f"[distill] done, {args.iterations} iterations in {time.time() - t0:.0f} s", flush=True)
logf.close()
# Kit sometimes never returns from close (see eval_suite's watchdog); everything is on disk
os._exit(0)
