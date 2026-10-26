# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate many policies side by side with the eval-suite protocol (tools/eval_suite.py, read 2026-10-26): 20 s
deterministic episodes from a standing start, cells = terrain level x type (each type starts in the higher column of
its pair), one DR cell (friction, payload, motor strength), fixed forward command, per-robot start jitter (xy, yaw,
random gait phase), P3 success / falls / mean speed from the env's own episode_stats. Many policy slices share one
batch (stacked weights), so a whole snapshot curve costs a few batches instead of one eval-suite run per policy.

Policies: --checkpoint FILE (label "frozen") and/or --snapshots DIR (every adapt_*.pt written by
tools/deploy_sweep_updater.py --snapshot_adapt_min; one policy per deployment slice and minute).
Output: <out>.json (per policy: success, falls, mean speed, per-cell results, group / replicate / adapt-min /
robot-hours from the snapshot) and <out>.md (per group and minute, pooled over replicates, with SE).

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_eval_policies.py --headless \\
      --checkpoint <model.pt> --snapshots <sweep>/snapshots --dr_cell 1.0,0.0,0.6 --robots_per_cell 1 --out <path>
"""

import argparse
import glob
import json
import math
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="frozen pi_0 (also defines the env through its params/)")
parser.add_argument("--run_dir", default=None)
parser.add_argument("--snapshots", default=None)
parser.add_argument("--groups", nargs="*", default=None, help="only these snapshot groups")
parser.add_argument("--minutes", type=int, nargs="*", default=None, help="only these adapt-minutes")
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--dr_cell", default="1.0,0.0,1.0", help="friction,dmass_kg,motor_strength")
parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
parser.add_argument("--cmd", type=float, default=0.28)
parser.add_argument("--robots_per_cell", type=int, default=4, help="per policy")
parser.add_argument("--frozen_robots_per_cell", type=int, default=None, help="for the --checkpoint policy (default = robots_per_cell)")
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--start_jitter", type=float, nargs=2, default=[0.05, 10.0])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--out", required=True, help="output path without extension")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import BatchedGaussianActor, load_rsl_rl_checkpoint  # noqa: E402
from hexapod.deploy.envutil import deploy_env_cfg  # noqa: E402

torch.manual_seed(args.seed)
mu, dm, s_ = (float(x) for x in args.dr_cell.split(","))

# ---- policies ------------------------------------------------------------------------------------------------------
pols = []  # (label, meta, actor_state (rsl_rl layout), slice index or None)
assert args.checkpoint, "--checkpoint is needed (it defines the env and the network layout)"
info = load_rsl_rl_checkpoint(args.checkpoint, "cpu")
pols.append({"label": "frozen", "group": "frozen", "replicate": 0, "adapt_min": 0, "robot_hours": 0.0, "sd": info["actor"], "slice": None, "rpc": args.frozen_robots_per_cell or args.robots_per_cell})
if args.snapshots:
    for f in sorted(glob.glob(os.path.join(args.snapshots, "adapt_*.pt"))):
        snap = torch.load(f, map_location="cpu", weights_only=False)
        m = int(snap["adapt_min"])
        if args.minutes is not None and m not in args.minutes:
            continue
        for sl, (g, r) in enumerate(zip(snap["group_of_slice"], snap["replicate_of_slice"])):
            gname = snap["groups"][g]["name"]
            if args.groups is not None and gname not in args.groups:
                continue
            pols.append({"label": f"{gname}/r{r}/m{m}", "group": gname, "replicate": int(r), "adapt_min": m, "robot_hours": float(snap["robot_hours_of_slice"][sl]), "sd": snap["actor"], "slice": sl, "rpc": args.robots_per_cell})
P = len(pols)
print(f"[evalp] {P} policies", flush=True)

# ---- env -------------------------------------------------------------------------------------------------------------
dcfg = C.DeployCfg(task=args.task, cmd_speed=args.cmd)
env_cfg, applied = deploy_env_cfg(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir, dr=False, episode_s=args.episode_s)
import gymnasium as gym  # noqa: E402

import hexapod  # noqa: F401, E402

env = gym.make(args.task, cfg=env_cfg).unwrapped
dev = env.device
NE = env.num_envs
n_cols = env._terrain.terrain_origins.shape[1]
seen = {}
for col in range(n_cols):
    seen.setdefault(env.terrain_type_names[col], []).append(col)
cells = [(lvl, name, max(cols)) for lvl in args.levels for name, cols in seen.items()]
jobs = [(p, ci) for p in range(P) for ci in range(len(cells)) for _ in range(pols[p]["rpc"])]
print(f"[evalp] {len(cells)} cells, {len(jobs)} episodes, {math.ceil(len(jobs) / NE)} batches of {NE}", flush=True)

# ---- stacked actor (one slice per policy) -------------------------------------------------------------------------
actor = BatchedGaussianActor(P, info["obs_dim"], tuple(info["hidden_dims"]), info["act_dim"]).to(dev)
with torch.no_grad():
    for i, p in enumerate(pols):
        if p["slice"] is None:
            actor.load_rsl_rl(p["sd"], idx=torch.tensor([i], device=dev))
        else:
            sd = p["sd"]
            for li in range(actor.mlp.n_layers):
                actor.mlp.weights[li][i] = sd[f"mlp.weights.{li}"][p["slice"]].to(dev)
                actor.mlp.biases[li][i] = sd[f"mlp.biases.{li}"][p["slice"]].to(dev)


def jitter(n):
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    xy, yaw_deg = args.start_jitter
    ids = torch.arange(n, device=dev)
    pose = env._robot.data.root_pose_w.torch[:n].clone()
    pose[:, 0:2] += (torch.rand(n, 2, device=dev) * 2 - 1) * xy
    yaw = (torch.rand(n, device=dev) * 2 - 1) * math.radians(yaw_deg)
    zero = torch.zeros_like(yaw)
    pose[:, 3:7] = quat_mul(quat_from_euler_xyz(zero, zero, yaw), pose[:, 3:7])
    env._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids.to(torch.int32))
    env._gait_phase[:n] = torch.rand(n, device=dev) * 2 * math.pi


results = {k: np.zeros(len(jobs)) for k in ("success", "fell", "mean_v_along_cmd")}
for b0 in range(0, len(jobs), NE):
    batch = jobs[b0 : b0 + NE]
    n = len(batch)
    ids = torch.arange(n, device=dev)
    lv = torch.tensor([cells[ci][0] for _, ci in batch], device=dev)
    col = torch.tensor([cells[ci][2] for _, ci in batch], device=dev)
    pidx = torch.tensor([p for p, _ in batch] + [0] * (NE - n), device=dev)
    env.set_terrain_cells(ids, lv, col)
    env.reset()
    env.set_friction(ids, mu, hold=True)
    env.set_payload(ids, dm, hold=True)
    env.set_motor_strength(ids, s_, hold=True)
    env._commands[:] = 0.0
    env._commands[ids, 1] = -args.cmd
    env._gait_phase[:] = 0.0
    jitter(n)
    env.episode_length_buf[:] = 0
    obs = env._get_observations()["policy"]
    done = torch.zeros(NE, dtype=torch.bool, device=dev)
    out = {k: torch.zeros(NE, device=dev) for k in results}
    for i in range(int(args.episode_s / env.step_dt) + 2):
        with torch.no_grad():
            act, _ = actor.forward_gather(obs, pidx)
        o, _, term, trunc, _ = env.step(act)
        obs = o["policy"]
        fin = (term | trunc) & ~done
        if bool(fin.any()):
            for k in out:
                out[k][fin] = env.episode_stats[k][fin]
            done |= fin
        if bool(done[:n].all()):
            break
        env._commands[ids, 0] = 0.0
        env._commands[ids, 1] = -args.cmd
        env._commands[ids, 2] = 0.0
    env.release_dr(torch.arange(NE, device=dev))
    for k in results:
        results[k][b0 : b0 + n] = out[k][:n].cpu().numpy()
    print(f"[evalp] batch {b0 // NE + 1}/{math.ceil(len(jobs) / NE)} done", flush=True)

# ---- summaries -------------------------------------------------------------------------------------------------------
pol_of_job = np.array([p for p, _ in jobs])
cell_of_job = np.array([ci for _, ci in jobs])
rows = []
for p, pol in enumerate(pols):
    sel = pol_of_job == p
    succ, fell, v = results["success"][sel], results["fell"][sel], results["mean_v_along_cmd"][sel]
    per_cell = [{"level": cells[ci][0], "type": cells[ci][1], "success": float(results["success"][sel & (cell_of_job == ci)].mean()), "fell": float(results["fell"][sel & (cell_of_job == ci)].mean()), "mean_v": float(results["mean_v_along_cmd"][sel & (cell_of_job == ci)].mean())} for ci in range(len(cells))]
    rows.append({k: pol[k] for k in ("label", "group", "replicate", "adapt_min", "robot_hours")} | {"n": int(sel.sum()), "success": float(succ.mean()), "falls": float(fell.mean()), "mean_v": float(v.mean()), "success_list": succ.tolist(), "cells": per_cell})
res = {"checkpoint": args.checkpoint, "snapshots": args.snapshots, "dr_cell": [mu, dm, s_], "cmd": args.cmd, "levels": args.levels, "cells": [list(c) for c in cells], "robots_per_cell": args.robots_per_cell, "start_jitter": args.start_jitter, "seed": args.seed, "env_applied": applied, "policies": rows}
json.dump(res, open(args.out + ".json", "w"), indent=1)
# pooled per group and minute
md = [f"# Policy eval: friction {mu}, payload {dm:+g} kg, motor {s_}, cmd {args.cmd}, levels {args.levels} x {len(seen)} types, jitter {args.start_jitter}", "",
      "Pooled over replicates; SE binomial over episodes / over replicates (deployment-level).", "",
      "| group | adapt-min | robot-hours per deployment | deployments | episodes | success [SE ep / SE dep] | falls | mean speed [m/s] |", "|---|---|---|---|---|---|---|---|"]
keys = sorted({(r["group"], r["adapt_min"]) for r in rows}, key=lambda x: (x[0] != "frozen", x[0], x[1]))
for g, m in keys:
    rs = [r for r in rows if r["group"] == g and r["adapt_min"] == m]
    allsucc = np.concatenate([r["success_list"] for r in rs])
    se_ep = math.sqrt(allsucc.mean() * (1 - allsucc.mean()) / len(allsucc))
    dep = np.array([r["success"] for r in rs])
    se_dep = dep.std(ddof=1) / math.sqrt(len(dep)) if len(dep) > 1 else float("nan")
    md.append(f"| {g} | {m} | {rs[0]['robot_hours']:.2f} | {len(rs)} | {len(allsucc)} | {allsucc.mean():.3f} [{se_ep:.3f} / {se_dep:.3f}] | {np.mean([r['falls'] for r in rs]):.3f} | {np.mean([r['mean_v'] for r in rs]):.3f} |")
open(args.out + ".md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("[evalp] wrote", args.out + ".json")
env.close()
app.close()
