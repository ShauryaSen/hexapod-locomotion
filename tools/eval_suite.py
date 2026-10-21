# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation suite (decisions.md P3/P4/P8): deterministic 20 s episodes over a grid of cells.

Cells = (terrain level x terrain type) x (DR grid) x (command speed bucket), N robots per cell.
  * terrain: levels 0-9 x every sub-terrain column type of the rough env; on the flat env a single cell "flat".
  * DR grid: friction {0.5, 0.875, 1.25} x added mass {-0.75, 0, +0.75} kg x motor strength {0.8, 1.0, 1.2}
    (27 cells), or nominal only (1 cell) with --nominal_only.
  * command speeds: --cmds (default: the env's command range in 0.05 m/s buckets), forward along -y, no yaw.
Per cell: success (P3: time-out without a fall AND |mean speed along cmd - cmd| < 25% cmd), mean speed error,
mean speed, falls. The per-episode outcome is computed by the env itself (HexapodEnv.episode_stats), so the suite
and the training logs use one definition.

Checkpoints: --checkpoint FILE, or --run_dir DIR to evaluate every model_*.pt in it. Env settings (gait clock,
command range, action scale, DR ranges, ...) are read from DIR/params/env.yaml unless --no_params; extra overrides
as key=value (--env_override gait_period_s=0.6).

Outputs next to each checkpoint: eval/<ckpt>.json and eval/<ckpt>.md (tables per level x type, per DR cell,
per command bucket).

Run inside the container (from the Isaac Lab root):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/eval_suite.py --headless \
      --task Hexapod-Flat-Direct-v0 --run_dir /workspace/isaaclab/logs/rsl_rl/hexapod_flat_direct/<run> \
      --nominal_only --robots_per_cell 32 --num_envs 512
"""

import argparse
import glob
import itertools
import json
import os
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Flat-Direct-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--run_dir", default=None, help="evaluate every model_*.pt in this directory")
parser.add_argument("--latest_only", action="store_true", help="with --run_dir: only the highest-numbered checkpoint")
parser.add_argument("--num_envs", type=int, default=512, help="envs per batch")
parser.add_argument("--robots_per_cell", type=int, default=16)
parser.add_argument("--nominal_only", action="store_true", help="DR grid = nominal only")
parser.add_argument("--levels", type=int, nargs="+", default=list(range(10)))
parser.add_argument("--cmds", type=float, nargs="+", default=None, help="command speeds [m/s]; default = env range in 0.05 buckets")
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--env_override", nargs="*", default=[], help="key=value env cfg overrides (applied after params)")
parser.add_argument("--no_params", action="store_true", help="do not read <run>/params/env.yaml")
parser.add_argument("--tag", default="", help="suffix for the output files")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import hexapod  # noqa: F401, E402  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FRICTIONS = [0.5, 0.875, 1.25]
MASSES = [-0.75, 0.0, 0.75]
STRENGTHS = [0.8, 1.0, 1.2]
PARAM_KEYS = (  # scalar env.yaml fields that change the policy's interface or the task; everything else stays default
    "action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "command_lin_vel_min",
    "command_lin_vel_max", "command_yaw_rate_max", "perceptive", "command_curriculum_max", "command_curriculum",
)


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


# ---- checkpoints
if args.checkpoint:
    ckpts = [args.checkpoint]
    run_dir = args.run_dir or os.path.dirname(args.checkpoint)
else:
    assert args.run_dir, "--checkpoint or --run_dir"
    run_dir = args.run_dir
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]) if re.findall(r"model_(\d+)\.pt", p) else -1)
    if args.latest_only:
        ckpts = ckpts[-1:]
assert ckpts, "no checkpoints found"

# ---- env cfg
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
params_path = os.path.join(run_dir, "params", "env.yaml")
applied = {}
if not args.no_params and os.path.exists(params_path):
    params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)  # Isaac Lab dumps python/tuple tags
    for k in PARAM_KEYS:
        if k in params and isinstance(params[k], (bool, int, float, str)):
            setattr(env_cfg, k, params[k])
            applied[k] = params[k]
for ov in args.env_override:
    k, v = ov.split("=", 1)
    setattr(env_cfg, k, parse_value(v))
    applied[k] = parse_value(v)
env_cfg.spread_resets = False
env_cfg.terrain_curriculum = False
env_cfg.command_curriculum = False
env_cfg.dr.enabled = False  # the suite sets the DR values itself per cell
env_cfg.episode_length_s = args.episode_s
if env_cfg.terrain.terrain_type == "generator":
    env_cfg.terrain.max_init_terrain_level = None
env = gym.make(args.task, cfg=env_cfg).unwrapped
dev = env.device
num_envs = env.num_envs
print("[eval] env settings from params/overrides:", applied)

# ---- cells
cmds = args.cmds
if cmds is None:
    lo = max(0.05, float(getattr(env_cfg, "command_lin_vel_min", 0.0)))
    hi = float(env_cfg.command_curriculum_max if env_cfg.command_curriculum else env_cfg.command_lin_vel_max)
    cmds = [round(lo + 0.05 * i, 3) for i in range(int((hi - lo) / 0.05 + 1e-6) + 1)]
if env._has_terrain_levels:
    n_cols = env._terrain.terrain_origins.shape[1]
    type_names = env.terrain_type_names
    seen = {}
    for col in range(n_cols):
        seen.setdefault(type_names[col], []).append(col)
    terrain_cells = [(lvl, name, cols) for lvl in args.levels for name, cols in seen.items()]
else:
    terrain_cells = [(0, "flat", [0])]
dr_cells = [(float(env_cfg.sim.physics_material.static_friction), 0.0, 1.0)] if args.nominal_only else list(itertools.product(FRICTIONS, MASSES, STRENGTHS))
cells = [
    {"level": lvl, "type": name, "cols": cols, "friction": f, "dmass": m, "strength": s, "cmd": c}
    for (lvl, name, cols) in terrain_cells
    for (f, m, s) in dr_cells
    for c in cmds
]
jobs = [i for i in range(len(cells)) for _ in range(args.robots_per_cell)]
print(f"[eval] {len(cells)} cells x {args.robots_per_cell} robots = {len(jobs)} episodes, {num_envs} envs per batch, cmds {cmds}")


def load_actor(path: str) -> nn.Module:
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    mods = []
    for i, li in enumerate(layers):
        w = sd[f"mlp.{li}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"mlp.{li}.bias"])
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ELU())
    for k in sd:
        if "normalizer" in k:
            raise RuntimeError(f"checkpoint has an observation normalizer ({k}); the suite does not apply it")
    return nn.Sequential(*mods).to(dev).eval()


def run_batch(actor, batch_jobs):
    n = len(batch_jobs)
    ids = torch.arange(n, device=dev)
    cell_rows = [cells[j] for j in batch_jobs]
    # terrain cells: spread the cell's robots over the columns of that type
    if env._has_terrain_levels:
        levels = torch.tensor([c["level"] for c in cell_rows], device=dev)
        types = torch.tensor([c["cols"][k % len(c["cols"])] for k, c in enumerate(cell_rows)], device=dev)
        env.set_terrain_cells(ids, levels, types)
    env.reset()
    env.set_friction(ids, torch.tensor([c["friction"] for c in cell_rows], device=dev), hold=True)
    env.set_payload(ids, torch.tensor([c["dmass"] for c in cell_rows], device=dev), hold=True)
    env.set_motor_strength(ids, torch.tensor([c["strength"] for c in cell_rows], device=dev), hold=True)
    env._commands[:] = 0.0
    env._commands[ids, 1] = -torch.tensor([c["cmd"] for c in cell_rows], device=dev)  # forward = -y
    env._gait_phase[:] = 0.0
    env.episode_length_buf[:] = 0
    obs = env._get_observations()["policy"]
    done = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    out = {k: torch.zeros(num_envs, device=dev) for k in ("success", "fell", "speed_err", "mean_v_along_cmd", "length_s")}
    steps = int(args.episode_s / env.step_dt) + 2
    for _ in range(steps):
        with torch.no_grad():
            act = actor(obs)
        obs_d, _, terminated, truncated, _ = env.step(act)
        obs = obs_d["policy"]
        finished = (terminated | truncated) & ~done
        if finished.any():
            for k in out:
                out[k][finished] = env.episode_stats[k][finished]
            done |= finished
        if done[:n].all():
            break
        # keep the pinned commands for the envs still running (a reset would resample them)
        env._commands[ids, 0] = 0.0
        env._commands[ids, 1] = -torch.tensor([c["cmd"] for c in cell_rows], device=dev)
        env._commands[ids, 2] = 0.0
    env.release_dr(torch.arange(num_envs, device=dev))
    return {k: v[:n].cpu() for k, v in out.items()}


def summarize(results):
    per_cell = []
    for i, c in enumerate(cells):
        r = results[i]
        n = len(r["success"])
        per_cell.append(
            {**{k: v for k, v in c.items() if k != "cols"}, "n": n,
             "success": r["success"].mean().item(), "falls": r["fell"].mean().item(),
             "speed_err": r["speed_err"].mean().item(), "mean_v": r["mean_v_along_cmd"].mean().item()}
        )
    return per_cell


def table(per_cell, key_fn, header):
    groups = {}
    for pc in per_cell:
        g = groups.setdefault(key_fn(pc), {"success": 0.0, "falls": 0.0, "speed_err": 0.0, "mean_v": 0.0, "n": 0})
        for k in ("success", "falls", "speed_err", "mean_v"):
            g[k] += pc[k] * pc["n"]
        g["n"] += pc["n"]
    lines = [f"| {header} | success | falls | speed err [m/s] | mean v [m/s] | n |", "|---|---|---|---|---|---|"]
    for key in sorted(groups):
        g = groups[key]
        lines.append(f"| {key} | {g['success']/g['n']:.3f} | {g['falls']/g['n']:.3f} | {g['speed_err']/g['n']:.3f} | {g['mean_v']/g['n']:.3f} | {g['n']} |")
    return "\n".join(lines)


for ckpt in ckpts:
    actor = load_actor(ckpt)
    results = {i: {k: [] for k in ("success", "fell", "speed_err", "mean_v_along_cmd", "length_s")} for i in range(len(cells))}
    for b in range(0, len(jobs), num_envs):
        batch = jobs[b : b + num_envs]
        r = run_batch(actor, batch)
        for k, j in enumerate(batch):
            for key in results[j]:
                results[j][key].append(r[key][k])
        print(f"[eval] {os.path.basename(ckpt)}: batch {b // num_envs + 1}/{(len(jobs) + num_envs - 1) // num_envs} done")
    results = {i: {k: torch.stack(v) for k, v in d.items()} for i, d in results.items()}
    per_cell = summarize(results)
    md = [f"# eval {os.path.basename(ckpt)}", f"task {args.task}, {args.robots_per_cell} robots/cell, episode {args.episode_s} s, deterministic actor", f"env settings applied: {applied}", ""]
    md += ["## per terrain level x type", table(per_cell, lambda c: f"L{c['level']} {c['type']}", "level type"), ""]
    md += ["## per command speed", table(per_cell, lambda c: f"{c['cmd']:.2f}", "cmd [m/s]"), ""]
    md += ["## per DR cell (friction, dmass, strength)", table(per_cell, lambda c: f"{c['friction']:.3f} {c['dmass']:+.2f} {c['strength']:.1f}", "mu dm s"), ""]
    md += ["## per terrain level", table(per_cell, lambda c: f"L{c['level']}", "level"), ""]
    overall = table(per_cell, lambda c: "all", "")
    md += ["## overall", overall, ""]
    text = "\n".join(md)
    print(text)
    out_dir = os.path.join(os.path.dirname(ckpt), "eval")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ckpt))[0] + (f"_{args.tag}" if args.tag else "")
    open(os.path.join(out_dir, base + ".md"), "w").write(text)
    json.dump({"checkpoint": ckpt, "task": args.task, "applied": applied, "cmds": cmds, "cells": per_cell}, open(os.path.join(out_dir, base + ".json"), "w"), indent=1)
    print("[eval] wrote", os.path.join(out_dir, base + ".md"))

env.close()
app.close()
