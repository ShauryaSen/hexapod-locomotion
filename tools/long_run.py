# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-run bar (decisions.md P5): a fixed-command deployment for --minutes without time-out resets.

A fall (body contact) resets that robot in place (counted, as the deployment will do). Per --window minutes it
records falls, mean reward per step, mean speed error along the command and mean speed, and reports falls per
robot-hour and the drift of the last window relative to the first. Bar: <= 1 fall/h nominal, and reward and speed
error in the last 10 min within 5% of the first 10 min over 60 min.

Terrain: on the rough env the robots are pinned to --levels (default 0-5, every type); terrain curriculum off.
DR: nominal unless --dr (then the cfg ranges, resampled only when a robot falls).

Run inside the container (from the Isaac Lab root):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/long_run.py --headless \
      --task Hexapod-Rough-Direct-v0 --checkpoint <model.pt> --cmd 0.16 --minutes 60 --num_envs 64
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None, help="for params/env.yaml (default: the checkpoint's directory)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--cmd", type=float, default=0.16, help="fixed forward command [m/s]")
parser.add_argument("--minutes", type=float, default=60.0)
parser.add_argument("--window", type=float, default=10.0, help="reporting window [min]")
parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
parser.add_argument("--dr", action="store_true", help="randomize friction/mass/strength (cfg ranges) instead of nominal")
parser.add_argument("--env_override", nargs="*", default=[])
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import hexapod  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

PARAM_KEYS = ("action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "command_yaw_rate_max", "perceptive", "step_cap")

run_dir = args.run_dir or os.path.dirname(args.checkpoint)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
params_path = os.path.join(run_dir, "params", "env.yaml")
applied = {}
if os.path.exists(params_path):
    params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)
    for k in PARAM_KEYS:
        if k in params and isinstance(params[k], (bool, int, float, str)):
            setattr(env_cfg, k, params[k])
            applied[k] = params[k]
for ov in args.env_override:
    k, v = ov.split("=", 1)
    setattr(env_cfg, k, yaml.safe_load(v))
    applied[k] = yaml.safe_load(v)
env_cfg.spread_resets = False
env_cfg.terrain_curriculum = False
env_cfg.command_curriculum = False
env_cfg.dr.enabled = args.dr
env_cfg.command_forward_only = True
env_cfg.command_lin_vel_min = env_cfg.command_lin_vel_max = args.cmd
env_cfg.command_backward_fraction = 0.0
env_cfg.command_yaw_rate_max = 0.0
env_cfg.episode_length_s = args.minutes * 60.0 + 60.0  # no time-outs inside the run
if env_cfg.terrain.terrain_type == "generator":
    env_cfg.terrain.max_init_terrain_level = None
env = gym.make(args.task, cfg=env_cfg).unwrapped
dev = env.device
N = env.num_envs
print("[long_run] env settings:", applied)

ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
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
actor = nn.Sequential(*mods).to(dev).eval()

ids = torch.arange(N, device=dev)
if env._has_terrain_levels:
    n_cols = env._terrain.terrain_origins.shape[1]
    levels = torch.tensor(args.levels, device=dev)[ids % len(args.levels)]
    types = (ids // len(args.levels)) % n_cols
    env.set_terrain_cells(ids, levels, types)
obs = env.reset()[0]["policy"]
env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -args.cmd, 0.0
obs = env._get_observations()["policy"]

steps_per_window = int(args.window * 60.0 / env.step_dt)
n_windows = int(args.minutes / args.window + 1e-6)
windows = []
for w in range(n_windows):
    falls, rew_sum, verr_sum, v_sum, n = 0, 0.0, 0.0, 0.0, 0
    for _ in range(steps_per_window):
        with torch.no_grad():
            act = actor(obs)
        obs_d, rew, terminated, truncated, _ = env.step(act)
        obs = obs_d["policy"]
        falls += int(terminated.sum().item())
        v = -env._robot.data.root_lin_vel_b.torch[:, 1]
        rew_sum += rew.mean().item()
        verr_sum += (v - args.cmd).abs().mean().item()
        v_sum += v.mean().item()
        n += 1
        env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -args.cmd, 0.0  # resets resample
    robot_hours = N * steps_per_window * env.step_dt / 3600.0
    windows.append({"window": w, "falls": falls, "falls_per_robot_hour": falls / robot_hours, "mean_reward": rew_sum / n, "mean_speed_err": verr_sum / n, "mean_speed": v_sum / n})
    print(f"[long_run] window {w + 1}/{n_windows}: falls {falls} ({falls / robot_hours:.3f}/robot-h), reward {rew_sum / n:.4f}, speed err {verr_sum / n:.4f}, speed {v_sum / n:.4f}")

first, last = windows[0], windows[-1]
drift = {k: (last[k] - first[k]) / max(abs(first[k]), 1e-9) for k in ("mean_reward", "mean_speed_err")}
total_falls = sum(w["falls"] for w in windows)
summary = {
    "checkpoint": args.checkpoint, "task": args.task, "cmd": args.cmd, "minutes": args.minutes, "num_envs": N,
    "levels": args.levels if env._has_terrain_levels else ["flat"], "dr": args.dr, "applied": applied,
    "falls_total": total_falls, "falls_per_robot_hour": total_falls / (N * args.minutes / 60.0),
    "drift_last_vs_first": drift, "windows": windows,
}
print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, indent=1))
out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(args.checkpoint))[0] + "_longrun" + (f"_{args.tag}" if args.tag else "")
json.dump(summary, open(os.path.join(out_dir, base + ".json"), "w"), indent=1)
print("[long_run] wrote", os.path.join(out_dir, base + ".json"))
env.close()
app.close()
