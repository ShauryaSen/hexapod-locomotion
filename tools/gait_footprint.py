# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait footprint on flat ground at a fixed command: stride length and foot-placement spread per leg.

Runs a checkpoint (or the scripted tripod with --teacher) at --cmd on the flat env, detects touchdowns per foot
(contact force rises above 1 N), and reports per leg: stride length (distance between successive touchdowns of the
same foot, median), touchdown spread along and across the walking direction (p5-p95 of the touchdown positions in
the body frame), swing duration, duty factor, and the toe clearance proxy. These numbers size the terrain v2 stair
treads and box grid width.

Run inside the container:
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/gait_footprint.py --headless \
      --checkpoint <model.pt> --cmd 0.28 --num_envs 32 --seconds 20
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--teacher", action="store_true", help="scripted tripod (hip 0.5, period 0.4) instead of a checkpoint")
parser.add_argument("--cmd", type=float, default=0.28)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--settle", type=float, default=3.0)
parser.add_argument("--out", default="/workspace/isaaclab/logs/kinematics")
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402

cfg = HexapodFlatEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.dr.enabled = False
cfg.spread_resets = False
cfg.command_forward_only = True
cfg.command_lin_vel_min = cfg.command_lin_vel_max = args.cmd
cfg.command_backward_fraction = 0.0
cfg.command_yaw_rate_max = 0.0
cfg.gait_clock_enabled = True
cfg.gait_period_s = 0.4
cfg.episode_length_s = args.seconds + args.settle + 5.0
if args.checkpoint:
    pp = os.path.join(os.path.dirname(args.checkpoint), "params", "env.yaml")
    if os.path.exists(pp):
        params = yaml.load(open(pp), Loader=yaml.UnsafeLoader)
        for k in ("action_scale", "gait_clock_enabled", "gait_period_s"):
            if k in params:
                setattr(cfg, k, params[k])
env = HexapodEnv(cfg)
dev = env.device
N = env.num_envs
legs = ("lf", "lm", "lr", "rf", "rm", "rr")
tib = [env._robot.body_names.index(f"{l}_tibia") for l in legs]
feet_sensor_ids = env._feet_ids
toe_off = torch.tensor([0.130, 0.0, 0.0], device=dev).expand(N, 6, 3)

if args.teacher:
    from hexapod.scripted_tripod import ScriptedTripod

    teacher = ScriptedTripod(env, hip_max=0.5, speed_at_hip_max=0.176)
    dphi = 2 * math.pi * env.step_dt / cfg.gait_period_s

    def policy(obs):
        return teacher.delta(env, env._gait_phase + dphi) / cfg.action_scale
else:
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    mods = []
    for i, li in enumerate(layers):
        w = sd[f"mlp.{li}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0]); lin.weight.data.copy_(w); lin.bias.data.copy_(sd[f"mlp.{li}.bias"]); mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ELU())
    actor = nn.Sequential(*mods).to(dev).eval()

    def policy(obs):
        with torch.no_grad():
            return actor(obs)


obs = env.reset()[0]["policy"]
env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -args.cmd, 0.0
obs = env._get_observations()["policy"]
steps = int(args.seconds / env.step_dt)
settle = int(args.settle / env.step_dt)
in_contact = torch.zeros(N, 6, dtype=torch.bool, device=dev)
last_td_pos = torch.full((N, 6, 3), float("nan"), device=dev)
last_td_t = torch.full((N, 6), float("nan"), device=dev)
last_lo_t = torch.full((N, 6), float("nan"), device=dev)
strides, swing_t, stance_t, td_body = [], [], [], []
falls = 0
for i in range(steps + settle):
    obs_d, _, terminated, _, _ = env.step(policy(obs))
    obs = obs_d["policy"]
    env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -args.cmd, 0.0
    falls += int(terminated.sum().item())
    t = i * env.step_dt
    f = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[:, feet_sensor_ids], dim=-1)
    contact = f > 1.0
    pos = env._robot.data.body_link_pos_w.torch[:, tib]
    quat = env._robot.data.body_link_quat_w.torch[:, tib]
    toe = pos + quat_apply(quat.reshape(-1, 4), toe_off.reshape(-1, 3)).reshape(N, 6, 3)
    td = contact & ~in_contact
    lo = ~contact & in_contact
    if i >= settle:
        root = env._robot.data.root_pos_w.torch[:, None, :].expand(N, 6, 3)
        rq = env._robot.data.root_quat_w.torch[:, None, :].expand(N, 6, 4)
        toe_b = quat_apply_inverse(rq.reshape(-1, 4), (toe - root).reshape(-1, 3)).reshape(N, 6, 3)
        if td.any():
            prev = last_td_pos[td]
            d = torch.linalg.norm((toe[td] - prev)[:, :2], dim=-1)
            strides.append(d[~torch.isnan(d)])
            st = t - last_lo_t[td]
            swing_t.append(st[~torch.isnan(st)])
            td_body.append(toe_b[td])
        if lo.any():
            s = t - last_td_t[lo]
            stance_t.append(s[~torch.isnan(s)])
    last_td_pos[td] = toe[td]
    last_td_t[td] = t
    last_lo_t[lo] = t
    in_contact = contact
strides = torch.cat(strides) if strides else torch.zeros(0)
swing_t = torch.cat(swing_t) if swing_t else torch.zeros(0)
stance_t = torch.cat(stance_t) if stance_t else torch.zeros(0)
td_body = torch.cat(td_body) if td_body else torch.zeros(0, 3)
v = -env._robot.data.root_lin_vel_b.torch[:, 1].mean().item()
q = lambda x, p: float(torch.quantile(x, p)) if x.numel() else float("nan")
out = {
    "source": "teacher" if args.teacher else args.checkpoint, "cmd": args.cmd, "num_envs": N, "seconds": args.seconds,
    "falls": falls, "final_speed_mps": v, "n_touchdowns": int(strides.numel()),
    "stride_m": {"median": q(strides, 0.5), "p5": q(strides, 0.05), "p95": q(strides, 0.95)},
    "swing_s": {"median": q(swing_t, 0.5), "p5": q(swing_t, 0.05), "p95": q(swing_t, 0.95)},
    "stance_s": {"median": q(stance_t, 0.5)},
    "duty_factor": q(stance_t, 0.5) / max(q(stance_t, 0.5) + q(swing_t, 0.5), 1e-6),
    "touchdown_body_x_m": {"p5": q(td_body[:, 0], 0.05), "p95": q(td_body[:, 0], 0.95)},
    "touchdown_body_y_m": {"p5": q(td_body[:, 1], 0.05), "p95": q(td_body[:, 1], 0.95)},
    "touchdown_abs_body_x_spread_m": q(td_body[:, 0].abs(), 0.95) - q(td_body[:, 0].abs(), 0.05),
}
print(json.dumps(out, indent=1))
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, f"gait_footprint{('_' + args.tag) if args.tag else ''}.json"), "w"), indent=1)
env.close()
app.close()
