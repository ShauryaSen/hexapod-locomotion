# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a short headless video of a checkpoint at a fixed forward command (and pinned terrain levels on the rough env).

Run inside the container (from the Isaac Lab root):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/record_video.py --headless \
      --enable_cameras --task Hexapod-Flat-Direct-v0 --checkpoint <model.pt> --cmd 0.28 --num_envs 8 --steps 400 \
      --out /workspace/isaaclab/logs/videos/<name>
The mp4 lands in <out>/rl-video-step-0.mp4 (gymnasium RecordVideo naming). Camera: the env's ViewerCfg (next to env 0).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Flat-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--cmd", type=float, default=0.28)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--levels", type=int, nargs="+", default=None, help="rough env: terrain levels to spawn on")
parser.add_argument("--dr", action="store_true")
parser.add_argument("--out", required=True)
parser.add_argument("--camera", default="side", choices=["side", "behind", "high"], help="rough env static camera: side (1 m to +x), behind (0.6 m behind, 0.3 m up, looking along -y), high (3 m back, 2 m up)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
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
if os.path.exists(params_path):
    params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)
    for k in PARAM_KEYS:
        if k in params and isinstance(params[k], (bool, int, float, str)):
            setattr(env_cfg, k, params[k])
env_cfg.spread_resets = False
env_cfg.terrain_curriculum = False
env_cfg.command_curriculum = False
env_cfg.dr.enabled = args.dr
env_cfg.command_forward_only = True
env_cfg.command_lin_vel_min = env_cfg.command_lin_vel_max = args.cmd
env_cfg.command_backward_fraction = 0.0
env_cfg.command_yaw_rate_max = 0.0
env_cfg.episode_length_s = args.steps / 50.0 + 5.0
if env_cfg.terrain.terrain_type == "generator":
    env_cfg.terrain.max_init_terrain_level = None
env_cfg.scene.env_spacing = 1.0 if env_cfg.terrain.terrain_type == "plane" else env_cfg.scene.env_spacing
if env_cfg.terrain.terrain_type == "generator" and args.levels:
    # headless recording: the viewport camera controller is not created, so the ViewerCfg must already point at the
    # terrain cell env 0 is pinned to (level = args.levels[0], column 1). Static side view of the first ~2 m of walking.
    g = env_cfg.terrain.terrain_generator
    ox = (args.levels[0] + 0.5) * g.size[0] - g.num_rows * g.size[0] / 2
    oy = (1 + 0.5) * g.size[1] - g.num_cols * g.size[1] / 2
    env_cfg.viewer.origin_type = "world"
    if args.camera == "side":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox + 1.0, oy - 1.0, 0.45), (ox, oy - 1.0, 0.05)
    elif args.camera == "behind":  # the robot walks along -y, away from the camera
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox, oy + 0.6, 0.3), (ox, oy - 0.8, 0.0)
    else:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox, oy + 3.0, 2.0), (ox, oy - 1.2, 0.0)
    print(f"[record_video] camera at {env_cfg.viewer.eye} looking at {env_cfg.viewer.lookat} (cell origin {ox:.2f}, {oy:.2f})")
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
raw = env.unwrapped
os.makedirs(args.out, exist_ok=True)
env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0, video_length=args.steps, disable_logger=True)
dev = raw.device
N = raw.num_envs

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
if raw._has_terrain_levels and args.levels:
    n_cols = raw._terrain.terrain_origins.shape[1]
    levels = torch.tensor(args.levels, device=dev)[ids % len(args.levels)]
    types = (ids * 2 + 1) % n_cols  # odd columns: the next tile along -y is the same type
    raw.set_terrain_cells(ids, levels, types)
obs, _ = env.reset()
raw._commands[:, 0], raw._commands[:, 1], raw._commands[:, 2] = 0.0, -args.cmd, 0.0
obs = raw._get_observations()["policy"]
falls = 0
cam = getattr(raw, "viewport_camera_controller", None)
eye_off = torch.tensor(env_cfg.viewer.eye, device=dev)
look_off = torch.tensor(env_cfg.viewer.lookat, device=dev)


def follow_env0():
    # the viewer's asset_root tracking does not follow a robot that was teleported to a terrain cell; place the
    # camera relative to env 0's robot every step instead (same offsets as the ViewerCfg)
    if cam is None:
        return
    pos = raw._robot.data.root_pos_w.torch[0]
    cam.update_view_location(eye=(pos + eye_off).tolist(), lookat=(pos + look_off).tolist())


print("[record_video] viewport camera controller:", cam)
follow_env0()
for _ in range(args.steps):
    with torch.no_grad():
        act = actor(obs)
    follow_env0()
    obs_d, _, terminated, _, _ = env.step(act)
    obs = obs_d["policy"]
    falls += int(terminated.sum().item())
    raw._commands[:, 0], raw._commands[:, 1], raw._commands[:, 2] = 0.0, -args.cmd, 0.0
print(f"[record_video] {args.steps} steps, {N} envs, cmd {args.cmd}, falls {falls}, mean speed along cmd {(-raw._robot.data.root_lin_vel_b.torch[:, 1]).mean().item():.3f}")
env.close()
print("[record_video] wrote", os.listdir(args.out))
app.close()
