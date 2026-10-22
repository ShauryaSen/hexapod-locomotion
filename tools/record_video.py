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
parser.add_argument("--camera", default="side", choices=["side", "behind", "high"], help="static camera relative to env 0's cell (rough) or tracking offsets (flat): side (1 m to +x), behind (0.6 m behind, 0.3 m up, looking along -y), high (3 m back, 2 m up)")
parser.add_argument("--cam_eye", type=float, nargs=3, default=None, help="explicit world camera eye (overrides --camera)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=None)
parser.add_argument("--lateral_spread", type=float, default=0.0, help="rough env: spread the robots of one cell across the tile, x offsets in [-a, +a] m (they walk along -y)")
parser.add_argument("--column", type=int, default=1, help="rough env: terrain column for all envs (and the camera); use the higher column of a type pair")
parser.add_argument("--spread", action="store_true", help="rough env: spread the envs over every level and column (env i -> level i %% rows, column (i // rows) %% cols)")
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

PARAM_KEYS = ("action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "command_yaw_rate_max", "perceptive", "step_cap", "terrain_version", "terrain_stair_tread", "terrain_stair_platform")
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
if env_cfg.terrain.terrain_type == "generator" and (getattr(env_cfg, "terrain_version", 1) != 1 or getattr(env_cfg, "step_cap", None) is not None):
    # same rebuild the env does in __init__, so the camera uses the real grid size
    from hexapod.hexapod_terrains_cfg import make_hexapod_terrains_cfg, terrain_type_names

    _kw = {"version": getattr(env_cfg, "terrain_version", 1)}
    if env_cfg.step_cap is not None:
        _kw["step_cap"] = env_cfg.step_cap
    if getattr(env_cfg, "terrain_stair_tread", None) is not None:
        _kw["tread"] = env_cfg.terrain_stair_tread
    if getattr(env_cfg, "terrain_stair_platform", None) is not None:
        _kw["stair_platform"] = env_cfg.terrain_stair_platform
    env_cfg.terrain.terrain_generator = make_hexapod_terrains_cfg(**_kw)
    env_cfg.step_cap, env_cfg.terrain_version = None, 1  # already applied; stop the env from rebuilding it again
    env_cfg.terrain_stair_tread = env_cfg.terrain_stair_platform = None
    print(f"[record_video] terrain rebuilt with {_kw}")
    print("[record_video] terrain columns:", terrain_type_names(env_cfg.terrain.terrain_generator))
if env_cfg.terrain.terrain_type == "generator" and (args.levels or args.spread):
    if args.spread and not args.levels:
        args.levels = [0]
    # headless recording: the viewport camera controller is not created, so the ViewerCfg must already point at the
    # terrain cell env 0 is pinned to (level = args.levels[0], column 1). Static side view of the first ~2 m of walking.
    g = env_cfg.terrain.terrain_generator
    ox = (args.levels[0] + 0.5) * g.size[0] - g.num_rows * g.size[0] / 2
    oy = (args.column + 0.5) * g.size[1] - g.num_cols * g.size[1] / 2
    env_cfg.viewer.origin_type = "world"
    if args.camera == "side":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox + 1.0, oy - 1.0, 0.45), (ox, oy - 1.0, 0.05)
    elif args.camera == "behind":  # the robot walks along -y, away from the camera
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox, oy + 0.6, 0.3), (ox, oy - 0.8, 0.0)
    else:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (ox, oy + 3.0, 2.0), (ox, oy - 1.2, 0.0)
elif env_cfg.terrain.terrain_type == "plane":
    # flat: the viewer tracks env 0's robot (asset_root), so only the offsets change
    if args.camera == "behind":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.0, 0.6, 0.3), (0.0, -0.8, 0.0)
    elif args.camera == "high":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.0, 3.0, 2.0), (0.0, -1.2, 0.0)
if args.cam_eye is not None:
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = tuple(args.cam_eye)
    env_cfg.viewer.lookat = tuple(args.cam_lookat) if args.cam_lookat is not None else (0.0, 0.0, 0.0)
print(f"[record_video] viewer {env_cfg.viewer.origin_type} eye {env_cfg.viewer.eye} lookat {env_cfg.viewer.lookat}")
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
if raw._has_terrain_levels and args.spread:
    # every column (type) in turn, rows from --levels (default 0 3 6 9): 40 envs cover 10 columns x 4 levels
    n_rows, n_cols = raw._terrain.terrain_origins.shape[:2]
    rows = torch.tensor(args.levels if args.levels and args.levels != [0] else [0, 3, 6, 9], device=dev)
    raw.set_terrain_cells(ids, rows[(ids // n_cols) % len(rows)], ids % n_cols)
elif raw._has_terrain_levels and args.levels:
    n_cols = raw._terrain.terrain_origins.shape[1]
    levels = torch.tensor(args.levels, device=dev)[ids % len(args.levels)]
    types = torch.full_like(ids, args.column)  # one column for every env (pick the higher column of a type pair)
    raw.set_terrain_cells(ids, levels, types)
    if args.lateral_spread > 0 and N > 1:
        raw._terrain.env_origins[:, 0] += torch.linspace(-args.lateral_spread, args.lateral_spread, N, device=dev)
        # off the centre platform the ground can be higher (pits, inverted slopes): spawn a little higher and let the
        # robots drop onto the terrain instead of starting inside it
        raw._terrain.env_origins[:, 2] += 0.06
obs, _ = env.reset()
raw._commands[:, 0], raw._commands[:, 1], raw._commands[:, 2] = 0.0, -args.cmd, 0.0
obs = raw._get_observations()["policy"]
falls = 0
speed_sum, speed_n = 0.0, 0
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
    speed_sum += (-raw._robot.data.root_lin_vel_b.torch[:, 1]).mean().item()
    speed_n += 1
    raw._commands[:, 0], raw._commands[:, 1], raw._commands[:, 2] = 0.0, -args.cmd, 0.0
print(f"[record_video] {args.steps} steps, {N} envs, cmd {args.cmd}, falls {falls}, mean speed along cmd {speed_sum / max(speed_n, 1):.3f}, final speed {(-raw._robot.data.root_lin_vel_b.torch[:, 1]).mean().item():.3f}")
def _exit_watchdog(seconds: float = 120.0):
    """Kit sometimes never returns from env.close()/app.close() (measured: finished evals lingering for hours at 110%
    CPU, holding a GPU slot). Force the process to exit after the results are on disk."""
    import os, threading, time

    def _kill():
        time.sleep(seconds)
        os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()

_exit_watchdog()
env.close()
print("[record_video] wrote", os.listdir(args.out))
app.close()
