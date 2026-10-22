# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Still images (PNG) of the training terrain with robots on it, from several cameras, in one env instance.

One robot per terrain cell (env i -> level i // cols, column i % cols), a checkpoint walking at a fixed forward
command for --settle_s seconds, then the Kit perspective camera is moved to each shot and one frame is saved.
The terrain is rebuilt from the run's params (terrain_version, step_cap, tread, platform, tile size).

Run inside the container:
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/terrain_stills.py --headless \
      --enable_cameras --checkpoint <model.pt> --out /workspace/isaaclab/logs/stills
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cmd", type=float, default=0.28)
parser.add_argument("--settle_s", type=float, default=1.2)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--out", required=True)
parser.add_argument("--prefix", default="terrain_v21")
parser.add_argument("--rows", type=int, nargs="+", default=[0, 3, 5, 9], help="levels that get a robot on every column (camera rendering is killed for memory above ~40-60 robots on the terrain mesh)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

import hexapod  # noqa: F401, E402
from hexapod.hexapod_terrains_cfg import make_hexapod_terrains_cfg, terrain_type_names  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

task = "Hexapod-Rough-Direct-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
params = yaml.load(open(os.path.join(os.path.dirname(args.checkpoint), "params", "env.yaml")), Loader=yaml.UnsafeLoader)
for k in ("action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "perceptive"):
    if k in params:
        setattr(env_cfg, k, params[k])
kw = {"version": params.get("terrain_version", 1)}
for src, dst in (("step_cap", "step_cap"), ("terrain_stair_tread", "tread"), ("terrain_stair_platform", "stair_platform"), ("terrain_tile_size", "tile_size")):
    if params.get(src) is not None:
        kw[dst] = params[src]
gen = make_hexapod_terrains_cfg(**kw)
env_cfg.terrain.terrain_generator = gen
env_cfg.step_cap, env_cfg.terrain_version = None, 1
env_cfg.terrain_stair_tread = env_cfg.terrain_stair_platform = None
if hasattr(env_cfg, "terrain_tile_size"):
    env_cfg.terrain_tile_size = None
env_cfg.terrain.max_init_terrain_level = None
rows, cols = gen.num_rows, gen.num_cols
env_cfg.scene.num_envs = len(args.rows) * cols
env_cfg.spread_resets = False
env_cfg.terrain_curriculum = False
env_cfg.command_curriculum = False
env_cfg.dr.enabled = False
env_cfg.command_forward_only = True
env_cfg.command_lin_vel_min = env_cfg.command_lin_vel_max = args.cmd
env_cfg.command_backward_fraction = 0.0
env_cfg.command_yaw_rate_max = 0.0
env_cfg.video_recorder.window_width = args.width
env_cfg.video_recorder.window_height = args.height
env_cfg.viewer.origin_type = "world"
env_cfg.viewer.eye, env_cfg.viewer.lookat = (20.0, -18.0, 14.0), (0.0, 0.0, 0.0)
print(f"[stills] terrain {kw}: {rows} levels x {cols} columns {terrain_type_names(gen)}")
env = gym.make(task, cfg=env_cfg, render_mode="rgb_array").unwrapped
dev = env.device
N = env.num_envs
ids = torch.arange(N, device=dev)
row_t = torch.tensor(args.rows, device=dev)
env.set_terrain_cells(ids, row_t[ids // cols], ids % cols)
print(f"[stills] env created: {N} robots on levels {args.rows} x {cols} columns", flush=True)

ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
sd = ck["actor_state_dict"]
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

obs = env.reset()[0]["policy"]
print("[stills] reset done, walking", flush=True)
for _ in range(int(args.settle_s / env.step_dt)):
    env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -args.cmd, 0.0
    with torch.no_grad():
        obs = env.step(actor(obs))[0]["policy"]

origins = env._terrain.terrain_origins.cpu().numpy()  # (rows, cols, 3)
print("[stills] walking done, rendering shots", flush=True)
names = terrain_type_names(gen)
col_of = {n: max(i for i, m in enumerate(names) if m == n) for n in set(names)}
L5 = min(5, rows - 1)
cx = float(origins[L5, :, 0].mean())
shots = [
    ("overview_wide", (20.0, -18.0, 14.0), (0.0, 0.0, -1.0)),
    ("row_L5_side", (cx + 14.0, 0.0, 6.0), (cx, 0.0, -0.5)),
    ("row_L5_along", (cx + 2.5, float(origins[L5, -1, 1]) + 2.0, 2.2), (cx, float(origins[L5, 0, 1]) + 4.0, 0.0)),
]
for t in ("stairs", "stairs_inv", "boxes"):
    for lvl in (5, 9):
        lvl = min(lvl, rows - 1)
        o = origins[lvl, col_of[t]]
        shots.append((f"{t}_L{lvl}", (o[0] + 1.1, o[1] + 1.0, o[2] + 0.75), (o[0], o[1] - 0.35, o[2])))

from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402

os.makedirs(args.out, exist_ok=True)
for name, eye, target in shots:
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[float(x) for x in eye], target=[float(x) for x in target])
    frame = None
    for _ in range(8):  # let the renderer converge after the camera jump
        frame = env.render()
    if frame is None or np.asarray(frame).size == 0:
        print(f"[stills] {name}: no frame")
        continue
    img = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
    path = os.path.join(args.out, f"{args.prefix}_{name}.png")
    img.save(path)
    print(f"[stills] wrote {path} {img.size} eye {tuple(round(float(x), 2) for x in eye)} target {tuple(round(float(x), 2) for x in target)}")


def _exit_watchdog(seconds: float = 120.0):
    import os as _os, threading, time

    def _kill():
        time.sleep(seconds)
        _os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()


_exit_watchdog()
env.close()
app.close()
