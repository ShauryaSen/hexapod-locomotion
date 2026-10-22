# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stairs doability, kinematic ceiling of the tripod gait family (open loop, no policy).

The scripted tripod is run open loop on terrain v2 stairs (pyramid up) and inverted stairs (pit) at levels
--levels, for each (gait, lift) setting in a sweep, 16 robots per cell. Each robot starts on the centre platform of
its tile (stairs: the apex; stairs_inv: the pit floor) and walks along -y into the next tile of the same type. A
robot "crosses" if it covers --cross_m along -y (default 1.75 m = from its tile centre to the far platform edge of
the next tile, i.e. it went down/up one full flight and up/down the next) without a body contact (fall).

Reports per setting x type x level: crossing rate, falls, median progress [m], and per setting x type the highest
level (riser) crossed by >= 50% of robots. Riser per level from the generator formula (code) is printed alongside.

Run inside the container:
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/stairs_openloop.py --headless
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 10)))
parser.add_argument("--robots", type=int, default=16)
parser.add_argument("--seconds", type=float, default=25.0)
parser.add_argument("--cross_m", type=float, default=1.75)
parser.add_argument("--step_cap", type=float, default=0.052)
parser.add_argument(
    "--settings", nargs="+",
    default=["0.5:0.4:0.35:-0.30", "0.5:0.4:0.70:0.00", "0.5:0.4:1.00:0.50", "0.5:0.4:1.30:1.00",
             "0.35:0.8:0.35:-0.30", "0.35:0.8:0.70:0.00", "0.35:0.8:1.00:0.50", "0.35:0.8:1.30:1.00"],
    help="hip_max:period:lift_femur:lift_tibia (lifts are deltas from the standing pose during swing) [rad, s]",
)
parser.add_argument("--out", default="/workspace/isaaclab/logs/stairs_doability")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodRoughEnvCfg  # noqa: E402
from hexapod.hexapod_terrains_cfg import make_hexapod_terrains_cfg, terrain_type_names  # noqa: E402
from hexapod.scripted_tripod import ScriptedTripod  # noqa: E402

cfg = HexapodRoughEnvCfg()
cfg.terrain.terrain_generator = make_hexapod_terrains_cfg(step_cap=args.step_cap, version=2)
cfg.terrain.max_init_terrain_level = None
names = terrain_type_names(cfg.terrain.terrain_generator)
types = {"stairs": max(i for i, n in enumerate(names) if n == "stairs"), "stairs_inv": max(i for i, n in enumerate(names) if n == "stairs_inv")}
cells = [(t, lvl) for t in types for lvl in args.levels]
N = len(cells) * args.robots
cfg.scene.num_envs = N
cfg.dr.enabled = False
cfg.spread_resets = False
cfg.terrain_curriculum = False
cfg.command_forward_only, cfg.command_lin_vel_min, cfg.command_lin_vel_max, cfg.command_yaw_rate_max = True, 0.1, 1.0, 0.0
cfg.gait_clock_enabled = True
cfg.episode_length_s = args.seconds + 10.0
env = HexapodEnv(cfg)
dev = env.device
ids = torch.arange(N, device=dev)
cell_of = ids // args.robots
lvl_t = torch.tensor([c[1] for c in cells], device=dev)[cell_of]
col_t = torch.tensor([types[c[0]] for c in cells], device=dev)[cell_of]
teacher = ScriptedTripod(env, hip_max=0.5)
g = cfg.terrain.terrain_generator
riser_mm = {l: 1000 * (0.005 + (args.step_cap - 0.005) * (l + 0.5) / g.num_rows) for l in range(g.num_rows)}
print(f"[stairs] columns {names}, using stairs col {types['stairs']}, stairs_inv col {types['stairs_inv']}; {N} robots; nominal riser per level [mm]: { {l: round(v, 1) for l, v in riser_mm.items()} }")

results = []
for spec in args.settings:
    hip, period, lf, lt = (float(x) for x in spec.split(":"))
    teacher.lift_femur, teacher.lift_tibia = lf, lt
    env.set_terrain_cells(ids, lvl_t, col_t)
    env.reset()
    env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -0.5, 0.0
    start = env._robot.data.root_pos_w.torch[:, :2].clone()
    phase = torch.zeros(N, device=dev)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    progress = torch.zeros(N, device=dev)
    for i in range(int(args.seconds / env.step_dt)):
        phase = torch.remainder(phase + 2 * math.pi * env.step_dt / period, 2 * math.pi)
        a = teacher.delta(env, phase, hip=hip) / cfg.action_scale
        _, _, died, _, _ = env.step(a)
        newly = died & ~fell
        fell |= died
        alive = ~fell
        # progress along -y, frozen once a robot has fallen (the env resets it to its origin)
        prog_now = start[:, 1] - env._robot.data.root_pos_w.torch[:, 1]
        progress = torch.where(alive, torch.maximum(progress, prog_now), progress)
        env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -0.5, 0.0
    crossed = (progress >= args.cross_m) & ~fell
    for k, (t, lvl) in enumerate(cells):
        sel = cell_of == k
        results.append({
            "hip_max": hip, "period_s": period, "lift_femur": lf, "lift_tibia": lt, "type": t, "level": lvl,
            "riser_mm_nominal": riser_mm[lvl], "crossed": crossed[sel].float().mean().item(),
            "falls": fell[sel].float().mean().item(), "progress_median_m": progress[sel].median().item(),
        })
    print(f"[stairs] setting {spec} done")

lines = ["| hip/period | lift f/t | type | " + " | ".join(f"L{l} ({riser_mm[l]:.0f} mm)" for l in args.levels) + " | highest riser crossed ≥50% |",
         "|---|---|---|" + "---|" * len(args.levels) + "---|"]
for spec in args.settings:
    hip, period, lf, lt = (float(x) for x in spec.split(":"))
    for t in types:
        row = [r for r in results if r["type"] == t and (r["hip_max"], r["period_s"], r["lift_femur"], r["lift_tibia"]) == (hip, period, lf, lt)]
        best = max([r["riser_mm_nominal"] for r in row if r["crossed"] >= 0.5], default=None)
        cellstr = " | ".join(f"{r['crossed']:.2f}/{r['falls']:.2f}/{r['progress_median_m']:.2f}" for r in row)
        lines.append(f"| {hip:.2f}/{period:.1f} | {lf:+.2f}/{lt:+.2f} | {t} | {cellstr} | {('%.0f mm' % best) if best else '-'} |")
text = "cells: crossed / falls / median progress [m]\n" + "\n".join(lines)
print(text)
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "stairs_openloop.md"), "w").write(text + "\n")
json.dump({"args": vars(args), "results": results}, open(os.path.join(args.out, "stairs_openloop.json"), "w"), indent=1)
print("[stairs] wrote", args.out)


def _exit_watchdog(seconds: float = 120.0):
    import os as _os, threading, time

    def _kill():
        time.sleep(seconds)
        _os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()


_exit_watchdog()
env.close()
app.close()
