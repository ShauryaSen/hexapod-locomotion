# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop teacher grid on flat ground (speed sweep part A) and swing-clearance measurement (step-height cap).

Rolls the scripted tripod open loop (joint targets straight from the gait, no policy) over a grid of
hip amplitude cap x gait period, lift fixed, on the nominal robot (no DR), and records per setting:

* body speed along the command [m/s] (after a settle time; envs that fell are excluded)
* falls: fraction of envs whose body touched the ground at least once
* torque saturation: fraction of (step, joint) samples with |torque| > 95% of the cap
* swing clearance [m]: toe height above its stance height, max per gait cycle and leg, median over cycles, legs and
  envs. The toe is the far end-cap of the tibia capsule, 0.130 m along the tibia link's +x axis
  (assets/usd/robot_simple.usda: translate 0.0533 + height/2 0.0591 + radius 0.0176).

Selection rule (brief item 6): the setting with the highest speed among fall-free settings (falls <= --max_falls).
Step-height cap (decisions.md P1/P2): 0.8 x median clearance of the selected setting.

Every setting runs its own phase clock with its own period, so the whole grid runs in one batch of envs.

Run inside the container (from the Isaac Lab root):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/measure_clearance.py --headless \
      --envs_per_setting 16 --out /workspace/isaaclab/logs/speed_sweep/teacher_grid
"""

import argparse
import itertools
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hip_max", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5], help="coxa amplitude caps [rad]")
parser.add_argument("--period", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.8], help="gait periods [s]")
parser.add_argument("--lift_femur", type=float, default=0.35)
parser.add_argument("--lift_tibia", type=float, default=-0.30)
parser.add_argument("--envs_per_setting", type=int, default=16)
parser.add_argument("--duration", type=float, default=10.0, help="seconds per rollout")
parser.add_argument("--settle", type=float, default=2.0, help="seconds ignored at the start of each rollout")
parser.add_argument("--max_falls", type=float, default=0.0, help="largest fall fraction still counted as fall-free")
parser.add_argument("--out", default="/workspace/isaaclab/logs/speed_sweep/teacher_grid")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402
from hexapod.scripted_tripod import ScriptedTripod  # noqa: E402

TOE_OFFSET_TIBIA = (0.130, 0.0, 0.0)  # [m] toe tip in the tibia link frame (see module docstring)

settings = list(itertools.product(args.hip_max, args.period))
n_set = len(settings)
num_envs = n_set * args.envs_per_setting

cfg = HexapodFlatEnvCfg()
cfg.scene.num_envs = num_envs
cfg.command_forward_only, cfg.command_lin_vel_min, cfg.command_lin_vel_max, cfg.command_yaw_rate_max = True, 0.1, 1.0, 0.0
cfg.gait_clock_enabled = True
cfg.dr.enabled = False
cfg.spread_resets = False
cfg.episode_length_s = args.duration + 5.0  # no time-outs inside the measurement window
env = HexapodEnv(cfg)
dev = env.device
teacher = ScriptedTripod(env, hip_max=max(args.hip_max), lift_femur=args.lift_femur, lift_tibia=args.lift_tibia)

setting_of_env = torch.arange(num_envs, device=dev) // args.envs_per_setting
hip_of_env = torch.tensor([s[0] for s in settings], device=dev)[setting_of_env]
period_of_env = torch.tensor([s[1] for s in settings], device=dev)[setting_of_env]

legs = ("lf", "lm", "lr", "rf", "rm", "rr")
tibia_ids = [env._robot.body_names.index(f"{l}_tibia") for l in legs]
toe_offset = torch.tensor(TOE_OFFSET_TIBIA, device=dev).expand(num_envs, len(legs), 3)


def toe_height() -> torch.Tensor:
    pos = env._robot.data.body_link_pos_w.torch[:, tibia_ids]  # (N, 6, 3)
    quat = env._robot.data.body_link_quat_w.torch[:, tibia_ids]  # (N, 6, 4) xyzw
    return (pos + quat_apply(quat.reshape(-1, 4), toe_offset.reshape(-1, 3)).reshape(num_envs, len(legs), 3))[..., 2]


obs, _ = env.reset()
# forward command (-y is the club's forward axis); the amplitude is pinned to hip_max per env, not command-derived
env._commands[:, 0] = 0.0
env._commands[:, 1] = -0.5
env._commands[:, 2] = 0.0
phase = torch.zeros(num_envs, device=dev)
steps = int(args.duration / env.step_dt)
settle_steps = int(args.settle / env.step_dt)
fell = torch.zeros(num_envs, dtype=torch.bool, device=dev)
v_sum = torch.zeros(num_envs, device=dev)
v_n = 0
sat_sum = torch.zeros(num_envs, device=dev)
toe_log = []  # per step (N, 6)
cycle_of_step = []  # per step (N,) cycle index
cap = env._actuator.effort_limit
for i in range(steps):
    phase = torch.remainder(phase + 2 * math.pi * env.step_dt / period_of_env, 2 * math.pi)
    # the env clock is not used; targets come straight from the per-env phase
    a = teacher.delta(env, phase, hip=hip_of_env) / cfg.action_scale
    obs, _, died, _, _ = env.step(a)
    fell |= died
    if i >= settle_steps:
        v_sum += -env._robot.data.root_lin_vel_b.torch[:, 1]  # speed along -y
        v_n += 1
        sat_sum += (env._robot.data.applied_torque.torch.abs() > 0.95 * cap).float().mean(dim=1)
        toe_log.append(toe_height().clone())
        cycle_of_step.append(((i - settle_steps) * env.step_dt / period_of_env).floor().long())

toe = torch.stack(toe_log)  # (T, N, 6)
cyc = torch.stack(cycle_of_step)  # (T, N)
rows = []
for k, (hip, period) in enumerate(settings):
    ids = (setting_of_env == k).nonzero().squeeze(1)
    ok = ids[~fell[ids]]
    n_cycles = int(cyc[-1, ids[0]].item())  # complete cycles in the window (last partial one dropped)
    clearances = []
    for e in ok.tolist():
        for c in range(n_cycles):
            m = cyc[:, e] == c
            z = toe[m, e]  # (steps in cycle, 6)
            clearances.append((z.max(dim=0).values - z.min(dim=0).values))  # per leg
    if clearances:
        cl = torch.cat(clearances)
        clearance_med, clearance_p10, clearance_p90 = (cl.median().item(), cl.quantile(0.1).item(), cl.quantile(0.9).item())
    else:
        clearance_med = clearance_p10 = clearance_p90 = float("nan")
    rows.append(
        {
            "hip_max": hip,
            "period_s": period,
            "speed_mps": (v_sum[ok] / max(v_n, 1)).mean().item() if len(ok) else float("nan"),
            "falls_frac": fell[ids].float().mean().item(),
            "torque_sat": (sat_sum[ids] / max(v_n, 1)).mean().item(),
            "clearance_med_m": clearance_med,
            "clearance_p10_m": clearance_p10,
            "clearance_p90_m": clearance_p90,
            "n_envs": len(ids),
            "n_cycles": n_cycles,
        }
    )

fall_free = [r for r in rows if r["falls_frac"] <= args.max_falls and not math.isnan(r["speed_mps"])]
selected = max(fall_free, key=lambda r: r["speed_mps"]) if fall_free else None
step_cap = 0.8 * selected["clearance_med_m"] if selected else float("nan")

os.makedirs(args.out, exist_ok=True)
md = ["| hip_max [rad] | period [s] | speed [m/s] | falls | torque sat | clearance med [m] | p10 | p90 |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    mark = " **selected**" if r is selected else ""
    md.append(
        f"| {r['hip_max']:.2f} | {r['period_s']:.2f} | {r['speed_mps']:.3f} | {r['falls_frac']:.2f} | {r['torque_sat']:.3f} |"
        f" {r['clearance_med_m']:.4f} | {r['clearance_p10_m']:.4f} | {r['clearance_p90_m']:.4f} |{mark}"
    )
md.append("")
md.append(f"Selected (highest speed with falls <= {args.max_falls}): {selected}")
md.append(f"STEP_CAP = 0.8 x median clearance = {step_cap:.4f} m")
md.append(f"lift_femur={args.lift_femur}, lift_tibia={args.lift_tibia}, duration={args.duration}s, settle={args.settle}s, envs/setting={args.envs_per_setting}")
text = "\n".join(md)
print(text)
open(os.path.join(args.out, "teacher_grid.md"), "w").write(text + "\n")
json.dump({"rows": rows, "selected": selected, "step_cap_m": step_cap, "args": vars(args)}, open(os.path.join(args.out, "teacher_grid.json"), "w"), indent=1, default=str)
print("saved", args.out)
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
app.close()
