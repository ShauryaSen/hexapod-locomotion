# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait metrics per leg: is the policy walking a tripod on the gait clock, and what are the front legs doing?

Clock convention (scripted_tripod.ScriptedTripod, the BC teacher the stage-1 policies descend from): legs
(lf, lm, lr, rf, rm, rr) have phase offsets (0, pi, 0, pi, 0, pi); a leg is in stance when
((phase + offset) / 2pi mod 1) < 0.5. Tripod A = lf, rm, lr (L1, R2, L3); tripod B = rf, lm, rr (R1, L2, R3).

Per cell (terrain level x type, or "flat" on the flat task) and per leg, over the steps after --settle_s while the
robot is still up (a robot that falls stops counting at the fall):
  * clock_match: fraction of steps where the leg's contact state (tibia contact force > 1 N) equals the clock's
    tripod stance/swing for that leg;
  * shift_match / shift_deg: the same with one common phase shift for all six legs, the shift chosen to maximise
    the six-leg mean (tripod coordination regardless of how the policy aligned itself with the clock);
  * rhythm, lag_deg: first Fourier component of the +-1 contact signal at the clock frequency: amplitude (0.64 for
    a clean square wave locked to the clock, ~0 for no rhythm at that frequency) and phase lag against the
    leg's expected stance centre (0 = on the tripod schedule, 180 = in the other tripod's phase);
  * duty: stance fraction; touchdowns_hz: contact onsets per second (the clock gives 1 / gait_period_s);
  * reach_m: forward-most toe position in the body frame (forward = -y), mean over gait-period windows;
    stride_m: fore-aft toe range per window; swing_h_m: toe height range in the body frame per window.

Outputs next to the checkpoint: eval/<ckpt>_gait_<tag>.json and .md.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/gait_metrics.py --headless \
      --task Hexapod-Rough-Direct-v0 --checkpoint <model.pt> --levels 5 --cmd 0.28 --robots 16 --tag L5
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None, help="params/env.yaml source; default: the checkpoint's directory")
parser.add_argument("--levels", type=int, nargs="+", default=[5])
parser.add_argument("--types", nargs="+", default=None)
parser.add_argument("--cmd", type=float, default=0.28)
parser.add_argument("--robots", type=int, default=16, help="robots per cell")
parser.add_argument("--episode_s", type=float, default=12.0)
parser.add_argument("--settle_s", type=float, default=2.0)
parser.add_argument("--start_jitter", type=float, nargs=2, default=(0.05, 10.0), metavar=("XY_M", "YAW_DEG"))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--env_override", nargs="*", default=[])
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.manual_seed(args.seed)

import hexapod  # noqa: F401, E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
OFFSETS = (0.0, math.pi, 0.0, math.pi, 0.0, math.pi)  # ScriptedTripod.offsets
PARAM_KEYS = (
    "action_scale", "gait_clock_enabled", "gait_period_s", "command_forward_only", "command_lin_vel_min",
    "command_lin_vel_max", "command_yaw_rate_max", "perceptive", "step_cap", "terrain_version",
    "terrain_stair_tread", "terrain_stair_platform", "terrain_tile_size",
)

run_dir = args.run_dir or os.path.dirname(args.checkpoint)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
applied = {}
params_path = os.path.join(run_dir, "params", "env.yaml")
if os.path.exists(params_path):
    params = yaml.load(open(params_path), Loader=yaml.UnsafeLoader)
    for k in PARAM_KEYS:
        if k in params and isinstance(params[k], (bool, int, float, str)) and hasattr(env_cfg, k):
            setattr(env_cfg, k, params[k])
            applied[k] = params[k]
for ov in args.env_override:
    k, v = ov.split("=", 1)
    v = {"true": True, "false": False}.get(v.lower(), v)
    for cast in (int, float):
        try:
            v = cast(v)
            break
        except (ValueError, TypeError):
            pass
    setattr(env_cfg, k, v)
    applied[k] = v
assert env_cfg.gait_clock_enabled, "gait metrics are defined against the gait clock"
env_cfg.spread_resets = False
env_cfg.terrain_curriculum = False
env_cfg.command_curriculum = False
env_cfg.dr.enabled = False
env_cfg.episode_length_s = args.episode_s + 5.0
if getattr(env_cfg, "terrain", None) is not None and env_cfg.terrain.terrain_type == "generator":
    env_cfg.terrain.max_init_terrain_level = None

# one env per (cell, robot); Kit cannot build a second env in the same process, so the cell count is fixed before
# the env exists: v2.1 has 6 terrain types (checked against the env's own names below)
TYPES = ("random_rough", "slope", "slope_inv", "stairs", "stairs_inv", "boxes")
rough = "Rough" in args.task
n_cells = len(args.levels) * len(args.types or TYPES) if rough else 1
env_cfg.scene.num_envs = n_cells * args.robots
env = gym.make(args.task, cfg=env_cfg).unwrapped
dev, n = env.device, env.num_envs
if env._has_terrain_levels:
    names = env.terrain_type_names
    cols = {}
    for c in range(env._terrain.terrain_origins.shape[1]):
        cols.setdefault(names[c], []).append(c)
    assert set(cols) == set(TYPES), f"terrain types {sorted(cols)} != {TYPES}"
    cells = [(lvl, t, max(cols[t])) for lvl in args.levels for t in (args.types or TYPES)]
else:
    cells = [(0, "flat", 0)]
assert len(cells) == n_cells
print(f"[gait] {len(cells)} cells x {args.robots} robots, settings {applied}", flush=True)

from hexapod.rma import load_policy  # noqa: E402

actor = load_policy(args.checkpoint, dev)

# leg order of the contact sensor vs LEGS
sensor_names = [env._contact_sensor.body_names[i] for i in env._feet_ids]
feet_by_leg = [env._feet_ids[sensor_names.index(f"{l}_tibia")] for l in LEGS]
print("[gait] contact sensor tibia order:", sensor_names, "-> LEGS order ids", feet_by_leg, flush=True)
tibia_ids = [env._robot.body_names.index(f"{l}_tibia") for l in LEGS]
toe_off = torch.tensor([0.130, 0.0, 0.0], device=dev)

ids = torch.arange(n, device=dev)
cell_of = ids // args.robots
if env._has_terrain_levels:
    env.set_terrain_cells(ids, torch.tensor([cells[i][0] for i in cell_of.tolist()], device=dev), torch.tensor([cells[i][2] for i in cell_of.tolist()], device=dev))
env.reset()
env._commands[:] = 0.0
env._commands[:, 1] = -args.cmd
# start jitter (same as eval_suite --start_jitter)
pose = env._robot.data.root_pose_w.torch.clone()
pose[:, 0:2] += (torch.rand(n, 2, device=dev) * 2 - 1) * args.start_jitter[0]
yaw = (torch.rand(n, device=dev) * 2 - 1) * math.radians(args.start_jitter[1])
z0 = torch.zeros_like(yaw)
pose[:, 3:7] = quat_mul(quat_from_euler_xyz(z0, z0, yaw), pose[:, 3:7])
env._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids.to(torch.int32))
env._gait_phase[:] = torch.rand(n, device=dev) * 2 * math.pi
env.episode_length_buf[:] = 0
obs = env._get_observations()

steps = int(args.episode_s / env.step_dt)
settle = int(args.settle_s / env.step_dt)
offsets = torch.tensor(OFFSETS, device=dev)
alive = torch.ones(n, dtype=torch.bool, device=dev)
C, PH, TOE, VALID = [], [], [], []
for i in range(steps):
    with torch.no_grad():
        act = actor(obs)
    obs, _, term, trunc, _ = env.step(act)
    alive &= ~(term | trunc)
    env._commands[:, 0] = 0.0
    env._commands[:, 1] = -args.cmd
    env._commands[:, 2] = 0.0
    if i < settle:
        continue
    f = torch.linalg.norm(env._contact_sensor.data.net_forces_w.torch[:, feet_by_leg], dim=-1)
    C.append(f > 1.0)
    PH.append(env._gait_phase.clone())
    pos = env._robot.data.body_link_pos_w.torch[:, tibia_ids]
    q = env._robot.data.body_link_quat_w.torch[:, tibia_ids]
    toe_w = pos + quat_apply(q.reshape(-1, 4), toe_off.expand(n * 6, 3)).reshape(n, 6, 3)
    rq = env._robot.data.root_quat_w.torch
    rp = env._robot.data.root_pos_w.torch
    toe_b = quat_apply_inverse(rq.repeat_interleave(6, 0), (toe_w - rp[:, None]).reshape(-1, 3)).reshape(n, 6, 3)
    TOE.append(toe_b)
    VALID.append(alive.clone())

C = torch.stack(C).float()  # (T, n, 6)
PH = torch.stack(PH)  # (T, n)
TOE = torch.stack(TOE)  # (T, n, 6, 3)
V = torch.stack(VALID).float()  # (T, n)
T = C.shape[0]
leg_ph = torch.remainder((PH[..., None] + offsets) / (2 * math.pi), 1.0)  # (T, n, 6)
exp_stance = (leg_ph < 0.5).float()
w = V[..., None]
denom = w.sum(0).clamp_min(1.0)  # (n, 1)

clock_match = ((C == exp_stance).float() * w).sum(0) / denom  # (n, 6)
duty = (C * w).sum(0) / denom
onsets = ((C[1:] > C[:-1]).float() * w[1:]).sum(0) / (denom * env.step_dt)
# common phase shift maximising the six-leg match
shifts = torch.linspace(0, 2 * math.pi, 37, device=dev)[:-1]
best = torch.full((n,), -1.0, device=dev)
best_shift = torch.zeros(n, device=dev)
best_leg = torch.zeros(n, 6, device=dev)
for s in shifts:
    e = (torch.remainder((PH[..., None] + offsets + s) / (2 * math.pi), 1.0) < 0.5).float()
    m_leg = ((C == e).float() * w).sum(0) / denom
    m = m_leg.mean(1)
    upd = m > best
    best = torch.where(upd, m, best)
    best_shift = torch.where(upd, s.expand(n), best_shift)
    best_leg = torch.where(upd[:, None], m_leg, best_leg)
# Fourier component of the +-1 contact signal at the clock frequency; expected stance centre at leg phase 0.25
sig = (2 * C - 1) * w
ang = 2 * math.pi * leg_ph - math.pi / 2  # 0 at the expected stance centre
z = (sig * torch.exp(torch.complex(torch.zeros_like(ang), -ang))).sum(0) / denom  # (n, 6) complex
rhythm = z.abs()
lag_deg = torch.rad2deg(torch.angle(z))
# toe geometry per gait-period window (only fully valid windows)
win = max(int(env_cfg.gait_period_s / env.step_dt), 2)
nw = T // win
tw = TOE[: nw * win].reshape(nw, win, n, 6, 3)
vw = V[: nw * win].reshape(nw, win, n).min(1).values  # (nw, n)
fwd = -tw[..., 1]
reach = fwd.max(1).values
stride = fwd.max(1).values - fwd.min(1).values
swing_h = tw[..., 2].max(1).values - tw[..., 2].min(1).values
vwn = vw.sum(0).clamp_min(1.0)[:, None]


def wmean(x):  # (nw, n, 6) -> (n, 6)
    return (x * vw[..., None]).sum(0) / vwn


reach, stride, swing_h = wmean(reach), wmean(stride), wmean(swing_h)

metrics = {"clock_match": clock_match, "shift_match": best_leg, "rhythm": rhythm, "lag_deg": lag_deg, "duty": duty,
           "touchdowns_hz": onsets, "reach_m": reach, "stride_m": stride, "swing_h_m": swing_h}
out_cells = []
for ci, (lvl, t, col) in enumerate(cells):
    sel = (cell_of == ci) & (V.sum(0) > 0)
    row = {"level": lvl, "type": t, "column": col, "robots": int(sel.sum()), "fell": int((~alive[cell_of == ci]).sum()),
           "six_leg_shift_match": best[sel].mean().item(), "shift_deg_median": math.degrees(best_shift[sel].median().item())}
    for k, v in metrics.items():
        if k == "lag_deg":  # circular mean over robots
            zz = z[sel]
            row[k] = {l: math.degrees(torch.angle(zz[:, j].mean()).item()) for j, l in enumerate(LEGS)}
        else:
            row[k] = {l: v[sel, j].mean().item() for j, l in enumerate(LEGS)}
    out_cells.append(row)


def grp(row, k, legs):
    return sum(row[k][l] for l in legs) / len(legs)


FRONT, MID, REAR = ("lf", "rf"), ("lm", "rm"), ("lr", "rr")
md = [f"# gait metrics {os.path.basename(args.checkpoint)}", f"task {args.task}, cmd {args.cmd} m/s, {args.robots} robots/cell,"
      f" {args.episode_s} s ({args.settle_s} s settle dropped), jitter {tuple(args.start_jitter)}, seed {args.seed}, clock period"
      f" {env_cfg.gait_period_s} s", f"settings {applied}", "",
      "## front / middle / rear legs (mean of left and right)", "",
      "| cell | fell | clock match F/M/R | 6-leg shift match | rhythm F/M/R | lag deg F/M/R | duty F/M/R | touchdowns Hz F/M/R | reach m F/M/R | swing h m F/M/R |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in out_cells:
    def fmr(k, fmt):
        return "/".join(fmt.format(grp(r, k, g)) for g in (FRONT, MID, REAR))
    md.append(f"| L{r['level']} {r['type']} | {r['fell']}/{r['robots']} | {fmr('clock_match', '{:.2f}')} | {r['six_leg_shift_match']:.2f} @ {r['shift_deg_median']:.0f} deg"
              f" | {fmr('rhythm', '{:.2f}')} | {fmr('lag_deg', '{:+.0f}')} | {fmr('duty', '{:.2f}')} | {fmr('touchdowns_hz', '{:.2f}')}"
              f" | {fmr('reach_m', '{:.3f}')} | {fmr('swing_h_m', '{:.3f}')} |")
md += ["", "## per leg", "", "| cell | metric | " + " | ".join(LEGS) + " |", "|---|---|" + "---|" * 6]
for r in out_cells:
    for k in metrics:
        md.append(f"| L{r['level']} {r['type']} | {k} | " + " | ".join(f"{r[k][l]:.3f}" if k != "lag_deg" else f"{r[k][l]:+.0f}" for l in LEGS) + " |")
text = "\n".join(md)
print(text, flush=True)
out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(args.checkpoint))[0] + "_gait" + (f"_{args.tag}" if args.tag else "")
open(os.path.join(out_dir, base + ".md"), "w").write(text)
json.dump({"checkpoint": args.checkpoint, "task": args.task, "cmd": args.cmd, "applied": applied, "sensor_order": sensor_names,
           "cells": out_cells}, open(os.path.join(out_dir, base + ".json"), "w"), indent=1)
print("[gait] wrote", os.path.join(out_dir, base + ".md"), flush=True)


def _exit_watchdog(seconds: float = 120.0):
    import threading
    import time

    def _kill():
        time.sleep(seconds)
        os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()


_exit_watchdog()
env.close()
app.close()
