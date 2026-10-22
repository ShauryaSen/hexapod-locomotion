# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does the friction setter act in the deployment env? Half the robots get --mu (default 0.05) via
env.set_friction, the other half stay nominal; all walk at the deployment command on terrain level 0 (near-flat)
with the deterministic pi_0. Reported per half:
  * the PhysX material read-back (static / dynamic friction of the robot's shapes),
  * the contact force ratio |F_tangential| / F_normal of the feet (tibia bodies) in contact, world z = normal
    (L0 slopes are small): Coulomb friction caps it at the effective mu, so its p99 must stay <= ~mu,
  * foot slip speed: horizontal speed of the tibia tip... approximated by the tibia body's horizontal speed while in
    contact (stance), mean and p90,
  * forward progress rate.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_check_friction.py --headless \
      --checkpoint <model.pt> --num_envs 16 --mu 0.05 --seconds 30
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None)
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--mu", type=float, default=0.05)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--out", default=None, help="default: <checkpoint dir>/ensemble/friction_check_<mu>.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint  # noqa: E402
from hexapod.deploy.envutil import forward_speed, make_env, pin_commands, policy_obs, reset_in_place  # noqa: E402

dcfg = C.DeployCfg(task=args.task, levels=(0,), reposition_s=15.0)
env, applied = make_env(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir)
dev, N, dt = env.device, env.num_envs, float(env.step_dt)
actor, _, info = build_from_checkpoint(args.checkpoint, N, dev)
ids = torch.arange(N, device=dev)
low = ids[: N // 2]
env.set_friction(low, args.mu, hold=True)
reset_in_place(env, ids, dcfg.cmd_speed)
mat = wp.to_torch(env._robot.root_view.get_material_properties())
readback = {"low_static": float(mat[low.cpu(), :, 0].mean()), "low_dynamic": float(mat[low.cpu(), :, 1].mean()),
            "nominal_static": float(mat[ids[N // 2 :].cpu(), :, 0].mean()), "nominal_dynamic": float(mat[ids[N // 2 :].cpu(), :, 1].mean())}
print("[friction] material read-back:", readback, flush=True)
ground_mu = float(env.cfg.terrain.physics_material.static_friction)
combine = env.cfg.terrain.physics_material.friction_combine_mode
feet = env._feet_ids
tibia_body_ids = env._robot.find_bodies(".*_tibia")[0]
obs = policy_obs(env)
ratio = {"low": [], "nominal": []}
slip = {"low": [], "nominal": []}
prog = torch.zeros(N, device=dev)
falls = torch.zeros(N, device=dev)
steps = int(args.seconds / dt)
settle = int(args.settle_s / dt)
repo = int(dcfg.reposition_s / dt)
for k in range(steps):
    with torch.no_grad():
        act, _ = actor(obs)
    o, rew, term, _, _ = env.step(act)
    obs = o["policy"]
    if bool(term.any()):
        pin_commands(env, dcfg.cmd_speed)
        obs = policy_obs(env)
    falls += term.float()
    if (k + 1) % repo == 0:
        reset_in_place(env, ids, dcfg.cmd_speed)
        obs = policy_obs(env)
        continue
    if k < settle:
        continue
    prog += forward_speed(env).clamp_max(dcfg.cmd_speed) * dt
    f = env._contact_sensor.data.net_forces_w.torch[:, feet]  # (N, 6, 3)
    fn = f[..., 2]
    ft = torch.linalg.norm(f[..., :2], dim=-1)
    contact = fn > 1.0
    v = env._robot.data.body_lin_vel_w.torch[:, tibia_body_ids, :2]  # (N, 6, 2)
    vs = torch.linalg.norm(v, dim=-1)
    for name, sel in (("low", low), ("nominal", ids[N // 2 :])):
        c = contact[sel]
        ratio[name].append((ft[sel] / fn[sel].clamp_min(1e-6))[c].cpu())
        slip[name].append(vs[sel][c].cpu())
res = {"checkpoint": args.checkpoint, "mu_set": args.mu, "ground_mu": ground_mu, "combine_mode": combine, "effective_mu_expected": args.mu * ground_mu if combine == "multiply" else None,
       "readback": readback, "seconds": args.seconds, "num_envs": N, "env_applied": applied}
for name in ("low", "nominal"):
    r = torch.cat(ratio[name]).numpy()
    s = torch.cat(slip[name]).numpy()
    sel = low if name == "low" else ids[N // 2 :]
    res[name] = {"force_ratio_p50_p90_p99_max": [float(np.percentile(r, q)) for q in (50, 90, 99)] + [float(r.max())],
                 "stance_tibia_speed_mean_p90": [float(s.mean()), float(np.percentile(s, 90))],
                 "progress_rate_m_per_min": float(prog[sel].mean() / ((steps - settle) * dt / 60.0)), "falls": float(falls[sel].sum()), "contact_samples": int(len(r))}
print(json.dumps(res, indent=1))
out = args.out or os.path.join(os.path.dirname(args.checkpoint), "ensemble", f"friction_check_{args.mu:g}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print("[friction] wrote", out)
env.close()
app.close()
