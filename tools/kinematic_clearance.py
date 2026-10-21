# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic toe lift: how high can a toe rise above its stance height within the joint limits?

The robot is held at its standing pose (root pose rewritten every step, so it cannot fall) and one leg's femur and
tibia targets are swept over a grid inside their joint limits; after settling, the toe height (tibia end-cap, 0.130 m
along tibia +x) relative to the stance toe height is recorded. Reports the max lift and the (femur, tibia) at which it
occurs, plus the lift at the gait's lift amplitudes for reference. The kinematic step cap = ASSUMED 0.8 x max lift.

Run inside the container:
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/kinematic_clearance.py --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grid", type=int, default=25, help="grid points per joint")
parser.add_argument("--settle_steps", type=int, default=25, help="env steps (50 Hz) per configuration")
parser.add_argument("--out", default="/workspace/isaaclab/logs/kinematics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402

cfg = HexapodFlatEnvCfg()
N = args.grid * args.grid
cfg.scene.num_envs = N
cfg.dr.enabled = False
cfg.spread_resets = False
cfg.episode_length_s = 1000.0
env = HexapodEnv(cfg)
dev = env.device
jn = env._robot.joint_names
leg = "lf"
i_f, i_t = jn.index(f"{leg}_femur_pitch"), jn.index(f"{leg}_tibia_pitch")
tib = env._robot.body_names.index(f"{leg}_tibia")
lim = env._robot.data.joint_pos_limits.torch[0]  # (J, 2)
f_lo, f_hi = lim[i_f].tolist()
t_lo, t_hi = lim[i_t].tolist()
print(f"[kin] femur limits {f_lo:.3f}..{f_hi:.3f}, tibia limits {t_lo:.3f}..{t_hi:.3f}, default femur {env._robot.data.default_joint_pos.torch[0, i_f]:.3f} tibia {env._robot.data.default_joint_pos.torch[0, i_t]:.3f}")
fem = torch.linspace(f_lo, f_hi, args.grid, device=dev)
tbi = torch.linspace(t_lo, t_hi, args.grid, device=dev)
F, T = torch.meshgrid(fem, tbi, indexing="ij")
targets = env._robot.data.default_joint_pos.torch.clone()
toe_off = torch.tensor([0.130, 0.0, 0.0], device=dev).expand(N, 3)


def toe_z():
    pos = env._robot.data.body_link_pos_w.torch[:, tib]
    quat = env._robot.data.body_link_quat_w.torch[:, tib]
    return (pos + quat_apply(quat, toe_off))[:, 2] - env._robot.data.root_pos_w.torch[:, 2]


env.reset()
root_pose = env._robot.data.default_root_pose.torch.clone()
root_pose[:, :3] += env._terrain.env_origins
root_vel = torch.zeros(N, 6, device=dev)
ids = torch.arange(N, device=dev, dtype=torch.int32)
# stance toe height (relative to the root) at the default pose
for _ in range(args.settle_steps):
    env._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
    env._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=ids)
    env.step(torch.zeros(N, 18, device=dev))
z_stance = toe_z().clone()
# sweep: actions are (target - default) / action_scale
targets[:, i_f] = F.reshape(-1)
targets[:, i_t] = T.reshape(-1)
act = (targets - env._robot.data.default_joint_pos.torch) / cfg.action_scale
for _ in range(args.settle_steps * 2):
    env._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
    env._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=ids)
    env.step(act)
lift = (toe_z() - z_stance).reshape(args.grid, args.grid)
q = env._robot.data.joint_pos.torch
reached = ((q[:, i_f] - F.reshape(-1)).abs() < 0.05) & ((q[:, i_t] - T.reshape(-1)).abs() < 0.05)
lift_ok = torch.where(reached.reshape(args.grid, args.grid), lift, torch.full_like(lift, float("nan")))
k = torch.nanmean(lift_ok) * 0 + torch.nan_to_num(lift_ok, nan=-1.0)
imax = int(torch.argmax(k))
fi, ti = divmod(imax, args.grid)
out = {
    "leg": leg, "femur_limits": [f_lo, f_hi], "tibia_limits": [t_lo, t_hi],
    "max_lift_m": float(lift_ok[fi, ti]), "at_femur_rad": float(fem[fi]), "at_tibia_rad": float(tbi[ti]),
    "reached_frac": float(reached.float().mean()), "stance_toe_z_rel_root_m": float(z_stance.mean()),
    "kinematic_cap_m_ASSUMED_0.8x": 0.8 * float(lift_ok[fi, ti]),
    "lift_at_gait_amplitudes_m": None,
}
# the scripted gait lift (femur +0.35, tibia -0.30 from default) for reference
d = env._robot.data.default_joint_pos.torch[0]
fg, tg = float(d[i_f] + 0.35), float(d[i_t] - 0.30)
fi_g = int(torch.argmin((fem - fg).abs())); ti_g = int(torch.argmin((tbi - tg).abs()))
out["lift_at_gait_amplitudes_m"] = float(lift_ok[fi_g, ti_g])
print(json.dumps(out, indent=1))
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "kinematic_clearance.json"), "w"), indent=1)
torch.save({"femur": fem.cpu(), "tibia": tbi.cpu(), "lift": lift_ok.cpu()}, os.path.join(args.out, "kinematic_lift_grid.pt"))
env.close()
app.close()
