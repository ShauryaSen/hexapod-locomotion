# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that the runtime DR setters take effect mid-episode (brief item 5).

Rolls the scripted tripod open loop on flat ground. Half the envs get a change 3 s in; the other half stay nominal.
  * set_friction(0.15): read back the PhysX material buffer; the body speed of the changed envs should drop
    (feet slip) relative to the nominal half over the following seconds.
  * set_payload(+2.5 kg): read back body_mass; the standing height should drop.
  * set_motor_strength(0.5): read back joint_stiffness / effort limits; the applied-torque magnitude and tracking
    error should change.
Prints before/after numbers for each; exits non-zero if a read-back does not match.

Run inside the container (from the Isaac Lab root):
  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/test_setters.py --headless --num_envs 32
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import warp as wp  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402
from hexapod.scripted_tripod import ScriptedTripod  # noqa: E402

cfg = HexapodFlatEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.command_forward_only, cfg.command_lin_vel_min, cfg.command_lin_vel_max, cfg.command_yaw_rate_max = True, 0.1, 0.15, 0.0
cfg.gait_clock_enabled = True
cfg.dr.enabled = False
cfg.spread_resets = False
env = HexapodEnv(cfg)
dev = env.device
teacher = ScriptedTripod(env)
dphi = 2 * math.pi * env.step_dt / cfg.gait_period_s
N = env.num_envs
half = torch.arange(N // 2, device=dev)
rest = torch.arange(N // 2, N, device=dev)
failures = []


def run(seconds: float):
    v, z, tau, err, n = 0.0, 0.0, 0.0, 0.0, 0
    for _ in range(int(seconds / env.step_dt)):
        a = teacher.delta(env, env._gait_phase + dphi) / cfg.action_scale
        env.step(a)
        v = v + (-env._robot.data.root_lin_vel_b.torch[:, 1])
        z = z + env._robot.data.root_pos_w.torch[:, 2]
        tau = tau + env._robot.data.applied_torque.torch.abs().mean(dim=1)
        err = err + (env._robot.data.joint_pos.torch - env._processed_actions).abs().mean(dim=1)
        n += 1
    return v / n, z / n, tau / n, err / n


def report(name, before, after, idx_changed, idx_nominal):
    b_c, b_n = before[idx_changed].mean().item(), before[idx_nominal].mean().item()
    a_c, a_n = after[idx_changed].mean().item(), after[idx_nominal].mean().item()
    print(f"  {name}: changed {b_c:.4f} -> {a_c:.4f} | nominal {b_n:.4f} -> {a_n:.4f}")


def episode(label, setter, readback):
    print(f"\n== {label}")
    env.reset()
    env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -0.15, 0.0
    before = run(3.0)
    setter()
    env._commands[:, 0], env._commands[:, 1], env._commands[:, 2] = 0.0, -0.15, 0.0
    ok, msg = readback()
    print(f"  read-back: {msg}")
    if not ok:
        failures.append(label)
    after = run(4.0)
    for name, b, a in zip(("speed along cmd [m/s]", "body height [m]", "|torque| mean [Nm]", "joint tracking err [rad]"), before, after):
        report(name, b, a, half, rest)
    env.set_nominal_dr(torch.arange(N, device=dev))  # setters persist across resets by design; restore for the next check


# 1. friction
def rb_friction():
    m = wp.to_torch(env._robot.root_view.get_material_properties())
    changed = m[: N // 2, :, 0]
    nominal = m[N // 2 :, :, 0]
    ok = torch.allclose(changed, torch.full_like(changed, 0.15)) and (nominal > 0.9).all().item()
    return ok, f"material static friction changed envs {changed.mean():.3f} (want 0.15), nominal {nominal.mean():.3f}"


episode("set_friction(0.15) on the first half", lambda: env.set_friction(half, 0.15), rb_friction)


# 2. payload
def rb_payload():
    m = env._robot.data.body_mass.torch[:, env._body_id].squeeze(1)
    nominal = env._nominal_body_mass.squeeze(1)
    ok = torch.allclose(m[: N // 2], nominal[: N // 2] + 2.5, atol=1e-3) and torch.allclose(m[N // 2 :], nominal[N // 2 :], atol=1e-3)
    return ok, f"body mass changed {m[: N // 2].mean():.3f} (nominal {nominal.mean():.3f} + 2.5), rest {m[N // 2 :].mean():.3f}"


episode("set_payload(+2.5 kg) on the first half", lambda: env.set_payload(half, 2.5), rb_payload)


# 3. motor strength
def rb_strength():
    kp = env._robot.data.joint_stiffness.torch
    lim = env._robot.data.joint_effort_limits.torch
    ok = torch.allclose(kp[: N // 2], env._nominal_stiffness[: N // 2] * 0.5) and torch.allclose(lim[: N // 2], env._nominal_effort_limit[: N // 2] * 0.5) and torch.allclose(kp[N // 2 :], env._nominal_stiffness[N // 2 :])
    return ok, f"stiffness changed {kp[: N // 2].mean():.3f} (want {env._nominal_stiffness.mean() * 0.5:.3f}), effort limit {lim[: N // 2].mean():.3f} (want {env._nominal_effort_limit.mean() * 0.5:.3f}), rest kp {kp[N // 2 :].mean():.3f}"


episode("set_motor_strength(0.5) on the first half", lambda: env.set_motor_strength(half, 0.5), rb_strength)

# 4. privileged observation reflects the values
z = env.privileged_obs()
print("\nprivileged obs (dmass, mu, s) after release + no resample:", z[0].tolist(), z[-1].tolist())

print("\nRESULT:", "FAIL " + str(failures) if failures else "all read-backs match")
env.close()
app.close()
sys.exit(1 if failures else 0)
