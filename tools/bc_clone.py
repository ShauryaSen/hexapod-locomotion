# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone the club's scripted tripod gait into an rsl_rl actor, then hand it to PPO as a starting point.

Why: six PPO runs from a standing start never discovered a coordinated six-leg sweep (logs/overnight/README.md).
Starting the network already walking removes that discovery problem; PPO only has to improve tracking.

Steps:
  1. Roll the scripted tripod in HexapodEnv (gait clock in the observation) and record (observation -> teacher action).
     The executed action gets small noise so the data covers nearby states; the label is always the clean teacher.
  2. Fit an ELU MLP with rsl_rl's layer layout by mean-squared error.
  3. DAgger rounds: let the clone drive (plus noise), label every state it reaches with the teacher, refit. The teacher
     is a function of (phase, command) only, so any state can be labelled. Fixes compounding error: 0.039 -> 0.072 m/s.
  4. Verify the deterministic clone walks from a standing start.
  5. --splice_into MODEL_0: copy the clone's weights into a checkpoint written by rsl_rl (keeps its critic/optimizer
     layout), set the exploration std, clear Adam's state, save as model_bc.pt next to it.

Run (from the Isaac Lab root inside the container):
  PYTHONPATH=/workspace/learning:$PYTHONPATH ./isaaclab.sh -p /workspace/learning/hexapod/tools/bc_clone.py --headless \
      --splice_into /workspace/isaaclab/logs/rsl_rl/hexapod_flat_direct/<1-iteration run>/model_0.pt
then train with:  train.py ... --resume --load_run <that run> --checkpoint model_bc.pt  (plus the same env overrides).
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600, help="steps per collection rollout (12 s at 50 Hz)")
parser.add_argument("--noise", type=float, default=0.05, help="std of the executed-action noise during collection")
parser.add_argument("--dagger_rounds", type=int, default=2)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--std", type=float, default=0.1, help="exploration std written into the actor for PPO fine-tuning")
parser.add_argument("--cmd_min", type=float, default=0.10)
parser.add_argument("--cmd_max", type=float, default=0.15)
parser.add_argument("--out", default="/workspace/learning/logs/bc")
parser.add_argument("--splice_into", default=None, help="an rsl_rl model_*.pt; writes model_bc.pt next to it")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from hexapod.hexapod_env import HexapodEnv  # noqa: E402
from hexapod.hexapod_env_cfg import HexapodFlatEnvCfg  # noqa: E402


class ScriptedTripod:
    """The club's tripod gait as joint-position deltas [rad] from the standing pose, as a function of the env's clock.

    Tripod A (lf, rm, lr) and tripod B (lm, rf, rr) are half a cycle apart. In a leg's stance half the coxa sweeps
    +A -> -A (pushing the body); in its swing half it returns -A -> +A while femur/tibia lift with a half-sine.
    Left and right legs sweep in opposite directions. Sweep sign +1 moves the body along -y (measured in the overnight
    Phase 0 open-loop test). Hip amplitude grows with the commanded speed (0.2 rad ~ 0.09-0.10 m/s, measured).
    """

    def __init__(self, env: HexapodEnv, hip_max: float = 0.35, lift_femur: float = 0.35, lift_tibia: float = -0.30):
        legs = ("lf", "lm", "lr", "rf", "rm", "rr")
        jn = env._robot.joint_names
        self.coxa = [jn.index(f"{l}_coxa_yaw") for l in legs]
        self.femur = [jn.index(f"{l}_femur_pitch") for l in legs]
        self.tibia = [jn.index(f"{l}_tibia_pitch") for l in legs]
        dev = env.device
        self.side = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=dev).unsqueeze(0)
        self.offsets = torch.tensor([0.0, math.pi, 0.0, math.pi, 0.0, math.pi], device=dev).unsqueeze(0)
        self.hip_max, self.lift_femur, self.lift_tibia = hip_max, lift_femur, lift_tibia
        self.n_joints = len(jn)
        self.cmd_max = env.cfg.command_lin_vel_max

    def delta(self, env: HexapodEnv, phase: torch.Tensor) -> torch.Tensor:
        ph = torch.remainder((phase.unsqueeze(1) + self.offsets) / (2 * math.pi), 1.0)  # (N, 6) in [0, 1)
        stance = ph < 0.5
        s_st, s_sw = ph / 0.5, (ph - 0.5) / 0.5
        sweep = torch.where(stance, 1 - 2 * s_st, -1 + 2 * s_sw)
        lift = torch.where(stance, torch.zeros_like(ph), torch.sin(math.pi * s_sw.clamp(0, 1)))
        cmd_y = env._commands[:, 1]
        moving = (cmd_y.abs() > 0.1 * self.cmd_max).float()
        dir_sign = -torch.sign(cmd_y) * moving
        hip = (cmd_y.abs() / 0.45).clamp(max=self.hip_max)
        d = torch.zeros(env.num_envs, self.n_joints, device=env.device)
        d[:, self.coxa] = (dir_sign * hip).unsqueeze(1) * sweep * self.side
        d[:, self.femur] = moving.unsqueeze(1) * self.lift_femur * lift
        d[:, self.tibia] = moving.unsqueeze(1) * self.lift_tibia * lift
        return d


os.makedirs(args.out, exist_ok=True)
cfg = HexapodFlatEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.command_forward_only, cfg.command_lin_vel_min, cfg.command_lin_vel_max, cfg.command_yaw_rate_max = True, args.cmd_min, args.cmd_max, 0.1
cfg.gait_clock_enabled = True
env = HexapodEnv(cfg)
dev = env.device
teacher = ScriptedTripod(env)
dphi = 2 * math.pi * env.step_dt / cfg.gait_period_s


def teacher_action() -> torch.Tensor:
    # _pre_physics_step advances the clock before building joint targets, so label with the phase the env will use
    return teacher.delta(env, env._gait_phase + dphi) / cfg.action_scale


def rollout(policy, steps: int, noise: float):
    """Drive with `policy` (+noise), label every visited state with the teacher. Returns X, Y, speed along cmd, deaths."""
    obs, _ = env.reset()
    obs = obs["policy"]
    cmd = env._commands[:, :2]
    chat = cmd / torch.linalg.norm(cmd, dim=1, keepdim=True)
    xs, ys, v_sum, n, deaths = [], [], 0.0, 0, 0.0
    for i in range(steps):
        a_star = teacher_action()
        xs.append(obs.clone())
        ys.append(a_star.clone())
        with torch.no_grad():
            a = a_star if policy is None else policy(obs)
        obs, _, died, _, _ = env.step(a + noise * torch.randn_like(a))
        obs = obs["policy"]
        deaths += died.float().mean().item()
        if i >= 100:
            v_sum += (env._robot.data.root_lin_vel_b.torch[:, :2] * chat).sum(1).mean().item()
            n += 1
    return torch.cat(xs), torch.cat(ys), v_sum / max(n, 1), deaths / steps


# 1. collect from the teacher
t0 = time.time()
X, Y, spd, dth = rollout(None, args.steps, args.noise)
print(f"teacher rollout: {X.shape[0]} samples, obs dim {X.shape[1]}, speed along cmd {spd:.4f} m/s, deaths/step {dth:.4f}, {time.time()-t0:.0f}s")

# 2. fit, rsl_rl layout: Linear, ELU, Linear, ELU, Linear, ELU, Linear -> state-dict keys mlp.0/2/4/6
obs_dim, act_dim = X.shape[1], Y.shape[1]
mlp = nn.Sequential(nn.Linear(obs_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, act_dim)).to(dev)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)


def fit(X, Y, tag):
    perm = torch.randperm(X.shape[0], device=dev)
    n_val = X.shape[0] // 10
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    vl = float("nan")
    for _ in range(args.epochs):
        mlp.train()
        p = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        for st in range(0, len(p), 4096):
            idx = p[st : st + 4096]
            loss = ((mlp(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            vl = ((mlp(X[val_idx]) - Y[val_idx]) ** 2).mean().item()
    print(f"{tag}: {X.shape[0]} samples, val mse {vl:.5f} (label variance {Y.var().item():.4f})")
    return vl


vl = fit(X, Y, "BC fit")
# 3. DAgger
for r in range(args.dagger_rounds):
    Xr, Yr, spd, dth = rollout(mlp, args.steps, args.noise)
    print(f"DAgger round {r + 1}: clone (+noise) speed along cmd {spd:.4f} m/s, deaths/step {dth:.4f}")
    X, Y = torch.cat([X, Xr]), torch.cat([Y, Yr])
    vl = fit(X, Y, f"DAgger round {r + 1} refit")

# 4. verify the deterministic clone from a standing start
_, _, spd, dth = rollout(mlp, 500, 0.0)
print(f"clone, deterministic, standing start: speed along cmd {spd:.4f} m/s (commands {args.cmd_min}-{args.cmd_max}), deaths/step {dth:.4f}")

actor = {f"mlp.{k}": v.detach().cpu() for k, v in mlp.state_dict().items()}
actor["distribution.std_param"] = args.std * torch.ones(act_dim)
torch.save(actor, os.path.join(args.out, "bc_actor.pt"))
json.dump({"samples": int(X.shape[0]), "obs_dim": obs_dim, "val_mse": vl, "speed_along_cmd": spd, "deaths_per_step": dth, "std": args.std}, open(os.path.join(args.out, "bc_fit.json"), "w"), indent=1)
print("saved", os.path.join(args.out, "bc_actor.pt"))

# 5. splice into an rsl_rl checkpoint
if args.splice_into:
    ck = torch.load(args.splice_into, map_location="cpu", weights_only=False)
    a = ck["actor_state_dict"]
    assert set(a.keys()) == set(actor.keys()), set(a.keys()) ^ set(actor.keys())
    for k in a:
        assert a[k].shape == actor[k].shape, (k, a[k].shape, actor[k].shape)
        a[k] = actor[k].to(a[k].dtype)
    ck["optimizer_state_dict"]["state"] = {}  # Adam moments belong to the old weights; start fresh
    ck["iter"] = 0
    out = os.path.join(os.path.dirname(args.splice_into), "model_bc.pt")
    torch.save(ck, out)
    print("spliced into", out)

env.close()
app.close()
