# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the dynamics ensemble (decisions.md F5) from a pi_0 checkpoint: nominal rollouts of the deterministic
policy at the deployment command across the DR training ranges (friction / mass / motor strength resampled at
every repositioning), L0-L5, fall and teleport transitions excluded. Then a fresh calibration rollout (new DR
draws) gives the nominal per-gait-cycle residual mean / sigma stored in the model and the nominal z streams
(for tools/deploy_calibrate_cusum.py).

Outputs in --out_dir: ensemble.pt, calib_streams.npz (residual, disagreement: (cycles, envs)), train.json.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_train_ensemble.py --headless \
      --checkpoint <model.pt> --num_envs 64 --steps 3000 --calib_steps 3000
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None)
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3000, help="rollout steps per env for training (3000 = 1 min)")
parser.add_argument("--calib_steps", type=int, default=3000, help="fresh rollout steps per env for the nominal residual calibration")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--hidden", type=int, nargs="+", default=None)
parser.add_argument("--levels", type=int, nargs="+", default=None)
parser.add_argument("--reposition_s", type=float, default=15.0)
parser.add_argument("--out_dir", default=None, help="default: <checkpoint dir>/ensemble")
parser.add_argument("--env_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint  # noqa: E402
from hexapod.deploy.ensemble import DynamicsEnsemble, train_ensemble  # noqa: E402
from hexapod.deploy.envutil import make_env, pin_commands, policy_obs, reset_in_place  # noqa: E402

dcfg = C.DeployCfg(task=args.task, env_seed=args.env_seed, reposition_s=args.reposition_s)
if args.levels:
    dcfg.levels = tuple(args.levels)
ecfg = C.EnsembleCfg()
if args.epochs:
    ecfg.epochs = args.epochs
if args.hidden:
    ecfg.hidden_dims = tuple(args.hidden)
out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "ensemble")
os.makedirs(out_dir, exist_ok=True)

env, applied = make_env(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir, dr=True)
dev, N, dt = env.device, env.num_envs, float(env.step_dt)
actor, _, info = build_from_checkpoint(args.checkpoint, N, dev)
ids = torch.arange(N, device=dev)
repo_steps = int(round(args.reposition_s / dt))
steps_per_cycle = int(round(dcfg.gait_cycle_s / dt))
print("[ensemble] env settings:", applied, "N", N, "dr ranges", env.cfg.dr.friction_static_range, env.cfg.dr.mass_add_range_kg, env.cfg.dr.motor_strength_range)


def rollout(n_steps: int, ensemble: DynamicsEnsemble | None = None):
    """Deterministic pi_0 rollout; returns (obs, act, next_obs, valid) per step and, with an ensemble, per-cycle
    residual / disagreement arrays (cycles, N)."""
    env.resample_dr(ids)
    reset_in_place(env, ids, dcfg.cmd_speed)
    obs = policy_obs(env)
    O, A, O2, V = [], [], [], []
    res_c, dis_c = [], []
    acc_r = torch.zeros(N, device=dev)
    acc_d = torch.zeros(N, device=dev)
    acc_n = torch.zeros(N, device=dev)
    falls = 0
    t0 = time.time()
    for step in range(n_steps):
        with torch.no_grad():
            act, _ = actor(obs)
        obs_next, rew, term, trunc, _ = env.step(act)
        obs_next = obs_next["policy"]
        fell = term.bool()
        falls += int(fell.sum())
        valid = ~fell
        O.append(obs)
        A.append(act)
        O2.append(obs_next)
        V.append(valid)
        if ensemble is not None:
            nmse, dis = ensemble.residual_step(obs, act, obs_next)
            acc_r += torch.where(valid, nmse, torch.zeros_like(nmse))
            acc_d += torch.where(valid, dis, torch.zeros_like(dis))
            acc_n += valid.float()
            if (step + 1) % steps_per_cycle == 0:
                nan = torch.full((N,), float("nan"), device=dev)
                res_c.append(torch.where(acc_n > 0, acc_r / acc_n.clamp_min(1), nan).cpu())
                dis_c.append(torch.where(acc_n > 0, acc_d / acc_n.clamp_min(1), nan).cpu())
                acc_r.zero_(), acc_d.zero_(), acc_n.zero_()
        if bool(fell.any()):
            pin_commands(env, dcfg.cmd_speed)
            obs_next = policy_obs(env)
        if repo_steps and (step + 1) % repo_steps == 0:
            reset_in_place(env, ids, dcfg.cmd_speed)  # dr.enabled: new DR draws for every env
            obs_next = policy_obs(env)
        obs = obs_next
        if (step + 1) % 500 == 0:
            print(f"[ensemble] rollout step {step + 1}/{n_steps}, {(step + 1) / (time.time() - t0):.1f} steps/s, falls {falls}", flush=True)
    out = (torch.cat(O), torch.cat(A), torch.cat(O2), torch.cat(V))
    if ensemble is not None:
        return out, torch.stack(res_c).numpy(), torch.stack(dis_c).numpy(), falls
    return out, None, None, falls


(obs, act, obs2, valid), _, _, falls_train = rollout(args.steps)
obs, act, obs2 = obs[valid], act[valid], obs2[valid]
print(f"[ensemble] training set: {obs.shape[0]} transitions from {N} envs x {args.steps} steps ({falls_train} falls excluded)")
ensemble = DynamicsEnsemble(info["obs_dim"], info["act_dim"], ecfg).to(dev)
res = train_ensemble(ensemble, obs, act, obs2, ecfg, dev)
ensemble.eval()

# fresh calibration rollout: per-cycle residual statistics of the nominal (in-range DR) robot
(_, _, _, valid_c), res_c, dis_c, falls_calib = rollout(args.calib_steps, ensemble)
r = res_c[~np.isnan(res_c)]
d = dis_c[~np.isnan(dis_c)]
with torch.no_grad():
    ensemble.nominal_mu.fill_(float(r.mean()))
    ensemble.nominal_sigma.fill_(float(r.std()))
    ensemble.nominal_mu_dis.fill_(float(d.mean()))
    ensemble.nominal_sigma_dis.fill_(float(d.std()))
stats = {
    "checkpoint": args.checkpoint, "num_envs": N, "train_steps": args.steps, "calib_steps": args.calib_steps, "levels": list(dcfg.levels),
    "n_train_transitions": int(obs.shape[0]), "falls_train": falls_train, "falls_calib": falls_calib, "ensemble_cfg": C.to_dict(ecfg),
    "history": res["history"], "heldout_step_nmse_mean": res["heldout_nmse_mean"], "heldout_step_nmse_std": res["heldout_nmse_std"],
    "calib_cycle_residual_mean": float(r.mean()), "calib_cycle_residual_std": float(r.std()),
    "calib_cycle_residual_p50_p95_p99": [float(np.quantile(r, q)) for q in (0.5, 0.95, 0.99)],
    "calib_cycle_disagreement_mean": float(d.mean()), "calib_cycle_disagreement_std": float(d.std()),
    "calib_cycles": int(res_c.shape[0]), "dr_ranges": {"friction": env.cfg.dr.friction_static_range, "mass": env.cfg.dr.mass_add_range_kg, "motor": env.cfg.dr.motor_strength_range},
}
ensemble.save(os.path.join(out_dir, "ensemble.pt"), extra=stats)
np.savez_compressed(os.path.join(out_dir, "calib_streams.npz"), residual=res_c, disagreement=dis_c, cycle_s=dcfg.gait_cycle_s)
json.dump(stats, open(os.path.join(out_dir, "train.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "history"}, indent=1))
print("[ensemble] wrote", out_dir)
env.close()
app.close()
