# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run N deployments of one arm in one process (hexapod/deploy). Each env = one deployment with its own policy
copy; env i follows schedule seed ``schedule_seeds[i % len]`` (so N = replicates x schedules), or no changes with
--no_schedule. Writes <out_dir>/{streams.npz, meta.json, schedule_env{i}.json, summary.json, final_policies.pt}.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_run.py --headless \
      --checkpoint <model.pt> --arm never --num_envs 16 --minutes 2 --no_schedule --tag smoke
"""

import argparse
import datetime
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="pi_0 (rsl_rl model_*.pt)")
parser.add_argument("--run_dir", default=None, help="for params/env.yaml (default: the checkpoint's directory)")
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--minutes", type=float, default=None, help="robot-time per deployment (default: the schedule length, or 2 with --no_schedule)")
parser.add_argument("--arm", default="never")
parser.add_argument("--period_min", type=float, default=10.0)
parser.add_argument("--hold_min", type=float, default=10.0)
parser.add_argument("--budget_min", type=float, default=None, help="adaptation budget per episode [robot-min] (default UpdaterCfg.budget_min)")
parser.add_argument("--schedule_seeds", type=int, nargs="*", default=[0])
parser.add_argument("--no_schedule", action="store_true")
parser.add_argument("--t50_min", type=float, default=None, help="T50 for the schedule generator (default ScheduleCfg.t50_min)")
parser.add_argument("--ensemble", default=None, help="dynamics ensemble .pt (tools/deploy_train_ensemble.py)")
parser.add_argument("--cusum_h", type=float, default=None, help="calibrated CUSUM threshold (tools/deploy_calibrate_cusum.py); the cusum arm's knob")
parser.add_argument("--three_sigma_consecutive", type=int, default=None, help="the three_sigma arm's knob")
parser.add_argument("--baseline_cycles", type=int, default=None, help="R5 per-deployment baseline length in cycles (default DetectorCfg)")
parser.add_argument("--steps_per_batch", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--kl_weight", type=float, default=None)
parser.add_argument("--levels", type=int, nargs="+", default=None)
parser.add_argument("--fall_recovery_s", type=float, default=None)
parser.add_argument("--reposition_s", type=float, default=None)
parser.add_argument("--env_seed", type=int, default=0)
parser.add_argument("--env_override", nargs="*", default=[])
parser.add_argument("--out_root", default="/workspace/isaaclab/logs/deploy")
parser.add_argument("--tag", default="")
parser.add_argument("--progress_every_min", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.arms import arm_label, make_arm  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint, build_preproc  # noqa: E402
from hexapod.deploy.ensemble import DynamicsEnsemble  # noqa: E402
from hexapod.deploy.envutil import make_env  # noqa: E402
from hexapod.deploy.runner import DeploymentRunner  # noqa: E402
from hexapod.deploy.schedule import ScheduleApplier, generate_schedule  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

dcfg = C.DeployCfg(task=args.task, env_seed=args.env_seed, env_override=tuple(args.env_override))
if args.levels:
    dcfg.levels = tuple(args.levels)
if args.fall_recovery_s is not None:
    dcfg.fall_recovery_s = args.fall_recovery_s
if args.reposition_s is not None:
    dcfg.reposition_s = args.reposition_s
ucfg = C.UpdaterCfg()
if args.steps_per_batch is not None:
    ucfg.steps_per_batch = args.steps_per_batch
if args.lr is not None:
    ucfg.learning_rate = args.lr
if args.kl_weight is not None:
    ucfg.kl_anchor_weight = args.kl_weight
if args.budget_min is not None:
    ucfg.budget_min = args.budget_min
scfg = C.ScheduleCfg()
if args.t50_min:
    scfg.t50_min = args.t50_min
acfg = C.ArmCfg(name=args.arm, period_min=args.period_min, hold_min=args.hold_min, budget_min=args.budget_min, cusum_h=args.cusum_h, three_sigma_consecutive=args.three_sigma_consecutive)
detcfg = C.DetectorCfg()
if args.baseline_cycles is not None:
    detcfg.baseline_cycles = args.baseline_cycles

stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join(args.out_root, f"{stamp}_{args.arm}" + (f"_{args.tag}" if args.tag else ""))
os.makedirs(out_dir, exist_ok=True)

env, applied = make_env(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir)
dev = env.device
N = env.num_envs
print("[deploy] env settings:", applied, "N", N, "step_dt", env.step_dt)

schedules = [None] * N
if not args.no_schedule:
    seeds = args.schedule_seeds
    uniq = {s: generate_schedule(s, scfg) for s in seeds}
    schedules = [uniq[seeds[i % len(seeds)]] for i in range(N)]
    for s in uniq.values():
        s.save(os.path.join(out_dir, f"schedule_seed{s.seed}.json"))
    print("[deploy] schedules:", {s: [(round(e.t_s), e.factor, e.on) for e in sch.events] for s, sch in uniq.items()})
minutes = args.minutes if args.minutes is not None else (2.0 if args.no_schedule else max(s.length_s for s in schedules) / 60.0)

actor, critic, info = build_from_checkpoint(args.checkpoint, N, dev)
preproc = build_preproc(info, N, env.single_observation_space["policy"].shape[-1], dev)  # R1: RMA student or None
updater = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, info["obs_dim"])
ensemble = DynamicsEnsemble.load(args.ensemble, dev) if args.ensemble else None
arm = make_arm(args.arm, N, acfg, ucfg, detcfg, dev, dcfg.gait_cycle_s, schedules)
applier = ScheduleApplier(env, schedules, scfg)
runner = DeploymentRunner(env, dcfg, updater, arm, applier, ensemble, pi0_critic=info["critic"], preproc=preproc)

meta = {
    "checkpoint": args.checkpoint, "task": args.task, "num_envs": N, "arm": args.arm, "arm_label": arm_label(acfg, detcfg), "tag": args.tag, "minutes": minutes,
    "rma_student": info.get("rma_student") is not None, "schedule_split": [None if s is None else ("tune" if s.seed in scfg.tune_seeds else "test" if s.seed in scfg.test_seeds else "other") for s in schedules],
    "schedule_seeds": None if args.no_schedule else args.schedule_seeds, "env_schedule_seed": [None if s is None else s.seed for s in schedules],
    "ensemble": args.ensemble, "cusum_h": args.cusum_h, "env_applied": applied, "step_dt": env.step_dt,
    "deploy_cfg": C.to_dict(dcfg), "updater_cfg": C.to_dict(ucfg), "schedule_cfg": C.to_dict(scfg), "arm_cfg": C.to_dict(acfg), "detector_cfg": C.to_dict(detcfg),
    "levels_per_env": env._terrain.terrain_levels.tolist() if env._has_terrain_levels else None,
    "types_per_env": env._terrain.terrain_types.tolist() if env._has_terrain_levels else None,
}
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
print("[deploy] out_dir", out_dir)
summary = runner.run(minutes, args.progress_every_min)
runner.stream.save(out_dir, meta)
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
torch.save({"actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()}, "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()}}, os.path.join(out_dir, "final_policies.pt"))
print(json.dumps({k: v for k, v in summary.items() if k not in ("progress_per_env_m", "falls_per_env", "updates_per_env", "events")}, indent=1))
print("[deploy] wrote", out_dir)
env.close()
app.close()
