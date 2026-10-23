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
parser.add_argument("--arms", nargs="*", default=None, help="several arms in one process: name or name:key=val,key=val (period_min, hold_min, budget_min, cusum_h, three_sigma_consecutive); deployment i runs arm i %% len(arms) and schedule seed schedule_seeds[(i // len(arms)) %% len(seeds)], so every arm sees the same schedules and terrain pattern")
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
parser.add_argument("--init", nargs="*", default=[], help="world state from t = 0 for every deployment through the runtime setters (hold), e.g. friction=0.175 payload=2.5 motor=0.6; the PhysX read-back of each env is stored in meta.json")
parser.add_argument("--out_root", default="/workspace/isaaclab/logs/deploy")
parser.add_argument("--tag", default="")
parser.add_argument("--progress_every_min", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.arms import arm_label, make_arm, make_composite  # noqa: E402
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
out_dir = os.path.join(args.out_root, f"{stamp}_{'multi' if args.arms else args.arm}" + (f"_{args.tag}" if args.tag else ""))
os.makedirs(out_dir, exist_ok=True)

env, applied = make_env(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir)
dev = env.device
N = env.num_envs
print("[deploy] env settings:", applied, "N", N, "step_dt", env.step_dt)

A = len(args.arms) if args.arms else 1
replicate_of_env = [i // A for i in range(N)]
if A > 1 and env._has_terrain_levels:
    # every arm's replicate r gets the same terrain cell (and schedule) as the other arms' replicate r
    rep = torch.tensor(replicate_of_env, device=dev)
    n_cols = env._terrain.terrain_origins.shape[1]
    lv = torch.tensor(list(dcfg.levels), device=dev)[rep % len(dcfg.levels)]
    ty = (rep // len(dcfg.levels)) % n_cols
    env.set_terrain_cells(torch.arange(N, device=dev), lv, ty)
    env.reset()
    from hexapod.deploy.envutil import pin_commands

    pin_commands(env, dcfg.cmd_speed)
schedules = [None] * N
if not args.no_schedule:
    seeds = args.schedule_seeds
    uniq = {s: generate_schedule(s, scfg) for s in seeds}
    schedules = [uniq[seeds[(i // A) % len(seeds)]] for i in range(N)]
    for s in uniq.values():
        s.save(os.path.join(out_dir, f"schedule_seed{s.seed}.json"))
    print("[deploy] schedules:", {s: [(round(e.t_s), e.factor, e.on) for e in sch.events] for s, sch in uniq.items()})
minutes = args.minutes if args.minutes is not None else (2.0 if args.no_schedule else max(s.length_s for s in schedules) / 60.0)

actor, critic, info = build_from_checkpoint(args.checkpoint, N, dev)
preproc = build_preproc(info, N, env.single_observation_space["policy"].shape[-1], dev)  # R1: RMA student or None
updater = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, info["obs_dim"])
ensemble = DynamicsEnsemble.load(args.ensemble, dev) if args.ensemble else None
if args.arms:
    specs = []
    for spec in args.arms:
        name, _, kv = spec.partition(":")
        kw = {k: float(v) if k != "three_sigma_consecutive" else int(v) for k, v in (x.split("=") for x in kv.split(",") if x)}
        specs.append((name, C.ArmCfg(name=name, **{"period_min": args.period_min, "hold_min": args.hold_min, "budget_min": args.budget_min, **kw})))
    arm = make_composite(specs, N, ucfg, detcfg, dev, dcfg.gait_cycle_s, schedules)
    arm_of_env = [specs[i % A][0] for i in range(N)]
    label_of_env = [arm.labels[i % A] for i in range(N)]
    args.arm = "multi"
else:
    arm = make_arm(args.arm, N, acfg, ucfg, detcfg, dev, dcfg.gait_cycle_s, schedules)
    arm_of_env = [args.arm] * N
    label_of_env = [arm_label(acfg, detcfg)] * N
applier = ScheduleApplier(env, schedules, scfg)
init_readback = None
if args.init:
    import warp as wp

    kv = {k: float(v) for k, v in (x.split("=") for x in args.init)}
    applier.set_all(payload=kv.get("payload"), motor=kv.get("motor"), friction=kv.get("friction"))
    mat = wp.to_torch(env._robot.root_view.get_material_properties())
    init_readback = {
        "requested": kv,
        "material_static_per_env": mat[:, :, 0].mean(1).tolist(), "material_static_min_max": [float(mat[:, :, 0].min()), float(mat[:, :, 0].max())],
        "material_dynamic_min_max": [float(mat[:, :, 1].min()), float(mat[:, :, 1].max())],
        "dr_friction_per_env": env.dr_friction.tolist(), "dr_delta_mass_per_env": env.dr_delta_mass.tolist(), "dr_motor_strength_per_env": env.dr_motor_strength.tolist(),
        "ground_static": float(env.cfg.terrain.physics_material.static_friction), "ground_combine": env.cfg.terrain.physics_material.friction_combine_mode,
    }
    print("[deploy] init state:", kv, "material static min/max", init_readback["material_static_min_max"], "dr_friction min/max", min(init_readback["dr_friction_per_env"]), max(init_readback["dr_friction_per_env"]), flush=True)
runner = DeploymentRunner(env, dcfg, updater, arm, applier, ensemble, pi0_critic=info["critic"], preproc=preproc)

meta = {
    "checkpoint": args.checkpoint, "task": args.task, "num_envs": N, "arm": args.arm, "arm_label": arm_label(acfg, detcfg) if not args.arms else "multi", "arm_of_env": arm_of_env, "arm_label_of_env": label_of_env, "replicate_of_env": replicate_of_env, "arms_spec": args.arms, "tag": args.tag, "minutes": minutes,
    "rma_student": info.get("rma_student") is not None, "init_state": init_readback, "schedule_split": [None if s is None else ("tune" if s.seed in scfg.tune_seeds else "test" if s.seed in scfg.test_seeds else "other") for s in schedules],
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
