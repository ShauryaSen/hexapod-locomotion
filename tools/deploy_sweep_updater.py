# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep the updater's nuisance knobs (decisions.md F7: LR and KL weight, tuned once in the pilot) in ONE process:
the deployments are split into groups, each with its own learning rate, KL-anchor weight and exploration switch
(per-slice values in the batched updater; everything else shared). Every deployment gets the same change at
--t_on; the oracle arm adapts from the change for --budget_min, then holds (deterministic actions, adapted weights
kept) until the end, so the last part of the run measures the adapted policy without exploration noise.

Groups are given as name:lr:kl_weight:explore, e.g.
  control:0:1:0          lr 0, no exploration  (= frozen pi_0, the never arm, in the same process)
  explore_only:0:1:1     lr 0, exploration on  (cost of exploration alone)
  lr1e-4_kl1:1e-4:1:1    the harness default
Per group: progress rate before the change, while adapting, and in the hold phase; falls; KL to pi_0 at the end.
Throughput (MEASURED 2026-10-21/22 on the shared Spark): ~1.5 steps/s at both 16 and 64 robots, so groups in one
process cost no extra wall time.

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_sweep_updater.py --headless \
      --checkpoint <model.pt> --factor motor --value 0.6 --per_group 12 --t_on 60 --budget_min 8 --minutes 11 \
      --groups control:0:1:0 explore_only:0:1:1 lr1e-4_kl1:1e-4:1:1 lr1e-3_kl1:1e-3:1:1 lr1e-3_kl0.1:1e-3:0.1:1
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
parser.add_argument("--factor", default="motor", choices=["payload", "motor", "friction"])
parser.add_argument("--value", type=float, required=True)
parser.add_argument("--groups", nargs="+", required=True, help="name:lr:kl_weight:explore(0/1)[:std_scale[:batch_steps]] (std_scale multiplies pi_0's action std at the start of the fine-tune; batch_steps = this group's PPO batch length)")
parser.add_argument("--per_group", type=int, default=12)
parser.add_argument("--t_on", type=float, default=60.0)
parser.add_argument("--budget_min", type=float, default=8.0)
parser.add_argument("--minutes", type=float, default=11.0)
parser.add_argument("--steps_per_batch", type=int, default=None)
parser.add_argument("--hold_skip_s", type=float, default=10.0, help="ignore this much of the hold phase (transient) in the hold rate")
parser.add_argument("--out_dir", default=None, help="default: <checkpoint dir>/sweep_<factor><value>[_tag]")
parser.add_argument("--tag", default="")
parser.add_argument("--env_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.arms import make_arm  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint, build_preproc  # noqa: E402
from hexapod.deploy.envutil import make_env  # noqa: E402
from hexapod.deploy.metrics import bootstrap_ci  # noqa: E402
from hexapod.deploy.runner import DeploymentRunner  # noqa: E402
from hexapod.deploy.schedule import ChangeEvent, Schedule, ScheduleApplier  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

groups = []
for g in args.groups:
    parts = g.split(":")
    name, lr, kl, ex = parts[:4]
    std_scale = float(parts[4]) if len(parts) > 4 else 1.0
    batch = int(parts[5]) if len(parts) > 5 else None
    groups.append({"name": name, "lr": float(lr), "kl_weight": float(kl), "explore": bool(int(ex)), "std_scale": std_scale, "batch_steps": batch})
G, R = len(groups), args.per_group
N = G * R
dcfg = C.DeployCfg(task=args.task, env_seed=args.env_seed)
ucfg = C.UpdaterCfg(budget_min=args.budget_min)
if args.steps_per_batch is not None:
    ucfg.steps_per_batch = args.steps_per_batch
scfg = C.ScheduleCfg()
setattr(scfg, {"payload": "payload_kg", "motor": "motor_strength", "friction": "friction"}[args.factor], args.value)
detcfg = C.DetectorCfg()
out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), f"sweep_{args.factor}{args.value:g}" + (f"_{args.tag}" if args.tag else ""))
os.makedirs(out_dir, exist_ok=True)

env, applied = make_env(dcfg, args.checkpoint, N, args.device, args.run_dir)
dev = env.device
actor, critic, info = build_from_checkpoint(args.checkpoint, N, dev)
preproc = build_preproc(info, N, env.single_observation_space["policy"].shape[-1], dev)
updater = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, info["obs_dim"])
# group i = deployments [i*R, (i+1)*R); interleaving would spread groups over terrain cells more evenly, so do that
gid = torch.arange(N, device=dev) % G
for i, g in enumerate(groups):
    m = gid == i
    updater.opt.lr[m] = g["lr"]
    updater.kl_w[m] = g["kl_weight"]
    updater.explore_enabled[m] = g["explore"]
    updater.update_enabled[m] = g["lr"] > 0
    if g["std_scale"] != 1.0:
        with torch.no_grad():
            actor.std_param[m] *= g["std_scale"]  # ASSUMED: smaller exploration noise for the fine-tune (pi_0 itself unchanged)
    if g["batch_steps"]:
        updater.set_batch_steps(m, g["batch_steps"])
sched = Schedule(seed=-1, length_s=args.minutes * 60.0, events=[ChangeEvent(args.t_on, args.factor, True)], cfg={"manual": True, "value": args.value})
schedules = [sched] * N
arm = make_arm("oracle", N, C.ArmCfg(name="oracle", budget_min=args.budget_min), ucfg, detcfg, dev, dcfg.gait_cycle_s, schedules)
applier = ScheduleApplier(env, schedules, scfg)
runner = DeploymentRunner(env, dcfg, updater, arm, applier, None, pi0_critic=info["critic"], preproc=preproc)
meta = {"checkpoint": args.checkpoint, "factor": args.factor, "value": args.value, "groups": groups, "per_group": R, "group_of_env": gid.tolist(), "t_on": args.t_on, "budget_min": args.budget_min, "minutes": args.minutes, "updater_cfg": C.to_dict(ucfg), "deploy_cfg": C.to_dict(dcfg), "env_applied": applied,
        "levels_per_env": env._terrain.terrain_levels.tolist() if env._has_terrain_levels else None, "types_per_env": env._terrain.terrain_types.tolist() if env._has_terrain_levels else None}
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
summary = runner.run(args.minutes)
runner.stream.save(out_dir, meta)
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

arr = runner.stream.arrays()
t = arr["t_s"][:, 0]
cyc = dcfg.gait_cycle_s
t_end_adapt = args.t_on + args.budget_min * 60.0
phases = {"before": t <= args.t_on, "adapting": (t > args.t_on) & (t <= t_end_adapt), "hold": t > t_end_adapt + args.hold_skip_s}
gid_np = np.array(meta["group_of_env"])
rows = []
for i, g in enumerate(groups):
    sel = gid_np == i
    row = {"group": g["name"], **g}
    for ph, mask in phases.items():
        mins = mask.sum() * cyc / 60.0
        rate = arr["progress_m"][mask][:, sel].sum(0) / max(mins, 1e-9)  # per robot, m/min
        lo, hi = bootstrap_ci(rate, 1000, 0.95)
        row[f"rate_{ph}"] = {"mean": float(rate.mean()), "ci": [lo, hi], "per_robot": rate.tolist()}
        row[f"falls_{ph}"] = float(arr["falls"][mask][:, sel].sum())
        row[f"minutes_{ph}"] = float(mins)
    row["updates"] = float(arr["updates"][:, sel].sum() / R)
    kl_end = arr["kl_pi0"][-1, sel]
    row["kl_pi0_end"] = float(np.nanmean(kl_end)) if np.any(~np.isnan(kl_end)) else None
    rows.append(row)
ctrl = next((r for r in rows if r["lr"] == 0 and not r["explore"]), None)
md = [f"# Updater sweep: {args.factor} = {args.value} at {args.t_on:.0f} s, {R} robots per group, adapt {args.budget_min:g} min then hold (deterministic) to {args.minutes:g} min", "",
      "groups: " + "; ".join(f"{g['name']} = lr {g['lr']:g}, KL {g['kl_weight']:g}, explore {int(g['explore'])}, std x{g['std_scale']:g}, batch {g['batch_steps'] or ucfg.steps_per_batch}" for g in groups), "",
      f"checkpoint `{os.path.basename(os.path.dirname(args.checkpoint))}/{os.path.basename(args.checkpoint)}`; {ucfg.steps_per_batch} steps/batch, {ucfg.num_learning_epochs} epochs, entropy {ucfg.entropy_coef}; progress capped at the command (R4). Rates in m/min per robot, mean [95% bootstrap CI over robots].", "",
      "| group | lr | KL weight | explore | before change | adapting | hold (adapted weights, no noise) | hold - control hold | falls before / adapting / hold | updates per robot | KL to pi_0 at end |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    f = lambda k: f"{r[k]['mean']:.2f} [{r[k]['ci'][0]:.2f}, {r[k]['ci'][1]:.2f}]"
    d = f"{r['rate_hold']['mean'] - ctrl['rate_hold']['mean']:+.2f}" if ctrl else "–"
    md.append(f"| {r['group']} | {r['lr']:g} | {r['kl_weight']:g} | {int(r['explore'])} | {f('rate_before')} | {f('rate_adapting')} | {f('rate_hold')} | {d} | {r['falls_before']:.0f} / {r['falls_adapting']:.0f} / {r['falls_hold']:.0f} | {r['updates']:.0f} | {r['kl_pi0_end'] if r['kl_pi0_end'] is None else round(r['kl_pi0_end'], 4)} |")
open(os.path.join(out_dir, "sweep.md"), "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.join(out_dir, "sweep.json"), "w"), indent=1)
print("\n".join(md))
print("[sweep] wrote", out_dir)
env.close()
app.close()
