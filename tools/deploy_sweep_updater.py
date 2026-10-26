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
parser.add_argument("--groups", nargs="+", required=True, help="name:lr:kl_weight:explore(0/1)[:std_scale[:batch_steps]][:key=val...] (std_scale multiplies pi_0's action std at the start of the fine-tune; batch_steps = this group's PPO batch length per robot; keys: k = robots per deployment (option 2, default 1), subset = all | last_layer (option 3a))")
parser.add_argument("--snapshot_adapt_min", type=float, nargs="*", default=[], help="save every deployment's actor at these adaptation minutes (whole robot-minutes after t_on) to <out>/snapshots/ for tools/deploy_eval_policies.py")
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
    pos = [x for x in parts[4:] if "=" not in x]
    kw = dict(x.split("=", 1) for x in parts[4:] if "=" in x)
    std_scale = float(pos[0]) if len(pos) > 0 else 1.0
    batch = int(pos[1]) if len(pos) > 1 else None
    groups.append({"name": name, "lr": float(lr), "kl_weight": float(kl), "explore": bool(int(ex)), "std_scale": std_scale, "batch_steps": batch, "k": int(kw.get("k", 1)), "subset": kw.get("subset", "all")})
G, R = len(groups), args.per_group
NS = G * R  # deployments (policy slices): slice s = replicate s // G, group s % G
KMAX = max(g["k"] for g in groups)
slice_mode = KMAX > 1
sid_list, rob_list = [], []
for sl in range(NS):
    for j in range(groups[sl % G]["k"]):
        sid_list.append(sl)
        rob_list.append(j)
N = len(sid_list)  # robots
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
actor, critic, info = build_from_checkpoint(args.checkpoint, NS, dev)
preproc = build_preproc(info, N, env.single_observation_space["policy"].shape[-1], dev)
sid = torch.tensor(sid_list, device=dev)
updater = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, info["obs_dim"], slice_of_env=sid if slice_mode else None)
# deployment s: group s % G, replicate s // G; every replicate's G deployments share terrain cells, so groups are
# compared on identical terrain (2026-10-23: with G = 6 and 6 levels, cells assigned by env index put each group on
# a single level and confounded the groups with terrain in sweep 2). With k robots per deployment, robot j of a
# replicate-r deployment walks cell r * KMAX + j (robot 0 of every group shares the replicate's first cell).
gid_s = torch.arange(NS, device=dev) % G
rep_s = torch.arange(NS, device=dev) // G
gid = gid_s[sid]
rep = rep_s[sid]
cell_idx = rep * KMAX + torch.tensor(rob_list, device=dev) if slice_mode else rep
if env._has_terrain_levels:
    from hexapod.deploy.envutil import terrain_cells

    lv, ty = terrain_cells(env, cell_idx, dcfg.levels)
    env.set_terrain_cells(torch.arange(N, device=dev), lv, ty)
    env.reset()
    from hexapod.deploy.envutil import pin_commands

    pin_commands(env, dcfg.cmd_speed)
for i, g in enumerate(groups):
    m = gid_s == i  # per deployment
    if g["subset"] != "all":
        updater.set_param_subset(m, g["subset"])
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
arm = make_arm("oracle", NS, C.ArmCfg(name="oracle", budget_min=args.budget_min), ucfg, detcfg, dev, dcfg.gait_cycle_s, [sched] * NS)
applier = ScheduleApplier(env, schedules, scfg)
runner = DeploymentRunner(env, dcfg, updater, arm, applier, None, pi0_critic=info["critic"], preproc=preproc)
meta = {"checkpoint": args.checkpoint, "factor": args.factor, "value": args.value, "groups": groups, "per_group": R, "group_of_env": gid.tolist(), "replicate_of_env": rep.tolist(), "slice_of_env": sid.tolist(), "group_of_slice": gid_s.tolist(), "replicate_of_slice": rep_s.tolist(), "slice_mode": slice_mode, "snapshot_adapt_min": args.snapshot_adapt_min, "t_on": args.t_on, "budget_min": args.budget_min, "minutes": args.minutes, "updater_cfg": C.to_dict(ucfg), "deploy_cfg": C.to_dict(dcfg), "env_applied": applied,
        "levels_per_env": env._terrain.terrain_levels.tolist() if env._has_terrain_levels else None, "types_per_env": env._terrain.terrain_types.tolist() if env._has_terrain_levels else None}
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
snap_dir = os.path.join(out_dir, "snapshots")
want = {int(round(m)) for m in args.snapshot_adapt_min}


def on_minute(t_s):
    m = int(round((t_s - args.t_on) / 60.0))
    if abs((t_s - args.t_on) - 60.0 * m) < 1e-6 and m in want:
        os.makedirs(snap_dir, exist_ok=True)
        torch.save({"adapt_min": m, "t_s": t_s, "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}, "group_of_slice": gid_s.tolist(), "replicate_of_slice": rep_s.tolist(), "groups": groups,
                    "robot_hours_of_slice": [groups[int(g)]["k"] * m / 60.0 for g in gid_s.tolist()]}, os.path.join(snap_dir, f"adapt_{m:03d}.pt"))
        print(f"[sweep] snapshot at {m} adapt-min", flush=True)


if 0 in want:
    on_minute(args.t_on) if args.t_on % 60 == 0 else None
summary = runner.run(args.minutes, on_minute=on_minute if want else None)
runner.stream.save(out_dir, meta)
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

arr = runner.stream.arrays()
t = arr["t_s"][:, 0]
cyc = dcfg.gait_cycle_s
t_end_adapt = args.t_on + args.budget_min * 60.0
phases = {"before": t <= args.t_on, "adapting": (t > args.t_on) & (t <= t_end_adapt), "hold": t > t_end_adapt + args.hold_skip_s}
sid_np = np.array(meta["slice_of_env"])
gid_np = np.array(meta["group_of_slice"])  # analysis per deployment = mean over its robots
k_of_slice = np.bincount(sid_np, minlength=NS)


def per_slice(x):  # (robots,) -> (deployments,)
    return np.bincount(sid_np, weights=x, minlength=NS) / k_of_slice


rows = []
for i, g in enumerate(groups):
    sel = gid_np == i
    row = {"group": g["name"], **g}
    for ph, mask in phases.items():
        mins = mask.sum() * cyc / 60.0
        rate = per_slice(arr["progress_m"][mask].sum(0) / max(mins, 1e-9))[sel]  # per deployment (robot mean), m/min
        lo, hi = bootstrap_ci(rate, 1000, 0.95)
        row[f"rate_{ph}"] = {"mean": float(rate.mean()), "ci": [lo, hi], "per_robot": rate.tolist()}
        row[f"falls_{ph}"] = float(per_slice(arr["falls"][mask].sum(0))[sel].sum())  # falls per robot, summed over deployments
        row[f"minutes_{ph}"] = float(mins)
    row["updates"] = float(per_slice(arr["updates"].sum(0))[sel].mean())
    kl_end = per_slice(arr["kl_pi0"][-1])[sel]
    row["kl_pi0_end"] = float(np.nanmean(kl_end)) if np.any(~np.isnan(kl_end)) else None
    rows.append(row)
ctrl = next((r for r in rows if r["lr"] == 0 and not r["explore"]), None)
# paired differences vs control per replicate (same terrain cell), hold phase
if ctrl is not None:
    rep_np = np.array(meta["replicate_of_slice"])
    ci_ctrl = groups.index(next(g for g in groups if g["name"] == ctrl["group"]))
    for i, r in enumerate(rows):
        sel, csel = gid_np == i, gid_np == ci_ctrl
        a = np.array(r["rate_hold"]["per_robot"])
        c = np.array(ctrl["rate_hold"]["per_robot"])
        ra, rc = rep_np[sel], rep_np[csel]
        common = sorted(set(ra) & set(rc))
        d = np.array([a[list(ra).index(k)] - c[list(rc).index(k)] for k in common])
        lo, hi = bootstrap_ci(d, 2000, 0.95)
        r["paired_hold_vs_control"] = {"mean": float(d.mean()), "ci": [lo, hi], "n": len(d)}
md = [f"# Updater sweep: {args.factor} = {args.value} at {args.t_on:.0f} s, {R} robots per group, adapt {args.budget_min:g} min then hold (deterministic) to {args.minutes:g} min", "",
      "groups: " + "; ".join(f"{g['name']} = lr {g['lr']:g}, KL {g['kl_weight']:g}, explore {int(g['explore'])}, std x{g['std_scale']:g}, batch {g['batch_steps'] or ucfg.steps_per_batch}" for g in groups), "",
      f"checkpoint `{os.path.basename(os.path.dirname(args.checkpoint))}/{os.path.basename(args.checkpoint)}`; {ucfg.steps_per_batch} steps/batch, {ucfg.num_learning_epochs} epochs, entropy {ucfg.entropy_coef}; progress capped at the command (R4). Rates in m/min per robot, mean [95% bootstrap CI over robots].", "",
      ("robots per deployment: " + ", ".join(f"{g['name']} k={g['k']}" + (f" ({g['subset']})" if g["subset"] != "all" else "") for g in groups) + "; rates are per deployment (mean over its robots)") if slice_mode or any(g["subset"] != "all" for g in groups) else "", "",
      "| group | lr | KL weight | explore | before change | adapting | hold (adapted weights, no noise) | hold - control, paired per replicate [95% CI] | falls before / adapting / hold | updates per robot | KL to pi_0 at end |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    f = lambda k: f"{r[k]['mean']:.2f} [{r[k]['ci'][0]:.2f}, {r[k]['ci'][1]:.2f}]"
    pd = r.get("paired_hold_vs_control")
    d = f"{pd['mean']:+.2f} [{pd['ci'][0]:+.2f}, {pd['ci'][1]:+.2f}]" if pd else "–"
    md.append(f"| {r['group']} | {r['lr']:g} | {r['kl_weight']:g} | {int(r['explore'])} | {f('rate_before')} | {f('rate_adapting')} | {f('rate_hold')} | {d} | {r['falls_before']:.0f} / {r['falls_adapting']:.0f} / {r['falls_hold']:.0f} | {r['updates']:.0f} | {r['kl_pi0_end'] if r['kl_pi0_end'] is None else round(r['kl_pi0_end'], 4)} |")
open(os.path.join(out_dir, "sweep.md"), "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.join(out_dir, "sweep.json"), "w"), indent=1)
print("\n".join(md))
print("[sweep] wrote", out_dir)
env.close()
app.close()
