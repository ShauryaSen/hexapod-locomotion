# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual vs time across a change event: N nominal deployments of frozen pi_0 (never arm), one factor switched
on at --t_on (and optionally off at --t_off), gait-cycle streams through the deployment runner, then a plot of the
residual / z / disagreement / progress across the event (mean and 10-90% band over deployments).

  PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/deploy_eval_residual.py --headless \
      --checkpoint <model.pt> --ensemble <ensemble.pt> --num_envs 16 --factor payload --t_on 60 --minutes 3
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_dir", default=None)
parser.add_argument("--ensemble", required=True)
parser.add_argument("--task", default="Hexapod-Rough-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--factor", default="payload", choices=["payload", "motor", "friction"])
parser.add_argument("--t_on", type=float, default=60.0)
parser.add_argument("--t_off", type=float, default=None)
parser.add_argument("--minutes", type=float, default=3.0)
parser.add_argument("--arm", default="never")
parser.add_argument("--levels", type=int, nargs="+", default=None)
parser.add_argument("--out_dir", default=None, help="default: <ensemble dir>/residual_<factor>")
parser.add_argument("--env_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.arms import make_arm  # noqa: E402
from hexapod.deploy.batched import build_from_checkpoint  # noqa: E402
from hexapod.deploy.ensemble import DynamicsEnsemble  # noqa: E402
from hexapod.deploy.envutil import make_env  # noqa: E402
from hexapod.deploy.runner import DeploymentRunner  # noqa: E402
from hexapod.deploy.schedule import ChangeEvent, Schedule, ScheduleApplier  # noqa: E402
from hexapod.deploy.updater import KLAnchoredPPO  # noqa: E402

dcfg = C.DeployCfg(task=args.task, env_seed=args.env_seed)
if args.levels:
    dcfg.levels = tuple(args.levels)
scfg, ucfg, detcfg = C.ScheduleCfg(), C.UpdaterCfg(), C.DetectorCfg()
out_dir = args.out_dir or os.path.join(os.path.dirname(args.ensemble), f"residual_{args.factor}" + ("" if args.arm == "never" else f"_{args.arm}"))
os.makedirs(out_dir, exist_ok=True)

env, applied = make_env(dcfg, args.checkpoint, args.num_envs, args.device, args.run_dir)
dev, N = env.device, env.num_envs
events = [ChangeEvent(args.t_on, args.factor, True)] + ([ChangeEvent(args.t_off, args.factor, False)] if args.t_off else [])
sched = Schedule(seed=-1, length_s=args.minutes * 60.0, events=events, cfg={"manual": True})
schedules = [sched] * N
actor, critic, info = build_from_checkpoint(args.checkpoint, N, dev)
updater = KLAnchoredPPO(actor, critic, info["actor"], ucfg, dev, info["obs_dim"])
ensemble = DynamicsEnsemble.load(args.ensemble, dev)
arm = make_arm(args.arm, N, C.ArmCfg(name=args.arm), ucfg, detcfg, dev, dcfg.gait_cycle_s, schedules)
applier = ScheduleApplier(env, schedules, scfg)
runner = DeploymentRunner(env, dcfg, updater, arm, applier, ensemble, pi0_critic=info["critic"])
summary = runner.run(args.minutes)
meta = {"checkpoint": args.checkpoint, "ensemble": args.ensemble, "num_envs": N, "factor": args.factor, "t_on": args.t_on, "t_off": args.t_off, "arm": args.arm, "env_applied": applied, "nominal_mu": float(ensemble.nominal_mu), "nominal_sigma": float(ensemble.nominal_sigma), "deploy_cfg": C.to_dict(dcfg)}
runner.stream.save(out_dir, meta)
sched.save(os.path.join(out_dir, "schedule_manual.json"))
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

arr = runner.stream.arrays()
t = arr["t_s"][:, 0]
res = {}
for key in ("residual", "z", "disagreement", "progress_m", "track_err", "falls"):
    a = arr[key]
    res[key] = {"mean": np.nanmean(a, 1).tolist(), "p10": np.nanpercentile(a, 10, 1).tolist(), "p90": np.nanpercentile(a, 90, 1).tolist()}
before = t < args.t_on
after = (t >= args.t_on + 2.0) & ((t < args.t_off) if args.t_off else True)
z = arr["z"]
stats = {
    "z_mean_before": float(np.nanmean(z[before])), "z_mean_after": float(np.nanmean(z[after])),
    "residual_mean_before": float(np.nanmean(arr["residual"][before])), "residual_mean_after": float(np.nanmean(arr["residual"][after])),
    "progress_rate_before_m_per_min": float(np.nansum(arr["progress_m"][before]) / N / (args.t_on / 60.0)),
    "progress_rate_after_m_per_min": float(np.nansum(arr["progress_m"][after]) / N / (after.sum() * dcfg.gait_cycle_s / 60.0)),
    "frac_cycles_z_gt_3_before": float(np.nanmean(z[before] > 3)), "frac_cycles_z_gt_3_after": float(np.nanmean(z[after] > 3)),
    "falls_total": int(np.nansum(arr["falls"])),
}
json.dump({"t_s": t.tolist(), "series": res, "stats": stats}, open(os.path.join(out_dir, "residual_vs_time.json"), "w"))
print(json.dumps(stats, indent=1))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, key, label in zip(axes, ("residual", "z", "disagreement", "progress_m"), ("ensemble residual (nmse)", "z = (r - mu) / sigma nominal", "disagreement", "progress per cycle [m]")):
        ax.fill_between(t, res[key]["p10"], res[key]["p90"], alpha=0.25)
        ax.plot(t, res[key]["mean"])
        ax.axvline(args.t_on, color="k", ls="--")
        if args.t_off:
            ax.axvline(args.t_off, color="k", ls=":")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].axhline(3.0, color="r", ls="--", lw=0.8)
    axes[-1].set_xlabel("robot time [s]")
    fig.suptitle(f"{args.factor} on at {args.t_on:.0f} s, {N} deployments, {os.path.basename(args.checkpoint)}, arm {args.arm}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_vs_time.png"), dpi=120)
    print("[residual] wrote", os.path.join(out_dir, "residual_vs_time.png"))
except Exception as e:  # noqa: BLE001
    print("[residual] no plot:", e)
env.close()
app.close()
