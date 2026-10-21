# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize deployment runs (decisions.md E2 / E4): one markdown table across arms x pi_0 seeds x schedules with
bootstrap CIs over deployments. Regret = oracle - arm progress, matched on (checkpoint, schedule seed, env index);
no-change cost = never - arm on nominal time, matched the same way. The hindsight-best schedule is the
scheduled(T) period with the highest mean progress (a post-hoc pick, not a run). No Isaac Sim needed.

  python tools/deploy_summarize.py --root /workspace/isaaclab/logs/deploy --out summary.md
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.metrics import bootstrap_ci, deployment_metrics, fmt_ci  # noqa: E402
from hexapod.deploy.schedule import Schedule  # noqa: E402
from hexapod.deploy.streams import StreamLog  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, help="directory of deploy_run.py outputs (searched recursively)")
parser.add_argument("--out", default=None, help="markdown output (default <root>/summary.md)")
parser.add_argument("--detect_window_min", type=float, default=None)
parser.add_argument("--bootstrap_n", type=int, default=1000)
args = parser.parse_args()
mcfg = C.MetricsCfg(detect_window_min=args.detect_window_min, bootstrap_n=args.bootstrap_n)

rows = []  # one per deployment
for meta_path in sorted(glob.glob(os.path.join(args.root, "**", "meta.json"), recursive=True)):
    run_dir = os.path.dirname(meta_path)
    if not os.path.exists(os.path.join(run_dir, "streams.npz")):
        continue
    arrays, meta = StreamLog.load(run_dir)
    if "arm" not in meta or "env_schedule_seed" not in meta:
        continue
    budget = (meta.get("arm_cfg") or {}).get("budget_min") or (meta.get("updater_cfg") or {}).get("budget_min") or 1.0
    scheds = {}
    for i, seed in enumerate(meta["env_schedule_seed"]):
        if seed is None:
            s = None
        else:
            if seed not in scheds:
                scheds[seed] = Schedule.load(os.path.join(run_dir, f"schedule_seed{seed}.json"))
            s = scheds[seed]
        m = deployment_metrics(arrays, i, s, mcfg, budget)
        if not m:
            continue
        m.update({"arm": meta["arm"], "checkpoint": os.path.basename(os.path.dirname(meta["checkpoint"])) + "/" + os.path.basename(meta["checkpoint"]), "schedule_seed": seed, "env": i, "env_seed": (meta.get("deploy_cfg") or {}).get("env_seed", 0), "period_min": (meta.get("arm_cfg") or {}).get("period_min"), "run_dir": run_dir})
        rows.append(m)
print(f"[summarize] {len(rows)} deployments from {len({r['run_dir'] for r in rows})} runs")


def key(r):
    return (r["checkpoint"], r["schedule_seed"], r["env_seed"], r["env"])


oracle = {key(r): r["progress_m"] for r in rows if r["arm"] == "oracle"}
never = {key(r): r for r in rows if r["arm"] == "never"}
for r in rows:
    k = key(r)
    r["regret_m"] = oracle[k] - r["progress_m"] if k in oracle else float("nan")
    r["no_change_cost_per_h"] = never[k]["no_change_progress_per_h"] - r["no_change_progress_per_h"] if k in never else float("nan")

arm_label = lambda r: r["arm"] + (f"(T={r['period_min']:g})" if r["arm"] in ("scheduled", "periodic_reset") and r["period_min"] is not None else "")
groups: dict[str, list] = {}
for r in rows:
    groups.setdefault(arm_label(r), []).append(r)
# hindsight-best schedule: the scheduled(T) with the highest mean progress
sched_groups = {g: v for g, v in groups.items() if g.startswith("scheduled(")}
if sched_groups:
    best = max(sched_groups, key=lambda g: np.mean([r["progress_m"] for r in sched_groups[g]]))
    groups[f"hindsight_best_schedule[{best}]"] = sched_groups[best]

cols = [("progress_m", "progress [m]"), ("regret_m", "regret vs oracle [m]"), ("falls_per_h", "falls/h"), ("track_err", "tracking err [m/s]"), ("false_alarms_per_h", "false alarms/h"), ("detection_delay_s", "detection delay [s]"), ("missed", "missed changes"), ("no_change_cost_per_h", "no-change cost [m/h]"), ("recurrence_gain_m", "recurrence gain [m]"), ("adapt_min", "adapt time [min]")]
order = ["never", "always", "scheduled", "periodic_reset", "pause_hold", "three_sigma", "cusum", "triggered", "oracle", "hindsight"]
names = sorted(groups, key=lambda g: next((i for i, o in enumerate(order) if g.startswith(o)), 99))
md = [f"# Deployment summary ({len(rows)} deployments, {len({r['checkpoint'] for r in rows})} pi_0 checkpoints, {len({r['schedule_seed'] for r in rows})} schedules)", "",
      "mean [95% bootstrap CI over deployments]; regret and no-change cost are matched per (checkpoint, schedule, env).", "",
      "| arm | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
for g in names:
    rs = groups[g]
    md.append(f"| {g} | {len(rs)} | " + " | ".join(fmt_ci([r[c] for r in rs], args.bootstrap_n) for c, _ in cols) + " |")
md += ["", "## Per checkpoint x schedule (progress [m], mean over envs)", ""]
cks = sorted({r["checkpoint"] for r in rows})
seeds = sorted({str(r["schedule_seed"]) for r in rows})
md.append("| arm | checkpoint | " + " | ".join(f"sched {s}" for s in seeds) + " |")
md.append("|---|---|" + "---|" * len(seeds))
for g in names:
    for ck in cks:
        vals = []
        for s in seeds:
            v = [r["progress_m"] for r in groups[g] if r["checkpoint"] == ck and str(r["schedule_seed"]) == s]
            vals.append(f"{np.mean(v):.1f}" if v else "–")
        if any(v != "–" for v in vals):
            md.append(f"| {g} | {ck} | " + " | ".join(vals) + " |")
out = args.out or os.path.join(args.root, "summary.md")
open(out, "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.splitext(out)[0] + ".json", "w"), indent=1, default=float)
print("\n".join(md))
print("[summarize] wrote", out)
