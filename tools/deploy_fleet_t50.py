# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T50 of k-robot fleets (k robots share one policy) from chained deploy_sweep_updater jobs: the jobs' per-cycle
progress streams are concatenated in time, then per replicate r (fleet r of the adapting group, control fleet r on the
same cells), as deploy_t50 defines them:

  R0  nominal rate before the change (fleet + control, m/min per robot)
  Rf  the control fleet's rate after the change
  T50 first adaptation time (after the first window) at which the fleet's trailing-window rate (pooled over its
      robots) reaches Rf + 0.5 (R0 - Rf)

Also the gap fraction closed in the last window. No Isaac Sim needed.

  python tools/deploy_fleet_t50.py <job_00> <job_01> ... --group onpolicy --control control --t_on 30
"""

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("jobs", nargs="+", help="job out dirs in order (streams.npz + meta.json each)")
parser.add_argument("--group", default="onpolicy")
parser.add_argument("--control", default="control")
parser.add_argument("--t_on", type=float, required=True, help="change time in job 0 [s]")
parser.add_argument("--window_min", type=float, default=1.0)
parser.add_argument("--out", default=None)
args = parser.parse_args()

prog, ts, meta0 = [], [], None
offset = 0.0
for j in args.jobs:
    z = np.load(os.path.join(j, "streams.npz"))
    m = json.load(open(os.path.join(j, "meta.json")))
    meta0 = meta0 or m
    if m["group_of_env"] != meta0["group_of_env"] or m["replicate_of_env"] != meta0["replicate_of_env"]:
        raise ValueError(f"{j}: robot layout differs from {args.jobs[0]}")
    prog.append(z["progress_m"])
    ts.append(z["t_s"][:, 0] + offset)
    offset += float(z["t_s"][-1, 0])
prog, t = np.concatenate(prog), np.concatenate(ts)
dt_c = t[1] - t[0]
names = [g["name"] for g in meta0["groups"]]
gi, ci = names.index(args.group), names.index(args.control)
genv, renv = np.array(meta0["group_of_env"]), np.array(meta0["replicate_of_env"])
before = t <= args.t_on + 1e-6
after = ~before
win = int(round(args.window_min * 60.0 / dt_c))
rows = []
for r in sorted(set(renv.tolist())):
    fl = (genv == gi) & (renv == r)
    co = (genv == ci) & (renv == r)
    rate = lambda mask, sel: prog[sel][:, mask].sum() / (sel.sum() * dt_c / 60.0) / mask.sum()
    R0 = 0.5 * (rate(fl, before) + rate(co, before))
    Rf = rate(co, after)
    target = Rf + 0.5 * (R0 - Rf)
    fl_c = prog[:, fl].mean(1)  # per-cycle progress per robot, fleet mean
    idx_after = np.where(after)[0]
    t50, last = None, None
    for k in range(win, len(idx_after) + 1):
        w = idx_after[k - win : k]
        rw = fl_c[w].sum() / (win * dt_c / 60.0)
        last = rw
        if t50 is None and rw >= target:
            t50 = float(t[w[-1]] - args.t_on)
    adapt_min = float((t[-1] - args.t_on) / 60.0)
    R0, Rf, target = float(R0), float(Rf), float(target)
    last = None if last is None else float(last)
    rows.append({"replicate": int(r), "R0": R0, "Rf": Rf, "target": target, "T50_s": t50, "adapt_min": adapt_min, "last_window_rate": last,
                 "gap_closed_last_window": None if last is None else float((last - Rf) / (R0 - Rf)), "frozen_drop": float(1 - Rf / R0)})
md = ["| seed (fleet) | R0 nominal | Rf frozen after | frozen drop | target (50%) | T50 | adapt-min run | gap closed, last 1-min window |", "|---|---|---|---|---|---|---|---|"]
for x in rows:
    md.append(f"| {x['replicate']} | {x['R0']:.2f} | {x['Rf']:.2f} | {x['frozen_drop']:.0%} | {x['target']:.2f} | {'%.1f min' % (x['T50_s'] / 60) if x['T50_s'] is not None else 'not reached'} | {x['adapt_min']:.2f} | {x['gap_closed_last_window']:+.0%} |")
print("\n".join(md))
if args.out:
    json.dump(rows, open(args.out + ".json", "w"), indent=1)
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
