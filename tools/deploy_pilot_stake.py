# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stake of a test value for a frozen pi_0 from deploy_t50 frozen-arm runs (``<out>/value_x/never`` or ``<out>/never``):
progress-rate drop after the change against the same robots before it, per terrain type and type-balanced, with a
bootstrap 95% CI over robots (resampled within each type), and falls per robot-minute after the change. The protocol
of the 2026-10-26 pilots on 7299: before = t <= t_on, after = the last --after_min minutes. No Isaac Sim needed.

  python tools/deploy_pilot_stake.py <run>/value_0.1/never [<run2>/never ...] --after_min 2 [--labels friction_0.1 ...]
"""

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--labels", nargs="*", default=None)
parser.add_argument("--after_min", type=float, required=True, help="the last minutes of the run used for the after-change rate")
parser.add_argument("--boot", type=int, default=2000)
parser.add_argument("--out", default=None, help="write a .json and .md here (path without extension)")
args = parser.parse_args()
rng = np.random.default_rng(0)


def stake(run):
    z = np.load(os.path.join(run, "streams.npz"))
    meta = json.load(open(os.path.join(run, "meta.json")))
    t, prog, falls = z["t_s"][:, 0], z["progress_m"], z["falls"]
    t_on = float(meta["t_on"])
    before = t <= t_on + 1e-6
    after = t > t[-1] - args.after_min * 60.0 + 1e-6
    rb = prog[before].sum(0) / (before.sum() * (t[1] - t[0]) / 60.0)  # m/min per robot
    ra = prog[after].sum(0) / (after.sum() * (t[1] - t[0]) / 60.0)
    fa = falls[after].sum(0) / (after.sum() * (t[1] - t[0]) / 60.0)
    types = np.array(meta["types_per_env"])
    names = sorted(set(types.tolist()))
    rows = {}
    boots_tb = np.zeros(args.boot)
    for nm in names:
        m = types == nm
        b, a, f = rb[m], ra[m], fa[m]
        drop = 1.0 - a.mean() / b.mean()
        idx = rng.integers(0, m.sum(), size=(args.boot, m.sum()))
        bd = 1.0 - a[idx].mean(1) / b[idx].mean(1)
        boots_tb += bd / len(names)
        rows[nm] = {"robots": int(m.sum()), "rate_before": float(b.mean()), "rate_after": float(a.mean()), "drop": float(drop), "drop_ci": [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))], "falls_per_robot_min_after": float(f.mean())}
    tb = float(np.mean([r["drop"] for r in rows.values()]))
    return {"run": run, "t_on": t_on, "after_min": args.after_min, "robots": int(len(types)), "types": rows, "type_balanced_drop": tb,
            "type_balanced_ci": [float(np.quantile(boots_tb, 0.025)), float(np.quantile(boots_tb, 0.975))],
            "falls_per_robot_min_after": float(fa.mean()), "falls_before": float(falls[before].sum())}


res = [stake(r) for r in args.runs]
labels = args.labels or [os.path.basename(os.path.dirname(r.rstrip("/"))) for r in args.runs]
names = sorted(res[0]["types"])
md = ["| terrain type | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
for nm in names:
    md.append(f"| {nm} ({res[0]['types'][nm]['robots']} robots) | " + " | ".join(f"{r['types'][nm]['drop']:.0%} [{r['types'][nm]['drop_ci'][0]:.0%}, {r['types'][nm]['drop_ci'][1]:.0%}] / {r['types'][nm]['falls_per_robot_min_after']:.2f}" for r in res) + " |")
md.append("| **type-balanced** | " + " | ".join(f"**{r['type_balanced_drop']:.0%}** [{r['type_balanced_ci'][0]:.0%}, {r['type_balanced_ci'][1]:.0%}] / {r['falls_per_robot_min_after']:.2f}" for r in res) + " |")
md.append("")
md.append("Cells: progress-rate drop after the change vs before, same robots [bootstrap 95% CI over robots] / falls per robot-minute after. "
          + "; ".join(f"{l}: rate before {np.mean([v['rate_before'] for v in r['types'].values()]):.2f} m/min, falls before {r['falls_before']:.0f}" for l, r in zip(labels, res)))
print("\n".join(md))
if args.out:
    json.dump(dict(zip(labels, res)), open(args.out + ".json", "w"), indent=1)
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
