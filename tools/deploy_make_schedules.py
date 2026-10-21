# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate and list the seeded change schedules (decisions.md E1) for a given T50; writes JSON + a markdown
table. No Isaac Sim needed.

  python tools/deploy_make_schedules.py --t50_min 2.0 --seeds 0 1 2 3 4 5 6 7 8 9 --out_dir schedules/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hexapod.deploy import cfg as C  # noqa: E402
from hexapod.deploy.schedule import generate_schedule  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--t50_min", type=float, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()
cfg = C.ScheduleCfg(t50_min=args.t50_min)
os.makedirs(args.out_dir, exist_ok=True)
md = [f"# Change schedules, T50 = {args.t50_min} min, mean gap {cfg.mean_gap_factor * args.t50_min:.1f} min, length {(cfg.warmup_s + cfg.n_changes_target * cfg.mean_gap_factor * args.t50_min * 60) / 60:.1f} min", "", "| seed | changes | nominal frac | recurrence | events (min: factor on/off) |", "|---|---|---|---|---|"]
for seed in args.seeds:
    s = generate_schedule(seed, cfg)
    s.save(os.path.join(args.out_dir, f"schedule_seed{seed}.json"))
    md.append(f"| {seed} | {len(s.events)} | {s.nominal_fraction():.2f} | {s.has_recurrence()} | " + ", ".join(f"{e.t_s / 60:.1f}: {e.factor} {'on' if e.on else 'off'}" for e in s.events) + " |")
open(os.path.join(args.out_dir, "schedules.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
